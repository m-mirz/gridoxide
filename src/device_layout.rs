//! Host-side flattening for the CUDA batch path — the plain-data half of
//! `src/gpu.rs`, deliberately kept out of it.
//!
//! Everything here turns gridoxide's ordinary CPU-side structures
//! ([`YBusSparse`]'s adjacency lists, [`Bus`]'s `zip_terms`, a batch's
//! per-scenario `Vec<Bus>` states) into the flat, strided, index-typed arrays
//! a CUDA kernel can read. None of it touches CUDA, and **none of it is
//! feature-gated**, which is the point: the GPU path is developed on machines
//! with no NVIDIA GPU (see `scripts/GPU_RUNBOOK.md`), where nothing behind
//! `gpu`/`cudss` even compiles. Every layout decision the kernels depend on is
//! therefore made here, where an ordinary `cargo test` can check it against
//! the CPU implementation that is already cross-validated against five other
//! solvers.
//!
//! The tests at the bottom are the contract:
//!
//! - [`YbusCsr`] reproduces [`YBusSparse::mul_vec`], so
//!   `cuda/gridoxide_kernels.cu`'s `go_power_injections` reproduces
//!   [`network::power_injections`](crate::network::power_injections).
//! - [`ZipCoeffs`] reproduces
//!   [`network::effective_injection`](crate::network::effective_injection) at
//!   any voltage, so the mismatch kernel can evaluate the ZIP model with three
//!   fused multiply-adds instead of a per-bus loop over a `Vec<ZipTerm>`.
//! - [`FlatStates`] round-trips a batch's `Vec<Vec<Bus>>` through the
//!   scenario-major layout every kernel indexes by.

use num_complex::Complex;

use crate::network::YBusSparse;
use crate::types::{Bus, BusType, ZipKind};

/// The Y-bus in the CSR form the device SpMV wants: `i32` indices (cuDSS's
/// and cuSPARSE's convention, and plenty for gridoxide's network sizes), with
/// the complex values split into separate real and imaginary arrays so the
/// kernel's loads stay coalesced instead of strided by two.
///
/// [`YBusSparse`] already stores each row's `(col, admittance)` pairs sorted
/// by column ([`YBusSparse::row`]'s own contract), so this is a concatenation,
/// not a sort — and the resulting row order is identical to the one
/// [`YBusSparse::mul_vec`] sums in, which is what
/// [`csr_matches_mul_vec`](self::tests::csr_matches_mul_vec) pins down.
#[derive(Clone, Debug, PartialEq)]
pub struct YbusCsr {
    pub n: usize,
    pub row_ptr: Vec<i32>,
    pub col_idx: Vec<i32>,
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl YbusCsr {
    pub fn from_ybus(ybus: &YBusSparse) -> Self {
        let n = ybus.n();
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut re = Vec::new();
        let mut im = Vec::new();

        row_ptr.push(0i32);
        for i in 0..n {
            for &(col, y) in ybus.row(i) {
                col_idx.push(col as i32);
                re.push(y.re);
                im.push(y.im);
            }
            row_ptr.push(col_idx.len() as i32);
        }

        Self { n, row_ptr, col_idx, re, im }
    }

    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// The CPU twin of `go_power_injections`, in the kernel's exact
    /// accumulation order: `I = Y·V` row by row, then `S = V ⊙ conj(I)`.
    ///
    /// Exists so the kernel has a reference that can be diffed without a GPU.
    /// It is *not* the production path —
    /// [`network::power_injections`](crate::network::power_injections) is —
    /// and the two need not agree bit-for-bit, since `faer`'s column-major
    /// SpMV sums each output in a different order than a row-major one does.
    pub fn power_injections_reference(&self, vm: &[f64], va: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        let mut p = vec![0.0; n];
        let mut q = vec![0.0; n];
        for k in 0..n {
            let (mut i_re, mut i_im) = (0.0f64, 0.0f64);
            for slot in self.row_ptr[k] as usize..self.row_ptr[k + 1] as usize {
                let j = self.col_idx[slot] as usize;
                let (v_re, v_im) = (vm[j] * va[j].cos(), vm[j] * va[j].sin());
                i_re += self.re[slot] * v_re - self.im[slot] * v_im;
                i_im += self.re[slot] * v_im + self.im[slot] * v_re;
            }
            let (v_re, v_im) = (vm[k] * va[k].cos(), vm[k] * va[k].sin());
            // S = V * conj(I)
            p[k] = v_re * i_re + v_im * i_im;
            q[k] = v_im * i_re - v_re * i_im;
        }
        (p, q)
    }
}

/// A bus's ZIP load model, collapsed from a `Vec<ZipTerm>` into three complex
/// coefficients per bus.
///
/// [`network::effective_injection`](crate::network::effective_injection)
/// evaluates `S_spec + Σ_terms f(kind, |V|)` where `f` is one of `s`, `s·|V|`,
/// `s·|V|²`. Since the sum is linear in the terms and the three basis
/// functions are fixed, summing each kind's `s_const` once at setup gives the
/// identical result from a branch-free
///
/// ```text
/// p_eff = p_spec + p_const + p_curr·vm + p_imp·vm²
/// ```
///
/// which is what the mismatch kernel evaluates. The coefficients are
/// **per-bus, not per-scenario**: [`BusOverride`](crate::batch::BusOverride)
/// can only change `p_spec`/`q_spec`/`voltage_mag`, never `zip_terms`, so
/// these are shared by the whole batch and uploaded once.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZipCoeffs {
    pub p_const: Vec<f64>,
    pub q_const: Vec<f64>,
    pub p_curr: Vec<f64>,
    pub q_curr: Vec<f64>,
    pub p_imp: Vec<f64>,
    pub q_imp: Vec<f64>,
}

impl ZipCoeffs {
    pub fn from_buses(buses: &[Bus]) -> Self {
        let n = buses.len();
        let mut out = Self {
            p_const: vec![0.0; n],
            q_const: vec![0.0; n],
            p_curr: vec![0.0; n],
            q_curr: vec![0.0; n],
            p_imp: vec![0.0; n],
            q_imp: vec![0.0; n],
        };
        for (b, i) in buses.iter().zip(0..n) {
            let mut c = Complex::new(0.0, 0.0);
            let mut curr = Complex::new(0.0, 0.0);
            let mut imp = Complex::new(0.0, 0.0);
            for zt in &b.zip_terms {
                match zt.kind {
                    ZipKind::ConstPower => c += zt.s_const,
                    ZipKind::ConstCurrent => curr += zt.s_const,
                    ZipKind::ConstImpedance => imp += zt.s_const,
                }
            }
            out.p_const[i] = c.re;
            out.q_const[i] = c.im;
            out.p_curr[i] = curr.re;
            out.q_curr[i] = curr.im;
            out.p_imp[i] = imp.re;
            out.q_imp[i] = imp.im;
        }
        out
    }

    /// The CPU twin of the kernel's ZIP evaluation, for tests. `p_spec`/
    /// `q_spec` come from the *scenario's* bus, the coefficients from the
    /// shared template.
    pub fn effective(&self, bus: usize, vm: f64, p_spec: f64, q_spec: f64) -> (f64, f64) {
        let vm2 = vm * vm;
        (
            p_spec + self.p_const[bus] + self.p_curr[bus] * vm + self.p_imp[bus] * vm2,
            q_spec + self.q_const[bus] + self.q_curr[bus] * vm + self.q_imp[bus] * vm2,
        )
    }
}

/// Which physical bus each unknown belongs to, as `u32` device arrays.
///
/// Identical in content to the `non_slack_idx`/`pq_idx` vectors
/// [`crate::bde`] and [`crate::jacobian::JacobianPattern`] build — the same
/// ordering, so unknown `r < n_angle` is angle `non_slack[r]` and unknown
/// `n_angle + r` is magnitude `pq[r]`, exactly as the CPU Newton loop indexes
/// its right-hand side.
#[derive(Clone, Debug, PartialEq)]
pub struct UnknownMaps {
    pub non_slack: Vec<u32>,
    pub pq: Vec<u32>,
}

impl UnknownMaps {
    pub fn from_buses(buses: &[Bus]) -> Self {
        Self {
            non_slack: buses
                .iter()
                .filter(|b| !matches!(b.bus_type, BusType::Slack))
                .map(|b| b.idx as u32)
                .collect(),
            pq: buses.iter().filter(|b| matches!(b.bus_type, BusType::PQ)).map(|b| b.idx as u32).collect(),
        }
    }

    pub fn n_angle(&self) -> usize {
        self.non_slack.len()
    }

    /// Unknowns per scenario — the block stride, matching
    /// [`crate::bde::BlockDiagonal::block_size`].
    pub fn block_size(&self) -> usize {
        self.non_slack.len() + self.pq.len()
    }
}

/// A batch's per-scenario bus state, flattened scenario-major
/// (`scenario * n_buses + bus`) — the stride every kernel in
/// `cuda/gridoxide_kernels.cu` assumes, and the same one today's assembly
/// kernel already uses.
///
/// `p_spec`/`q_spec` are flattened from the *already-constructed* scenario
/// states rather than re-derived from `&[Scenario]`, so there is exactly one
/// place where a [`BusOverride`](crate::batch::BusOverride) is applied and no
/// way for the device copy to drift from what the CPU path would have used.
#[derive(Clone, Debug, PartialEq)]
pub struct FlatStates {
    pub n_buses: usize,
    pub n_scenarios: usize,
    pub vm: Vec<f64>,
    pub va: Vec<f64>,
    pub p_spec: Vec<f64>,
    pub q_spec: Vec<f64>,
}

impl FlatStates {
    pub fn from_states(states: &[Vec<Bus>]) -> Self {
        let n_scenarios = states.len();
        let n_buses = states.first().map_or(0, |s| s.len());
        assert!(
            states.iter().all(|s| s.len() == n_buses),
            "every scenario's bus array must be the same length"
        );

        let total = n_scenarios * n_buses;
        let mut out = Self {
            n_buses,
            n_scenarios,
            vm: Vec::with_capacity(total),
            va: Vec::with_capacity(total),
            p_spec: Vec::with_capacity(total),
            q_spec: Vec::with_capacity(total),
        };
        for s in states {
            out.vm.extend(s.iter().map(|b| b.voltage_mag));
            out.va.extend(s.iter().map(|b| b.voltage_ang));
            out.p_spec.extend(s.iter().map(|b| b.p_spec));
            out.q_spec.extend(s.iter().map(|b| b.q_spec));
        }
        out
    }

    /// Writes `vm`/`va` back into per-scenario bus arrays — the one readback
    /// the device-resident loop does, once, after convergence.
    pub fn write_voltages_into(&self, states: &mut [Vec<Bus>]) {
        assert_eq!(states.len(), self.n_scenarios);
        for (s, buses) in states.iter_mut().enumerate() {
            let base = s * self.n_buses;
            for (i, b) in buses.iter_mut().enumerate() {
                b.voltage_mag = self.vm[base + i];
                b.voltage_ang = self.va[base + i];
            }
        }
    }
}

/// The array of device pointers a uniform-batch API wants: `count` pointers
/// `count` strides apart into one contiguous allocation.
///
/// cuDSS's `cudssMatrixCreateBatchCsr`/`cudssMatrixCreateBatchDn` take
/// `void**` for the values, row-start and column-index arrays. Because
/// block-diagonal embedding makes every block *uniform* — identical size,
/// identical sparsity pattern — the structure pointers degenerate to `count`
/// copies of one pointer ([`repeat_device_ptr`]) and only the values stride.
///
/// Pure arithmetic on purpose: it is the one piece of the batched-matrix
/// setup that can be wrong in a way no compiler catches, and it is checked by
/// [`strided_ptrs_are_evenly_spaced`](self::tests::strided_ptrs_are_evenly_spaced)
/// with no GPU present.
pub fn strided_device_ptrs(base: u64, stride_bytes: usize, count: usize) -> Vec<u64> {
    (0..count).map(|s| base + (s * stride_bytes) as u64).collect()
}

/// `count` copies of one device pointer — the uniform batch's shared CSR
/// structure, where every block reads the *same* `row_ptr`/`col_idx`.
pub fn repeat_device_ptr(ptr: u64, count: usize) -> Vec<u64> {
    vec![ptr; count]
}

/// Builds a CSR structure (row pointers + sorted column indices) from a set of
/// `(row, col)` index pairs, merging duplicates — identical accumulation
/// semantics to `sparse_pardiso::build_csr_structure` (row-major, the same
/// duplicate-summing convention `sparse_klu`'s CSC builder uses).
///
/// Returns `(row_ptr, col_idx, groups)`, where `groups[k]` lists the original
/// entry indices contributing to the `k`-th CSR position.
///
/// Lives here rather than in `sparse_cudss` (where it started) so it is
/// reachable — and testable — without the `cudss` feature and without a GPU:
/// the CSR layout is what *both* the cuDSS matrix and the assembly kernel's
/// scatter map are derived from, so getting it wrong is a whole-batch failure
/// and it deserves to be checked on every `cargo test`, not only on a box with
/// cuDSS installed.
pub fn build_csr_structure(n: usize, pairs: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>, Vec<Vec<usize>>) {
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&i| (pairs[i].0, pairs[i].1));

    let mut row_ptr = vec![0i32; n + 1];
    let mut col_idx: Vec<i32> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();

    let mut idx = 0;
    while idx < order.len() {
        let (row, col) = pairs[order[idx]];
        let mut group = vec![order[idx]];
        let mut k = idx + 1;
        while k < order.len() && pairs[order[k]] == (row, col) {
            group.push(order[k]);
            k += 1;
        }
        col_idx.push(col as i32);
        groups.push(group);
        row_ptr[row + 1] += 1;
        idx = k;
    }
    for r in 0..n {
        row_ptr[r + 1] += row_ptr[r];
    }
    (row_ptr, col_idx, groups)
}

/// For a **single block's** `(row, col)` pairs in [`JacobianPattern`] entries
/// order, computes the CSR position each entry maps to — the inverse of
/// [`build_csr_structure`]'s grouping. This is what lets the assembly kernel
/// write its per-entry outputs directly at the CSR position cuDSS expects, in
/// one pass, with no host-side reorder.
///
/// [`JacobianPattern`]: crate::jacobian::JacobianPattern
///
/// Only defined when every CSR position merges exactly one original entry —
/// true for every gridoxide Jacobian pattern, and true by construction:
/// [`YBusSparse`] never has a duplicate `(row, col)` neighbor (parallel lines
/// are already summed at Y-bus assembly), so `JacobianPattern`, which walks
/// each unknown row's Y-bus neighbors exactly once, can't produce a duplicate
/// `(row, col)` triplet either. Panics otherwise — a real merge would need
/// atomic adds in the kernel this map feeds, not a plain permutation, and
/// silently dropping a contribution would be a much worse failure than a panic.
///
/// Because block-diagonal embedding gives every scenario's block the exact
/// same relative `(row, col)` structure, just offset by `s * block_size`
/// ([`crate::bde::BlockDiagonal::analyze`]), sorting the *whole* stacked
/// matrix's pairs by `(row, col)` reproduces this same single-block
/// permutation inside each scenario's own contiguous `nnz`-sized segment — so
/// one single-block scatter map is all any batch size needs, and under cuDSS's
/// *uniform batch* API a single block's CSR structure is all that is uploaded
/// at all.
pub fn csr_scatter_map(n: usize, pairs: &[(usize, usize)]) -> Vec<u32> {
    let (_, _, groups) = build_csr_structure(n, pairs);
    let mut scatter = vec![0u32; pairs.len()];
    for (csr_pos, group) in groups.iter().enumerate() {
        assert_eq!(
            group.len(),
            1,
            "CSR position {csr_pos} merges {} entries; the device scatter map assumes a pure permutation",
            group.len()
        );
        scatter[group[0]] = csr_pos as u32;
    }
    scatter
}

/// Permutes a values array from [`JacobianPattern`] entries order into CSR
/// order, summing any merged positions.
///
/// [`JacobianPattern`]: crate::jacobian::JacobianPattern
pub fn pack_values_slice(values: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    groups.iter().map(|g| g.iter().map(|&i| values[i]).sum()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::NetworkData;
    use crate::network::{build_ybus, effective_injection, linear_initial_guess, power_injections};
    use crate::types::ZipTerm;
    use std::fs;
    use std::path::PathBuf;

    fn load_network() -> NetworkData {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/network.json");
        let raw = fs::read_to_string(path).expect("read network.json");
        serde_json::from_str(&raw).expect("parse network.json")
    }

    /// The load-bearing claim under `go_power_injections`: concatenating
    /// `YBusSparse::row`'s adjacency lists yields a CSR matrix whose SpMV is
    /// the same operator `mul_vec` applies, hence the same `power_injections`.
    #[test]
    fn csr_matches_mul_vec() {
        let network = load_network();
        let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();
        let csr = YbusCsr::from_ybus(&ybus);

        let mut buses = network.buses.clone();
        linear_initial_guess(&mut buses, &ybus);
        let vm: Vec<f64> = buses.iter().map(|b| b.voltage_mag).collect();
        let va: Vec<f64> = buses.iter().map(|b| b.voltage_ang).collect();

        let (p_ref, q_ref) = power_injections(&buses, &ybus);
        let (p_csr, q_csr) = csr.power_injections_reference(&vm, &va);

        assert_eq!(p_csr.len(), p_ref.len());
        let worst = p_csr
            .iter()
            .zip(&p_ref)
            .chain(q_csr.iter().zip(&q_ref))
            .map(|(&a, &b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-14, "CSR power injections disagree with power_injections (worst rel {worst:.3e})");
    }

    #[test]
    fn csr_row_ptr_is_monotone_and_covers_every_nonzero() {
        let network = load_network();
        let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();
        let csr = YbusCsr::from_ybus(&ybus);

        assert_eq!(csr.row_ptr.len(), csr.n + 1);
        assert_eq!(csr.row_ptr[0], 0);
        assert_eq!(*csr.row_ptr.last().unwrap() as usize, csr.nnz());
        assert_eq!(csr.re.len(), csr.nnz());
        assert_eq!(csr.im.len(), csr.nnz());
        for i in 0..csr.n {
            assert!(csr.row_ptr[i] <= csr.row_ptr[i + 1], "row_ptr not monotone at row {i}");
            assert_eq!(
                (csr.row_ptr[i + 1] - csr.row_ptr[i]) as usize,
                ybus.row(i).len(),
                "row {i} has the wrong nonzero count"
            );
        }
    }

    /// The ZIP flattening must reproduce `effective_injection` at *any*
    /// voltage, not just the one it was built at — the mismatch kernel
    /// re-evaluates it against the live iterate every Newton step.
    #[test]
    fn zip_coeffs_match_effective_injection() {
        let network = load_network();
        let mut buses = network.buses.clone();

        // The committed fixture has no ZIP terms, so plant a mixed set: this
        // test would otherwise pass on the `zip_terms.is_empty()` path alone
        // and prove nothing about the three basis functions.
        buses[1].zip_terms = vec![
            ZipTerm { s_const: Complex::new(-0.2, 0.05), kind: ZipKind::ConstPower },
            ZipTerm { s_const: Complex::new(-0.1, -0.03), kind: ZipKind::ConstCurrent },
            ZipTerm { s_const: Complex::new(0.07, 0.11), kind: ZipKind::ConstImpedance },
            // Two of the same kind, to check they sum rather than overwrite.
            ZipTerm { s_const: Complex::new(0.01, -0.02), kind: ZipKind::ConstCurrent },
        ];

        let zip = ZipCoeffs::from_buses(&buses);

        for &vm in &[0.5f64, 0.85, 1.0, 1.07, 1.4] {
            for (i, b) in buses.iter().enumerate() {
                let mut probe = b.clone();
                probe.voltage_mag = vm;
                let (p_ref, q_ref) = effective_injection(&probe);
                let (p_got, q_got) = zip.effective(i, vm, probe.p_spec, probe.q_spec);
                assert!(
                    (p_got - p_ref).abs() < 1e-15 && (q_got - q_ref).abs() < 1e-15,
                    "bus {i} at vm={vm}: got ({p_got}, {q_got}), want ({p_ref}, {q_ref})"
                );
            }
        }
    }

    #[test]
    fn unknown_maps_match_the_newton_loops_own_indexing() {
        let network = load_network();
        let maps = UnknownMaps::from_buses(&network.buses);

        let non_slack: Vec<u32> = network
            .buses
            .iter()
            .filter(|b| !matches!(b.bus_type, BusType::Slack))
            .map(|b| b.idx as u32)
            .collect();
        let pq: Vec<u32> =
            network.buses.iter().filter(|b| matches!(b.bus_type, BusType::PQ)).map(|b| b.idx as u32).collect();

        assert_eq!(maps.non_slack, non_slack);
        assert_eq!(maps.pq, pq);
        assert_eq!(maps.block_size(), non_slack.len() + pq.len());
    }

    #[test]
    fn flat_states_round_trip() {
        let network = load_network();
        let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();

        let mut states: Vec<Vec<Bus>> = (0..4)
            .map(|s| {
                let mut buses = network.buses.clone();
                let f = 0.7 + 0.15 * s as f64;
                for b in buses.iter_mut() {
                    b.p_spec *= f;
                    b.q_spec *= f;
                }
                linear_initial_guess(&mut buses, &ybus);
                buses
            })
            .collect();

        let flat = FlatStates::from_states(&states);
        assert_eq!(flat.n_scenarios, 4);
        assert_eq!(flat.n_buses, network.buses.len());

        // Scenario-major stride: scenario s's slice is contiguous.
        for (s, buses) in states.iter().enumerate() {
            let base = s * flat.n_buses;
            for (i, b) in buses.iter().enumerate() {
                assert_eq!(flat.vm[base + i], b.voltage_mag);
                assert_eq!(flat.va[base + i], b.voltage_ang);
                assert_eq!(flat.p_spec[base + i], b.p_spec);
                assert_eq!(flat.q_spec[base + i], b.q_spec);
            }
        }

        // Writing back is the identity when nothing changed on the device.
        let before = states.clone();
        flat.write_voltages_into(&mut states);
        for (a, b) in states.iter().flatten().zip(before.iter().flatten()) {
            assert_eq!(a.voltage_mag, b.voltage_mag);
            assert_eq!(a.voltage_ang, b.voltage_ang);
        }
    }

    /// [`csr_scatter_map`] assumes every CSR position merges exactly one
    /// original `JacobianPattern` entry — no summing, a pure permutation.
    /// This holds by construction (see that function's doc comment), and this
    /// pins the invariant down on the committed fixture so a future change to
    /// `JacobianPattern`'s emission order or logic can't silently break the
    /// scatter kernel.
    ///
    /// Moved here from `sparse_cudss.rs` when the CSR helpers did: it used to
    /// run only with `--features cudss`, i.e. only on a box with cuDSS
    /// installed, which is the last place you want to first learn the
    /// permutation assumption broke.
    #[test]
    fn csr_groups_are_all_singletons() {
        let network = load_network();
        let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();
        let pattern = crate::jacobian::JacobianPattern::analyze(&network.buses, &ybus);

        let pairs: Vec<(usize, usize)> =
            pattern.rows().iter().zip(pattern.cols()).map(|(&r, &c)| (r as usize, c as usize)).collect();
        let (_, _, groups) = build_csr_structure(pattern.n_unknowns, &pairs);
        assert_eq!(pairs.len(), groups.len(), "some CSR position merges >1 original entry");
    }

    /// Pins down [`csr_scatter_map`]'s doc-comment claim numerically: one
    /// single-block scatter map, offset by `scenario * nnz`, correctly locates
    /// every scenario's entries in the *stacked* block-diagonal CSR structure.
    ///
    /// This is what licenses the uniform-batch shortcut — every block sharing
    /// one `row_ptr`/`col_idx` — so it is checked position by position against
    /// `build_csr_structure` run directly on the full stacked pairs, not just
    /// argued structurally.
    #[test]
    fn stacked_scatter_matches_per_block_assumption() {
        use crate::bde::BlockDiagonal;
        use crate::jacobian::JacobianPattern;

        let network = load_network();
        let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();

        let nb = 3usize;
        let bd = BlockDiagonal::analyze(&network.buses, &ybus, nb);
        let base = JacobianPattern::analyze(&network.buses, &ybus);
        let blk = base.n_unknowns;
        let nnz = base.len();

        let block_pairs: Vec<(usize, usize)> =
            base.rows().iter().zip(base.cols()).map(|(&r, &c)| (r as usize, c as usize)).collect();
        let scatter = csr_scatter_map(blk, &block_pairs);
        let (block_row_ptr, block_col_idx, _) = build_csr_structure(blk, &block_pairs);

        let dummy = vec![0.0f64; bd.len()];
        let full_pairs: Vec<(usize, usize)> = bd.to_triplets(&dummy).iter().map(|&(r, c, _)| (r, c)).collect();
        let (full_row_ptr, full_col_idx, full_groups) = build_csr_structure(bd.n_unknowns(), &full_pairs);

        assert_eq!(full_col_idx.len(), nnz * nb);
        assert!(full_groups.iter().all(|g| g.len() == 1));

        for s in 0..nb {
            for e in 0..nnz {
                let expected_col = block_col_idx[scatter[e] as usize] as usize + s * blk;
                let actual_col = full_col_idx[s * nnz + scatter[e] as usize] as usize;
                assert_eq!(expected_col, actual_col, "column mismatch at scenario {s}, entry {e}");
            }
            for r in 0..=blk {
                let expected = block_row_ptr[r] + (s * nnz) as i32;
                let actual = full_row_ptr[s * blk + r];
                assert_eq!(expected, actual, "row_ptr mismatch at scenario {s}, row {r}");
            }
        }
    }

    #[test]
    fn pack_values_slice_sums_merged_positions() {
        // Two entries landing on (0,0), one each on (0,1) and (1,0).
        let pairs = [(0usize, 0usize), (0, 1), (0, 0), (1, 0)];
        let (row_ptr, col_idx, groups) = build_csr_structure(2, &pairs);
        assert_eq!(row_ptr, vec![0, 2, 3]);
        assert_eq!(col_idx, vec![0, 1, 0]);

        let values = [1.5, 2.0, 0.5, 7.0];
        assert_eq!(pack_values_slice(&values, &groups), vec![2.0, 2.0, 7.0]);
    }

    #[test]
    fn strided_ptrs_are_evenly_spaced() {
        let base = 0x7f00_0000_0000u64;
        let ptrs = strided_device_ptrs(base, 4096, 5);
        assert_eq!(ptrs, vec![base, base + 4096, base + 8192, base + 12288, base + 16384]);

        // The degenerate uniform-structure case: every block reads the same
        // row_ptr/col_idx.
        assert_eq!(repeat_device_ptr(base, 3), vec![base, base, base]);
        assert_eq!(strided_device_ptrs(base, 0, 3), vec![base, base, base]);
        assert!(strided_device_ptrs(base, 8, 0).is_empty());
    }
}

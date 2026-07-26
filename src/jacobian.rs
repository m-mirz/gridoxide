//! Topology-derived Jacobian assembly: analyze the sparsity pattern once,
//! then refill values every iteration at fixed offsets.
//!
//! `solver::build_jacobian_triplets` rebuilds a fresh `Vec<(usize, usize,
//! f64)>` every Newton iteration, re-deriving row and column indices that are
//! by construction identical each time — the sparsity pattern is fixed for
//! the lifetime of a topology, which is the very property
//! `solver::LinearSolver` relies on to cache its symbolic factorization.
//! [`JacobianPattern`] hoists all of that out of the loop: the `(row, col)`
//! arrays and a per-entry "recipe" are computed once by
//! [`analyze`](JacobianPattern::analyze), and [`fill`](JacobianPattern::fill)
//! then writes only the `f64` values into a caller-owned buffer.
//!
//! Two payoffs:
//!
//! - **On the CPU**, per-iteration allocation and index arithmetic disappear.
//!   Jacobian assembly is 36–41% of iteration time on the cases measured in
//!   `plans/GPU_PLAN.md` §1.
//! - **On a GPU**, this *is* the kernel shape. [`Entry`] is a flat,
//!   fixed-size, branch-light record; `fill` is one independent write per
//!   entry into a preallocated array at a precomputed offset, with all reads
//!   gathers from small per-bus arrays. `plans/GPU_PLAN.md` §3 property 4
//!   ("assembly becomes one flat kernel") is exactly this layout, extended to
//!   a batch by adding a scenario stride.
//!
//! [`fill`] is a transliteration of `build_jacobian_triplets`' H/N/M/L
//! formulas, not a rederivation — `tests/jacobian_pattern_test.rs` asserts
//! the two agree bit-for-bit in f64.

use num_complex::Complex;

use crate::network::YBusSparse;
use crate::types::{Bus, BusType};

/// Which of the Jacobian's eight distinct formulas an [`Entry`] evaluates.
///
/// ```text
/// J = [ H  N ]   H = dP/d_ang, N = dP/d_vmag
///     [ M  L ]   M = dQ/d_ang, L = dQ/d_vmag
/// ```
///
/// Diagonal and off-diagonal cases are separate variants because they read
/// different inputs, not merely different indices: the diagonal terms depend
/// on `p_calc`/`q_calc` at the bus, the off-diagonal ones on the angle
/// difference to a neighbor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EntryKind {
    Hii = 0,
    Nii = 1,
    Hik = 2,
    Nik = 3,
    Mii = 4,
    Lii = 5,
    Mik = 6,
    Lik = 7,
}

/// One Jacobian nonzero's recipe. Deliberately flat and `Copy` — this is the
/// record a GPU kernel would read one-per-thread.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub kind: EntryKind,
    /// Row bus (physical index).
    pub i: u32,
    /// Column bus (physical index). Equals `i` for the diagonal kinds.
    pub k: u32,
    /// `Y[i][k]`, captured at analyze time. The Y-bus is fixed for the
    /// lifetime of a topology, so carrying the admittance here turns what
    /// would be a pointer-chase through `YBusSparse::row` into a value the
    /// entry already owns.
    pub y: Complex<f64>,
}

/// A topology's Jacobian sparsity pattern plus the per-entry recipes needed
/// to refill its values. Build once per topology via
/// [`analyze`](Self::analyze); call [`fill`](Self::fill) every iteration.
///
/// `rows`/`cols`/`entries` are parallel arrays of the same length, in the
/// exact emission order `solver::build_jacobian_triplets` used, so a cached
/// symbolic factorization keyed on that order stays valid.
pub struct JacobianPattern {
    pub n_unknowns: usize,
    rows: Vec<u32>,
    cols: Vec<u32>,
    entries: Vec<Entry>,
}

impl JacobianPattern {
    /// Derives the pattern from a topology and a bus-type assignment.
    ///
    /// Must be recomputed if either changes — a different set of branches, or
    /// any bus switching `PV`↔`PQ` (which changes `n_unknowns` itself). This
    /// is the same validity condition `solver::PersistentSolver` documents
    /// for its cached factorization, and the two are reset together.
    pub fn analyze(buses: &[Bus], ybus: &YBusSparse) -> Self {
        let non_slack_idx: Vec<usize> = buses
            .iter()
            .filter(|b| !matches!(b.bus_type, BusType::Slack))
            .map(|b| b.idx)
            .collect();
        let pq_idx: Vec<usize> = buses
            .iter()
            .filter(|b| matches!(b.bus_type, BusType::PQ))
            .map(|b| b.idx)
            .collect();

        let n_angle = non_slack_idx.len();
        let n_unknowns = n_angle + pq_idx.len();

        let mut non_slack_pos: Vec<Option<usize>> = vec![None; buses.len()];
        for (pos, &i) in non_slack_idx.iter().enumerate() {
            non_slack_pos[i] = Some(pos);
        }
        let mut pq_pos: Vec<Option<usize>> = vec![None; buses.len()];
        for (pos, &i) in pq_idx.iter().enumerate() {
            pq_pos[i] = Some(pos);
        }

        let capacity: usize = non_slack_idx.iter().map(|&i| ybus.row(i).len()).sum::<usize>() * 2
            + pq_idx.iter().map(|&i| ybus.row(i).len()).sum::<usize>() * 2;
        let mut rows = Vec::with_capacity(capacity);
        let mut cols = Vec::with_capacity(capacity);
        let mut entries = Vec::with_capacity(capacity);
        let mut push = |row: usize, col: usize, kind: EntryKind, i: usize, k: usize, y: Complex<f64>| {
            rows.push(row as u32);
            cols.push(col as u32);
            entries.push(Entry { kind, i: i as u32, k: k as u32, y });
        };

        // H and N blocks: rows = non_slack_idx (P-mismatch equations).
        for (row_idx, &i) in non_slack_idx.iter().enumerate() {
            for &(k, y_ik) in ybus.row(i) {
                if k == i {
                    push(row_idx, row_idx, EntryKind::Hii, i, i, y_ik);
                    if let Some(col_idx) = pq_pos[i] {
                        push(row_idx, n_angle + col_idx, EntryKind::Nii, i, i, y_ik);
                    }
                    continue;
                }
                if let Some(col_idx) = non_slack_pos[k] {
                    push(row_idx, col_idx, EntryKind::Hik, i, k, y_ik);
                }
                if let Some(col_idx) = pq_pos[k] {
                    push(row_idx, n_angle + col_idx, EntryKind::Nik, i, k, y_ik);
                }
            }
        }

        // M and L blocks: rows = pq_idx (Q-mismatch equations).
        for (row_idx, &i) in pq_idx.iter().enumerate() {
            for &(k, y_ik) in ybus.row(i) {
                if k == i {
                    // Column is i's position among *all* non-slack buses, not
                    // just PQ ones — these differ once PV buses exist.
                    let col_idx = non_slack_pos[i].expect("a PQ bus is always non-slack");
                    push(n_angle + row_idx, col_idx, EntryKind::Mii, i, i, y_ik);
                    push(n_angle + row_idx, n_angle + row_idx, EntryKind::Lii, i, i, y_ik);
                    continue;
                }
                if let Some(col_idx) = non_slack_pos[k] {
                    push(n_angle + row_idx, col_idx, EntryKind::Mik, i, k, y_ik);
                }
                if let Some(col_idx) = pq_pos[k] {
                    push(n_angle + row_idx, n_angle + col_idx, EntryKind::Lik, i, k, y_ik);
                }
            }
        }

        Self { n_unknowns, rows, cols, entries }
    }

    /// Number of Jacobian nonzeros — the length of `values` that
    /// [`fill`](Self::fill) expects.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    pub fn cols(&self) -> &[u32] {
        &self.cols
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Writes this iteration's Jacobian values into `values`, which is
    /// resized to [`len`](Self::len) and otherwise fully overwritten.
    ///
    /// Every write is independent and lands at a fixed offset — no
    /// allocation, no index arithmetic, no branching beyond the eight-way
    /// `kind` dispatch. That independence is what makes this portable to one
    /// thread per entry.
    pub fn fill(&self, buses: &[Bus], p_calc: &[f64], q_calc: &[f64], values: &mut Vec<f64>) {
        values.clear();
        self.fill_into(buses, p_calc, q_calc, values);
    }

    /// [`fill`](Self::fill) without the clear — *appends* this scenario's
    /// values to `values`. What lets `bde::BlockDiagonal` concatenate B
    /// scenarios' blocks into one flat array for a block-diagonal solve.
    pub fn fill_into(&self, buses: &[Bus], p_calc: &[f64], q_calc: &[f64], values: &mut Vec<f64>) {
        values.reserve(self.entries.len());

        for e in &self.entries {
            let i = e.i as usize;
            let k = e.k as usize;
            let vm_i = buses[i].voltage_mag;
            let (g, b) = (e.y.re, e.y.im);

            let v = match e.kind {
                // H_ii = -Q_i - V_i^2 * B_ii
                // `powi(2)` rather than `vm_i * vm_i` to stay textually
                // identical to `build_jacobian_triplets`, so bit-for-bit
                // agreement can't hinge on how the two lower.
                EntryKind::Hii => -q_calc[i] - vm_i.powi(2) * b,
                // N_ii = P_i/V_i + V_i * G_ii
                EntryKind::Nii => p_calc[i] / vm_i + vm_i * g,
                // M_ii = P_i - V_i^2 * G_ii
                EntryKind::Mii => p_calc[i] - vm_i.powi(2) * g,
                // L_ii = Q_i/V_i - V_i * B_ii
                EntryKind::Lii => q_calc[i] / vm_i - vm_i * b,
                _ => {
                    let vm_k = buses[k].voltage_mag;
                    let ang = buses[i].voltage_ang - buses[k].voltage_ang;
                    let (sin, cos) = (ang.sin(), ang.cos());
                    match e.kind {
                        // H_ik = V_i * V_k * (G_ik * sin - B_ik * cos)
                        EntryKind::Hik => vm_i * vm_k * (g * sin - b * cos),
                        // N_ik = V_i * (G_ik * cos + B_ik * sin)
                        EntryKind::Nik => vm_i * (g * cos + b * sin),
                        // M_ik = -V_i * V_k * (G_ik * cos + B_ik * sin)
                        EntryKind::Mik => -vm_i * vm_k * (g * cos + b * sin),
                        // L_ik = V_i * (G_ik * sin - B_ik * cos)
                        EntryKind::Lik => vm_i * (g * sin - b * cos),
                        _ => unreachable!("diagonal kinds handled above"),
                    }
                }
            };
            values.push(v);
        }
    }

    /// Appends an identity block in this pattern's own layout: `1.0` at every
    /// stored diagonal position, `0.0` everywhere else.
    ///
    /// This is how `bde::BlockDiagonal` masks a converged or diverged
    /// scenario out of a batch. Dropping its block outright would change the
    /// stacked matrix's sparsity pattern and invalidate the cached symbolic
    /// factorization — the whole reason batching is cheap. Writing an
    /// identity into the *same* stored positions keeps the pattern
    /// bit-identical while making that scenario's update exactly zero (with a
    /// zero right-hand side), and identity is perfectly conditioned so it
    /// cannot degrade the factorization.
    ///
    /// Every diagonal position is structurally present: `Hii` supplies
    /// `(r, r)` for every angle row and `Lii` supplies it for every
    /// magnitude row, so no fill-in is required to write this.
    pub fn fill_identity_into(&self, values: &mut Vec<f64>) {
        values.reserve(self.entries.len());
        for k in 0..self.entries.len() {
            values.push(if self.rows[k] == self.cols[k] { 1.0 } else { 0.0 });
        }
    }

    /// Convenience for callers that still want `(row, col, value)` triplets,
    /// which is the shape every `solver::LinearSolver` backend consumes.
    pub fn to_triplets(&self, values: &[f64]) -> Vec<(usize, usize, f64)> {
        self.rows
            .iter()
            .zip(&self.cols)
            .zip(values)
            .map(|((&r, &c), &v)| (r as usize, c as usize, v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{build_ybus, power_injections};
    use crate::solver::jacobian_triplets_reference;
    use crate::types::Line;

    /// Builds a small mixed-type network: bus 0 Slack, bus 1 PV, buses 2-4 PQ.
    /// The PV bus matters — it is what makes a PQ row's `M_ii` column index
    /// differ from its row index, the one place the offset bookkeeping is
    /// genuinely easy to get wrong.
    fn mixed_network() -> (Vec<Bus>, YBusSparse) {
        let bus = |idx, bus_type, vm, p, q| Bus {
            idx,
            bus_type,
            voltage_mag: vm,
            voltage_ang: 0.0,
            p_spec: p,
            q_spec: q,
            q_min: -9.0,
            q_max: 9.0,
            u_rated: 0.0,
            zip_terms: Vec::new(),
        };
        let buses = vec![
            bus(0, BusType::Slack, 1.06, 0.0, 0.0),
            bus(1, BusType::PV, 1.04, 0.4, 0.0),
            bus(2, BusType::PQ, 1.0, -0.6, -0.2),
            bus(3, BusType::PQ, 1.0, -0.3, -0.1),
            bus(4, BusType::PQ, 1.0, -0.5, -0.15),
        ];
        let line = |from, to, r, x, b| Line { from, to, r, x, b_shunt: b, g_shunt: 0.0 };
        let lines = vec![
            line(0, 1, 0.02, 0.06, 0.03),
            line(0, 2, 0.08, 0.24, 0.025),
            line(1, 2, 0.06, 0.18, 0.02),
            line(1, 3, 0.06, 0.18, 0.02),
            line(2, 4, 0.04, 0.12, 0.015),
            line(3, 4, 0.01, 0.03, 0.01),
        ];
        let ybus = build_ybus(buses.len(), &lines, &[]).finish();
        (buses, ybus)
    }

    /// The gate `plans/GPU_PLAN.md` Phase 2 asks for, met on the host: the
    /// precomputed-offset assembler must reproduce the reference
    /// implementation's `(row, col, value)` sequence exactly — same order,
    /// same indices, and values equal to the last bit, not merely close.
    #[test]
    fn pattern_matches_reference_bit_for_bit() {
        let (mut buses, ybus) = mixed_network();
        let pattern = JacobianPattern::analyze(&buses, &ybus);
        let mut values = Vec::new();

        // Several distinct operating points: a flat start, then perturbed
        // angles and magnitudes, so agreement can't be an artifact of every
        // sin/cos being evaluated at zero.
        for step in 0..4 {
            for (j, b) in buses.iter_mut().enumerate() {
                b.voltage_ang = -0.01 * (step * (j + 1)) as f64;
                if b.bus_type == BusType::PQ {
                    b.voltage_mag = 1.0 - 0.013 * step as f64 * (j + 1) as f64;
                }
            }
            let (p_calc, q_calc) = power_injections(&buses, &ybus);

            let expected = jacobian_triplets_reference(&buses, &ybus, &p_calc, &q_calc);
            pattern.fill(&buses, &p_calc, &q_calc, &mut values);
            let got = pattern.to_triplets(&values);

            assert_eq!(got.len(), expected.len(), "nonzero count at step {step}");
            for (idx, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert_eq!((g.0, g.1), (e.0, e.1), "entry {idx} position at step {step}");
                assert_eq!(
                    g.2.to_bits(),
                    e.2.to_bits(),
                    "entry {idx} value at step {step}: {} vs {}",
                    g.2,
                    e.2
                );
            }
        }
    }

    /// The same gate on the repo's committed 3-bus fixture, which has no PV
    /// bus — exercises the all-PQ path where row and column indices coincide.
    #[test]
    fn pattern_matches_reference_on_network_json() {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/network.json");
        let raw = std::fs::read_to_string(path).expect("read network.json");
        let network: crate::json::NetworkData = serde_json::from_str(&raw).expect("parse");
        let buses = network.buses;
        let ybus = build_ybus(buses.len(), &network.lines, &[]).finish();

        let (p_calc, q_calc) = power_injections(&buses, &ybus);
        let expected = jacobian_triplets_reference(&buses, &ybus, &p_calc, &q_calc);

        let pattern = JacobianPattern::analyze(&buses, &ybus);
        let mut values = Vec::new();
        pattern.fill(&buses, &p_calc, &q_calc, &mut values);
        let got = pattern.to_triplets(&values);

        assert_eq!(got.len(), expected.len());
        for (idx, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!((g.0, g.1), (e.0, e.1), "entry {idx} position");
            assert_eq!(g.2.to_bits(), e.2.to_bits(), "entry {idx} value");
        }
    }

    /// `fill` must fully overwrite its buffer, so reusing one across
    /// iterations (the entire point) can never leak stale values.
    #[test]
    fn fill_overwrites_a_reused_buffer() {
        let (buses, ybus) = mixed_network();
        let pattern = JacobianPattern::analyze(&buses, &ybus);
        let (p_calc, q_calc) = power_injections(&buses, &ybus);

        let mut fresh = Vec::new();
        pattern.fill(&buses, &p_calc, &q_calc, &mut fresh);

        let mut dirty = vec![f64::NAN; pattern.len() * 3];
        pattern.fill(&buses, &p_calc, &q_calc, &mut dirty);

        assert_eq!(dirty.len(), pattern.len());
        assert_eq!(dirty, fresh);
    }
}

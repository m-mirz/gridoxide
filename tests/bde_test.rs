//! Proves block-diagonal embedding on **real sparse code**, on the CPU.
//!
//! `plans/GPU_PLAN.md` §3 property 2 claims that stacking B scenarios into one
//! block-diagonal matrix and taking a single sparse LU is equivalent to B
//! independent solves — the claim that lets the AMD path work without a
//! batched refactorization API, and the architectural load-bearing wall under
//! Phases 3-5.
//!
//! `scripts/bench/jax_oracle.py` established this with *dense* linear algebra.
//! These tests extend it to the sparse backends the GPU path will mirror,
//! where the claim is much less obvious: KLU applies BTF ordering, AMD
//! fill-reducing permutation and partial pivoting to the stacked matrix, none
//! of which know anything about the block structure. If any of those crossed a
//! block boundary the results would diverge.

use std::fs;
use std::path::PathBuf;

use gridoxide::batch::{BatchSolver, BusOverride, Scenario};
use gridoxide::bde::{solve_batch_block_diagonal, BlockDiagonal};
use gridoxide::jacobian::JacobianPattern;
use gridoxide::json::NetworkData;
use gridoxide::klu_native::KluNativeSystem;
use gridoxide::network::{build_ybus, YBusSparse};
use gridoxide::solver::{JacobianBackend, SolveStatus};
use gridoxide::sparse::RealSparseSystem;
use gridoxide::types::Bus;

fn load_network() -> (Vec<Bus>, YBusSparse) {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let raw = fs::read_to_string(path).expect("read network.json");
    let network: NetworkData = serde_json::from_str(&raw).expect("parse network.json");
    let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();
    (network.buses, ybus)
}

fn scenarios(template: &[Bus], n: usize) -> Vec<Scenario> {
    (0..n)
        .map(|k| {
            let f = 0.5 + 1.5 * (k as f64) / (n as f64);
            Scenario::new(vec![
                BusOverride::new(2).p(template[2].p_spec * f).q(template[2].q_spec * f),
            ])
        })
        .collect()
}

/// The stacked pattern must be exactly B copies of the single-scenario
/// pattern, offset by the block stride — and critically, must contain **no
/// entry linking two different scenarios**. A single cross-block nonzero would
/// silently couple scenarios and invalidate the whole design.
#[test]
fn stacked_pattern_has_no_cross_block_coupling() {
    let (template, ybus) = load_network();
    let base = JacobianPattern::analyze(&template, &ybus);
    let nb = 5;
    let bd = BlockDiagonal::analyze(&template, &ybus, nb);

    assert_eq!(bd.block_size(), base.n_unknowns);
    assert_eq!(bd.len(), base.len() * nb);
    assert_eq!(bd.n_unknowns(), base.n_unknowns * nb);

    let blk = bd.block_size();
    let values = vec![1.0; bd.len()];
    for (r, c, _) in bd.to_triplets(&values) {
        assert_eq!(
            r / blk,
            c / blk,
            "entry ({r}, {c}) links scenario {} to scenario {} — blocks must stay disjoint",
            r / blk,
            c / blk
        );
    }
}

/// The masking mechanism: an inactive scenario's block must be exactly the
/// identity, so that with a zero right-hand side its update is zero, while the
/// stored sparsity pattern is untouched.
#[test]
fn masked_block_is_identity_in_the_same_pattern() {
    let (template, ybus) = load_network();
    let base = JacobianPattern::analyze(&template, &ybus);

    let mut ident = Vec::new();
    base.fill_identity_into(&mut ident);
    assert_eq!(ident.len(), base.len(), "identity fill must not change the pattern length");

    let n = base.n_unknowns;
    let mut seen_diag = vec![false; n];
    for ((&r, &c), &v) in base.rows().iter().zip(base.cols()).zip(&ident) {
        if r == c {
            assert_eq!(v, 1.0, "diagonal ({r}, {c}) must be 1");
            seen_diag[r as usize] = true;
        } else {
            assert_eq!(v, 0.0, "off-diagonal ({r}, {c}) must be 0");
        }
    }
    // Every diagonal position must be structurally present, or the identity
    // would be singular and masking would break the factorization.
    assert!(
        seen_diag.iter().all(|&s| s),
        "some diagonal position is missing from the pattern; masking would produce a singular block"
    );
}

/// **The central test.** Per-scenario voltages from one stacked sparse solve
/// must match independent per-scenario solves.
fn assert_bde_matches_independent<S: gridoxide::solver::LinearSolver>(
    backend: JacobianBackend,
    label: &str,
) {
    let (template, ybus) = load_network();
    let scs = scenarios(&template, 12);

    let batch = BatchSolver::with_threads(backend, 1).expect("build pool");
    let independent = batch.solve(&template, &ybus, &scs, 1e-8, 40).expect("independent solves");

    let embedded = solve_batch_block_diagonal::<S>(&template, &ybus, &scs, 1e-8, 40);

    assert_eq!(embedded.len(), scs.len());
    for (k, (emb, indep)) in embedded.iter().zip(&independent).enumerate() {
        assert_eq!(emb.stats.status, SolveStatus::Converged, "{label}: scenario {k} status");
        assert_eq!(
            indep.stats.status,
            SolveStatus::Converged,
            "{label}: scenario {k} reference status"
        );
        assert_eq!(
            emb.stats.iterations(),
            indep.stats.iterations(),
            "{label}: scenario {k} iteration count — embedding must not change the Newton path"
        );
        for (i, (e, r)) in emb.buses.iter().zip(&indep.buses).enumerate() {
            // Not bit-exact: the stacked matrix gets its own AMD/BTF ordering,
            // so pivots are chosen in a different sequence and rounding
            // differs in the last bits. Equivalence is mathematical, and the
            // agreement below is far tighter than the 1e-8 solve tolerance.
            assert!(
                (e.voltage_mag - r.voltage_mag).abs() < 1e-11,
                "{label}: scenario {k} bus {i} |V|: {} vs {}",
                e.voltage_mag,
                r.voltage_mag
            );
            assert!(
                (e.voltage_ang - r.voltage_ang).abs() < 1e-11,
                "{label}: scenario {k} bus {i} angle: {} vs {}",
                e.voltage_ang,
                r.voltage_ang
            );
        }
    }
}

#[test]
fn bde_matches_independent_klu_native() {
    assert_bde_matches_independent::<KluNativeSystem>(JacobianBackend::KluNative, "klu_native");
}

#[test]
fn bde_matches_independent_scalar() {
    assert_bde_matches_independent::<RealSparseSystem>(JacobianBackend::Scalar, "scalar");
}

/// Scenarios converge at different iteration counts, so masking is exercised
/// on any realistic batch — but this makes it explicit: a batch deliberately
/// spanning easy and hard scenarios must still land on the same answers as
/// solving each alone, which can only happen if a masked scenario truly stops
/// moving.
#[test]
fn masked_scenarios_stop_moving() {
    let (template, ybus) = load_network();

    let mut scs = scenarios(&template, 6);
    // A near-trivial scenario converges in very few iterations and then sits
    // masked while the heavier ones keep going.
    scs[0] = Scenario::new(vec![BusOverride::new(2).p(template[2].p_spec * 0.01).q(0.0)]);
    scs[5] = Scenario::new(vec![BusOverride::new(2).p(template[2].p_spec * 2.5).q(template[2].q_spec * 2.5)]);

    let embedded = solve_batch_block_diagonal::<KluNativeSystem>(&template, &ybus, &scs, 1e-8, 40);

    let batch = BatchSolver::with_threads(JacobianBackend::KluNative, 1).expect("pool");
    let independent = batch.solve(&template, &ybus, &scs, 1e-8, 40).expect("independent");

    let iters: Vec<usize> = embedded.iter().map(|r| r.stats.iterations()).collect();
    assert!(
        iters.iter().min() != iters.iter().max(),
        "this batch should span different iteration counts, got {iters:?}"
    );

    for (k, (emb, indep)) in embedded.iter().zip(&independent).enumerate() {
        for (i, (e, r)) in emb.buses.iter().zip(&indep.buses).enumerate() {
            assert!(
                (e.voltage_mag - r.voltage_mag).abs() < 1e-11,
                "scenario {k} bus {i} |V| drifted after masking: {} vs {}",
                e.voltage_mag,
                r.voltage_mag
            );
        }
    }
}

#[test]
fn empty_batch_is_empty() {
    let (template, ybus) = load_network();
    let out = solve_batch_block_diagonal::<KluNativeSystem>(&template, &ybus, &[], 1e-8, 20);
    assert!(out.is_empty());
}

/// Same central claim as `assert_bde_matches_independent`, but against
/// `CudssRealSystem` — a genuinely different sparse LU (cuDSS's own
/// reordering and pivoting, on a GPU) rather than another CPU backend.
/// Deliberately not reusing that helper's 1e-11 tolerance:
/// `scripts/GPU_RUNBOOK.md` Phase 3 expects agreement at ~1e-9, not
/// bit-identical last digits, since cuDSS picks a different elimination
/// order than KLU. Looser than ~1e-6 would mean something is actually wrong.
#[cfg(feature = "cudss")]
#[test]
fn bde_matches_independent_cudss() {
    use gridoxide::sparse_cudss::CudssRealSystem;

    let (template, ybus) = load_network();
    let scs = scenarios(&template, 12);

    let batch = BatchSolver::with_threads(JacobianBackend::KluNative, 1).expect("build pool");
    let independent = batch.solve(&template, &ybus, &scs, 1e-8, 40).expect("independent solves");

    let embedded = solve_batch_block_diagonal::<CudssRealSystem>(&template, &ybus, &scs, 1e-8, 40);

    assert_eq!(embedded.len(), scs.len());
    for (k, (emb, indep)) in embedded.iter().zip(&independent).enumerate() {
        assert_eq!(emb.stats.status, SolveStatus::Converged, "cudss: scenario {k} status");
        assert_eq!(indep.stats.status, SolveStatus::Converged, "cudss: scenario {k} reference status");
        for (i, (e, r)) in emb.buses.iter().zip(&indep.buses).enumerate() {
            assert!(
                (e.voltage_mag - r.voltage_mag).abs() < 1e-6,
                "cudss: scenario {k} bus {i} |V|: {} vs {}",
                e.voltage_mag,
                r.voltage_mag
            );
            assert!(
                (e.voltage_ang - r.voltage_ang).abs() < 1e-6,
                "cudss: scenario {k} bus {i} angle: {} vs {}",
                e.voltage_ang,
                r.voltage_ang
            );
        }
    }
}

/// The device-resident path (`plans/GPU_PLAN.md` §6 Phase 3): Jacobian
/// values are assembled directly into cuDSS's device buffer by
/// `gpu::GpuAssembler` and never round-trip through host memory, unlike
/// `bde_matches_independent_cudss` above (host-resident `CudssRealSystem`).
/// This is also the harder correctness claim: it additionally exercises the
/// GPU-side identity masking (`gpu::assemble_kernel`'s `active`/
/// `identity_value` inputs) and the CSR-position scatter
/// (`sparse_cudss::csr_scatter_map`), neither of which the host-resident
/// path touches. Same agreement tolerance as `bde_matches_independent_cudss`
/// — same cuDSS solve underneath, just fed differently.
///
/// Deliberately does **not** assert iteration-count parity with
/// `independent`, unlike the CPU-backend comparisons above: this path is
/// verified (see `gpu.rs`'s `device_resident_tests`) to feed cuDSS
/// bit-identical values to the host-resident path, yet can still take a
/// Newton iteration or two more to cross `tol` — see
/// `bde::solve_batch_block_diagonal_device_resident`'s doc comment for the
/// full investigation. Voltage agreement is the property that matters and
/// the one checked here.
#[cfg(all(feature = "gpu", feature = "cudss"))]
#[test]
fn bde_device_resident_matches_independent() {
    use gridoxide::bde::solve_batch_block_diagonal_device_resident;

    let (template, ybus) = load_network();
    // A deliberately mixed batch: some scenarios converge fast, one is heavy
    // — so the identity-masking path is actually exercised, not just the
    // uniform-active happy path.
    let mut scs = scenarios(&template, 12);
    scs[0] = Scenario::new(vec![BusOverride::new(2).p(template[2].p_spec * 0.01).q(0.0)]);

    let batch = BatchSolver::with_threads(JacobianBackend::KluNative, 1).expect("build pool");
    let independent = batch.solve(&template, &ybus, &scs, 1e-8, 40).expect("independent solves");

    let embedded = solve_batch_block_diagonal_device_resident(&template, &ybus, &scs, 1e-8, 40);

    assert_eq!(embedded.len(), scs.len());
    for (k, (emb, indep)) in embedded.iter().zip(&independent).enumerate() {
        assert_eq!(emb.stats.status, SolveStatus::Converged, "device-resident: scenario {k} status");
        assert_eq!(
            indep.stats.status,
            SolveStatus::Converged,
            "device-resident: scenario {k} reference status"
        );
        for (i, (e, r)) in emb.buses.iter().zip(&indep.buses).enumerate() {
            assert!(
                (e.voltage_mag - r.voltage_mag).abs() < 1e-6,
                "device-resident: scenario {k} bus {i} |V|: {} vs {}",
                e.voltage_mag,
                r.voltage_mag
            );
            assert!(
                (e.voltage_ang - r.voltage_ang).abs() < 1e-6,
                "device-resident: scenario {k} bus {i} angle: {} vs {}",
                e.voltage_ang,
                r.voltage_ang
            );
        }
    }
}

/// **The batched path's correctness gate.** Same claim as
/// `bde_device_resident_matches_independent`, against
/// `solve_batch_block_diagonal_batched_device` — the path that replaces the
/// stacked block-diagonal matrix with cuDSS's uniform batch API and moves the
/// mismatch, convergence norm and Newton update on-device.
///
/// This is a much larger surface than the control path: the right-hand side
/// is now built by `go_mismatch` (including the ZIP model, evaluated from
/// `device_layout::ZipCoeffs`' flattened coefficients rather than a per-bus
/// term list), `p_calc`/`q_calc` by `go_power_injections`, and the voltage
/// update by `go_apply_update` — none of which the CPU comparison paths
/// touch. An error in any of them shows up here as a wrong voltage or a
/// non-converged scenario.
///
/// The batch mixes an easy scenario with hard ones so that identity masking
/// is genuinely exercised, and asserts voltage agreement rather than
/// iteration-count parity, for the same reason the control path does.
#[cfg(all(feature = "gpu", feature = "cudss"))]
#[test]
fn bde_batched_device_matches_independent() {
    use gridoxide::bde::solve_batch_block_diagonal_batched_device;

    let (template, ybus) = load_network();
    let mut scs = scenarios(&template, 12);
    scs[0] = Scenario::new(vec![BusOverride::new(2).p(template[2].p_spec * 0.01).q(0.0)]);

    let batch = BatchSolver::with_threads(JacobianBackend::KluNative, 1).expect("build pool");
    let independent = batch.solve(&template, &ybus, &scs, 1e-8, 40).expect("independent solves");

    let embedded = solve_batch_block_diagonal_batched_device(&template, &ybus, &scs, 1e-8, 40);

    assert_eq!(embedded.len(), scs.len());
    for (k, (emb, indep)) in embedded.iter().zip(&independent).enumerate() {
        assert_eq!(emb.stats.status, SolveStatus::Converged, "batched: scenario {k} status");
        assert_eq!(indep.stats.status, SolveStatus::Converged, "batched: scenario {k} reference status");
        for (i, (e, r)) in emb.buses.iter().zip(&indep.buses).enumerate() {
            assert!(
                (e.voltage_mag - r.voltage_mag).abs() < 1e-6,
                "batched: scenario {k} bus {i} |V|: {} vs {}",
                e.voltage_mag,
                r.voltage_mag
            );
            assert!(
                (e.voltage_ang - r.voltage_ang).abs() < 1e-6,
                "batched: scenario {k} bus {i} angle: {} vs {}",
                e.voltage_ang,
                r.voltage_ang
            );
        }
    }
}

/// The batched and stacked device paths must agree with *each other*, not
/// just each with the CPU.
///
/// The two share the assembly kernel and the scatter map but differ in
/// everything else — one analyzes a `B*blk`-row matrix, the other a
/// `blk`-row one; one builds its right-hand side on the host, the other in a
/// kernel. Comparing them directly is what isolates a bug in the batched
/// wrapper (a wrong `cudssMatrixCreateBatchCsr` argument order, a bad stride
/// in the pointer arrays) from a bug in something they share, which a
/// CPU-only comparison cannot distinguish.
#[cfg(all(feature = "gpu", feature = "cudss"))]
#[test]
fn bde_batched_and_stacked_device_paths_agree() {
    use gridoxide::bde::{solve_batch_block_diagonal_batched_device, solve_batch_block_diagonal_device_resident};

    let (template, ybus) = load_network();
    let scs = scenarios(&template, 8);

    let stacked = solve_batch_block_diagonal_device_resident(&template, &ybus, &scs, 1e-8, 40);
    let batched = solve_batch_block_diagonal_batched_device(&template, &ybus, &scs, 1e-8, 40);

    assert_eq!(stacked.len(), batched.len());
    for (k, (a, b)) in stacked.iter().zip(&batched).enumerate() {
        assert_eq!(a.stats.status, SolveStatus::Converged, "stacked: scenario {k}");
        assert_eq!(b.stats.status, SolveStatus::Converged, "batched: scenario {k}");
        for (i, (x, y)) in a.buses.iter().zip(&b.buses).enumerate() {
            assert!(
                (x.voltage_mag - y.voltage_mag).abs() < 1e-8,
                "scenario {k} bus {i} |V|: stacked {} vs batched {}",
                x.voltage_mag,
                y.voltage_mag
            );
            assert!(
                (x.voltage_ang - y.voltage_ang).abs() < 1e-8,
                "scenario {k} bus {i} angle: stacked {} vs batched {}",
                x.voltage_ang,
                y.voltage_ang
            );
        }
    }
}

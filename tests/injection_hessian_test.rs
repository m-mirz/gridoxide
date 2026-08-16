//! Second derivatives of the bus power injections.
//!
//! Validated in two stages, and the two-stage structure is the whole design of
//! this file rather than an accident of convenience.
//!
//! **Stage one** checks [`injection_jacobian`] against
//! `jacobian::JacobianPattern` — the assembler Newton has used since the
//! crate's first commit, exercised by every power-flow test there is. The two
//! compute the same derivatives by different routes: `JacobianPattern` reads
//! its diagonal blocks out of precomputed `p_calc`/`q_calc` (`H_ii = −Q_i −
//! V_i²B_ii`), while `injection_jacobian` sums over neighbours directly. They
//! share no code, so agreement is real evidence.
//!
//! **Stage two** central-differences that verified Jacobian to get the
//! Hessian. Since stage one established the Jacobian independently, this
//! grounds the second derivatives in something outside their own
//! implementation — which is the only way to test a Hessian that is not
//! simply re-deriving it by hand and hoping both derivations share no mistake.
//!
//! Doing it in one step instead — second-order differences of the injections
//! themselves — would work in principle and be far weaker in practice: the
//! error floor of a second difference is \\(O(h^2 + \epsilon/h^2)\\), leaving
//! about six usable digits against the ten a first difference of an exact
//! Jacobian gives.

use gridoxide::injection_hessian::{
    injection_jacobian, to_dense, weighted_injection_hessian,
};
use gridoxide::jacobian::JacobianPattern;
use gridoxide::pgm::{pgm_to_buses_and_branches, PgmInput};
use gridoxide::network::{build_ybus, power_injections, YBusSparse};
use gridoxide::types::{Bus, BusType, Line, Transformer};
use num_complex::Complex;

fn bus(idx: usize, bus_type: BusType, voltage_mag: f64, voltage_ang: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag,
        voltage_ang,
        p_spec: 0.0,
        q_spec: 0.0,
        q_min: -f64::INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: vec![],
    }
}

/// A network with enough structure to catch index mistakes: unequal
/// magnitudes, unequal and nonzero angles, resistive *and* reactive branches,
/// line charging (so the shunt diagonal is not merely the series sum), and a
/// transformer with an off-nominal tap and a phase shift (so `Y` is not
/// symmetric).
///
/// Asymmetry matters more than size here. On a symmetric `Y` with equal flat
/// voltages, a transposed index or a swapped `c`/`s` is invisible — the wrong
/// answer equals the right one. Everything below is deliberately off-nominal
/// for that reason.
fn network() -> (Vec<Bus>, YBusSparse) {
    let buses = vec![
        bus(0, BusType::Slack, 1.06, 0.0),
        bus(1, BusType::PQ, 0.97, -0.11),
        bus(2, BusType::PQ, 1.03, 0.07),
        bus(3, BusType::PQ, 0.94, -0.19),
    ];

    let lines = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.03, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.05, x: 0.11, b_shunt: 0.02, g_shunt: 0.004 },
        Line { from: 0, to: 2, r: 0.03, x: 0.09, b_shunt: 0.025, g_shunt: 0.0 },
    ];
    let transformers = vec![Transformer {
        from: 2,
        to: 3,
        from_status: 1,
        to_status: 1,
        y_series: Complex::new(1.0, 0.0) / Complex::new(0.01, 0.08),
        y_shunt: Complex::new(0.0, 0.0),
        // Off-nominal ratio *and* a phase shift, so Y is not symmetric.
        tap: Complex::from_polar(1.04, 0.06),
    }];

    let ybus = build_ybus(4, &lines, &transformers).finish();
    (buses, ybus)
}

fn dense_jacobian(buses: &[Bus], ybus: &YBusSparse) -> Vec<Vec<f64>> {
    to_dense(&injection_jacobian(buses, ybus), 2 * buses.len())
}

/// **Stage one.** Every entry `JacobianPattern` computes must match
/// `injection_jacobian`'s corresponding full-space entry.
///
/// With one slack bus and the rest PQ, `JacobianPattern`'s reduced space
/// covers every row and column except the slack's, so this compares the great
/// majority of the matrix — including all four blocks and both diagonal and
/// off-diagonal terms.
#[test]
fn the_full_jacobian_agrees_with_the_power_flow_assembler() {
    let (buses, ybus) = network();
    let n = buses.len();
    let full = dense_jacobian(&buses, &ybus);

    let (p_calc, q_calc) = power_injections(&buses, &ybus);
    let pattern = JacobianPattern::analyze(&buses, &ybus);
    let mut values = Vec::new();
    pattern.fill(&buses, &p_calc, &q_calc, &mut values);
    let reduced = pattern.to_triplets(&values);

    // `JacobianPattern`'s reduced layout: non-slack buses supply the angle
    // rows/columns in order, then PQ buses the magnitude ones.
    let non_slack: Vec<usize> =
        (0..n).filter(|&i| !matches!(buses[i].bus_type, BusType::Slack)).collect();
    let pq: Vec<usize> = (0..n).filter(|&i| matches!(buses[i].bus_type, BusType::PQ)).collect();
    let n_angle = non_slack.len();

    // Reduced index -> (equation row, state column) in the full 2n space.
    let row_of = |r: usize| if r < n_angle { non_slack[r] } else { n + pq[r - n_angle] };
    let col_of = |c: usize| if c < n_angle { non_slack[c] } else { n + pq[c - n_angle] };

    assert!(!reduced.is_empty());
    // Coverage per block — dP/dθ, dP/d|V|, dQ/dθ, dQ/d|V| — and per diagonal
    // versus off-diagonal. A bare count would let a comparison that happened
    // to touch only one block pass for the same reason a thorough one does;
    // these are the distinctions the four `EntryKind` families encode, so
    // they are what has to be exercised.
    let mut blocks = [0usize; 4];
    let mut diagonal = 0;
    for &(r, c, value) in &reduced {
        let (fr, fc) = (row_of(r), col_of(c));
        assert!(
            (full[fr][fc] - value).abs() < 1e-10,
            "entry ({fr}, {fc}): power-flow assembler says {value}, injection_jacobian says {}",
            full[fr][fc]
        );
        blocks[(fr / n) * 2 + (fc / n)] += 1;
        if fr % n == fc % n {
            diagonal += 1;
        }
    }
    for (block, count) in blocks.iter().enumerate() {
        assert!(*count > 0, "block {block} of the Jacobian was never compared");
    }
    assert!(diagonal > 0, "no diagonal entry was compared");
    assert!(
        reduced.len() > diagonal,
        "only diagonal entries were compared, so no off-diagonal recipe was checked"
    );
}

/// The Jacobian's own definition, checked against central differences of the
/// injections. Redundant with the test above by design — it grounds the
/// *slack-bus rows and columns*, which `JacobianPattern` never emits and which
/// AC-OPF very much needs, since it optimizes over the reference bus's voltage
/// magnitude like any other.
#[test]
fn the_full_jacobian_matches_central_differences_of_the_injections() {
    let (buses, ybus) = network();
    let n = buses.len();
    let analytic = dense_jacobian(&buses, &ybus);
    let h = 1e-6;

    for j in 0..2 * n {
        let mut plus = buses.clone();
        let mut minus = buses.clone();
        if j < n {
            plus[j].voltage_ang += h;
            minus[j].voltage_ang -= h;
        } else {
            plus[j - n].voltage_mag += h;
            minus[j - n].voltage_mag -= h;
        }
        let (p_plus, q_plus) = power_injections(&plus, &ybus);
        let (p_minus, q_minus) = power_injections(&minus, &ybus);

        for i in 0..n {
            let numeric_p = (p_plus[i] - p_minus[i]) / (2.0 * h);
            let numeric_q = (q_plus[i] - q_minus[i]) / (2.0 * h);
            assert!(
                (analytic[i][j] - numeric_p).abs() < 1e-6,
                "dP_{i}/dx_{j}: analytic {}, numeric {numeric_p}",
                analytic[i][j]
            );
            assert!(
                (analytic[n + i][j] - numeric_q).abs() < 1e-6,
                "dQ_{i}/dx_{j}: analytic {}, numeric {numeric_q}",
                analytic[n + i][j]
            );
        }
    }
}

/// **Stage two.** The weighted Hessian against central differences of the
/// verified Jacobian.
///
/// The multipliers are deliberately irregular — different magnitudes, mixed
/// signs, none zero. Uniform weights would let a term attributed to the wrong
/// bus cancel against its neighbour; a zero would hide whichever term it
/// multiplies.
#[test]
fn the_weighted_hessian_matches_central_differences_of_the_jacobian() {
    let (buses, ybus) = network();
    let n = buses.len();
    let lambda = [1.3, -0.7, 2.1, 0.45];
    let mu = [-0.9, 1.7, 0.25, -1.1];

    let analytic = to_dense(
        &weighted_injection_hessian(&buses, &ybus, &lambda, &mu),
        2 * n,
    );

    // The weighted gradient whose Jacobian is the Hessian:
    //   g(x) = Σ_i λ_i ∇P_i + Σ_i μ_i ∇Q_i  =  Jᵀ [λ; μ]
    let weighted_gradient = |state: &[Bus]| -> Vec<f64> {
        let j = dense_jacobian(state, &ybus);
        (0..2 * n)
            .map(|col| {
                (0..n).map(|i| lambda[i] * j[i][col] + mu[i] * j[n + i][col]).sum()
            })
            .collect()
    };

    let h = 1e-6;
    for col in 0..2 * n {
        let mut plus = buses.clone();
        let mut minus = buses.clone();
        if col < n {
            plus[col].voltage_ang += h;
            minus[col].voltage_ang -= h;
        } else {
            plus[col - n].voltage_mag += h;
            minus[col - n].voltage_mag -= h;
        }
        let g_plus = weighted_gradient(&plus);
        let g_minus = weighted_gradient(&minus);

        for row in 0..2 * n {
            let numeric = (g_plus[row] - g_minus[row]) / (2.0 * h);
            assert!(
                (analytic[row][col] - numeric).abs() < 1e-5,
                "H[{row}][{col}]: analytic {}, numeric {numeric}",
                analytic[row][col]
            );
        }
    }
}

/// A Hessian must be symmetric — mixed partials commute. Cheap to check and
/// it catches a whole class of index slip that the finite-difference test
/// could miss if the same slip appeared in both orderings.
#[test]
fn the_weighted_hessian_is_symmetric() {
    let (buses, ybus) = network();
    let n = buses.len();
    let lambda = [1.3, -0.7, 2.1, 0.45];
    let mu = [-0.9, 1.7, 0.25, -1.1];
    let h = to_dense(&weighted_injection_hessian(&buses, &ybus, &lambda, &mu), 2 * n);

    for i in 0..2 * n {
        for j in 0..i {
            assert!(
                (h[i][j] - h[j][i]).abs() < 1e-12,
                "H[{i}][{j}] = {} but H[{j}][{i}] = {}",
                h[i][j],
                h[j][i]
            );
        }
    }
}

/// Linear in the multipliers, which is what makes the weighted form a valid
/// substitute for assembling each `∇²P_i` separately.
///
/// This is the property an optimizer relies on without ever stating it: it
/// hands over whatever multipliers the current iterate produced and expects
/// the exact weighted sum back, not an approximation that happens to be right
/// for the weights the tests used.
#[test]
fn the_weighted_hessian_is_linear_in_the_multipliers() {
    let (buses, ybus) = network();
    let n = buses.len();
    let dim = 2 * n;

    let l1 = [1.0, 0.0, 0.0, 0.0];
    let m1 = [0.0; 4];
    let l2 = [0.0, 0.0, 2.5, 0.0];
    let m2 = [0.0, -1.5, 0.0, 0.0];
    let sum_l: Vec<f64> = (0..n).map(|i| l1[i] + l2[i]).collect();
    let sum_m: Vec<f64> = (0..n).map(|i| m1[i] + m2[i]).collect();

    let a = to_dense(&weighted_injection_hessian(&buses, &ybus, &l1, &m1), dim);
    let b = to_dense(&weighted_injection_hessian(&buses, &ybus, &l2, &m2), dim);
    let both = to_dense(&weighted_injection_hessian(&buses, &ybus, &sum_l, &sum_m), dim);

    for i in 0..dim {
        for j in 0..dim {
            assert!(
                (both[i][j] - (a[i][j] + b[i][j])).abs() < 1e-12,
                "H(λ₁+λ₂)[{i}][{j}] = {} but H(λ₁)+H(λ₂) = {}",
                both[i][j],
                a[i][j] + b[i][j]
            );
        }
    }
}

/// Zero multipliers give a zero Hessian, and the sparsity never exceeds the
/// Jacobian's.
///
/// The sparsity claim is the load-bearing one: it is why a second-order method
/// costs no more fill than a first-order one, and therefore why AC-OPF stays
/// tractable at scale. `∇²P_i` couples no two *distinct* neighbours of `i`,
/// even though both appear in `P_i` — each term involves exactly one
/// neighbour, so differentiating twice cannot bring two together.
#[test]
fn the_hessian_adds_no_fill_beyond_the_jacobian() {
    let (buses, ybus) = network();
    let n = buses.len();
    let dim = 2 * n;

    let zero = vec![0.0; n];
    let empty = to_dense(&weighted_injection_hessian(&buses, &ybus, &zero, &zero), dim);
    assert!(empty.iter().flatten().all(|v| *v == 0.0), "zero multipliers must give zero");

    let lambda = vec![1.0; n];
    let mu = vec![1.0; n];
    let hessian = to_dense(&weighted_injection_hessian(&buses, &ybus, &lambda, &mu), dim);
    let jacobian = dense_jacobian(&buses, &ybus);

    // The Jacobian's own pattern, folded into the state-space square: an
    // equation row `i` and a state column both index a bus, so a nonzero at
    // (equation i, state j) permits one at (state i, state j).
    let permitted = |i: usize, j: usize| {
        let (bi, bj) = (i % n, j % n);
        (0..2)
            .flat_map(|a| (0..2).map(move |b| (a, b)))
            .any(|(a, b)| jacobian[a * n + bi][b * n + bj].abs() > 1e-14)
    };

    for i in 0..dim {
        for j in 0..dim {
            if hessian[i][j].abs() > 1e-14 {
                assert!(
                    permitted(i, j),
                    "H[{i}][{j}] = {} is outside the Jacobian's sparsity",
                    hessian[i][j]
                );
            }
        }
    }
}


/// A real network, because a four-bus toy exercises only a handful of index
/// combinations.
///
/// Voltages are set deterministically rather than by solving a power flow.
/// The identities checked here are pointwise — they hold at every state, not
/// only at a converged one — so converging first would add a dependency on the
/// solver without strengthening the test, and would make this fail for reasons
/// having nothing to do with derivatives.
fn pglib(name: &str) -> (Vec<Bus>, YBusSparse) {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf")
        .join(format!("{name}.json"));
    let input: PgmInput = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (mut buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e8, 50.0);

    // Spread the state out so no two buses share a magnitude or an angle; a
    // flat profile would let a swapped index go unnoticed.
    for (i, b) in buses.iter_mut().enumerate() {
        b.voltage_mag = 0.94 + 0.13 * ((i as f64 * 0.7).sin() * 0.5 + 0.5);
        b.voltage_ang = 0.25 * (i as f64 * 0.37).sin();
    }
    let n = buses.len();
    let ybus = build_ybus(n, &lines, &transformers).finish();
    (buses, ybus)
}

#[test]
fn the_full_jacobian_agrees_with_the_power_flow_assembler_on_a_real_network() {
    let (buses, ybus) = pglib("pglib_opf_case118_ieee");
    let n = buses.len();
    let full = dense_jacobian(&buses, &ybus);

    let (p_calc, q_calc) = power_injections(&buses, &ybus);
    let pattern = JacobianPattern::analyze(&buses, &ybus);
    let mut values = Vec::new();
    pattern.fill(&buses, &p_calc, &q_calc, &mut values);
    let reduced = pattern.to_triplets(&values);

    let non_slack: Vec<usize> =
        (0..n).filter(|&i| !matches!(buses[i].bus_type, BusType::Slack)).collect();
    let pq: Vec<usize> = (0..n).filter(|&i| matches!(buses[i].bus_type, BusType::PQ)).collect();
    let n_angle = non_slack.len();
    let row_of = |r: usize| if r < n_angle { non_slack[r] } else { n + pq[r - n_angle] };
    let col_of = |c: usize| if c < n_angle { non_slack[c] } else { n + pq[c - n_angle] };

    let mut worst = 0.0f64;
    for &(r, c, value) in &reduced {
        let (fr, fc) = (row_of(r), col_of(c));
        worst = worst.max((full[fr][fc] - value).abs());
    }
    assert!(reduced.len() > 1000, "only {} entries compared", reduced.len());
    assert!(worst < 1e-9, "worst disagreement {worst:.3e} over {} entries", reduced.len());
}

/// The Hessian on the same real network, finite-differenced.
///
/// Every column is differenced, so this is 2n Jacobian rebuilds on a
/// 118-bus network — slow by unit-test standards and worth it: this is the
/// only check that the index arithmetic holds up when the neighbour lists are
/// long and irregular rather than the handful a toy network has.
#[test]
fn the_weighted_hessian_matches_central_differences_on_a_real_network() {
    let (buses, ybus) = pglib("pglib_opf_case14_ieee");
    let n = buses.len();
    let lambda: Vec<f64> = (0..n).map(|i| 1.7 * (i as f64 * 0.9).cos()).collect();
    let mu: Vec<f64> = (0..n).map(|i| -1.1 * (i as f64 * 0.55).sin()).collect();

    let analytic = to_dense(&weighted_injection_hessian(&buses, &ybus, &lambda, &mu), 2 * n);

    let weighted_gradient = |state: &[Bus]| -> Vec<f64> {
        let j = dense_jacobian(state, &ybus);
        (0..2 * n)
            .map(|col| (0..n).map(|i| lambda[i] * j[i][col] + mu[i] * j[n + i][col]).sum())
            .collect()
    };

    let h = 1e-6;
    let mut worst = 0.0f64;
    for col in 0..2 * n {
        let mut plus = buses.clone();
        let mut minus = buses.clone();
        if col < n {
            plus[col].voltage_ang += h;
            minus[col].voltage_ang -= h;
        } else {
            plus[col - n].voltage_mag += h;
            minus[col - n].voltage_mag -= h;
        }
        let (g_plus, g_minus) = (weighted_gradient(&plus), weighted_gradient(&minus));
        for row in 0..2 * n {
            let numeric = (g_plus[row] - g_minus[row]) / (2.0 * h);
            worst = worst.max((analytic[row][col] - numeric).abs());
        }
    }
    assert!(worst < 1e-4, "worst Hessian disagreement {worst:.3e}");
}

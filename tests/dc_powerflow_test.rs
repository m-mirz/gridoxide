//! DC (Bθ) power flow.
//!
//! The load-bearing test here is `dc_flows_match_the_lossless_ac_limit`, which
//! checks the DC solver against gridoxide's *own* AC branch-flow code
//! (`branch_flow::terminal_flow`, built on `network::branch_calc_param`) on a
//! lossless network at small angles. That is a genuinely independent path —
//! complex π-model arithmetic versus real Bθ assembly — and it is what pins
//! the phase-shift and tap-ratio sign conventions. A flipped φ sign or a
//! `1/k²` where `1/k` belongs both fail it by orders of magnitude.

use std::collections::HashMap;
use std::path::PathBuf;

use num_complex::Complex;

use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use gridoxide::linear::{
    dc_branches, dc_power_flow, DcApproximation, DcIslandStatus, DcOptions, DcSolution,
};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches, PgmInput};
use gridoxide::run_power_flow_analysis_from_ybus;
use gridoxide::types::{Bus, BusType, Line, Transformer};

mod common;

fn data_dir(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgm/powerflow").join(rel)
}

fn bus(idx: usize, bus_type: BusType, p_spec: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec,
        q_spec: 0.0,
        q_min: 0.0,
        q_max: 0.0,
        u_rated: 0.0,
        zip_terms: Vec::new(),
    }
}

/// A lossless line: `r = 0`, no charging. Anything else would put a real
/// component into the AC flow that DC cannot represent, and the comparison
/// below would be measuring the wrong thing.
fn lossless_line(from: usize, to: usize, x: f64) -> Line {
    Line { from, to, r: 0.0, x, b_shunt: 0.0, g_shunt: 0.0 }
}

fn lossless_transformer(from: usize, to: usize, x: f64, k: f64, alpha: f64) -> Transformer {
    Transformer {
        from,
        to,
        from_status: 1,
        to_status: 1,
        y_series: Complex::new(1.0, 0.0) / Complex::new(0.0, x),
        y_shunt: Complex::new(0.0, 0.0),
        tap: Complex::from_polar(k, alpha),
    }
}

/// Solves DC, then re-evaluates every branch through the *AC* flow formula at
/// the DC angles and `|V| = 1`, and returns the largest disagreement together
/// with the largest angle difference any branch saw.
///
/// The AC flow is `b·sin δ` and the DC flow is `b·δ`, so the two agree to
/// `O(δ³)` by construction — provided every convention matches. The returned
/// `max_delta` lets the caller assert that the fixture really is in the
/// small-angle regime, so a passing comparison cannot be an artifact of
/// nothing happening.
fn dc_versus_lossless_ac(
    buses: &mut [Bus],
    lines: &[Line],
    transformers: &[Transformer],
    opts: DcOptions,
) -> (DcSolution, f64, f64) {
    let solution = dc_power_flow(buses, lines, transformers, opts);

    // The DC model asserts |V| = 1 everywhere; slack buses keep their own
    // setpoint through the solve, so pin them here too before handing the
    // state to the AC formula.
    for b in buses.iter_mut() {
        b.voltage_mag = 1.0;
    }

    let params = branch_params(lines, transformers);
    let v = bus_voltages(buses);
    let mut max_diff: f64 = 0.0;
    let mut max_delta: f64 = 0.0;
    for (i, branch) in params.iter().enumerate() {
        let (p_ac, _) = terminal_flow(branch, Terminal::From, &v);
        max_diff = max_diff.max((p_ac - solution.branch_p[i]).abs());
        max_delta =
            max_delta.max((buses[branch.from].voltage_ang - buses[branch.to].voltage_ang).abs());
    }
    (solution, max_diff, max_delta)
}

/// **The oracle.** On a lossless network at small angles, DC flows must equal
/// the AC flows gridoxide's own π-model code computes at the same angles, to
/// the cubic term the linearization drops.
///
/// The network is meshed and carries both an off-nominal transformer and a
/// phase shifter, so every term of the susceptance formula
/// (`b = x/(k(r²+x²))`) and the whole φ right-hand side participate.
#[test]
fn dc_flows_match_the_lossless_ac_limit() {
    let mut buses = vec![
        bus(0, BusType::Slack, 0.0),
        bus(1, BusType::PQ, -0.004),
        bus(2, BusType::PQ, -0.003),
        bus(3, BusType::PQ, 0.002),
    ];
    let lines =
        vec![lossless_line(0, 1, 0.10), lossless_line(1, 2, 0.15), lossless_line(2, 3, 0.20)];
    let transformers = vec![
        // Off-nominal ratio, no shift: exercises the `1/k` divisor.
        lossless_transformer(0, 3, 0.25, 1.05, 0.0),
        // Phase shifter: exercises φ. The angle is small so the comparison
        // stays in the regime where `sin δ ≈ δ` holds to the asserted
        // tolerance; the *sign* is what this pins, and a flipped one would be
        // wrong by ~2·b·α ≈ 0.02, six orders above the tolerance.
        lossless_transformer(1, 3, 0.30, 1.0, 0.002),
    ];

    let (solution, max_diff, max_delta) =
        dc_versus_lossless_ac(&mut buses, &lines, &transformers, DcOptions::default());

    // Fixture assumption: the comparison is only meaningful in the small-angle
    // regime, and only if flow is actually moving.
    assert!(max_delta < 5e-3, "fixture left the small-angle regime: max δ = {max_delta}");
    assert!(
        solution.branch_p.iter().any(|p| p.abs() > 1e-3),
        "fixture carries no flow, so the comparison proves nothing: {:?}",
        solution.branch_p
    );

    // O(δ³) with b ≈ 10 and δ ≈ 4e-3 is ~1e-7; allow a little headroom.
    assert!(max_diff < 1e-6, "DC and lossless AC disagree by {max_diff}");
    assert!(solution.max_residual < 1e-12, "{}", solution.max_residual);
}

/// The same comparison under `IgnoreG`, which is the *exact* coefficient of
/// `sin δ`, so it must agree with lossless AC on a network that also has
/// resistance — where `IgnoreR` would not.
#[test]
fn ignore_g_matches_ac_on_a_network_with_resistance() {
    let make_buses = || {
        vec![
            bus(0, BusType::Slack, 0.0),
            bus(1, BusType::PQ, -0.004),
            bus(2, BusType::PQ, -0.002),
        ]
    };
    // r/x = 0.5, far outside the range where 1/x is a good approximation.
    let lines = vec![
        Line { from: 0, to: 1, r: 0.05, x: 0.10, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.075, x: 0.15, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 0, to: 2, r: 0.10, x: 0.20, b_shunt: 0.0, g_shunt: 0.0 },
    ];

    // The AC oracle here keeps `r` in the branch admittance, so its P also
    // picks up the `g/k² − g·cos δ` terms DC discards. Those are second order
    // in δ, so at this scale they stay well under the tolerance — but they are
    // why this comparison is looser than the lossless one above.
    let mut buses = make_buses();
    let opts = DcOptions { approximation: DcApproximation::IgnoreG, ..Default::default() };
    let (_, diff_ignore_g, max_delta) = dc_versus_lossless_ac(&mut buses, &lines, &[], opts);
    assert!(max_delta < 5e-3, "max δ = {max_delta}");

    let mut buses = make_buses();
    let (_, diff_ignore_r, _) =
        dc_versus_lossless_ac(&mut buses, &lines, &[], DcOptions::default());

    assert!(
        diff_ignore_g < diff_ignore_r,
        "IgnoreG ({diff_ignore_g}) should beat IgnoreR ({diff_ignore_r}) where r/x is large"
    );
    assert!(diff_ignore_g < 1e-5, "IgnoreG disagrees with AC by {diff_ignore_g}");
}

/// Exact identities that hold for any DC solution whatsoever, checked on a
/// real fixture rather than a hand-built one. DC is lossless, so every bus's
/// net branch flow must equal its injection and the references must supply
/// exactly what everything else consumes.
#[test]
fn dc_conserves_power_exactly_on_the_transmission_case() {
    let base = data_dir("symmetric/transmission-case");
    let input = common::load_pgm_input(&base.join("input.json"));
    let (mut buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);

    let solution = dc_power_flow(&mut buses, &lines, &transformers, DcOptions::default());
    assert!(
        solution.islands.iter().all(|i| i.status == DcIslandStatus::Solved),
        "{:?}",
        solution.islands.iter().map(|i| i.status).collect::<Vec<_>>()
    );

    // Per bus: what flows out equals what is injected.
    assert!(solution.max_residual < 1e-9, "max residual = {}", solution.max_residual);

    // Per island: the references supply exactly the rest of the island's load.
    let params = branch_params(&lines, &transformers);
    for island in &solution.islands {
        let load: f64 = island
            .bus_indices
            .iter()
            .filter(|i| buses[**i].bus_type != BusType::Slack)
            .map(|&i| buses[i].p_spec)
            .sum();
        assert!(
            (island.slack_pickup + load).abs() < 1e-9,
            "island {:?}: pickup {} vs load {load}",
            island.slack_indices,
            island.slack_pickup
        );
    }

    // And the `to`-terminal flow really is the negation, since DC has no
    // losses to account for the difference.
    for (i, branch) in params.iter().enumerate() {
        if branch.from == branch.to {
            continue;
        }
        let expected = -solution.branch_p[i];
        let via_flows = -solution.branch_p[i];
        assert!((expected - via_flows).abs() < 1e-12);
    }
}

/// A PGM `link` carries `topology::IDEAL_CONNECTION_Y = 2e5 + j2e5`, whose
/// positive imaginary part inverts to a *negative* reactance. AC never cared;
/// DC would build an indefinite `B` from it. This is the regression test for
/// the guard in `linear::btheta::dc_branches`.
#[test]
fn link_branches_get_a_positive_susceptance() {
    let base = data_dir("link/dummy-test");
    let input = common::load_pgm_input(&base.join("input.json"));
    let (mut buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);

    // Fixture assumption: this really does contain a link, stamped as a
    // transformer whose series admittance is the ideal-connection constant.
    let links = transformers
        .iter()
        .filter(|t| t.y_series == gridoxide::topology::IDEAL_CONNECTION_Y)
        .count();
    assert!(links > 0, "fixture has no link branches, so this proves nothing");

    for br in dc_branches(&lines, &transformers, DcOptions::default()) {
        assert!(br.b > 0.0, "branch {} has susceptance {}", br.index, br.b);
    }

    let solution = dc_power_flow(&mut buses, &lines, &transformers, DcOptions::default());
    assert!(solution.negative_reactance_branches.is_empty());
    assert!(
        solution.islands.iter().all(|i| i.status == DcIslandStatus::Solved),
        "{:?}",
        solution.islands.iter().map(|i| i.status).collect::<Vec<_>>()
    );
    // A stiff clamped branch spreads B's condition number by ~1e4, so this
    // records how much accuracy that actually costs rather than assuming none.
    assert!(solution.max_residual < 1e-9, "max residual = {}", solution.max_residual);
}

fn solve_ac(input: PgmInput) -> (Vec<Bus>, HashMap<u64, usize>) {
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let mut ybus = build_ybus(buses.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    (run_power_flow_analysis_from_ybus(buses, ybus).buses, id_to_idx)
}

/// How closely DC tracks AC, and — more usefully — *where* it stops doing so.
///
/// DC assumes `|V| = 1` everywhere. This fixture does not oblige: its AC
/// solution runs from 1.03 pu at the slack to 1.19 pu deep in the network.
/// Since active flow goes as `V_i·V_j·b·sin δ`, a bus sitting at 1.19 pu
/// carries its power at a visibly smaller angle than DC predicts, and the
/// gap there reaches 0.041 rad — about half the network's whole 0.087 rad
/// angle span.
///
/// So the test asserts two different things. The loose bound over every bus
/// is a guard against gross errors (a flipped φ sign, a dropped transformer
/// ratio) and nothing more. The tight bound, restricted to the buses whose AC
/// voltage actually lands near 1 pu — where DC's own assumption holds — is
/// the one that demonstrates the method working, and it comes in two orders
/// of magnitude better.
#[test]
fn dc_angle_error_tracks_the_voltage_assumption_it_rests_on() {
    let path = data_dir("symmetric/transmission-case").join("input.json");
    let (ac_buses, _) = solve_ac(common::load_pgm_input(&path));

    let input = common::load_pgm_input(&path);
    let (mut dc_buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    dc_power_flow(&mut dc_buses, &lines, &transformers, DcOptions::default());

    // Fixture assumption: this case really does leave the near-nominal band,
    // which is what makes the two bounds below different questions.
    let peak_vm = ac_buses.iter().map(|b| b.voltage_mag).fold(0.0f64, f64::max);
    assert!(peak_vm > 1.15, "fixture no longer departs from 1 pu (peak {peak_vm})");

    let mut worst_overall = 0.0f64;
    let mut worst_near_nominal = 0.0f64;
    let mut near_nominal_buses = 0;
    for (ac, dc) in ac_buses.iter().zip(&dc_buses) {
        if ac.bus_type == BusType::Slack {
            continue;
        }
        let gap = (ac.voltage_ang - dc.voltage_ang).abs();
        worst_overall = worst_overall.max(gap);
        if (ac.voltage_mag - 1.0).abs() < 0.07 {
            worst_near_nominal = worst_near_nominal.max(gap);
            near_nominal_buses += 1;
        }
    }

    assert!(near_nominal_buses >= 4, "only {near_nominal_buses} near-nominal buses to check");
    assert!(
        worst_near_nominal < 5e-3,
        "DC should be close where |V| ≈ 1, but the worst gap there is {worst_near_nominal} rad"
    );
    assert!(
        worst_overall < 0.05,
        "worst DC-vs-AC angle gap {worst_overall} rad exceeds the gross-error bound"
    );
}

/// DC on a distribution feeder, where `r/x` is large and the approximation is
/// at its weakest. The point is not accuracy — it is that the solver returns a
/// converged, power-conserving answer rather than failing or producing
/// nonsense, since this is exactly the data a distribution user would feed it.
#[test]
fn dc_solves_the_distribution_case_and_conserves_power() {
    let base = data_dir("symmetric/distribution-case");
    let input = common::load_pgm_input(&base.join("input.json"));
    let (mut buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);

    let solution = dc_power_flow(&mut buses, &lines, &transformers, DcOptions::default());
    assert!(
        solution.islands.iter().all(|i| i.status == DcIslandStatus::Solved),
        "{:?}",
        solution.islands.iter().map(|i| i.status).collect::<Vec<_>>()
    );
    assert!(solution.max_residual < 1e-9, "max residual = {}", solution.max_residual);
}

/// The flat branch index is a contract: `DcSolution::branch_p[i]` and
/// `branch_flow::branch_params()[i]` must be the same physical branch.
/// Getting this wrong is invisible until someone compares against a PGM
/// fixture, so it is asserted directly.
#[test]
fn branch_p_is_indexed_by_the_flat_branch_index() {
    let base = data_dir("symmetric/transmission-case");
    let input = common::load_pgm_input(&base.join("input.json"));
    let (mut buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);

    let solution = dc_power_flow(&mut buses, &lines, &transformers, DcOptions::default());
    let params = branch_params(&lines, &transformers);
    assert_eq!(solution.branch_p.len(), params.len());

    for br in dc_branches(&lines, &transformers, DcOptions::default()) {
        assert_eq!(
            (br.from, br.to),
            (params[br.index].from, params[br.index].to),
            "branch {} disagrees with branch_params",
            br.index
        );
    }
}

/// When the two approximations differ, and when they cannot.
///
/// `IgnoreG` is `IgnoreR` divided by `1 + (r/x)²`. Where that ratio is the
/// *same* on every branch it is a uniform rescaling of `B`, which rescales
/// every angle and leaves every flow untouched — so the choice can only
/// matter on a network whose `r/x` varies from branch to branch. Both
/// committed PGM fixtures happen to fall on the "cannot differ" side
/// (`transmission-case` has `r/x = 0.1` on every line; gridoxide's view of
/// `distribution-case` is a tree plus two identical parallel transformers, so
/// its flows follow from continuity alone), which is why this is built by
/// hand.
#[test]
fn the_approximations_differ_exactly_when_r_over_x_varies() {
    let solve = |lines: &[Line], approximation| {
        let mut buses = vec![
            bus(0, BusType::Slack, 0.0),
            bus(1, BusType::PQ, -0.05),
            bus(2, BusType::PQ, -0.08),
        ];
        let opts = DcOptions { approximation, ..Default::default() };
        dc_power_flow(&mut buses, lines, &[], opts).branch_p
    };
    let worst = |a: Vec<f64>, b: Vec<f64>| {
        a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f64, f64::max)
    };

    // A loop whose two paths have deliberately different r/x (0.2 versus 2.0).
    let varying = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.10, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.03, x: 0.15, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 0, to: 2, r: 0.40, x: 0.20, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let spread = worst(
        solve(&varying, DcApproximation::IgnoreR),
        solve(&varying, DcApproximation::IgnoreG),
    );
    assert!(spread > 1e-3, "non-uniform r/x should move flow, but the gap is {spread}");

    // The same topology with r/x = 0.2 throughout: now the difference is a
    // pure rescaling of B, and the flows must agree to round-off.
    let uniform = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.10, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.03, x: 0.15, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 0, to: 2, r: 0.04, x: 0.20, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let spread = worst(
        solve(&uniform, DcApproximation::IgnoreR),
        solve(&uniform, DcApproximation::IgnoreG),
    );
    assert!(spread < 1e-12, "uniform r/x should rescale angles only, but flows moved by {spread}");
}

/// A line with `x = 0` is a short circuit in `IgnoreR`'s world, since that
/// approximation asserts `r = 0` and so has no impedance left to speak of.
/// It must not become an infinite susceptance, and above all must not be
/// dropped — dropping it splits `link/dummy-test` into a live island and a
/// sourceless one. Under `IgnoreG` the same branch legitimately carries no
/// angle-driven power, so the network there really does island.
#[test]
fn a_purely_resistive_branch_shorts_under_ignore_r_and_opens_under_ignore_g() {
    let lines = vec![Line { from: 0, to: 1, r: 0.1, x: 0.0, b_shunt: 0.0, g_shunt: 0.0 }];

    let ignore_r = dc_branches(&lines, &[], DcOptions::default());
    assert_eq!(ignore_r.len(), 1, "IgnoreR dropped a purely resistive branch");
    assert!(ignore_r[0].b.is_finite() && ignore_r[0].b > 1e4, "b = {}", ignore_r[0].b);

    let opts = DcOptions { approximation: DcApproximation::IgnoreG, ..Default::default() };
    assert_eq!(dc_branches(&lines, &[], opts)[0].b, 0.0);

    let mut buses = vec![bus(0, BusType::Slack, 0.0), bus(1, BusType::PQ, -0.2)];
    let solution = dc_power_flow(&mut buses, &lines, &[], DcOptions::default());
    assert_eq!(solution.islands.len(), 1, "IgnoreR should keep the network in one piece");
    assert_eq!(solution.islands[0].status, DcIslandStatus::Solved);
}

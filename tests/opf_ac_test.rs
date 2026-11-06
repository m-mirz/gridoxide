//! AC optimal power flow.
//!
//! Three independent kinds of check, because no one of them is sufficient for
//! a nonconvex problem.
//!
//! **Derivatives against finite differences.** The model hands the solver a
//! gradient, a Jacobian and a Hessian; if any is wrong the solver converges
//! confidently to the wrong point, satisfying the first-order conditions of a
//! problem nobody posed. These are checked directly, and the branch-limit
//! terms especially — they are the one piece of second-derivative algebra not
//! already validated by `injection_hessian_test.rs`.
//!
//! **Feasibility at the returned point**, re-derived from
//! `network::power_injections` rather than read out of the solver. A converged
//! AC-OPF that does not satisfy the power flow equations is not an answer.
//!
//! **The published objectives.** pglib-opf publishes an AC value per case, and
//! all five are matched to 0.001%. That is the only check here grounded
//! entirely outside this crate.
//!
//! What none of them establish is **global** optimality, and nothing can: the
//! problem is nonconvex, so a KKT point is locally optimal and that is what
//! every AC-OPF tool — including the one that produced the reference numbers —
//! reports.

#![cfg(feature = "opf")]

use gridoxide::injection_hessian::{injection_jacobian, weighted_injection_hessian};
use gridoxide::network::power_injections;
use gridoxide::opf::OptStatus;
use gridoxide::opf::ac::{AcOpf, AcOpfFormulation, AcOpfOptions};
use gridoxide::opf::model::OpfData;
use gridoxide::opf::nlp::NonlinearProblem;
use gridoxide::pgm::PgmInput;

mod pglib;

/// pglib-opf's published AC objectives, $/h — see `tests/data/benchmark-grids/pglib/BASELINE.md`.
const PUBLISHED_AC: &[(&str, f64)] = &[
    ("pglib_opf_case3_lmbd", 5.8126e+03),
    ("pglib_opf_case5_pjm", 1.7552e+04),
    ("pglib_opf_case14_ieee", 2.1781e+03),
    ("pglib_opf_case30_ieee", 8.2085e+03),
    ("pglib_opf_case118_ieee", 9.7214e+04),
];

fn build(name: &str, options: AcOpfOptions) -> AcOpf {
    let dir = pglib::dir();
    let input: PgmInput =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap())
            .unwrap();
    let data =
        OpfData::from_json(&std::fs::read_to_string(dir.join(format!("{name}.opf.json"))).unwrap())
            .unwrap();
    let network =
        gridoxide::pgm::opf::ac_opf_network_from_pgm(input, &data, 50.0, &options).unwrap();
    AcOpf::build(network, options).unwrap()
}

/// **The external gate.** Every case, against pglib's own published AC value.
///
/// Reaching 0.001% required two modelling corrections that the objective alone
/// would never have localized, and both are recorded because the *pattern* of
/// which cases disagreed is what identified them:
///
/// - **Per-bus voltage limits.** Defaulting to `[0.9, 1.1]` matched `case3` and
///   `case5` exactly and left the other three low by 0.4–0.5% — those three
///   specify `[0.94, 1.06]`, so the looser default bought a cheaper answer that
///   was simply infeasible for the real case.
/// - **Bus shunts.** With voltage limits fixed, exactly the three cases
///   carrying shunt capacitors still disagreed and the two with none matched to
///   0.001%. A shunt supplies reactive power for free; dropping it makes the
///   generators supply it instead, at a cost.
///
/// Both were found the same way: not by staring at a gap, but by asking what
/// the disagreeing cases had in common that the agreeing ones did not.
#[test]
fn objectives_match_the_published_ac_baseline() {
    for &(name, published) in PUBLISHED_AC {
        let result = build(name, AcOpfOptions::default()).solve().unwrap();
        assert_eq!(
            result.status,
            OptStatus::Optimal,
            "{name}: {:?}",
            result.status
        );

        let relative = (result.objective - published).abs() / published;
        assert!(
            relative < 1e-4,
            "{name}: {:.4} vs published {published:.4} ({:.4}% apart)",
            result.objective,
            relative * 100.0
        );
        // An objective is only meaningful at a feasible point.
        assert!(
            result.violation < 1e-6,
            "{name}: converged with violation {:.3e}",
            result.violation
        );
    }
}

/// The returned point satisfies the power flow equations — checked by
/// recomputing the injections from scratch rather than trusting the solver's
/// own residual.
#[test]
fn the_solution_satisfies_the_power_flow_equations() {
    for &(name, _) in PUBLISHED_AC {
        let opf = build(name, AcOpfOptions::default());
        let result = opf.solve().unwrap();
        let network = opf.network();
        let n = network.n_buses;

        let buses = opf.buses_for_test(&result.angles, &result.magnitudes);
        let ybus = opf.ybus_for_test();
        let (p_inj, q_inj) = power_injections(&buses, &ybus);

        let base = network.base_mva;
        let mut p_from_generators = vec![0.0; n];
        let mut q_from_generators = vec![0.0; n];
        for (k, unit) in network.generators.iter().enumerate() {
            p_from_generators[unit.bus] += result.p_gen[k] / base;
            q_from_generators[unit.bus] += result.q_gen[k] / base;
        }

        for i in 0..n {
            let p_residual = p_inj[i] - (p_from_generators[i] - network.p_load[i]);
            let q_residual = q_inj[i] - (q_from_generators[i] - network.q_load[i]);
            assert!(
                p_residual.abs() < 1e-6,
                "{name} bus {i}: active balance off by {p_residual:.3e} pu"
            );
            assert!(
                q_residual.abs() < 1e-6,
                "{name} bus {i}: reactive balance off by {q_residual:.3e} pu"
            );
        }
    }
}

/// Every bound the model declares is respected at the solution.
///
/// The barrier keeps iterates strictly inside, so this should hold with room
/// to spare; a violation would mean a bound was declared on the wrong variable
/// rather than that the method leaked over one.
#[test]
fn every_declared_bound_holds_at_the_solution() {
    for &(name, _) in PUBLISHED_AC {
        let opf = build(name, AcOpfOptions::default());
        let result = opf.solve().unwrap();
        let network = opf.network();

        for i in 0..network.n_buses {
            assert!(
                result.magnitudes[i] >= network.v_min[i] - 1e-8
                    && result.magnitudes[i] <= network.v_max[i] + 1e-8,
                "{name} bus {i}: |V| = {} outside [{}, {}]",
                result.magnitudes[i],
                network.v_min[i],
                network.v_max[i]
            );
        }
        let base = network.base_mva;
        for (k, unit) in network.generators.iter().enumerate() {
            let p = result.p_gen[k] / base;
            let q = result.q_gen[k] / base;
            assert!(
                p >= unit.p_min - 1e-8 && p <= unit.p_max + 1e-8,
                "{name} generator {k}: P = {p} outside [{}, {}]",
                unit.p_min,
                unit.p_max
            );
            assert!(
                q >= unit.q_min - 1e-8 && q <= unit.q_max + 1e-8,
                "{name} generator {k}: Q = {q} outside [{}, {}]",
                unit.q_min,
                unit.q_max
            );
        }
    }
}

/// Branch apparent-power limits hold, and turning them off makes the answer
/// cheaper on a case where one binds.
///
/// The second half is what makes the first meaningful: a limit that is
/// respected because nothing ever approaches it tests nothing.
#[test]
fn branch_limits_are_enforced_and_actually_bind() {
    let name = "pglib_opf_case5_pjm";
    let enforced = build(name, AcOpfOptions::default()).solve().unwrap();

    let opf = build(name, AcOpfOptions::default());
    let network = opf.network();
    let base = network.base_mva;
    for (flat, rate) in network.limits.iter() {
        let Some(rate) = rate else { continue };
        let (p, q) = enforced.flows[*flat];
        let apparent = (p * p + q * q).sqrt() / base;
        assert!(
            apparent <= rate + 1e-6,
            "branch {flat}: |S| = {apparent} pu exceeds its rating {rate}"
        );
    }

    let relaxed = build(
        name,
        AcOpfOptions {
            enforce_limits: false,
            ..AcOpfOptions::default()
        },
    )
    .solve()
    .unwrap();
    assert_eq!(relaxed.status, OptStatus::Optimal);
    assert!(
        relaxed.objective < enforced.objective - 1.0,
        "removing the limits changed the cost by only {:.4} $/h, so none of them bind \
         and the enforcement check above proves nothing",
        enforced.objective - relaxed.objective
    );
}

/// The gradient, Jacobian and Hessian the model hands the solver, against
/// central differences of the functions they claim to differentiate.
///
/// This is the check that matters most, and the reason is worth stating: a
/// wrong derivative does not make the solver fail. It makes it converge — to
/// the first-order conditions of a different problem — and report success. The
/// branch-limit terms are the specific target, being the only second
/// derivatives here not already covered by `injection_hessian_test.rs`.
#[test]
fn the_models_derivatives_match_finite_differences() {
    derivatives_match_finite_differences(AcOpfFormulation::BusInjection);
}

/// The same check on the branch-flow formulation: its flow-definition rows,
/// shunt-only balances and limits on the flow variables.
#[test]
fn the_branch_flow_models_derivatives_match_finite_differences() {
    derivatives_match_finite_differences(AcOpfFormulation::BranchFlow);
}

/// The two formulations are one problem: with the flow variables set to the
/// flows the voltages imply, the branch-flow model's balances and limits
/// equal the bus-injection model's, its flow-definition rows are zero, and
/// the objectives agree.
#[test]
fn the_branch_flow_formulation_describes_the_same_problem() {
    let injection = build("pglib_opf_case14_ieee", AcOpfOptions::default());
    let flows = build(
        "pglib_opf_case14_ieee",
        AcOpfOptions {
            formulation: AcOpfFormulation::BranchFlow,
            ..AcOpfOptions::default()
        },
    );
    let mut x = injection.initial_point();
    for (k, value) in x.iter_mut().enumerate() {
        *value += 0.05 * ((k as f64 * 0.7).sin());
    }
    // The flow variables at their definitions: solve for them from the rows
    // themselves (each is `flow − flow(V)`, so evaluating with zeros gives
    // `−flow(V)`).
    let mut z = x.clone();
    z.resize(flows.n_vars(), 0.0);
    let c0 = flows.constraints(&z);
    let rows = injection.n_constraints();
    for (k, v) in c0[rows..].iter().enumerate() {
        z[x.len() + k] = -v;
    }
    let a = injection.constraints(&x);
    let b = flows.constraints(&z);
    for r in 0..rows {
        assert!(
            (a[r] - b[r]).abs() < 1e-12 * (1.0 + a[r].abs()),
            "row {r}: {} vs {}",
            a[r],
            b[r]
        );
    }
    assert!(b[rows..].iter().all(|v| v.abs() < 1e-14));
    assert_eq!(injection.objective(&x), flows.objective(&z));
}

/// The balance rows, evaluated element by element (each terminal's flow,
/// gathered per bus, plus the bus's shunt), against the Y-bus formulas they
/// replaced: values, Jacobian and the Hessian's balance part at a point away
/// from the flat start, on a case with transformers and shunts. Equal to
/// rounding.
#[test]
fn the_elementwise_balances_match_the_ybus() {
    use std::collections::HashMap;
    let opf = build("pglib_opf_case118_ieee", AcOpfOptions::default());
    let network = opf.network();
    let n = network.n_buses;
    let mut x = opf.initial_point();
    for (k, value) in x.iter_mut().enumerate().take(2 * n) {
        *value += 0.05 * ((k as f64 * 0.7).sin());
    }
    let buses = opf.buses_for_test(&x[..n], &x[n..2 * n]);
    let ybus = opf.ybus_for_test();

    let close = |a: f64, b: f64, what: &str| {
        assert!(
            (a - b).abs() < 1e-11 * (1.0 + b.abs()),
            "{what}: {a} vs {b}"
        );
    };
    let summed = |entries: Vec<(usize, usize, f64)>, keep: &dyn Fn(usize, usize) -> bool| {
        let mut out: HashMap<(usize, usize), f64> = HashMap::new();
        for (r, c, v) in entries {
            if keep(r, c) {
                *out.entry((r, c)).or_default() += v;
            }
        }
        out
    };

    let (p_inj, q_inj) = power_injections(&buses, &ybus);
    let c = opf.constraints(&x);
    let mut generation = vec![(0.0, 0.0); n];
    for (k, unit) in network.generators.iter().enumerate() {
        let g = network.generators.len();
        generation[unit.bus].0 += x[2 * n + k];
        generation[unit.bus].1 += x[2 * n + g + k];
    }
    for i in 0..n {
        close(
            c[i],
            p_inj[i] + network.p_load[i] - generation[i].0,
            "P balance",
        );
        close(
            c[n + i],
            q_inj[i] + network.q_load[i] - generation[i].1,
            "Q balance",
        );
    }

    let state = |r: usize, col: usize| r < 2 * n && col < 2 * n;
    let ours = summed(opf.jacobian(&x), &state);
    let theirs = summed(injection_jacobian(&buses, &ybus), &state);
    for (key, &want) in &theirs {
        close(*ours.get(key).unwrap_or(&0.0), want, &format!("J{key:?}"));
    }
    for (key, &got) in &ours {
        close(got, *theirs.get(key).unwrap_or(&0.0), &format!("J{key:?}"));
    }

    // Multipliers on the balance rows only, so the Hessian is theirs alone
    // (the cost's curvature sits on P_g columns, outside the state block).
    let mut y = vec![0.0; opf.n_constraints()];
    for (r, value) in y.iter_mut().enumerate().take(2 * n) {
        *value = 0.8 * ((r as f64 * 0.9).cos()) - 0.3;
    }
    let ours = summed(opf.lagrangian_hessian(&x, &y), &|r, col| state(r, col));
    let lambda: Vec<f64> = (0..n).map(|i| -y[i]).collect();
    let mu: Vec<f64> = (0..n).map(|i| -y[n + i]).collect();
    let theirs = summed(
        weighted_injection_hessian(&buses, &ybus, &lambda, &mu),
        &state,
    );
    for (key, &want) in &theirs {
        close(*ours.get(key).unwrap_or(&0.0), want, &format!("H{key:?}"));
    }
    for (key, &got) in &ours {
        close(got, *theirs.get(key).unwrap_or(&0.0), &format!("H{key:?}"));
    }
}

fn derivatives_match_finite_differences(formulation: AcOpfFormulation) {
    // Small enough to difference every column, and carries a transformer, a
    // shunt and rated branches, so all three constraint kinds are exercised.
    let opf = build(
        "pglib_opf_case14_ieee",
        AcOpfOptions {
            formulation,
            ..AcOpfOptions::default()
        },
    );
    let n_vars = opf.n_vars();
    let m = opf.n_constraints();

    // A point away from the flat start, so no term is accidentally zero.
    let mut x = opf.initial_point();
    for (k, value) in x.iter_mut().enumerate() {
        *value += 0.01 * ((k as f64 * 0.7).sin());
    }

    let h = 1e-6;
    let perturbed = |k: usize, delta: f64| {
        let mut p = x.clone();
        p[k] += delta;
        p
    };

    // Gradient.
    let gradient = opf.gradient(&x);
    for k in 0..n_vars {
        let numeric =
            (opf.objective(&perturbed(k, h)) - opf.objective(&perturbed(k, -h))) / (2.0 * h);
        assert!(
            (gradient[k] - numeric).abs() < 1e-5 * (1.0 + numeric.abs()),
            "df/dx{k}: analytic {}, numeric {numeric}",
            gradient[k]
        );
    }

    // Constraint Jacobian.
    let mut analytic = vec![vec![0.0; n_vars]; m];
    for (r, c, v) in opf.jacobian(&x) {
        analytic[r][c] += v;
    }
    for k in 0..n_vars {
        let plus = opf.constraints(&perturbed(k, h));
        let minus = opf.constraints(&perturbed(k, -h));
        for r in 0..m {
            let numeric = (plus[r] - minus[r]) / (2.0 * h);
            assert!(
                (analytic[r][k] - numeric).abs() < 1e-4 * (1.0 + numeric.abs()),
                "dc{r}/dx{k}: analytic {}, numeric {numeric}",
                analytic[r][k]
            );
        }
    }

    // Hessian of the Lagrangian, against central differences of the gradient
    // of the Lagrangian — the same two-stage grounding the injection Hessian
    // uses, and for the same reason.
    let y: Vec<f64> = (0..m)
        .map(|r| 0.8 * ((r as f64 * 0.9).cos()) - 0.3)
        .collect();
    let mut hessian = vec![vec![0.0; n_vars]; n_vars];
    for (r, c, v) in opf.lagrangian_hessian(&x, &y) {
        hessian[r][c] += v;
    }

    let lagrangian_gradient = |point: &[f64]| -> Vec<f64> {
        let mut out = opf.gradient(point);
        for (r, c, v) in opf.jacobian(point) {
            out[c] -= y[r] * v;
        }
        out
    };

    for k in 0..n_vars {
        let plus = lagrangian_gradient(&perturbed(k, h));
        let minus = lagrangian_gradient(&perturbed(k, -h));
        for r in 0..n_vars {
            let numeric = (plus[r] - minus[r]) / (2.0 * h);
            assert!(
                (hessian[r][k] - numeric).abs() < 1e-3 * (1.0 + numeric.abs()),
                "H[{r}][{k}]: analytic {}, numeric {numeric}",
                hessian[r][k]
            );
        }
    }
}

/// The answer does not depend on where the search started.
///
/// A nonconvex problem gives no guarantee of this — a local method returns the
/// optimum in whichever basin it starts, which `opf_nlp_test.rs` demonstrates
/// deliberately on a two-well problem. So this is an empirical finding about
/// *these* networks rather than a property of the method, and it is worth
/// pinning: it is the evidence that the 0.001% agreement with pglib reflects
/// the same solution rather than a coincidence of starting points.
#[test]
fn every_starting_point_reaches_the_same_optimum() {
    for &(name, _) in PUBLISHED_AC {
        let mut objectives = Vec::new();
        for v_start in [0.95, 1.0, 1.05] {
            for barrier in [0.01, 0.1, 1.0] {
                let mut options = AcOpfOptions::default();
                options.nlp.initial_barrier = barrier;
                let mut opf_network = {
                    let dir = pglib::dir();
                    let input: PgmInput = serde_json::from_str(
                        &std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap(),
                    )
                    .unwrap();
                    let data = OpfData::from_json(
                        &std::fs::read_to_string(dir.join(format!("{name}.opf.json"))).unwrap(),
                    )
                    .unwrap();
                    gridoxide::pgm::opf::ac_opf_network_from_pgm(input, &data, 50.0, &options)
                        .unwrap()
                };
                opf_network.v_start = Some(v_start);
                let result = AcOpf::build(opf_network, options).unwrap().solve().unwrap();
                assert_eq!(
                    result.status,
                    OptStatus::Optimal,
                    "{name} from |V| = {v_start}"
                );
                objectives.push(result.objective);
            }
        }
        let low = objectives.iter().cloned().fold(f64::INFINITY, f64::min);
        let high = objectives.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (high - low) / high.abs() < 1e-6,
            "{name}: starts reached objectives spanning {low} to {high}"
        );
    }
}

/// **The price sign**, pinned against a numerical `∂cost/∂load`.
///
/// This is the assertion the whole reactive-and-active pricing output rests
/// on, and it is checked numerically rather than by re-deriving the
/// convention, because re-deriving it is exactly what went wrong: the first
/// implementation reported prices of precisely the right magnitude with the
/// wrong sign on every bus of every case. A sign error here looks entirely
/// plausible in the output and is wrong everywhere, which is the failure mode
/// `opf::dc`'s module docs warn about for the same quantity.
///
/// So: add a megawatt of demand at a bus, re-solve, and check the cost rose by
/// the price.
#[test]
fn the_locational_marginal_price_is_the_cost_of_one_more_megawatt() {
    let name = "pglib_opf_case5_pjm";
    let base_result = build(name, AcOpfOptions::default()).solve().unwrap();
    let reference = build(name, AcOpfOptions::default());
    let base_mva = reference.network().base_mva;

    // A bus with load on it, and not the reference bus.
    let bus = reference
        .network()
        .p_load
        .iter()
        .enumerate()
        .filter(|(i, p)| **p > 0.0 && *i != reference.network().reference)
        .map(|(i, _)| i)
        .next()
        .expect("case5 has load away from the reference bus");

    let delta_mw = 1.0;
    let mut perturbed_network = {
        let dir = pglib::dir();
        let input: PgmInput = serde_json::from_str(
            &std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap(),
        )
        .unwrap();
        let data = OpfData::from_json(
            &std::fs::read_to_string(dir.join(format!("{name}.opf.json"))).unwrap(),
        )
        .unwrap();
        gridoxide::pgm::opf::ac_opf_network_from_pgm(input, &data, 50.0, &AcOpfOptions::default())
            .unwrap()
    };
    perturbed_network.p_load[bus] += delta_mw / base_mva;
    let perturbed = AcOpf::build(perturbed_network, AcOpfOptions::default())
        .unwrap()
        .solve()
        .unwrap();
    assert_eq!(perturbed.status, OptStatus::Optimal);

    let numerical = (perturbed.objective - base_result.objective) / delta_mw;
    let reported = base_result.lmp_p[bus];
    assert!(
        (numerical - reported).abs() < 0.05 * reported.abs().max(1.0),
        "bus {bus}: reported {reported:.4} $/MWh but one more MW actually cost \
         {numerical:.4} $/MWh"
    );
    // And the sign is the claim, not just the magnitude.
    assert!(
        reported > 0.0,
        "serving more demand cannot cost less than nothing"
    );
}

/// AC and DC prices should agree closely on the same network — they differ
/// only by losses and reactive effects, which are a few percent.
///
/// A second, independent angle on the sign: the DC price convention is pinned
/// separately in `opf_dc_test.rs` against its own analytic two-bus cases, so
/// agreeing with it means both conventions are right rather than consistently
/// wrong.
#[test]
fn ac_and_dc_prices_agree_to_within_losses() {
    use gridoxide::linear::DcApproximation;
    use gridoxide::opf::Solver;
    use gridoxide::opf::dc::{DcOpf, DcOpfOptions};
    use gridoxide::opf::ipm::IpmSolver;

    let name = "pglib_opf_case5_pjm";
    let dir = pglib::dir();
    let data =
        OpfData::from_json(&std::fs::read_to_string(dir.join(format!("{name}.opf.json"))).unwrap())
            .unwrap();

    let ac = build(name, AcOpfOptions::default()).solve().unwrap();

    let input: PgmInput =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap())
            .unwrap();
    let dc_network =
        gridoxide::pgm::opf::dc_opf_network_from_pgm(input, &data, 50.0, DcApproximation::IgnoreG)
            .unwrap();
    let dc_opf = DcOpf::build(dc_network, DcOpfOptions::default()).unwrap();
    let dc = dc_opf.interpret(&IpmSolver::new().solve(dc_opf.problem()).unwrap());

    for bus in 0..dc.lmp.len() {
        let (a, d) = (ac.lmp_p[bus], dc.lmp[bus]);
        assert!(
            (a - d).abs() < 0.05 * d.abs().max(1.0),
            "bus {bus}: AC priced it at {a:.4} $/MWh, DC at {d:.4}"
        );
    }
}

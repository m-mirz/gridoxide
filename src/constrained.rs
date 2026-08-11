//! A switch as an exact equality constraint, rather than as a stiff branch.
//!
//! This is approach 3 of `docs/src/powerflow/zero_impedance_branches.md` and
//! phase 3 of `plans/NODE_BREAKER_PLAN.md`:
//! [`SwitchTreatment::Constrain`](crate::switches::SwitchTreatment::Constrain).
//! A closed switch does not become an admittance at all. Instead its power flow
//! becomes an *unknown*, and the physics it enforces — the two ends are the same
//! point electrically — becomes a pair of exact equations:
//!
//! ```text
//! θ_i − θ_j = 0        V_i − V_j = 0
//! ```
//!
//! # The system
//!
//! Two unknowns and two equations per switch, on top of the ordinary
//! Newton-Raphson state. The switch's flow `(P_s, Q_s)` enters the power balance
//! of both its ends, leaving `i` and arriving at `j`:
//!
//! ```text
//! [  J    Aᵀ ] [ Δx ]   [ mismatch ]
//! [  A    0  ] [ Δs ] = [ −c(x)    ]
//! ```
//!
//! where `J` is the Jacobian the unconstrained solver already builds, `A` holds
//! the ±1 of each constraint, and `Aᵀ` the ±1 with which each flow enters a
//! mismatch row. It is the same KKT shape
//! [`se::constraints`](crate::se::constraints) assembles for zero-injection
//! buses, for the same reason: a hard equality is not a heavily weighted
//! measurement.
//!
//! # What this buys over `Regularize`
//!
//! Not scale — [`switches`](crate::switches) records that the expected
//! conditioning ceiling did not materialize, and `Regularize` solves every
//! node-breaker configuration in the tree. Three other things:
//!
//! - **No arbitrary constant.** `Regularize` picks a stiffness; the answer
//!   depends on it, slightly, everywhere. Here there is no constant to pick, so
//!   a closed switch is *exactly* a short circuit rather than very nearly one.
//! - **An honest answer for a loop of closed switches.** Their flows are
//!   genuinely indeterminate — the network does not decide how a bus coupler
//!   splits current with a parallel one — and this reports them
//!   ([`ConstrainedSolution::indeterminate`]) rather than inventing a split from
//!   the stiffness it happened to choose.
//! - **An open switch costs nothing.** It contributes no admittance and no
//!   coupling, where `Regularize` still carries its structural entries.
//!
//! # What it costs
//!
//! The matrix grows by two rows and columns per switch, and it stops being the
//! power-flow Jacobian: the multiplier block on the diagonal is zero, so the
//! system is indefinite and no longer has a nonzero diagonal everywhere. Every
//! gridoxide backend is a general sparse LU, so this is assembly work rather
//! than a new solver — the same argument `se::constraints` makes.
//!
//! # The sparsity pattern still does not move
//!
//! An open switch keeps its rows and columns. It simply constrains a different
//! thing: `P_s = Q_s = 0` instead of `θ_i = θ_j` and `V_i = V_j`. Both forms'
//! entries are stamped in every state, one of the two carrying a structural
//! zero, so **flipping a switch changes values and not the pattern** — the
//! property `plans/NODE_BREAKER_PLAN.md` §4.2 asks for, and which
//! `switches::regularized_branches` reaches by the other route (an open
//! terminal's all-zero stamp).

use crate::jacobian::JacobianPattern;
use crate::network::{effective_injection, power_injections, YBusSparse};
use crate::solver::{LinearSolver, SolveStats, SolveStatus};
use crate::sparse::RealSparseSystem;
use crate::types::{Bus, BusType};

/// A switch to be represented as a constraint rather than as a branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstrainedSwitch {
    pub from: usize,
    pub to: usize,
    pub open: bool,
}

/// How one switch's two rows are being used at this switching state.
///
/// Not a configuration choice — it is derived, and the derivation is the part
/// worth naming, because a switch that *looks* closed can still fail to
/// constrain anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowUse {
    /// `θ_i − θ_j = 0` / `V_i − V_j = 0`, the real thing.
    Tie,
    /// `P_s = 0` / `Q_s = 0`: the flow is pinned because nothing determines it.
    Pin,
}

/// The outcome of a constrained solve.
#[derive(Clone, Debug)]
pub struct ConstrainedSolution {
    pub stats: SolveStats,
    /// Per switch, the flow `(P, Q)` leaving its `from` end, in per unit.
    ///
    /// Exactly zero for an open switch, and for every switch named in
    /// [`indeterminate`](Self::indeterminate).
    pub flows: Vec<(f64, f64)>,
    /// Switches whose flow the network does not determine, so this pinned it to
    /// zero to keep the system nonsingular.
    ///
    /// Three ways in, all reported the same way because the caller's question is
    /// the same one — *is this number physical?*:
    ///
    /// - **A loop of closed switches.** `θ_i = θ_j` around a cycle is one
    ///   redundant equation, and the flows around it are one free parameter.
    ///   `Regularize` hides this by resolving the split with its own stiffness;
    ///   there is no more information there than here.
    /// - **A self-loop**, both ends on one bus. Nothing to tie.
    /// - **Both ends fixed** — two `Slack` buses, or two `PV` buses for the
    ///   reactive half. The constraint is vacuous and the flow is absorbed by
    ///   whatever holds those buses.
    ///
    /// The reactive half can be indeterminate while the active half is not (a
    /// switch between two `PV` buses), in which case the switch appears here and
    /// its `Q` is zero while its `P` is real. See [`ConstrainedSolution::flows`].
    pub indeterminate: Vec<usize>,
}

/// Runs Newton-Raphson with each closed switch enforced as an exact equality.
///
/// `ybus` must **not** contain the switches — that is the entire point. Build it
/// from the network's own lines and transformers, and pass the switches here.
///
/// Islands are decided across closed switches, since a switch conducts even
/// though it contributes no admittance. Without that, the two ends of every
/// closed switch would look like separate components and
/// [`mark_unreferenced_islands`](crate::network) would de-energize whichever
/// side does not hold the reference — turning a perfectly ordinary network into
/// a dead one.
pub fn solve_constrained(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    switches: &[ConstrainedSwitch],
    tol: f64,
    max_iter: usize,
) -> ConstrainedSolution {
    solve_constrained_with::<RealSparseSystem>(buses, ybus, switches, tol, max_iter)
}

/// [`solve_constrained`] against a chosen linear-solver backend.
pub fn solve_constrained_with<S: LinearSolver>(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    switches: &[ConstrainedSwitch],
    tol: f64,
    max_iter: usize,
) -> ConstrainedSolution {
    let n_sw = switches.len();
    let empty = ConstrainedSolution {
        stats: SolveStats::from_loop(SolveStatus::Singular, Vec::new()),
        flows: vec![(0.0, 0.0); n_sw],
        indeterminate: Vec::new(),
    };

    if switches.iter().any(|s| s.from >= buses.len() || s.to >= buses.len()) {
        return empty;
    }

    crate::network::mark_unreferenced_islands(
        buses,
        &crate::network::classify(buses, &components_across_switches(buses.len(), ybus, switches)),
    );

    // Which of each switch's two rows ties its ends, and which pins its flow.
    //
    // A tie is only independent if it says something the other rows do not, and
    // union-find is what decides that — one disjoint-set forest per half,
    // because the two halves fix different things. The subtlety is the seeding:
    // every bus whose angle is *already* fixed starts in one shared set, and
    // likewise every bus whose magnitude is. Then "already connected" covers
    // both ways a constraint can be redundant, without either being special-
    // cased:
    //
    // - a cycle of closed switches — the last one closing it adds no
    //   information, and its share of the flow is a free parameter;
    // - a chain whose ends are both held — two `Slack` buses joined through a
    //   run of switches makes the second tie a copy of the first, which is a
    //   duplicate row and a singular matrix. This one is not hypothetical: it is
    //   what SmallGrid's full node-breaker view does, and checking only each
    //   switch's own two ends does not catch it.
    let mut dsu_angle: Vec<usize> = (0..buses.len()).collect();
    let mut dsu_mag: Vec<usize> = (0..buses.len()).collect();
    seed_fixed(&mut dsu_angle, buses, has_angle);
    seed_fixed(&mut dsu_mag, buses, has_magnitude);

    let mut rows = Vec::with_capacity(n_sw);
    let mut indeterminate = Vec::new();
    for (k, sw) in switches.iter().enumerate() {
        let (i, j) = (sw.from, sw.to);
        let connects = !sw.open && i != j;
        let use_p = tie_or_pin(&mut dsu_angle, connects, i, j);
        let use_q = tie_or_pin(&mut dsu_mag, connects, i, j);
        if !sw.open && (use_p == RowUse::Pin || use_q == RowUse::Pin) {
            indeterminate.push(k);
        }
        rows.push((use_p, use_q));
    }

    // The base state vector, exactly as the unconstrained solver lays it out.
    let non_slack_idx: Vec<usize> =
        buses.iter().filter(|b| b.bus_type != BusType::Slack).map(|b| b.idx).collect();
    let pq_idx: Vec<usize> =
        buses.iter().filter(|b| b.bus_type == BusType::PQ).map(|b| b.idx).collect();
    let n_angle = non_slack_idx.len();
    let n_base = n_angle + pq_idx.len();
    let n = n_base + 2 * n_sw;

    let mut angle_col = vec![None; buses.len()];
    for (pos, &i) in non_slack_idx.iter().enumerate() {
        angle_col[i] = Some(pos);
    }
    let mut mag_col = vec![None; buses.len()];
    for (pos, &i) in pq_idx.iter().enumerate() {
        mag_col[i] = Some(n_angle + pos);
    }

    // The starting state, and why it needs its own Y-bus.
    //
    // `linear_initial_guess` is what every other entry point starts from
    // (`run_power_flow_analysis_from_ybus` calls it before touching Newton), and
    // on CGMES data the difference is not cosmetic: from a flat start this
    // network's first mismatch is 2.7e2 against the guess's 2.0e-3, and Newton
    // does not recover from the steps that produces — it oscillates for thirty
    // iterations without converging. Measured, not assumed.
    //
    // The guess has to see the switches, or every bus reachable only through one
    // is left flat, which is most of a node-breaker model. So it is taken
    // against a Y-bus with the closed switches stamped stiffly — the very thing
    // this formulation exists to avoid. That is not a contradiction: a starting
    // guess is discarded the moment the first step is taken, and the conditioning
    // argument is about the matrix Newton *factorizes*, which never contains it.
    let mut warm = crate::network::YBus::new(buses.len());
    for i in 0..buses.len() {
        for &(j, y) in ybus.row(i) {
            warm.add(i, j, y);
        }
    }
    let y_switch = crate::switches::switch_admittance();
    for sw in switches.iter().filter(|s| !s.open && s.from != s.to) {
        warm.add(sw.from, sw.from, y_switch);
        warm.add(sw.to, sw.to, y_switch);
        warm.add(sw.from, sw.to, -y_switch);
        warm.add(sw.to, sw.from, -y_switch);
    }
    crate::network::linear_initial_guess(buses, &warm.finish());

    let pattern = JacobianPattern::analyze(buses, ybus);
    let mut values: Vec<f64> = Vec::with_capacity(pattern.len());
    let mut flows = vec![(0.0, 0.0); n_sw];
    let mut history = Vec::with_capacity(max_iter);
    let mut system: Option<S> = None;

    for _ in 0..max_iter {
        let (p_calc, q_calc) = power_injections(buses, ybus);

        let mut rhs = vec![0.0; n];
        for (row, &i) in non_slack_idx.iter().enumerate() {
            rhs[row] = effective_injection(&buses[i]).0 - p_calc[i];
        }
        for (row, &i) in pq_idx.iter().enumerate() {
            rhs[n_angle + row] = effective_injection(&buses[i]).1 - q_calc[i];
        }
        // The switch flows leave `from` and arrive at `to`, so they enter the
        // two ends' balances with opposite signs.
        for (k, sw) in switches.iter().enumerate() {
            let (p_s, q_s) = flows[k];
            for (bus, sign) in [(sw.from, 1.0), (sw.to, -1.0)] {
                if let Some(row) = angle_col[bus] {
                    rhs[row] -= sign * p_s;
                }
                if let Some(row) = mag_col[bus] {
                    rhs[row] -= sign * q_s;
                }
            }
        }
        // And the constraints' own violations.
        for (k, sw) in switches.iter().enumerate() {
            let (use_p, use_q) = rows[k];
            rhs[n_base + 2 * k] = -match use_p {
                RowUse::Tie => buses[sw.from].voltage_ang - buses[sw.to].voltage_ang,
                RowUse::Pin => flows[k].0,
            };
            rhs[n_base + 2 * k + 1] = -match use_q {
                RowUse::Tie => buses[sw.from].voltage_mag - buses[sw.to].voltage_mag,
                RowUse::Pin => flows[k].1,
            };
        }

        let max_mis = rhs.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        history.push(max_mis);
        if max_mis < tol {
            return ConstrainedSolution {
                stats: SolveStats::from_loop(SolveStatus::Converged, history),
                flows,
                indeterminate,
            };
        }

        pattern.fill(buses, &p_calc, &q_calc, &mut values);
        let mut triplets = pattern.to_triplets(&values);
        for (k, sw) in switches.iter().enumerate() {
            let (p_col, q_col) = (n_base + 2 * k, n_base + 2 * k + 1);
            let (p_row, q_row) = (n_base + 2 * k, n_base + 2 * k + 1);
            let (use_p, use_q) = rows[k];

            // Aᵀ: how each flow enters the two ends' balance equations.
            for (bus, sign) in [(sw.from, 1.0), (sw.to, -1.0)] {
                if let Some(row) = angle_col[bus] {
                    triplets.push((row, p_col, sign));
                }
                if let Some(row) = mag_col[bus] {
                    triplets.push((row, q_col, sign));
                }
            }

            // A: the constraint rows. Both forms are stamped in every state,
            // the inactive one as a structural zero, so the pattern is the same
            // whatever the switch's position.
            for (row, col_of, flow_col, active) in [
                (p_row, &angle_col, p_col, use_p),
                (q_row, &mag_col, q_col, use_q),
            ] {
                let tie = f64::from(active == RowUse::Tie);
                for (bus, sign) in [(sw.from, 1.0), (sw.to, -1.0)] {
                    if let Some(c) = col_of[bus] {
                        triplets.push((row, c, sign * tie));
                    }
                }
                triplets.push((row, flow_col, 1.0 - tie));
            }
        }

        if system.is_none() {
            system = S::new(n, &triplets);
        }
        let Some(sys) = system.as_mut() else {
            return ConstrainedSolution {
                stats: SolveStats::from_loop(SolveStatus::Singular, history),
                flows,
                indeterminate,
            };
        };
        let Some(dx) = sys.factor_and_solve(&triplets, &rhs) else {
            return ConstrainedSolution {
                stats: SolveStats::from_loop(SolveStatus::Singular, history),
                flows,
                indeterminate,
            };
        };

        for (pos, &i) in non_slack_idx.iter().enumerate() {
            buses[i].voltage_ang += dx[pos];
        }
        for (pos, &i) in pq_idx.iter().enumerate() {
            buses[i].voltage_mag += dx[n_angle + pos];
        }
        for (k, flow) in flows.iter_mut().enumerate() {
            flow.0 += dx[n_base + 2 * k];
            flow.1 += dx[n_base + 2 * k + 1];
        }
    }

    ConstrainedSolution {
        stats: SolveStats::from_loop(SolveStatus::MaxIterationsReached, history),
        flows,
        indeterminate,
    }
}

/// Puts every bus whose unknown is *not* free into one shared set, so a
/// constraint between two of them reads as already-satisfied rather than as new
/// information.
fn seed_fixed(dsu: &mut [usize], buses: &[Bus], free: fn(&Bus) -> bool) {
    let mut anchor: Option<usize> = None;
    for (i, bus) in buses.iter().enumerate() {
        if free(bus) {
            continue;
        }
        match anchor {
            None => anchor = Some(i),
            Some(a) => {
                let (x, y) = (find(dsu, i), find(dsu, a));
                dsu[x] = y;
            }
        }
    }
}

/// Ties `i` to `j` if that is new information, and records the union if so.
fn tie_or_pin(dsu: &mut [usize], connects: bool, i: usize, j: usize) -> RowUse {
    if !connects {
        return RowUse::Pin;
    }
    let (a, b) = (find(dsu, i), find(dsu, j));
    if a == b {
        return RowUse::Pin;
    }
    dsu[a] = b;
    RowUse::Tie
}

fn has_angle(bus: &Bus) -> bool {
    bus.bus_type != BusType::Slack
}

fn has_magnitude(bus: &Bus) -> bool {
    bus.bus_type == BusType::PQ
}

fn find(dsu: &mut [usize], mut x: usize) -> usize {
    while dsu[x] != x {
        dsu[x] = dsu[dsu[x]];
        x = dsu[x];
    }
    x
}

/// Connected components of the Y-bus, further merged across closed switches.
///
/// A closed switch conducts without contributing admittance, which is precisely
/// the case `network::connected_components` cannot see on its own.
fn components_across_switches(
    n: usize,
    ybus: &YBusSparse,
    switches: &[ConstrainedSwitch],
) -> Vec<Vec<usize>> {
    let base = crate::network::connected_components(ybus);
    let mut dsu: Vec<usize> = (0..n).collect();
    for component in &base {
        for w in component.windows(2) {
            let (a, b) = (find(&mut dsu, w[0]), find(&mut dsu, w[1]));
            dsu[a] = b;
        }
    }
    for sw in switches.iter().filter(|s| !s.open) {
        if sw.from < n && sw.to < n {
            let (a, b) = (find(&mut dsu, sw.from), find(&mut dsu, sw.to));
            dsu[a] = b;
        }
    }

    let mut of_root: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut out: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let root = find(&mut dsu, i);
        let slot = *of_root.entry(root).or_insert_with(|| {
            out.push(Vec::new());
            out.len() - 1
        });
        out[slot].push(i);
    }
    out
}

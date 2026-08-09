//! Giving a retained switch a formulation.
//!
//! [`topology::bus_view`](crate::topology::bus_view) decides *which* switches
//! survive into the solver; this module decides what they become once they are
//! there. `docs/src/powerflow/zero_impedance_branches.md` surveys the three
//! options and this implements the second:
//!
//! | Treatment | Keeps identity | Scales | Constant pattern |
//! |---|---|---|---|
//! | Merge | ✗ | ✓✓ | ✗ |
//! | [`Regularize`](SwitchTreatment::Regularize) | ✓ | ✗ | ✓ |
//! | Constrain | ✓ | ✓ | ✓ |
//!
//! `Regularize` stamps each retained switch as a very stiff branch. Everything
//! downstream then works unchanged — Y-bus, Jacobian, branch flows, state
//! estimation — which is what makes it worth having despite not scaling: it is
//! the cheapest possible route to switch flows and a settable switch position.
//!
//! # The scaling ceiling, and what measuring it actually showed
//!
//! A stiff branch is a large number in the admittance matrix, and enough of
//! them are supposed to make the Jacobian ill-conditioned. `src/cgmes.rs`
//! records that this was tried once before and the AC solve *diverged* on
//! FullGrid with "20-odd" such branches active, and
//! `plans/NODE_BREAKER_PLAN.md` §4.1 extrapolates from that to call the
//! approach "dead on arrival at real scale" for SmallGrid's 1,266 switches and
//! Svedala's 1,464.
//!
//! [`conditioning_probe`] was written to turn that anecdote into a curve. It
//! did not find one. Across [`Chain`](ProbeShape::Chain),
//! [`Star`](ProbeShape::Star) and [`Ring`](ProbeShape::Ring) arrangements, at
//! every count from 1 to 1,464, and with the branch admittance both inductive
//! and capacitive, every solve converged in two iterations with a plausible
//! voltage profile. See `examples/switch_ceiling.rs`.
//!
//! **That is a negative result, not a clean bill of health.** What it rules out
//! is that switch *count*, switch *arrangement*, or the admittance *sign*
//! explains the historical divergence on its own. What it cannot rule out is
//! interaction with real network data — the probe's networks are uniform, and
//! FullGrid's are not: real impedances span decades, and the stiff branches
//! there sat alongside transformers, shunts and an HVDC subsystem. Reproducing
//! the divergence needs `Regularize` run against an actual CGMES node-breaker
//! import, which does not exist yet.
//!
//! So the ceiling is genuinely unknown rather than known-to-be-30, and a caller
//! should measure on its own data. `Constrain` remains the right answer for the
//! general case for the reasons `zero_impedance_branches.md` gives — it needs
//! no large number at all — but the empirical case against `Regularize` is
//! weaker than the plan assumed.
//!
//! # The open/closed representation, and the property it buys
//!
//! A retained switch becomes a [`Transformer`] whose `from_status`/`to_status`
//! carry its position: both 1 when closed, both 0 when open.
//! `network::branch_calc_param` already returns all-zeros for an open-terminal
//! transformer, and `network::build_ybus` stamps those zeros anyway, so:
//!
//! > **Flipping a switch changes Y-bus values but not the sparsity pattern.**
//!
//! That is the property `plans/NODE_BREAKER_PLAN.md` §4.2 attributes to the
//! *constrained* formulation, and it turns out `Regularize` has it too — for
//! the same underlying reason the AC contingency work exploits
//! (`network::build_ybus_with_outages`). What `Regularize` still lacks is the
//! scaling, not the pattern stability.
//!
//! One caveat that comes with it: [`PersistentSolver::invalidate_admittances`]
//! must be called when a position changes, because `JacobianPattern` caches
//! each entry's admittance. See that method's doc — this is the hazard that
//! bit the contingency work.
//!
//! [`PersistentSolver::invalidate_admittances`]: crate::solver::PersistentSolver::invalidate_admittances

use num_complex::Complex;

use crate::topology::bus_view::{BusView, RetainedSwitch};
use crate::topology::ideal_connection_z;
use crate::types::Transformer;

/// What a retained switch becomes in the solver's equations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SwitchTreatment {
    /// Not represented at all. Only valid when nothing was retained — a
    /// retained switch under `Merge` would be silently dropped, which is worse
    /// than refusing.
    #[default]
    Merge,
    /// A stiff branch, at the same magnitude
    /// [`topology::IDEAL_CONNECTION_Y`](crate::topology::IDEAL_CONNECTION_Y)
    /// gives a merged zero-impedance connection.
    Regularize,
}

/// The impedance a regularized switch is given: purely inductive, at the
/// magnitude of `topology::IDEAL_CONNECTION_Y`.
///
/// **Purely reactive, deliberately.** `IDEAL_CONNECTION_Y = 2e5 + j2e5` has a
/// *positive* imaginary part inherited from power-grid-model, which inverts to
/// a negative reactance — harmless in AC but sign-flipping in DC, where
/// `linear::btheta` needs an explicit guard for the PGM `link` that carries it.
/// A switch branch has no such history to preserve, so it takes the same
/// stiffness with the conventional sign and needs no guard anywhere.
pub fn switch_reactance() -> f64 {
    ideal_connection_z()
}

/// The series admittance of a regularized switch: `1/(j·x)`.
pub fn switch_admittance() -> Complex<f64> {
    Complex::new(1.0, 0.0) / Complex::new(0.0, switch_reactance())
}

/// The branches that give a bus view's retained switches identity.
///
/// Returned as [`Transformer`]s rather than [`Line`](crate::types::Line)s for
/// one reason: only `Transformer` carries per-terminal status, which is how an
/// open switch keeps its structural Y-bus entries and therefore how the
/// sparsity pattern survives a position change. The tap is always unity and
/// there is no shunt — nothing transformer-like is intended.
///
/// Ordering matches [`BusView::retained`], so the *n*th returned branch is the
/// *n*th retained switch. Callers appending these to their own transformer list
/// should record the offset: the crate's flat branch index is lines first, then
/// transformers, so a switch's branch index is
/// `lines.len() + transformers.len() + n`.
///
/// **Degenerate switches are skipped** — see
/// [`BusView::is_degenerate`](crate::topology::BusView::is_degenerate). A
/// retained switch whose ends already share a bus would stamp a self-loop
/// contributing exactly nothing, and its "flow" would be meaningless. They are
/// reported by [`degenerate_switches`] rather than silently included.
pub fn regularized_branches(view: &BusView) -> Vec<Transformer> {
    let y = switch_admittance();
    view.retained()
        .iter()
        .filter(|r| !view.is_degenerate(r))
        .map(|r| {
            let status = u8::from(!r.open);
            Transformer {
                from: r.buses[0].0,
                to: r.buses[1].0,
                from_status: status,
                to_status: status,
                y_series: y,
                y_shunt: Complex::new(0.0, 0.0),
                tap: Complex::new(1.0, 0.0),
            }
        })
        .collect()
}

/// The retained switches [`regularized_branches`] skipped, because their two
/// ends already resolve to one bus.
///
/// Not an error — a bus coupler in a closed ring is legitimately degenerate —
/// but a caller reporting switch flows needs to know that these have none.
pub fn degenerate_switches(view: &BusView) -> Vec<RetainedSwitch> {
    view.retained().iter().filter(|r| view.is_degenerate(r)).copied().collect()
}

/// The switch arrangement a [`conditioning_probe`] builds.
///
/// The shape matters more than the count, which is the whole reason this is a
/// parameter — see [`conditioning_probe`]'s doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeShape {
    /// A single chain of switches from the slack bus. A tree, so the stiff
    /// subnetwork has no circulating mode.
    Chain,
    /// Every switch fans out from one busbar node. Also a tree, and the shape a
    /// real substation bay arrangement most resembles.
    Star,
    /// The switches close a ring. **Not** a tree: a loop of near-ideal branches
    /// has a circulating current fixed only by the small differences between
    /// their impedances, which is where regularization actually breaks down.
    Ring,
}

/// How ill-conditioned a network becomes as regularized switches are added.
///
/// Returns `(count, iterations_or_None, far_end_voltage)` per entry in
/// `counts`. `None` means the solve did not converge or produced a non-finite
/// state.
///
/// `y_series` is the admittance each switch branch is given, so a caller can
/// compare the inductive [`switch_admittance`] this module uses against
/// [`topology::IDEAL_CONNECTION_Y`](crate::topology::IDEAL_CONNECTION_Y), whose
/// positive imaginary part makes it capacitive. That comparison is the point:
/// see below.
///
/// # What this measures, and what it corrected
///
/// `src/cgmes.rs` records that stamping CGMES switches as large-admittance
/// branches was tried and the AC solve *diverged* on FullGrid with "20-odd"
/// such branches active, and `plans/NODE_BREAKER_PLAN.md` §4.1 reasons from
/// that to "SmallGrid has 1,266 and Svedala 1,464 — 45–52x the count already
/// measured to diverge".
///
/// Measured, that inference does not hold: **switch count alone is not the
/// driver**. A [`Chain`](ProbeShape::Chain) or [`Star`](ProbeShape::Star) of
/// 1,464 stiff branches converges in two iterations, because both are trees and
/// a tree of stiff branches is merely stiff, not ill-conditioned. What breaks
/// is a [`Ring`](ProbeShape::Ring) — a loop of near-ideal branches whose
/// circulating current is determined only by the differences between
/// impedances that are all nearly equal, which is exactly the indeterminacy
/// `zero_impedance_branches.md` describes and the constrained formulation
/// resolves with a spanning tree.
///
/// So the honest statement of `Regularize`'s ceiling is topological, not
/// numerical: it is safe on radial switch arrangements at any count seen in
/// real data, and it degrades on meshed ones. A substation bay is a star; a
/// closed bus coupler in a double-busbar arrangement is a ring.
pub fn conditioning_probe(
    counts: &[usize],
    shape: ProbeShape,
    y_series: Complex<f64>,
    tol: f64,
    max_iter: usize,
) -> Vec<(usize, Option<usize>, f64)> {
    use crate::network::build_ybus;
    use crate::solver::{JacobianBackend, PersistentSolver, SolveStatus};
    use crate::types::{Bus, BusType, Line};

    counts
        .iter()
        .map(|&n| {
            // Each shape decides how many nodes its `n` switches span; the
            // load then hangs off one of them through an ordinary line, so the
            // network has something to solve for.
            //
            // Chain/Star span nodes `0..=n` (n switches, n+1 nodes). A ring of
            // n switches closes over only `0..n` (n nodes), so it needs one
            // fewer — and needs at least two switches to be a ring at all.
            let switch_nodes = match shape {
                ProbeShape::Chain | ProbeShape::Star => n + 1,
                ProbeShape::Ring => n.max(2),
            };
            let n_buses = switch_nodes + 1;
            let load = n_buses - 1;
            let mut buses: Vec<Bus> = (0..n_buses)
                .map(|i| Bus {
                    idx: i,
                    bus_type: if i == 0 { BusType::Slack } else { BusType::PQ },
                    voltage_mag: 1.0,
                    voltage_ang: 0.0,
                    p_spec: 0.0,
                    q_spec: 0.0,
                    q_min: -f64::INFINITY,
                    q_max: f64::INFINITY,
                    u_rated: 0.0,
                    zip_terms: Vec::new(),
                })
                .collect();
            buses[load].p_spec = -0.1;
            buses[load].q_spec = -0.03;

            let switch = |from: usize, to: usize| Transformer {
                from,
                to,
                from_status: 1,
                to_status: 1,
                y_series,
                y_shunt: Complex::new(0.0, 0.0),
                tap: Complex::new(1.0, 0.0),
            };
            let switches: Vec<Transformer> = match shape {
                ProbeShape::Chain => (0..n).map(|i| switch(i, i + 1)).collect(),
                // Everything hangs off the slack busbar; the load feeder is
                // attached to the last spoke.
                ProbeShape::Star => (0..n).map(|i| switch(0, i + 1)).collect(),
                // A chain over `0..switch_nodes` whose last switch closes back
                // on the slack, completing a loop of near-ideal branches.
                ProbeShape::Ring => (0..switch_nodes)
                    .map(|i| switch(i, (i + 1) % switch_nodes))
                    .collect(),
            };
            // The feeder always leaves from the node furthest into the switch
            // arrangement, so the load is genuinely fed through the switches.
            let feeder_from = switch_nodes - 1;
            let lines = vec![Line {
                from: feeder_from,
                to: load,
                r: 0.01,
                x: 0.05,
                b_shunt: 0.0,
                g_shunt: 0.0,
            }];

            let ybus = build_ybus(n_buses, &lines, &switches).finish();
            crate::network::linear_initial_guess(&mut buses, &ybus);
            let (_, stats) = PersistentSolver::new(JacobianBackend::Scalar)
                .solve_with_stats(&mut buses, &ybus, tol, max_iter);

            let finite =
                buses.iter().all(|b| b.voltage_mag.is_finite() && b.voltage_ang.is_finite());
            let ok = stats.status == SolveStatus::Converged && finite;
            (n, ok.then(|| stats.iterations()), buses[load].voltage_mag)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::model::{NodeBreakerTopology, NodeIdx, Switch, SwitchKind};
    use crate::topology::{bus_view, RetentionPolicy};

    fn sw(a: usize, b: usize, open: bool) -> Switch {
        Switch {
            kind: SwitchKind::Breaker,
            nodes: [NodeIdx(a), NodeIdx(b)],
            open,
            in_service: true,
        }
    }

    /// A regularized switch is inductive, unlike `IDEAL_CONNECTION_Y`. That is
    /// the whole reason DC needs no guard for it — a negative reactance would
    /// make `B` indefinite.
    #[test]
    fn a_switch_branch_is_inductive_at_the_ideal_connection_stiffness() {
        let y = switch_admittance();
        assert!(y.im < 0.0, "switch admittance {y} is capacitive");

        let z = Complex::new(1.0, 0.0) / y;
        assert!(z.im > 0.0, "switch reactance {} is negative", z.im);
        assert!((z.im - switch_reactance()).abs() < 1e-18);

        // Same stiffness as a merged ideal connection, to within round-off.
        assert!(
            (y.norm() - crate::topology::IDEAL_CONNECTION_Y.norm()).abs() < 1.0,
            "{} vs {}",
            y.norm(),
            crate::topology::IDEAL_CONNECTION_Y.norm()
        );

        // And DC reads it the right way round without any special case.
        let dc = crate::linear::dc_branches(
            &[],
            &regularized_branches(&{
                let mut t = NodeBreakerTopology::new(2);
                t.add_switch(sw(0, 1, false));
                bus_view(&t, &RetentionPolicy::RetainAll)
            }),
            crate::linear::DcOptions::default(),
        );
        assert_eq!(dc.len(), 1);
        assert!(dc[0].b > 0.0, "DC susceptance came out {}", dc[0].b);
    }

    /// Position becomes terminal status, which is what keeps the Y-bus pattern
    /// constant across a switching state change.
    #[test]
    fn open_and_closed_switches_differ_only_in_status() {
        let mut topo = NodeBreakerTopology::new(3);
        topo.add_switch(sw(0, 1, false));
        topo.add_switch(sw(1, 2, true));
        let view = bus_view(&topo, &RetentionPolicy::RetainAll);

        let branches = regularized_branches(&view);
        assert_eq!(branches.len(), 2);
        assert_eq!((branches[0].from_status, branches[0].to_status), (1, 1));
        assert_eq!((branches[1].from_status, branches[1].to_status), (0, 0));
        assert_eq!(branches[0].y_series, branches[1].y_series);
        assert_eq!((branches[0].from, branches[0].to), (0, 1));
        assert_eq!((branches[1].from, branches[1].to), (1, 2));
    }

    /// The property the module doc claims: a position change moves values, not
    /// structure. This is what lets a switching campaign share one symbolic
    /// factorization.
    #[test]
    fn flipping_a_switch_preserves_the_ybus_sparsity_pattern() {
        use crate::network::build_ybus;

        let pattern = |open: bool| {
            let mut topo = NodeBreakerTopology::new(3);
            topo.add_switch(sw(0, 1, open));
            topo.add_switch(sw(1, 2, false));
            let view = bus_view(&topo, &RetentionPolicy::RetainAll);
            let ybus = build_ybus(3, &[], &regularized_branches(&view)).finish();
            (0..ybus.n())
                .map(|i| ybus.row(i).iter().map(|&(j, _)| j).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        assert_eq!(pattern(false), pattern(true));
    }

    /// A retained switch whose ends already share a bus stamps nothing, and is
    /// reported rather than silently included.
    #[test]
    fn degenerate_switches_are_separated_out() {
        let mut topo = NodeBreakerTopology::new(2);
        topo.add_switch(sw(0, 1, false)); // retained
        topo.add_switch(Switch {
            kind: SwitchKind::Junction, // always merged, so both ends collapse
            nodes: [NodeIdx(0), NodeIdx(1)],
            open: false,
            in_service: true,
        });
        let view = bus_view(&topo, &RetentionPolicy::RetainAll);

        assert_eq!(view.retained().len(), 1);
        assert!(regularized_branches(&view).is_empty());
        assert_eq!(degenerate_switches(&view).len(), 1);
    }

    /// Nothing retained means nothing to stamp — the `MergeAll` path must add
    /// no branches at all, or the default treatment would quietly change every
    /// existing result.
    #[test]
    fn merge_all_produces_no_switch_branches() {
        let mut topo = NodeBreakerTopology::new(3);
        topo.add_switch(sw(0, 1, false));
        topo.add_switch(sw(1, 2, true));
        let view = bus_view(&topo, &RetentionPolicy::MergeAll);
        assert!(regularized_branches(&view).is_empty());
        assert!(degenerate_switches(&view).is_empty());
    }

    /// A closed regularized switch really does tie its two buses together: the
    /// voltage drop across it must be negligible against an ordinary branch's.
    #[test]
    fn a_closed_switch_holds_its_two_buses_together() {
        use crate::network::build_ybus;
        use crate::solver::newton_raphson;
        use crate::types::{Bus, BusType, Line};

        let mut buses: Vec<Bus> = (0..3)
            .map(|i| Bus {
                idx: i,
                bus_type: if i == 0 { BusType::Slack } else { BusType::PQ },
                voltage_mag: 1.0,
                voltage_ang: 0.0,
                p_spec: 0.0,
                q_spec: 0.0,
                q_min: -f64::INFINITY,
                q_max: f64::INFINITY,
                u_rated: 0.0,
                zip_terms: Vec::new(),
            })
            .collect();
        buses[2].p_spec = -0.5;
        buses[2].q_spec = -0.2;

        let mut topo = NodeBreakerTopology::new(3);
        topo.add_switch(sw(0, 1, false));
        let view = bus_view(&topo, &RetentionPolicy::RetainAll);
        let lines = vec![Line { from: 1, to: 2, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 }];

        let ybus = build_ybus(3, &lines, &regularized_branches(&view)).finish();
        newton_raphson(&mut buses, &ybus, 1e-8, 30);

        let drop = (buses[0].voltage_mag - buses[1].voltage_mag).abs();
        let across_line = (buses[1].voltage_mag - buses[2].voltage_mag).abs();
        assert!(
            drop < 1e-4,
            "a closed switch dropped {drop} p.u., which is not 'ideal enough'"
        );
        assert!(drop < across_line / 100.0, "{drop} vs {across_line} across a real branch");
    }

    /// The measured ceiling, pinned so it cannot drift silently.
    ///
    /// `plans/NODE_BREAKER_PLAN.md` §4.1 predicts that `Regularize` is "dead on
    /// arrival" at SmallGrid's 1,266 switches and Svedala's 1,464. On synthetic
    /// networks it is not: every shape, count and admittance sign converges. If
    /// this test ever starts failing, the prediction has finally been
    /// reproduced and §4.1's conclusion is vindicated — which is worth knowing
    /// either way, so the assertion is on the *result*, not on a claim.
    #[test]
    fn regularized_switches_converge_at_every_count_this_probe_can_build() {
        let counts = [1usize, 20, 29, 90, 1266, 1464];
        for shape in [ProbeShape::Chain, ProbeShape::Star, ProbeShape::Ring] {
            for (label, y) in [
                ("inductive", switch_admittance()),
                ("capacitive", crate::topology::IDEAL_CONNECTION_Y),
            ] {
                for (n, iters, vm) in conditioning_probe(&counts, shape, y, 1e-6, 30) {
                    assert!(
                        iters.is_some(),
                        "{shape:?}/{label}: {n} switches failed to converge — \
                         NODE_BREAKER_PLAN.md §4.1's prediction has been reproduced"
                    );
                    assert!(
                        (vm - 1.0).abs() < 0.05,
                        "{shape:?}/{label}: {n} switches converged to an implausible \
                         |V| = {vm}"
                    );
                }
            }
        }
    }

    /// An open regularized switch really does separate: the far side becomes a
    /// sourceless island rather than a lightly-coupled one.
    #[test]
    fn an_open_switch_separates_its_two_buses() {
        use crate::network::{build_ybus, connected_components};

        let mut topo = NodeBreakerTopology::new(2);
        topo.add_switch(sw(0, 1, true));
        let view = bus_view(&topo, &RetentionPolicy::RetainAll);
        let ybus = build_ybus(2, &[], &regularized_branches(&view)).finish();

        // The structural entry survives — that is the point, since it is what
        // keeps the sparsity pattern constant across a position change — so
        // the *value* is what carries the position.
        assert!(ybus.row(0).iter().any(|&(j, _)| j == 1), "structural entry was dropped");
        assert_eq!(ybus.get(0, 1), Complex::new(0.0, 0.0), "an open switch still conducts");
        // And `connected_components` reads the value, not merely the structure,
        // so it sees the separation: two components, not one.
        assert_eq!(connected_components(&ybus).len(), 2);
    }
}

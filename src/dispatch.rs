//! Which machine produced what reactive power.
//!
//! A `PV` bus has one reactive injection and the solver decides it. When one
//! machine holds that bus, that injection *is* the machine's output and there
//! is nothing to attribute. When six do — 62 of RealGrid's 417 regulated buses,
//! 2 of FullGrid's 3 — the power flow is still right, but nothing says which
//! machine is carrying the load.
//!
//! That matters most exactly where it is least visible.
//! [`ReactiveLimits`](crate::outerloop::ReactiveLimits) clamps a bus at the
//! *summed* capability of everything holding it, which is correct: the bus can
//! produce what its machines jointly can. But a bus comfortably inside its
//! summed limit can contain a machine that is not, with the others carrying it.
//! Only a split shows that.
//!
//! # What this is not
//!
//! **This changes no answer.** It is attribution after the fact, not a
//! constraint inside the solve. Every machine in the vendored corpus sits at
//! the bus it regulates, so the bus is an ordinary `PV` bus, its injection is
//! determined by the network, and the split is arithmetic on the result.
//!
//! The genuinely different problem — several machines at *different* buses
//! holding one remote bus — needs `n−1` extra equations inside the Newton
//! system (powsybl's `DISTR_Q`) because each controller bus then has its own
//! reactive unknown. No vendored fixture has that configuration, so it is not
//! built here. See `plans/REACTIVE_DISPATCH_PLAN.md` §2 for the measurement and
//! §6 for the formulation when a fixture appears.

use crate::network::{power_injections, YBusSparse};
use crate::types::{Bus, RegulatingMachine};

/// One bus's machines, and what they could account for.
#[derive(Clone, Debug, PartialEq)]
pub struct BusDispatch {
    /// The bus whose voltage these machines hold.
    pub bus: usize,
    /// What they jointly had to produce: the solved reactive injection at
    /// `bus`, less the part that is not theirs.
    pub required: f64,
    /// What they could actually be given, once each one's own limits were
    /// respected. Equal to `required` unless every machine is pinned.
    pub attributed: f64,
    /// `required − attributed`, and worth reading rather than ignoring.
    ///
    /// A non-zero value means the machines holding this bus cannot account for
    /// the reactive power the solve put there — they are all at their limits
    /// and it is still not enough. That is not an arithmetic failure here; it
    /// means the **solve** placed the bus outside its machines' joint
    /// capability, which happens because gridoxide bounds the wrong quantity.
    ///
    /// [`ReactiveLimits`](crate::outerloop::ReactiveLimits) compares
    /// `Bus::q_min`/`q_max` against the bus's *net* injection, while the CGMES
    /// importer fills those fields from the machines' *own* capability. Where a
    /// reactive load shares the bus, the machine has to cover it too, so the
    /// machine saturates before the net injection reaches the bound and the
    /// clamp fires late. `pgm::PgmVoltageRegulator`'s doc comment already names
    /// this as a deliberate simplification of the bus model; this field is the
    /// first thing that measures it.
    ///
    /// Measured on RealGrid: of 416 regulated buses, 50 share their bus with
    /// another reactive injection and 6 come out unattributable, 0.044 p.u. in
    /// total. Small, real, and previously invisible.
    pub unattributed: f64,
    /// What the shares were derived from.
    pub basis: KeyBasis,
    pub machines: Vec<MachineDispatch>,
}

/// One machine's share of the reactive power its bus produced.
#[derive(Clone, Debug, PartialEq)]
pub struct MachineDispatch {
    /// The source document's own identifier.
    pub id: String,
    pub at_bus: usize,
    pub controls_bus: usize,
    /// The allocated reactive output, per-unit.
    pub q: f64,
    pub q_min: f64,
    pub q_max: f64,
    /// This machine is at one of its own limits — which the bus-level clamp
    /// cannot tell you, because the bus may be well inside its summed range
    /// while this machine sits on the edge of its own.
    pub at_limit: bool,
    /// Its normalized share of the split, before limits were applied.
    pub share: f64,
}

/// How the shares were arrived at, so a reader can tell a considered split from
/// an even one.
///
/// The fallback is **wholesale, not per machine**: one implausible capability
/// discards the whole basis rather than leaving a mixture of two rules, which
/// is what `references/powsybl-open-loadflow`'s `Control.createReactiveKeys`
/// does and for the same reason — a split half-derived from capability and half
/// from nothing is harder to defend than either.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum KeyBasis {
    /// The document stated a share for every machine.
    Explicit,
    /// Proportional to each machine's own reactive range.
    Capability,
    /// Equal shares — no usable capability, so nothing distinguishes them.
    Uniform,
}

/// Bounds outside which a stated reactive range is treated as no information.
///
/// Transcribed from powsybl's `PlausibleValues`, in its per-unit form. A
/// machine declaring a 0 MVAr range says nothing about how to split; one
/// declaring 10 GVAr is a data error rather than a very large machine, and
/// letting either drive the split would hand it the whole bus.
const MIN_PLAUSIBLE_RANGE_PU: f64 = 0.01;
const MAX_PLAUSIBLE_RANGE_PU: f64 = 100.0;

/// The normalized split for one bus's machines, and what it was derived from.
///
/// Explicit keys where every machine has one; else capability, where every
/// machine's range is finite and plausible; else uniform.
pub fn reactive_keys(machines: &[&RegulatingMachine]) -> (Vec<f64>, KeyBasis) {
    let uniform = |n: usize| (vec![1.0 / n as f64; n], KeyBasis::Uniform);
    if machines.is_empty() {
        return (Vec::new(), KeyBasis::Uniform);
    }

    let normalize = |raw: Vec<f64>, basis| {
        let total: f64 = raw.iter().sum();
        if total > 0.0 && total.is_finite() {
            Some((raw.into_iter().map(|k| k / total).collect::<Vec<_>>(), basis))
        } else {
            None
        }
    };

    if machines.iter().all(|m| m.key.is_some_and(|k| k.is_finite() && k >= 0.0))
        && let Some(out) =
            normalize(machines.iter().map(|m| m.key.unwrap()).collect(), KeyBasis::Explicit)
    {
        return out;
    }

    let ranges: Vec<f64> = machines.iter().map(|m| m.q_max - m.q_min).collect();
    if ranges.iter().all(|r| (MIN_PLAUSIBLE_RANGE_PU..=MAX_PLAUSIBLE_RANGE_PU).contains(r))
        && let Some(out) = normalize(ranges, KeyBasis::Capability)
    {
        return out;
    }

    uniform(machines.len())
}

/// Splits `total` across `machines` by `keys`, respecting each machine's own
/// limits.
///
/// A machine whose share falls outside its own range is pinned there and its
/// excess re-split across the rest, repeating until nothing moves. Without that
/// the split can hand a machine more than it can produce while another sits
/// idle — arithmetic that sums correctly and describes nothing physical.
///
/// It takes **asymmetric** capability to reach that. With `q_max = −q_min` a
/// capability-proportional share is `total · range_i / Σrange`, which exceeds
/// `q_max_i = range_i / 2` only once `total > Σq_max` — when the bus is already
/// past its joint capability and every machine saturates together. Every
/// machine at RealGrid's shared buses is symmetric, so this path is exercised
/// by the unit tests rather than by that fixture.
///
/// If every machine ends up pinned the shares no longer sum to `total`; that is
/// reported by the caller rather than papered over, because it means the bus
/// itself is outside its summed capability and the *power flow* is the thing to
/// look at, not the split.
fn split(total: f64, machines: &[&RegulatingMachine], keys: &[f64]) -> (Vec<f64>, Vec<bool>) {
    let n = machines.len();
    let mut q = vec![0.0; n];
    let mut pinned = vec![false; n];

    for _ in 0..=n {
        let free_key: f64 =
            (0..n).filter(|&i| !pinned[i]).map(|i| keys[i]).sum();
        let pinned_q: f64 = (0..n).filter(|&i| pinned[i]).map(|i| q[i]).sum();
        let remaining = total - pinned_q;

        for i in 0..n {
            if !pinned[i] {
                // All keys pinned out is handled by the loop guard below; a zero
                // free key with machines left means every remaining share is
                // zero, which is a legitimate answer.
                q[i] = if free_key > 0.0 { remaining * keys[i] / free_key } else { 0.0 };
            }
        }

        let mut moved = false;
        for i in 0..n {
            if pinned[i] {
                continue;
            }
            if q[i] > machines[i].q_max {
                q[i] = machines[i].q_max;
                pinned[i] = true;
                moved = true;
            } else if q[i] < machines[i].q_min {
                q[i] = machines[i].q_min;
                pinned[i] = true;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }

    (q, pinned)
}

/// Attributes each bus's solved reactive injection to the machines holding it.
///
/// `buses` must be a **converged** solution and `ybus` the network it came
/// from; this reads whatever state it is handed and cannot tell a solved
/// network from an unsolved one, so the split of a non-converged state is
/// meaningless rather than wrong-looking. Same contract as
/// [`AcSensitivity::new`](crate::ac_sensitivity::AcSensitivity::new).
///
/// `nonregulating_q` is the per-bus reactive injection that is not a regulating
/// machine's — `cgmes::VoltageControlReport::nonregulating_q`, named rather
/// than linked because that module is behind the `cgmes` feature and this one
/// is not. The machines are responsible for the solved injection *minus* that.
pub fn allocate(
    machines: &[RegulatingMachine],
    nonregulating_q: &[f64],
    buses: &[Bus],
    ybus: &YBusSparse,
) -> Vec<BusDispatch> {
    let (_, q_calc) = power_injections(buses, ybus);

    // Grouped by the bus they *hold*, which is what shares a single reactive
    // injection — not by where they sit. The two coincide throughout the
    // vendored corpus and are different questions.
    let mut order: Vec<usize> = (0..machines.len()).collect();
    order.sort_by_key(|&i| machines[i].controls_bus);

    let mut out: Vec<BusDispatch> = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let bus = machines[order[start]].controls_bus;
        let mut end = start;
        while end < order.len() && machines[order[end]].controls_bus == bus {
            end += 1;
        }
        let group: Vec<&RegulatingMachine> =
            order[start..end].iter().map(|&i| &machines[i]).collect();

        // What the machines jointly produced, at the bus they hold.
        let required = q_calc[bus] - nonregulating_q.get(bus).copied().unwrap_or(0.0);
        let (keys, basis) = reactive_keys(&group);
        let (shares, pinned) = split(required, &group, &keys);
        let attributed: f64 = shares.iter().sum();

        out.push(BusDispatch {
            bus,
            required,
            attributed,
            unattributed: required - attributed,
            basis,
            machines: group
                .iter()
                .enumerate()
                .map(|(k, m)| MachineDispatch {
                    id: m.id.clone(),
                    at_bus: m.at_bus,
                    controls_bus: m.controls_bus,
                    q: shares[k],
                    q_min: m.q_min,
                    q_max: m.q_max,
                    at_limit: pinned[k],
                    share: keys[k],
                })
                .collect(),
        });
        start = end;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(id: &str, q_min: f64, q_max: f64, key: Option<f64>) -> RegulatingMachine {
        RegulatingMachine {
            id: id.to_string(),
            at_bus: 0,
            controls_bus: 0,
            q_min,
            q_max,
            q_scheduled: 0.0,
            key,
        }
    }

    #[test]
    fn explicit_keys_are_used_and_normalized() {
        let ms = [machine("a", -1.0, 1.0, Some(3.0)), machine("b", -1.0, 1.0, Some(1.0))];
        let (keys, basis) = reactive_keys(&ms.iter().collect::<Vec<_>>());
        assert_eq!(basis, KeyBasis::Explicit);
        assert!((keys[0] - 0.75).abs() < 1e-12 && (keys[1] - 0.25).abs() < 1e-12, "{keys:?}");
    }

    /// One missing key discards the whole basis rather than mixing two rules —
    /// powsybl's behaviour, and the defensible one: a split half-derived from
    /// stated shares and half from capability is harder to justify than either
    /// alone.
    #[test]
    fn one_missing_key_falls_back_wholesale() {
        let ms = [machine("a", -1.0, 1.0, Some(3.0)), machine("b", -2.0, 2.0, None)];
        let (keys, basis) = reactive_keys(&ms.iter().collect::<Vec<_>>());
        assert_eq!(basis, KeyBasis::Capability, "the stated key must not survive alone");
        // 2.0 range against 4.0 range.
        assert!((keys[0] - 1.0 / 3.0).abs() < 1e-12, "{keys:?}");
    }

    #[test]
    fn capability_keys_are_proportional_to_range() {
        let ms = [machine("a", -0.3, 0.3, None), machine("b", -0.1, 0.1, None)];
        let (keys, basis) = reactive_keys(&ms.iter().collect::<Vec<_>>());
        assert_eq!(basis, KeyBasis::Capability);
        assert!((keys[0] - 0.75).abs() < 1e-12, "{keys:?}");
    }

    /// An unbounded or absurd range is no information, and letting it drive the
    /// split would hand that machine the entire bus.
    #[test]
    fn implausible_ranges_fall_back_to_uniform() {
        for (lo, hi) in [
            (f64::NEG_INFINITY, f64::INFINITY),
            (0.0, 0.0),               // no stated capability
            (-1e6, 1e6),              // beyond any real machine
            (-0.001, 0.001),          // below the plausible floor
        ] {
            let ms = [machine("a", lo, hi, None), machine("b", -0.2, 0.2, None)];
            let (keys, basis) = reactive_keys(&ms.iter().collect::<Vec<_>>());
            assert_eq!(basis, KeyBasis::Uniform, "range ({lo}, {hi}) should be rejected");
            assert!((keys[0] - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn a_split_sums_to_the_total_and_respects_every_limit() {
        let ms = [machine("a", -0.3, 0.3, None), machine("b", -0.1, 0.1, None)];
        let refs: Vec<&RegulatingMachine> = ms.iter().collect();
        let (keys, _) = reactive_keys(&refs);
        let (q, pinned) = split(0.2, &refs, &keys);
        assert!((q.iter().sum::<f64>() - 0.2).abs() < 1e-12);
        assert!(pinned.iter().all(|p| !p));
        assert!((q[0] - 0.15).abs() < 1e-12 && (q[1] - 0.05).abs() < 1e-12, "{q:?}");
    }

    /// The case the redistribution exists for, and note what it takes to reach
    /// it: **asymmetric** capability.
    ///
    /// With symmetric limits (`q_max = −q_min = range/2`), a capability-
    /// proportional share is `total · range_i / Σrange`, which exceeds
    /// `q_max_i = range_i / 2` only when `total > Σrange / 2 = Σq_max` — that
    /// is, only when the *bus* is already past its joint capability, at which
    /// point every machine saturates together and there is nothing to
    /// redistribute. So a fixture of symmetric machines can never exercise this
    /// path, and one built that way would suggest the loop is dead code.
    #[test]
    fn a_saturated_machine_pins_and_the_rest_absorbs_it() {
        let ms = [machine("big", -1.0, 1.0, None), machine("small", -0.3, 0.1, None)];
        let refs: Vec<&RegulatingMachine> = ms.iter().collect();
        let (keys, _) = reactive_keys(&refs);
        let (q, pinned) = split(0.9, &refs, &keys);

        assert!((q.iter().sum::<f64>() - 0.9).abs() < 1e-12, "{q:?}");
        assert!(pinned[1] && !pinned[0], "the small machine should be the one to saturate");
        assert!((q[1] - 0.1).abs() < 1e-12, "pinned at its own max, not its share");
        assert!((q[0] - 0.8).abs() < 1e-12, "the big machine takes what is left");
    }

    /// When even every machine at its limit is not enough, the shortfall is
    /// reported rather than absorbed — it means the *solve* placed the bus
    /// outside its machines' joint capability.
    #[test]
    fn an_impossible_total_leaves_a_shortfall() {
        let ms = [machine("a", -0.1, 0.1, None), machine("b", -0.1, 0.1, None)];
        let refs: Vec<&RegulatingMachine> = ms.iter().collect();
        let (keys, _) = reactive_keys(&refs);
        let (q, pinned) = split(0.5, &refs, &keys);
        assert!(pinned.iter().all(|p| *p));
        assert!((q.iter().sum::<f64>() - 0.2).abs() < 1e-12, "capped at joint capability");
    }

    #[test]
    fn a_lone_machine_takes_everything() {
        let ms = [machine("only", -1.0, 1.0, None)];
        let refs: Vec<&RegulatingMachine> = ms.iter().collect();
        let (keys, _) = reactive_keys(&refs);
        assert_eq!(keys, vec![1.0]);
        let (q, _) = split(-0.42, &refs, &keys);
        assert!((q[0] + 0.42).abs() < 1e-12);
    }
}

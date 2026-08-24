//! Per-machine reactive dispatch, against RealGrid's own published solution.
//!
//! The invariants are exact and asserted as such: shares sum to what was
//! attributed, no machine is handed more than it can produce, and nothing is
//! silently dropped. The *split rule* is a modelling choice and is measured
//! rather than pinned tightly — the published dispatch came from another tool,
//! whose apportionment need not be capability-proportional, so an exact match
//! is a bonus and a tuned tolerance would be self-deception.
//!
//! What makes the comparison meaningful is separating two errors that are easy
//! to conflate. gridoxide's solve differs from the published one by ~7.5% on
//! these buses, which has nothing to do with attribution; measured against the
//! *published* bus total, the split itself lands within ~7.4% on the shared
//! buses — the only buses where a split is a choice at all.

mod cgmes_common;

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_to_network, load_profiles};
use gridoxide::dispatch::{allocate, reactive_keys, KeyBasis};
use gridoxide::network::{build_ybus, power_injections, stamp_shunts};
use gridoxide::solver::PowerFlowOptions;

const S_BASE_MVA: f64 = 100.0;

fn realgrid() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/RealGrid/RealGrid-Merged");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init \
             tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

/// Import, solve with reactive limits on, and attribute.
struct Solved {
    dispatch: Vec<gridoxide::dispatch::BusDispatch>,
    machines: Vec<gridoxide::types::RegulatingMachine>,
    q_calc: Vec<f64>,
    nonregulating_q: Vec<f64>,
    shared_buses: usize,
    published: std::collections::HashMap<String, f64>,
}

fn solve() -> Option<Solved> {
    let dir = realgrid()?;
    let ds = load_profiles(&[
        &dir.join("RealGrid_EQ.xml"),
        &dir.join("RealGrid_SSH.xml"),
        &dir.join("RealGrid_TP.xml"),
        &dir.join("RealGrid_SV.xml"),
    ])
    .expect("decode");
    let net = cgmes_to_network(&ds, S_BASE_MVA * 1e6).expect("convert");
    let vc = net.voltage_control;

    let report = gridoxide::run_power_flow(
        net.buses,
        &net.lines,
        &net.transformers,
        &net.shunts,
        gridoxide::TapData::none(),
        PowerFlowOptions {
            enforce_q_limits: true,
            max_iter: 60,
            max_outer_iter: 60,
            ..Default::default()
        },
    );
    assert_eq!(report.stats.status, gridoxide::solver::SolveStatus::Converged);

    let mut y = build_ybus(report.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut y, &net.shunts);
    let ybus = y.finish();
    let (_, q_calc) = power_injections(&report.buses, &ybus);

    let published = cgmes_common::expected_injections(&ds, S_BASE_MVA)
        .into_iter()
        .map(|e| (e.equipment_mrid, e.q))
        .collect();

    Some(Solved {
        dispatch: allocate(&vc.machines, &vc.nonregulating_q, &report.buses, &ybus),
        machines: vc.machines,
        q_calc,
        nonregulating_q: vc.nonregulating_q,
        shared_buses: vc.shared_buses,
        published,
    })
}

/// The arithmetic has to hold exactly, whatever the split rule is.
#[test]
fn the_split_is_exact_and_accounts_for_everything() {
    let Some(s) = solve() else { return };
    assert_eq!(s.machines.len(), 496, "every regulating machine is retained");
    assert!(s.shared_buses >= 62, "RealGrid's shared-control buses: {}", s.shared_buses);

    for bd in &s.dispatch {
        let summed: f64 = bd.machines.iter().map(|m| m.q).sum();
        assert!(
            (summed - bd.attributed).abs() < 1e-12,
            "bus {}: shares sum to {summed}, attributed says {}",
            bd.bus,
            bd.attributed
        );
        assert!(
            (bd.attributed + bd.unattributed - bd.required).abs() < 1e-12,
            "bus {}: attributed + unattributed must be what was required",
            bd.bus
        );
        // What was required is the bus's own solved injection less the part
        // that is not the machines' — recomputed here rather than trusted.
        let required = s.q_calc[bd.bus] - s.nonregulating_q[bd.bus];
        assert!(
            (bd.required - required).abs() < 1e-9,
            "bus {}: required {} vs independently computed {required}",
            bd.bus,
            bd.required
        );
        for m in &bd.machines {
            assert!(
                m.q <= m.q_max + 1e-9 && m.q >= m.q_min - 1e-9,
                "bus {}: machine {} given {} outside its own [{}, {}]",
                bd.bus,
                m.id,
                m.q,
                m.q_min,
                m.q_max
            );
            // One direction only. A share can land exactly on a limit without
            // having been clamped there, and calling that "at limit" would be
            // reading a coincidence as a constraint.
            if m.at_limit {
                assert!(
                    (m.q - m.q_max).abs() < 1e-9 || (m.q - m.q_min).abs() < 1e-9,
                    "bus {}: machine {} is flagged at its limit but sits at {}, \
                     inside [{}, {}]",
                    bd.bus,
                    m.id,
                    m.q,
                    m.q_min,
                    m.q_max
                );
            }
        }
    }
}

/// Unattributable reactive power is not noise — it is a measurement of a
/// documented simplification, and it should stay confined to the buses that
/// simplification affects.
///
/// `ReactiveLimits` bounds the bus's *net* injection with limits the importer
/// filled from the machines' *own* capability. Where nothing else sits at the
/// bus the two coincide and every bus is fully attributable. Where a reactive
/// load shares the bus, the machine must cover it too, so it saturates before
/// the net injection reaches the bound and the clamp fires late.
#[test]
fn unattributed_power_appears_only_where_a_load_shares_the_bus() {
    let Some(s) = solve() else { return };

    // A physical threshold, not a floating-point one: 1e-6 p.u. is 0.1 kVAr on
    // a 100 MVA base. Below it the residue is accumulation noise across a
    // 6252-bus network — one bus sits at 2e-8 — and calling that a shortfall
    // would be reading arithmetic as physics.
    const NEGLIGIBLE: f64 = 1e-6;
    let mut unattributable = 0;
    for bd in &s.dispatch {
        if bd.unattributed.abs() <= NEGLIGIBLE {
            continue;
        }
        unattributable += 1;
        assert!(
            s.nonregulating_q[bd.bus].abs() > 1e-9,
            "bus {} cannot account for {} but has nothing else injecting there — \
             that would be an allocation bug, not the bus-model simplification",
            bd.bus,
            bd.unattributed
        );
        assert!(
            bd.machines.iter().all(|m| m.at_limit),
            "bus {} left {} unattributed while a machine still had headroom",
            bd.bus,
            bd.unattributed
        );
    }

    let total: f64 =
        s.dispatch.iter().map(|b| b.unattributed.abs()).filter(|u| *u > NEGLIGIBLE).sum();
    // Recorded, not targeted. If either moves, the bus model changed and the
    // reason is worth knowing before this number is updated.
    assert_eq!(unattributable, 6, "buses whose machines cannot account for their own injection");
    assert!(
        (total - 0.0436).abs() < 5e-4,
        "total unattributed {total:.4} pu against the recorded 0.0436"
    );
}

/// Against the fixture's own published per-machine solution.
///
/// Two errors, deliberately separated. The bus totals differ because
/// gridoxide's solve differs from the published one — pre-existing, and nothing
/// to do with attribution. Measured against the *published* total instead, what
/// is left is the split rule alone.
#[test]
fn the_split_reproduces_the_published_attribution() {
    let Some(s) = solve() else { return };

    let (mut solve_err, mut solve_abs) = (0.0f64, 0.0f64);
    let (mut split_err, mut split_abs) = (0.0f64, 0.0f64);
    let mut compared = 0usize;

    for bd in &s.dispatch {
        let group: Vec<&gridoxide::types::RegulatingMachine> =
            s.machines.iter().filter(|m| m.controls_bus == bd.bus).collect();
        if !bd.machines.iter().all(|m| s.published.contains_key(&m.id)) {
            continue;
        }
        let published_total: f64 = bd.machines.iter().map(|m| s.published[&m.id]).sum();
        solve_err += (bd.attributed - published_total).abs();
        solve_abs += published_total.abs();

        if bd.machines.len() < 2 {
            continue; // a lone machine's "split" is not a choice
        }
        compared += bd.machines.len();
        let (keys, _) = reactive_keys(&group);
        for (k, m) in bd.machines.iter().enumerate() {
            split_err += (published_total * keys[k] - s.published[&m.id]).abs();
            split_abs += s.published[&m.id].abs();
        }
    }

    assert!(compared > 100, "expected RealGrid's shared buses to be compared, got {compared}");
    let solve_rel = 100.0 * solve_err / solve_abs;
    let split_rel = 100.0 * split_err / split_abs;
    eprintln!("bus totals (the solve): {solve_rel:.1}%   split rule: {split_rel:.1}%");

    // Recorded. The split is the thing under test; the solve error is the
    // backdrop it has to be read against, and asserting it here keeps the two
    // from being confused if either moves.
    assert!(solve_rel < 12.0, "solve-vs-published on these buses drifted to {solve_rel:.1}%");
    assert!(
        split_rel < 15.0,
        "capability-proportional split drifted to {split_rel:.1}% against the published \
         attribution — worth understanding before widening this"
    );
}

/// Nearly all of RealGrid's shared buses state a usable reactive range, so the
/// split has a real basis rather than falling back to equal shares.
///
/// One does not, and it is worth naming rather than rounding away: bus 6232's
/// two machines each declare a ±0.001 p.u. range — 0.2 MVAr, an order of
/// magnitude below anything a transmission machine plausibly has — so the
/// capability basis is rejected and the split is uniform. Since both declare the
/// *same* implausible range, uniform and capability agree there anyway; the
/// guard costs nothing and stops one mis-stated nameplate from taking a bus.
#[test]
fn realgrid_splits_on_capability_apart_from_one_implausible_pair() {
    let Some(s) = solve() else { return };
    let mut by_basis = std::collections::BTreeMap::new();
    for bd in &s.dispatch {
        if bd.machines.len() < 2 {
            continue;
        }
        *by_basis.entry(bd.basis).or_insert(0usize) += 1;
    }

    let shared: usize = by_basis.values().sum();
    assert!(shared >= 62, "expected at least RealGrid's 62 shared buses, got {shared}");
    assert_eq!(by_basis.get(&KeyBasis::Capability), Some(&61));
    assert_eq!(by_basis.get(&KeyBasis::Uniform), Some(&1));
    assert_eq!(by_basis.get(&KeyBasis::Explicit), None, "CGMES states no per-machine keys");

    let fallback = s
        .dispatch
        .iter()
        .find(|b| b.machines.len() > 1 && b.basis == KeyBasis::Uniform)
        .expect("the one uniform bus");
    for m in &fallback.machines {
        assert!(
            m.q_max - m.q_min < 0.01,
            "bus {} was expected to fall back because its ranges are implausibly narrow, \
             but machine {} spans {}",
            fallback.bus,
            m.id,
            m.q_max - m.q_min
        );
    }
}

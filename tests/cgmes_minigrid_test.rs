mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's MiniGrid conformance test configuration — a small
/// boundary-truncated area model (13 physical `TopologicalNode` buses, plus
/// two synthesized 3-winding-transformer star buses), the first fixture
/// this converter was tried against with *more than one* 3-winding
/// `PowerTransformer` in the same model. That exposed a real bug: the star
/// bus index was computed as `buses.len() + star_bus_count`, double-
/// counting star buses already pushed by earlier loop iterations once more
/// than one existed, corrupting the Y-bus with an out-of-range reference
/// (see `src/cgmes.rs`'s Steps 5+6 comment). Also exercises
/// `ExternalNetworkInjection` (its P/Q negation cross-checked against
/// `references/powsybl-core`'s own `ExternalNetworkInjectionConversion`,
/// which is unfortunately numerically silent here — both of this fixture's
/// instances happen to have P=Q=0 in this snapshot) and `AsynchronousMachine`
/// (3 real induction motors, ~9 MW/~5 MVAr total — *not* numerically
/// silent: adding this dropped the worst per-bus voltage deviation from
/// 4.52% to 3.93%. Sign convention cross-checked against
/// `references/powsybl-core`'s own `AsynchronousMachineConversion`, which
/// converts it to a plain `Load` with no sign flip at all — i.e. gridoxide's
/// own load-style *both-negated* convention, not `SynchronousMachine`'s Q
/// exception; empirically confirmed too, since the Q-exception convention
/// makes this fixture's worst deviation *worse* than not modeling
/// `AsynchronousMachine` at all, 5.05% vs 4.52%).
///
/// Tolerance matches MicroGrid-BE-MAS's own (boundary-truncated, simple
/// fixed-injection equivalents standing in for the rest of the
/// interconnected system — same root cause, not a correctness bug, see that
/// test's own doc comment).
#[test]
fn test_cgmes_minigrid() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MiniGrid/MiniGrid-Merged");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("MiniGrid_EQBD.xml");
    let eq = dir.join("MiniGrid_EQ.xml");
    let ssh = dir.join("MiniGrid_SSH.xml");
    let tp = dir.join("MiniGrid_TP.xml");
    let sv = dir.join("MiniGrid_SV.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    let expected = cgmes_common::expected_voltages(&ds);

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    // 13 physical TopologicalNode buses, plus one synthesized star bus per
    // 3-winding transformer (2 of them in this fixture).
    assert!(buses.len() > tn_mrids.len(), "expected extra synthesized star buses beyond the {} physical ones, got {}", tn_mrids.len(), buses.len());

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    cgmes_common::assert_matches_sv(&result, &tn_mrids, &expected, 5e-2);
}

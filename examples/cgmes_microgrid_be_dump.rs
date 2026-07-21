//! Solves ENTSO-E's MicroGrid-BE-MAS CGMES conformance case (the same
//! fixture `tests/cgmes_microgrid_be_test.rs` checks against its own
//! published `SvVoltage` values) and prints each `TopologicalNode`'s solved
//! voltage (plus its `bus_type`) as JSON on stdout, keyed by mRID.
//!
//! Not part of the test suite itself — this is the machine-readable half of
//! `scripts/bench/cross_validate_cgmes_microgrid_be.py`'s cross-validation
//! against pypowsybl's own independent CGMES import + AC load flow on the
//! same files: that script runs this binary as a subprocess (there's no
//! Python binding for CGMES import, only for PGM-JSON — see
//! `src/python.rs`), parses its JSON, and compares bus-by-bus. `bus_type`
//! specifically lets that script find gridoxide's own slack bus (the CGMES
//! `TopologicalIsland.AngleRefTopologicalNode`) so it can pin pypowsybl to
//! the same reference bus and make the two tools' angles comparable.
//!
//! Usage: cargo run --release --example cgmes_microgrid_be_dump --features cgmes

use gridoxide::cgmes::{cgmes_to_buses_and_branches, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let base = root.join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGid-BaseCase");
    let dir = base.join("MicroGrid-BE-MAS");
    if !dir.exists() {
        eprintln!(
            "{} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        std::process::exit(1);
    }

    // Same file set as tests/cgmes_microgrid_be_test.rs — see that test's
    // own doc comment for why EQ_BD is needed too (some BaseVoltage
    // references resolve only there, not in BE-MAS's own EQ file).
    let eq_bd = base.join("MicroGrid-BD-MAS/20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let eq = dir.join("20210325T1530Z_1D_BE_EQ_001.xml");
    let ssh = dir.join("20210325T1530Z_1D_BE_SSH_001.xml");
    let tp = dir.join("20210325T1530Z_1D_BE_TP_001.xml");
    let sv = dir.join("20210325T1530Z_1D_BE_SV_001.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus);

    // Only the physical TopologicalNode buses (tn_mrids.len() of them) —
    // synthesized buses (3-winding star points, boundary ConnectivityNodes)
    // have no TopologicalNode mRID of their own to key on.
    let mut out = Vec::with_capacity(tn_mrids.len());
    for (idx, mrid) in tn_mrids.iter().enumerate() {
        let bus = &result[idx];
        out.push(serde_json::json!({
            "mrid": mrid,
            "u_rated": bus.u_rated,
            "voltage_mag": bus.voltage_mag,
            "voltage_ang": bus.voltage_ang,
            "bus_type": bus.bus_type,
        }));
    }
    println!("{}", serde_json::to_string(&out).unwrap());
}

//! The node-breaker topology processor, against the exporter's own answer.
//!
//! CGMES carries connectivity twice. `ConnectivityNode` (EQ) is the
//! substation-level truth; `TopologicalNode` (TP) is a partial reduction of it
//! that the exporter already performed. So every configuration carrying both
//! supplies **free ground truth**: derive the bus view from EQ+SSH alone, and
//! compare it against the partition TP records.
//!
//! That is a much stronger check than gridoxide asserting against itself, and
//! it costs no fixture authoring — the data has been in the tree all along.
//!
//! # What the comparison actually found
//!
//! Neither partition contains the other in general, and both directions of
//! disagreement turn out to be real and explicable:
//!
//! | Config | CN | switches (conducting) | TN | buses | splits | multi-TN buses |
//! |---|---:|---:|---:|---:|---:|---:|
//! | MiniGrid | 103 | 90 (90) | 13 | 13 | 0 | 0 |
//! | FullGrid | 48 | 29 (28) | 25 | 22 | 1 | 2 |
//! | SmallGrid | 1,369 | 1,266 (1,203) | 167 | 167 | 4 | 1 |
//! | Svedala | 1,179 | 1,464 (988) | 191 | 228 | 29 | 0 |
//!
//! **MiniGrid agrees exactly** — 103 connectivity nodes and 90 switches reduce
//! to precisely the 13 topological nodes the exporter published. That is the
//! headline result: the processor reproduces a real exporter's reduction on
//! real data.
//!
//! A **multi-TN bus** is gridoxide merging *further* than the exporter: a
//! closed switch the export left unreduced. FullGrid is the known case — it is
//! why `merge_closed_switches` exists at all — and SmallGrid has one too.
//!
//! A **split** is the opposite, and is the more interesting direction: the
//! exporter placed connectivity nodes in one topological node that gridoxide
//! keeps apart. Every single one of the 34 splits across these four
//! configurations has the same cause — the nodes are joined *only* by switches
//! that are **open or out of service**. On FullGrid it is a single open
//! `GroundDisconnector` behind which one connectivity node sits; on Svedala it
//! is 29 of them, which is unsurprising for the most switch-dense model in the
//! tree (1,464 switches, only 988 conducting).
//!
//! gridoxide is right in that direction and the exporter is loose: an open
//! switch does not tie two points together. So the gate below asserts the
//! *cause* rather than the count — every split must be bridged only by
//! non-conducting switches — which is a statement that would break if the
//! reader ever missed an edge, while a bare count would not.



use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_node_breaker_topology, load_profiles, CgmesNodeBreaker};
use gridoxide::topology::model::{BusIdx, NodeIdx};
use gridoxide::topology::{bus_view, BusView, RetentionPolicy};

struct Fixture {
    name: &'static str,
    dir: &'static str,
    prefix: &'static str,
}

/// The four genuine node-breaker configurations in the tree. RealGrid and
/// PowerFlow are 1:1 CN:TN with no switches, so they exercise nothing here.
const FIXTURES: &[Fixture] = &[
    Fixture { name: "MiniGrid", dir: "MiniGrid/MiniGrid-Merged", prefix: "MiniGrid" },
    Fixture { name: "FullGrid", dir: "FullGrid/FullGrid-Merged", prefix: "FullGrid" },
    Fixture { name: "SmallGrid", dir: "SmallGrid/SmallGrid-Merged", prefix: "SmallGrid" },
    Fixture { name: "Svedala", dir: "Svedala/Svedala-Merged", prefix: "Svedala" },
];

fn profile_dir(fixture: &Fixture) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(fixture.dir)
}

fn available(fixture: &Fixture) -> Option<PathBuf> {
    let dir = profile_dir(fixture);
    if dir.exists() {
        return Some(dir);
    }
    eprintln!(
        "skipping {}: {} not found — run `git submodule update --init \
         tests/data/CGMES-Test-Configurations`",
        fixture.name,
        dir.display()
    );
    None
}

/// Loads the node-breaker graph. `with_tp` controls only whether the oracle is
/// available — nothing in the reader consults TP.
fn read(fixture: &Fixture, with_tp: bool) -> Option<CgmesNodeBreaker> {
    let dir = available(fixture)?;
    let eq_bd = dir.join(format!("{}_EQBD.xml", fixture.prefix));
    let eq = dir.join(format!("{}_EQ.xml", fixture.prefix));
    let ssh = dir.join(format!("{}_SSH.xml", fixture.prefix));
    let tp = dir.join(format!("{}_TP.xml", fixture.prefix));
    let mut paths: Vec<&Path> = vec![&eq_bd, &eq, &ssh];
    if with_tp {
        paths.push(&tp);
    }
    let ds = load_profiles(&paths).expect("failed to decode CGMES profiles");
    Some(cgmes_node_breaker_topology(&ds).expect("node-breaker read failed"))
}

fn nodes_by_topological_node(nb: &CgmesNodeBreaker) -> HashMap<&str, Vec<usize>> {
    let mut out: HashMap<&str, Vec<usize>> = HashMap::new();
    for (node, tn) in nb.tn_of_node.iter().enumerate() {
        if let Some(tn) = tn {
            out.entry(tn.as_str()).or_default().push(node);
        }
    }
    out
}

/// Every switch whose two ends landed on different buses, among `nodes`.
fn crossing_switches<'a>(
    nb: &'a CgmesNodeBreaker,
    view: &BusView,
    nodes: &[usize],
) -> Vec<&'a gridoxide::topology::Switch> {
    let set: HashSet<usize> = nodes.iter().copied().collect();
    nb.topology
        .switches
        .iter()
        .filter(|s| {
            set.contains(&s.nodes[0].0)
                && set.contains(&s.nodes[1].0)
                && view.bus_of(s.nodes[0]) != view.bus_of(s.nodes[1])
        })
        .collect()
}

/// **The gate.** Wherever the EQ+SSH bus view disagrees with the exporter's
/// topological nodes, the disagreement must be *explained*.
///
/// A `TopologicalNode` split across several gridoxide buses is only defensible
/// if those buses are joined solely by switches that are open or out of
/// service — that is gridoxide honoring a switch position the exporter
/// disregarded. If a split had **no** switch between its parts at all, the
/// reader would have missed an edge, and that is precisely what this asserts
/// against.
#[test]
fn every_disagreement_with_the_exporter_is_explained_by_an_open_switch() {
    let mut checked = 0;
    for fixture in FIXTURES {
        let Some(nb) = read(fixture, true) else { continue };
        let view = bus_view(&nb.topology, &RetentionPolicy::MergeAll);
        let by_tn = nodes_by_topological_node(&nb);
        checked += 1;

        // Fixture assumption: these are the node-breaker configurations.
        assert!(!nb.topology.switches.is_empty(), "{}: no switches read", fixture.name);
        assert!(
            nb.topology.n_nodes > by_tn.len(),
            "{}: {} CN is not more than {} TN — not a node-breaker export?",
            fixture.name,
            nb.topology.n_nodes,
            by_tn.len()
        );

        let mut splits = 0;
        let mut tns_of_bus: HashMap<usize, HashSet<&str>> = HashMap::new();
        for (tn, nodes) in &by_tn {
            let buses: HashSet<BusIdx> =
                nodes.iter().map(|&n| view.bus_of(NodeIdx(n))).collect();
            for b in &buses {
                tns_of_bus.entry(b.0).or_default().insert(tn);
            }
            if buses.len() == 1 {
                continue;
            }
            splits += 1;

            let crossing = crossing_switches(&nb, &view, nodes);
            assert!(
                !crossing.is_empty(),
                "{}: TopologicalNode {tn} is split across {} gridoxide buses with no switch \
                 between the parts at all — the reader missed an edge",
                fixture.name,
                buses.len()
            );
            for s in &crossing {
                assert!(
                    !s.conducting(),
                    "{}: TopologicalNode {tn} is split, but {:?} between its parts is \
                     conducting — gridoxide should have merged it",
                    fixture.name,
                    s.kind
                );
            }
        }

        let multi_tn = tns_of_bus.values().filter(|s| s.len() > 1).count();
        eprintln!(
            "{:<10} CN={:<5} sw={:<5} ({:<5} conducting)  bb={:<4} TN={:<4} buses={:<5} \
             splits={splits} multi-TN buses={multi_tn}",
            fixture.name,
            nb.topology.n_nodes,
            nb.topology.switches.len(),
            nb.topology.conducting_count(),
            nb.topology.busbars.len(),
            by_tn.len(),
            tns_of_bus.len(),
        );
    }
    assert!(checked > 0, "no CGMES fixture was available; the gate proved nothing");
}

/// MiniGrid is the clean case, and pinning it is what stops the gate above
/// from passing vacuously: 103 connectivity nodes and 90 switches must reduce
/// to exactly the 13 topological nodes the exporter published, with no
/// disagreement in either direction.
#[test]
fn minigrid_reproduces_the_exporter_partition_exactly() {
    let fixture = &FIXTURES[0];
    assert_eq!(fixture.name, "MiniGrid");
    let Some(nb) = read(fixture, true) else { return };

    let view = bus_view(&nb.topology, &RetentionPolicy::MergeAll);
    let by_tn = nodes_by_topological_node(&nb);

    assert_eq!(nb.topology.n_nodes, 103);
    assert_eq!(nb.topology.switches.len(), 90);
    assert_eq!(nb.topology.conducting_count(), 90, "every MiniGrid switch is closed");
    assert_eq!(by_tn.len(), 13);

    // Exact agreement: the TN partition and the bus partition are the same
    // partition of the connectivity nodes that carry a TN.
    let mut buses_seen: HashSet<BusIdx> = HashSet::new();
    for (tn, nodes) in &by_tn {
        let buses: HashSet<BusIdx> = nodes.iter().map(|&n| view.bus_of(NodeIdx(n))).collect();
        assert_eq!(buses.len(), 1, "TopologicalNode {tn} split across {} buses", buses.len());
        assert!(
            buses_seen.insert(*buses.iter().next().unwrap()),
            "two TopologicalNodes landed on one bus — gridoxide merged further than the exporter"
        );
    }
    assert_eq!(buses_seen.len(), 13);
}

/// FullGrid is the configuration that forced `merge_closed_switches` into
/// existence: its plain `Switch` is closed in SSH yet spans two distinct
/// `TopologicalNode`s. So it must still be a case where gridoxide merges
/// *further* than the exporter did — otherwise either the fixture changed or
/// the reader stopped seeing that switch.
#[test]
fn fullgrid_still_needs_merging_the_exporter_did_not_do() {
    let fixture = &FIXTURES[1];
    assert_eq!(fixture.name, "FullGrid");
    let Some(nb) = read(fixture, true) else { return };

    let view = bus_view(&nb.topology, &RetentionPolicy::MergeAll);
    let mut tns_of_bus: HashMap<BusIdx, HashSet<&str>> = HashMap::new();
    for (tn, nodes) in &nodes_by_topological_node(&nb) {
        for &n in nodes {
            tns_of_bus.entry(view.bus_of(NodeIdx(n))).or_default().insert(tn);
        }
    }
    assert!(
        tns_of_bus.values().any(|s| s.len() > 1),
        "FullGrid no longer has a bus spanning several TopologicalNodes"
    );
}

/// Busbar sections have to be found, or `RetentionPolicy::RetainAdjacentToBusbar`
/// silently retains nothing on real data — which would look like "no switches
/// worth keeping" rather than "the busbars were never read".
///
/// Also pins the ordering the whole retention idea rests on: retaining fewer
/// switches can only merge more.
#[test]
fn busbars_are_read_and_the_busbar_policy_selects_a_strict_subset() {
    let mut checked = 0;
    for fixture in FIXTURES {
        let Some(nb) = read(fixture, false) else { continue };
        checked += 1;

        assert!(!nb.topology.busbars.is_empty(), "{}: no busbar sections read", fixture.name);

        let all = bus_view(&nb.topology, &RetentionPolicy::RetainAll);
        let busbar = bus_view(&nb.topology, &RetentionPolicy::RetainAdjacentToBusbar);
        let merged = bus_view(&nb.topology, &RetentionPolicy::MergeAll);

        assert!(
            !busbar.retained().is_empty(),
            "{}: the busbar-adjacent policy retained nothing despite {} busbars",
            fixture.name,
            nb.topology.busbars.len()
        );
        assert!(
            busbar.retained().len() < all.retained().len(),
            "{}: the busbar-adjacent policy retained everything, so it is not a subset",
            fixture.name
        );
        assert!(merged.n_buses() <= busbar.n_buses());
        assert!(busbar.n_buses() <= all.n_buses());
        assert_eq!(all.n_buses(), nb.topology.n_nodes, "{}: RetainAll must merge nothing but junctions or nothing at all", fixture.name);

        eprintln!(
            "{:<10} retained: all={:<5} busbar-adjacent={:<5}   buses: merged={:<5} \
             busbar={:<5} all={}",
            fixture.name,
            all.retained().len(),
            busbar.retained().len(),
            merged.n_buses(),
            busbar.n_buses(),
            all.n_buses(),
        );
    }
    assert!(checked > 0, "no CGMES fixture was available");
}

/// The node-breaker read must not need TP at all — that is the whole point of
/// reading connectivity from EQ.
///
/// Compared as a **partition over mRIDs**, not index for index: `by_type` order
/// depends on which profiles were loaded, so both the node numbering and the
/// switch order legitimately differ between the two datasets. What must not
/// differ is which connectivity nodes end up together.
#[test]
fn the_bus_partition_does_not_depend_on_the_tp_profile() {
    let mut checked = 0;
    for fixture in FIXTURES {
        let (Some(without), Some(with)) = (read(fixture, false), read(fixture, true)) else {
            continue;
        };
        checked += 1;

        assert_eq!(without.topology.n_nodes, with.topology.n_nodes, "{}", fixture.name);
        assert_eq!(
            without.topology.switches.len(),
            with.topology.switches.len(),
            "{}",
            fixture.name
        );

        // TP supplies the oracle and nothing else, so it is the one difference.
        assert!(
            without.tn_of_node.iter().all(|t| t.is_none()),
            "{}: TopologicalNode references appeared without the TP profile",
            fixture.name
        );
        assert!(
            with.tn_of_node.iter().any(|t| t.is_some()),
            "{}: no TopologicalNode references even with TP loaded",
            fixture.name
        );

        let group = |nb: &CgmesNodeBreaker| -> HashMap<String, usize> {
            let view = bus_view(&nb.topology, &RetentionPolicy::MergeAll);
            nb.node_mrids
                .iter()
                .enumerate()
                .map(|(n, mrid)| (mrid.clone(), view.bus_of(NodeIdx(n)).0))
                .collect()
        };
        let (a, b) = (group(&without), group(&with));
        assert_eq!(a.len(), b.len(), "{}", fixture.name);

        // Same partition: two nodes share a bus in one iff they do in the other.
        let mut canonical: HashMap<usize, usize> = HashMap::new();
        for (mrid, bus_a) in &a {
            let bus_b = b[mrid];
            match canonical.get(bus_a) {
                None => {
                    canonical.insert(*bus_a, bus_b);
                }
                Some(&expected) => assert_eq!(
                    bus_b, expected,
                    "{}: node {mrid} groups differently with and without TP",
                    fixture.name
                ),
            }
        }
        assert_eq!(
            canonical.values().collect::<HashSet<_>>().len(),
            canonical.len(),
            "{}: two buses without TP collapsed into one with TP",
            fixture.name
        );
    }
    assert!(checked > 0, "no CGMES fixture was available");
}

/// Every switch endpoint and busbar must be in range, and the mrid vectors must
/// line up with the index spaces they describe.
#[test]
fn the_returned_indices_are_internally_consistent() {
    for fixture in FIXTURES {
        let Some(nb) = read(fixture, false) else { continue };

        assert_eq!(nb.node_mrids.len(), nb.topology.n_nodes, "{}", fixture.name);
        assert_eq!(nb.tn_of_node.len(), nb.topology.n_nodes, "{}", fixture.name);
        assert_eq!(nb.switch_mrids.len(), nb.topology.switches.len(), "{}", fixture.name);

        for s in &nb.topology.switches {
            for n in s.nodes {
                assert!(n.0 < nb.topology.n_nodes, "{}: node {n:?} out of range", fixture.name);
            }
        }
        for b in &nb.topology.busbars {
            assert!(b.0 < nb.topology.n_nodes, "{}: busbar {b:?} out of range", fixture.name);
        }

        // Node mrids are the identity of the index space, so a duplicate would
        // silently alias two substation points.
        let unique: HashSet<&String> = nb.node_mrids.iter().collect();
        assert_eq!(unique.len(), nb.node_mrids.len(), "{}: duplicate CN mrid", fixture.name);

        let view = bus_view(&nb.topology, &RetentionPolicy::MergeAll);
        let total: usize = (0..view.n_buses()).map(|b| view.nodes_of(BusIdx(b)).len()).sum();
        assert_eq!(total, nb.topology.n_nodes, "{}", fixture.name);
    }
}

//! Every MATPOWER case in benchmark-grids, solved twice: as [`matpower::read`] reads it, and
//! converted to CGMES by cimoxide-import and read back through [`cgmes_to_network_with`].
//!
//! Both are the same network, and both go through the same Newton solver, so the answers must
//! agree to rounding. That makes this a test of gridoxide's two readers, which tolerances
//! against a published solution are too loose to be: any difference is one of them reading
//! something differently. It found two in the CGMES reader (the slack's magnitude, and a
//! machine's reactive power sign), fixed in #42.
//!
//! Both topologies are read: the `TopologicalNode`s of the TP, and the breakers of the EQ, which
//! the converter puts at every branch end. Each solve starts from the case's own voltages, which
//! the converter writes to the SV: three RTE cases don't converge from a flat start, by either
//! path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use gridoxide::cgmes::{CgmesTopologyMode, CimDataset, cgmes_to_network_with, load_profiles};
use gridoxide::network::ShuntAdm;
use gridoxide::solver::SolveStatus;
use gridoxide::types::{Bus, BusType, Line, Transformer};
use gridoxide::{PowerFlowInit, PowerFlowOptions, PowerFlowReport, TapData, matpower};

/// Per bus, on |V| in per-unit and on the angle from the reference in radians. The two paths
/// agree to 3e-12 on every case; this leaves room for the larger cases' rounding.
const AGREE: f64 = 1e-9;

/// OPF benchmarks whose dispatch has no power-flow solution: both paths diverge.
const UNSOLVABLE: &[&str] = &["pglib_opf_case300_ieee", "pglib_opf_case3_lmbd"];

/// Above this, a case runs in [`the_large_cases_solve_the_same_through_cgmes`] only.
const LARGE: usize = 2_000;

fn grids() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/benchmark-grids");
    if dir.join("matpower").exists() {
        Some(dir)
    } else {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/benchmark-grids`",
            dir.display()
        );
        None
    }
}

/// Every case the converter takes, with its number of buses. It refuses the originals that edit
/// their matrices in MATLAB code; `matpower-plain/` has those as plain data.
fn cases(grids: &Path) -> Vec<(PathBuf, usize)> {
    let mut cases: Vec<(PathBuf, usize)> = ["matpower", "matpower-plain", "pglib"]
        .iter()
        .flat_map(|dir| std::fs::read_dir(grids.join(dir)).expect("read dir"))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().is_some_and(|x| x == "m"))
        .filter_map(|path| {
            let text = std::fs::read_to_string(&path).expect("read case");
            match cimimport::matpower::parse::parse(&text) {
                Ok(case) => Some((path, case.bus.len())),
                Err(cimimport::MatpowerError::Unsupported(_)) => None,
                Err(e) => panic!("{}: {e}", path.display()),
            }
        })
        .collect();
    cases.sort();
    cases
}

fn name(path: &Path) -> String {
    path.file_name().unwrap().to_string_lossy().into_owned()
}

fn options() -> PowerFlowOptions {
    PowerFlowOptions {
        // As set by the starting voltages below.
        init: PowerFlowInit::Flat,
        // 1e-10 is below what `case141`'s numbers allow.
        tol: 1e-8,
        max_iter: 40,
        ..Default::default()
    }
}

/// Starts each bus where `at` says it was: the angle everywhere, the magnitude where it isn't
/// held.
fn start(buses: &mut [Bus], at: impl IntoIterator<Item = (usize, f64, f64)>) {
    for (bus, vm, va_deg) in at {
        if buses[bus].bus_type == BusType::PQ {
            buses[bus].voltage_mag = vm;
        }
        buses[bus].voltage_ang = va_deg.to_radians();
    }
}

fn solve(
    buses: Vec<Bus>,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
) -> PowerFlowReport {
    gridoxide::run_power_flow(
        buses,
        lines,
        transformers,
        shunts,
        TapData::none(),
        options(),
    )
}

/// Each MATPOWER bus number's `TopologicalNode`, which the converter describes as
/// `MATPOWER bus {n}`.
fn nodes(ds: &CimDataset) -> HashMap<i64, String> {
    ds.by_type["TopologicalNode"]
        .iter()
        .filter_map(|id| {
            let number = ds.entries[id]
                .get_str("IdentifiedObject.description")?
                .strip_prefix("MATPOWER bus ")?
                .parse()
                .ok()?;
            Some((number, id.clone()))
        })
        .collect()
}

/// Each `TopologicalNode`'s `SvVoltage`, in kV and degrees.
fn sv(ds: &CimDataset) -> HashMap<String, (f64, f64)> {
    ds.by_type["SvVoltage"]
        .iter()
        .filter_map(|id| {
            let sv = &ds.entries[id];
            Some((
                sv.get_ref("SvVoltage.TopologicalNode")?.to_string(),
                (sv.get_f64("SvVoltage.v")?, sv.get_f64("SvVoltage.angle")?),
            ))
        })
        .collect()
}

/// The largest difference per bus between two solves, on |V| and on the angle from each one's
/// reference.
fn difference(a: &[Bus], b: &[Bus], pairs: &[(usize, usize)], reference: (usize, usize)) -> f64 {
    let angle =
        |buses: &[Bus], bus: usize, at: usize| buses[bus].voltage_ang - buses[at].voltage_ang;
    pairs
        .iter()
        .map(|&(i, j)| {
            let dv = (a[i].voltage_mag - b[j].voltage_mag).abs();
            let da = (angle(a, i, reference.0) - angle(b, j, reference.1)).abs();
            dv.max(da)
        })
        .fold(0.0, f64::max)
}

/// Solves one case both ways and returns what failed, if anything.
fn check(path: &Path) -> Option<String> {
    let case = name(path);
    let stem = path.file_stem().unwrap().to_string_lossy();
    let text = std::fs::read_to_string(path).expect("read case");
    let direct = matpower::read(path).expect("matpower::read");

    // The case's Vm and Va, by bus number: the direct side's start.
    let parsed = cimimport::matpower::parse::parse(&text).expect("parse");
    let given: HashMap<i64, (f64, f64)> = parsed
        .bus
        .iter()
        .map(|row| (row[0] as i64, (row[7], row[8])))
        .collect();
    let mut buses = direct.buses.clone();
    start(
        &mut buses,
        direct.bus_numbers.iter().enumerate().map(|(k, &n)| {
            let (vm, va) = given[&(n as i64)];
            (k, vm, va)
        }),
    );
    let solved = solve(buses, &direct.lines, &direct.transformers, &direct.shunts);
    let reference = direct
        .buses
        .iter()
        .position(|b| b.bus_type == BusType::Slack)
        .expect("a reference bus");

    let converted = cimimport::matpower(&case, &text, &Default::default()).expect("convert");
    let paths: Vec<PathBuf> = {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("matpower-via-cgmes")
            .join(stem.as_ref());
        std::fs::create_dir_all(&dir).expect("create dir");
        converted
            .files
            .iter()
            .map(|(file, xml)| {
                let path = dir.join(file);
                std::fs::write(&path, xml).expect("write profile");
                path
            })
            .collect()
    };
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let ds = load_profiles(&refs).expect("decode the converted profiles");
    let node = nodes(&ds);
    let sv = sv(&ds);

    let mut failed = Vec::new();
    let expect_solved = !UNSOLVABLE.contains(&stem.as_ref());
    if (solved.stats.status == SolveStatus::Converged) != expect_solved {
        failed.push(format!("direct: {:?}", solved.stats.status));
    }
    for mode in [
        CgmesTopologyMode::BusBranchFromTp,
        CgmesTopologyMode::NodeBreakerFromEq,
    ] {
        let net = cgmes_to_network_with(&ds, direct.base_mva * 1e6, mode).expect("read CGMES");
        let bus_of = |n: u64| net.node_bus[&node[&(n as i64)]];
        // The CGMES side starts from the SV.
        let mut buses = net.buses.clone();
        start(
            &mut buses,
            direct.bus_numbers.iter().map(|&n| {
                let bus = bus_of(n);
                let (kv, deg) = sv[&node[&(n as i64)]];
                (bus, kv * 1e3 / net.buses[bus].u_rated, deg)
            }),
        );
        let through = solve(buses, &net.lines, &net.transformers, &net.shunts);
        if through.stats.status != solved.stats.status {
            failed.push(format!(
                "{mode:?}: {:?} where the direct solve is {:?}",
                through.stats.status, solved.stats.status
            ));
            continue;
        }
        if !expect_solved {
            continue;
        }
        let pairs: Vec<(usize, usize)> = direct
            .bus_numbers
            .iter()
            .enumerate()
            .map(|(k, &n)| (k, bus_of(n)))
            .collect();
        let d = difference(
            &solved.buses,
            &through.buses,
            &pairs,
            (reference, bus_of(direct.bus_numbers[reference])),
        );
        if d >= AGREE {
            failed.push(format!("{mode:?}: differs by {d:.2e}"));
        }
    }

    // How far from solved the case was saved: reported, not failed.
    if expect_solved {
        let saved = direct
            .bus_numbers
            .iter()
            .enumerate()
            .map(|(k, &n)| (given[&(n as i64)].0 - solved.buses[k].voltage_mag).abs())
            .fold(0.0, f64::max);
        eprintln!("{case}: saved {saved:.1e} pu from its solution");
    }
    (!failed.is_empty()).then(|| format!("{case}: {}", failed.join("; ")))
}

fn run(large: bool) {
    let Some(grids) = grids() else { return };
    let cases: Vec<PathBuf> = cases(&grids)
        .into_iter()
        .filter(|(_, buses)| (*buses > LARGE) == large)
        .map(|(path, _)| path)
        .collect();
    assert!(!cases.is_empty());
    // One thread per case: they share nothing.
    let failed: Vec<String> = std::thread::scope(|scope| {
        let running: Vec<_> = cases
            .iter()
            .map(|path| scope.spawn(|| check(path)))
            .collect();
        running
            .into_iter()
            .filter_map(|case| case.join().expect("a case panicked"))
            .collect()
    });
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Every case up to [`LARGE`] buses, the radial feeders and the pglib OPF cases among them.
#[test]
fn every_case_solves_the_same_through_cgmes() {
    run(false);
}

/// The PEGASE, RTE and Polish cases above [`LARGE`] buses. Ignored for its time, 41 s
/// unoptimised: run with `--release -- --ignored`.
#[test]
#[ignore]
fn the_large_cases_solve_the_same_through_cgmes() {
    run(true);
}

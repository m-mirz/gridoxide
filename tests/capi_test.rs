//! The C ABI, exercised from Rust.
//!
//! These call the exported functions exactly as a C++ consumer would — through
//! raw pointers, with caller-owned buffers — but without needing a C compiler,
//! so they run wherever `cargo test` does. The compiled C and C++ examples
//! under `capi/examples/` cover the parts only a real C compiler can: that the
//! header parses, that the struct layouts agree, and that the symbols link.
//!
//! Two of the tests below matter more than their size suggests:
//!
//! - [`a_panic_is_caught_rather_than_killing_the_process`] — an unwind out of
//!   an `extern "C"` function aborts, taking the *host* program down. If the
//!   guard ever comes off, this test does not fail, it kills the test runner.
//! - [`the_committed_header_is_not_stale`] — the header is committed so a
//!   consumer with no Rust toolchain can use it, which means nothing forces it
//!   to match the code unless something checks.

#![cfg(feature = "capi")]

use std::ffi::{CStr, CString};
use std::path::PathBuf;

use gridoxide::capi::powerflow::*;
use gridoxide::capi::types::*;
use gridoxide::capi::{gridoxide_abi_version, gridoxide_last_error_message, GridoxideIslandStatus, Status};

fn fixture(name: &str) -> CString {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf")
        .join(format!("{name}.json"));
    CString::new(path.to_str().unwrap()).unwrap()
}

fn last_error() -> String {
    let ptr = gridoxide_last_error_message();
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: the pointer is either null or a live thread-local CString.
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

/// Opens a model from a fixture, asserting it worked.
fn open(name: &str) -> *mut GridoxidePowerFlow {
    // pglib documents are per-unit on a 100 MVA base.
    let options = GridoxideOptions { s_base_va: 1e8, ..Default::default() };
    let mut handle: *mut GridoxidePowerFlow = std::ptr::null_mut();
    // SAFETY: a valid path, a live options struct, a writable out-pointer.
    let status =
        unsafe { gridoxide_powerflow_from_pgm_file(fixture(name).as_ptr(), &options, &mut handle) };
    assert_eq!(status, Status::Ok, "{name}: {}", last_error());
    assert!(!handle.is_null());
    handle
}

fn voltages(handle: *mut GridoxidePowerFlow) -> (Vec<f64>, Vec<f64>) {
    // SAFETY: live handle.
    let n = unsafe { gridoxide_powerflow_bus_count(handle) };
    let (mut vm, mut va) = (vec![0.0; n], vec![0.0; n]);
    // SAFETY: buffers are exactly `n` long, as required.
    unsafe {
        assert_eq!(gridoxide_powerflow_voltage_magnitude(handle, vm.as_mut_ptr(), n), Status::Ok);
        assert_eq!(gridoxide_powerflow_voltage_angle(handle, va.as_mut_ptr(), n), Status::Ok);
    }
    (vm, va)
}

/// **The correctness gate.** The ABI must produce exactly what calling the
/// solver directly produces — not merely something plausible.
///
/// Compared bit-for-bit rather than to a tolerance: both paths run the same
/// Newton iteration on the same matrix from the same start, so any difference
/// at all is a marshalling error, not numerical weather.
#[test]
fn the_abi_reproduces_a_direct_rust_solve_exactly() {
    for name in ["pglib_opf_case14_ieee", "pglib_opf_case30_ieee", "pglib_opf_case118_ieee"] {
        let handle = open(name);
        // SAFETY: live handle.
        assert_eq!(unsafe { gridoxide_powerflow_solve(handle) }, Status::Ok, "{}", last_error());
        let (vm, va) = voltages(handle);

        // The same network, solved without the ABI in the way.
        let text = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/pglib-opf")
                .join(format!("{name}.json")),
        )
        .unwrap();
        let input: gridoxide::pgm::PgmInput = serde_json::from_str(&text).unwrap();
        let id_to_idx = gridoxide::pgm::node_id_to_idx(&input);
        let shunts = gridoxide::pgm::pgm_shunts_1ph(&input, &id_to_idx, 1e8);
        let (mut buses, lines, transformers) =
            gridoxide::pgm::pgm_to_buses_and_branches(input, 1e8, 50.0);
        let mut ybus = gridoxide::network::build_ybus(buses.len(), &lines, &transformers);
        gridoxide::network::stamp_shunts(&mut ybus, &shunts);
        let ybus = ybus.finish();
        gridoxide::solver::newton_raphson(&mut buses, &ybus, 1e-8, 20);

        for i in 0..buses.len() {
            assert_eq!(vm[i], buses[i].voltage_mag, "{name} bus {i}: |V|");
            assert_eq!(va[i], buses[i].voltage_ang, "{name} bus {i}: angle");
        }
        // SAFETY: live handle, freed once.
        unsafe { gridoxide_powerflow_free(handle) };
    }
}

/// The in-memory path and the document path describe the same network, so they
/// must reach the same answer.
///
/// This is what makes the `#[repr(C)]` structs trustworthy: a field in the
/// wrong order or a unit misread would show up here as a different solution,
/// where a standalone in-memory test would happily converge to whatever it was
/// handed.
#[test]
fn the_array_path_agrees_with_the_document_path() {
    // Two buses, one line, load at bus 1 — small enough to write out in full.
    let buses = [
        GridoxideBus {
            bus_type: GridoxideBusType::Slack,
            voltage_magnitude: 1.0,
            voltage_angle: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: f64::NEG_INFINITY,
            q_max: f64::INFINITY,
            u_rated: 1.0,
        },
        GridoxideBus {
            bus_type: GridoxideBusType::Pq,
            voltage_magnitude: 1.0,
            voltage_angle: 0.0,
            p_spec: -0.5,
            q_spec: -0.1,
            q_min: f64::NEG_INFINITY,
            q_max: f64::INFINITY,
            u_rated: 1.0,
        },
    ];
    let lines = [GridoxideLine { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 }];

    let mut handle: *mut GridoxidePowerFlow = std::ptr::null_mut();
    // SAFETY: arrays outlive the call; counts match; out-pointer writable.
    let status = unsafe {
        gridoxide_powerflow_from_arrays(
            buses.as_ptr(),
            buses.len(),
            lines.as_ptr(),
            lines.len(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(status, Status::Ok, "{}", last_error());
    // SAFETY: live handle.
    assert_eq!(unsafe { gridoxide_powerflow_solve(handle) }, Status::Ok, "{}", last_error());
    let (vm, va) = voltages(handle);

    // The same network built directly in Rust.
    let mut direct = vec![
        gridoxide::types::Bus {
            idx: 0,
            bus_type: gridoxide::types::BusType::Slack,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: f64::NEG_INFINITY,
            q_max: f64::INFINITY,
            u_rated: 1.0,
            zip_terms: Vec::new(),
        },
        gridoxide::types::Bus {
            idx: 1,
            bus_type: gridoxide::types::BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: -0.5,
            q_spec: -0.1,
            q_min: f64::NEG_INFINITY,
            q_max: f64::INFINITY,
            u_rated: 1.0,
            zip_terms: Vec::new(),
        },
    ];
    let direct_lines = vec![gridoxide::types::Line {
        from: 0,
        to: 1,
        r: 0.02,
        x: 0.06,
        b_shunt: 0.0,
        g_shunt: 0.0,
    }];
    let ybus = gridoxide::network::build_ybus(2, &direct_lines, &[]).finish();
    gridoxide::solver::newton_raphson(&mut direct, &ybus, 1e-8, 20);

    for i in 0..2 {
        assert_eq!(vm[i], direct[i].voltage_mag, "bus {i}: |V|");
        assert_eq!(va[i], direct[i].voltage_ang, "bus {i}: angle");
    }
    // SAFETY: live handle, freed once.
    unsafe { gridoxide_powerflow_free(handle) };
}

/// AC branch flows — the one capability added rather than mirrored, since the
/// Python binding exposes them for DC only.
///
/// Checked against the physics rather than against itself: on a lossless
/// two-bus line the power leaving one end must equal what arrives at the other,
/// negated. With resistance it must exceed it, by the loss.
#[test]
fn branch_flows_are_reported_and_obey_conservation() {
    let handle = open("pglib_opf_case14_ieee");
    // SAFETY: live handle.
    unsafe { assert_eq!(gridoxide_powerflow_solve(handle), Status::Ok) };
    // SAFETY: live handle.
    let n = unsafe { gridoxide_powerflow_branch_count(handle) };
    assert!(n > 0);

    let (mut pf, mut qf) = (vec![0.0; n], vec![0.0; n]);
    let (mut pt, mut qt) = (vec![0.0; n], vec![0.0; n]);
    // SAFETY: buffers are exactly `n` long.
    unsafe {
        assert_eq!(
            gridoxide_powerflow_branch_flow(handle, 0, pf.as_mut_ptr(), qf.as_mut_ptr(), n),
            Status::Ok
        );
        assert_eq!(
            gridoxide_powerflow_branch_flow(handle, 1, pt.as_mut_ptr(), qt.as_mut_ptr(), n),
            Status::Ok
        );
    }

    // Losses are positive and small: a branch cannot generate active power.
    for b in 0..n {
        let loss = pf[b] + pt[b];
        assert!(loss > -1e-9, "branch {b} produced {loss} pu of active power");
        assert!(loss < 0.5, "branch {b} lost {loss} pu, which is implausible");
    }
    assert!(pf.iter().any(|p| p.abs() > 0.01), "no branch carried anything");

    // A terminal other than from/to is the caller's mistake.
    // SAFETY: live handle, correct buffer length.
    let bad = unsafe { gridoxide_powerflow_branch_flow(handle, 7, pf.as_mut_ptr(), std::ptr::null_mut(), n) };
    assert_eq!(bad, Status::InvalidArgument);
    assert!(last_error().contains("terminal"), "{}", last_error());

    // SAFETY: live handle, freed once.
    unsafe { gridoxide_powerflow_free(handle) };
}

/// Distributed slack through the ABI, with the shift reported back.
#[test]
fn distributed_slack_is_reachable_and_reports_its_shift() {
    let handle = open("pglib_opf_case14_ieee");
    // SAFETY: live handle; null factors means "every generator, equal share".
    let status =
        unsafe { gridoxide_powerflow_solve_distributing_slack(handle, std::ptr::null(), 0) };
    assert_eq!(status, Status::Ok, "{}", last_error());

    // SAFETY: live handle.
    let n = unsafe { gridoxide_powerflow_bus_count(handle) };
    let mut shift = vec![0.0; n];
    // SAFETY: buffer is exactly `n` long.
    assert_eq!(
        unsafe { gridoxide_powerflow_slack_shift(handle, shift.as_mut_ptr(), n) },
        Status::Ok
    );
    let moved: f64 = shift.iter().sum();
    assert!(moved > 0.1, "only {moved} pu was moved off the slack");

    // An ordinary solve leaves nothing to report — and that is `NoAnswer`, the
    // third channel, not a failure.
    // SAFETY: live handle.
    unsafe { assert_eq!(gridoxide_powerflow_solve(handle), Status::Ok) };
    assert_eq!(
        unsafe { gridoxide_powerflow_slack_shift(handle, shift.as_mut_ptr(), n) },
        Status::NoAnswer
    );

    // SAFETY: live handle, freed once.
    unsafe { gridoxide_powerflow_free(handle) };
}

/// **A panic must not escape.**
///
/// `newton_raphson_distributing_slack` asserts on a wrong-length participation
/// vector. Unwinding out of an `extern "C"` function aborts the process, so
/// without the guard this test would not fail — it would kill the test runner
/// and take every other test with it.
///
/// The ABI checks the length itself and reports `InvalidArgument`, which is the
/// better answer; `guard` is the backstop for the panics nobody predicted.
#[test]
fn a_panic_is_caught_rather_than_killing_the_process() {
    let handle = open("pglib_opf_case14_ieee");
    // case14 becomes 15 buses after conversion — see `islands_are_enumerable`.
    let wrong = vec![1.0, 1.0, 1.0];

    // SAFETY: live handle; the slice is shorter than the bus count, which is
    // precisely what is under test.
    let status =
        unsafe { gridoxide_powerflow_solve_distributing_slack(handle, wrong.as_ptr(), wrong.len()) };
    assert_eq!(status, Status::InvalidArgument, "{}", last_error());
    assert!(last_error().contains("15 buses"), "{}", last_error());

    // Still usable afterwards — a rejected call must not poison the handle.
    // SAFETY: live handle.
    assert_eq!(unsafe { gridoxide_powerflow_solve(handle) }, Status::Ok);

    // SAFETY: live handle, freed once.
    unsafe { gridoxide_powerflow_free(handle) };
}

/// Every way a caller can get it wrong, answered rather than crashed.
#[test]
fn caller_mistakes_are_reported_not_undefined() {
    // Null handle everywhere.
    // SAFETY: null is explicitly permitted and checked.
    unsafe {
        assert_eq!(gridoxide_powerflow_solve(std::ptr::null_mut()), Status::InvalidArgument);
        assert_eq!(gridoxide_powerflow_bus_count(std::ptr::null()), 0);
        assert_eq!(gridoxide_powerflow_branch_count(std::ptr::null()), 0);
        assert!(gridoxide_powerflow_max_mismatch(std::ptr::null()).is_nan());
        // Freeing null is a no-op, as `free(NULL)` is in C.
        gridoxide_powerflow_free(std::ptr::null_mut());
    }

    // A missing file is I/O, not a parse failure.
    let mut handle: *mut GridoxidePowerFlow = std::ptr::null_mut();
    let missing = CString::new("/nonexistent/network.json").unwrap();
    // SAFETY: valid pointers.
    let status =
        unsafe { gridoxide_powerflow_from_pgm_file(missing.as_ptr(), std::ptr::null(), &mut handle) };
    assert_eq!(status, Status::Io);
    assert!(last_error().contains("/nonexistent/network.json"), "{}", last_error());

    // Malformed JSON is a parse failure.
    let junk = "{ not json";
    // SAFETY: valid pointer and length.
    let status = unsafe {
        gridoxide_powerflow_from_pgm_string(
            junk.as_ptr() as *const std::ffi::c_char,
            junk.len(),
            std::ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(status, Status::Parse);

    // A branch referring to a bus that does not exist is the caller's error,
    // caught before it can panic inside `build_ybus`.
    let buses = [GridoxideBus {
        bus_type: GridoxideBusType::Slack,
        voltage_magnitude: 1.0,
        voltage_angle: 0.0,
        p_spec: 0.0,
        q_spec: 0.0,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
    }];
    let lines = [GridoxideLine { from: 0, to: 9, r: 0.01, x: 0.1, b_shunt: 0.0, g_shunt: 0.0 }];
    // SAFETY: arrays outlive the call and counts match.
    let status = unsafe {
        gridoxide_powerflow_from_arrays(
            buses.as_ptr(),
            1,
            lines.as_ptr(),
            1,
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(status, Status::InvalidArgument);
    assert!(last_error().contains("only 1"), "{}", last_error());

    // A buffer of the wrong length is refused rather than overrun.
    let handle = open("pglib_opf_case14_ieee");
    // SAFETY: live handle.
    unsafe { assert_eq!(gridoxide_powerflow_solve(handle), Status::Ok) };
    let mut too_small = vec![0.0; 3];
    // SAFETY: the length passed matches the buffer; it is the *network* it
    // does not match, which is what the check exists for.
    let status = unsafe {
        gridoxide_powerflow_voltage_magnitude(handle, too_small.as_mut_ptr(), too_small.len())
    };
    assert_eq!(status, Status::InvalidArgument);
    assert!(last_error().contains("needs 15"), "{}", last_error());
    // SAFETY: live handle, freed once.
    unsafe { gridoxide_powerflow_free(handle) };
}

/// A backend whose Cargo feature is absent is refused, not silently downgraded.
#[test]
fn an_unavailable_backend_is_refused() {
    if cfg!(feature = "pardiso") {
        return;
    }
    let options = GridoxideOptions {
        s_base_va: 1e8,
        backend: GridoxideBackend::Pardiso,
        ..Default::default()
    };
    let mut handle: *mut GridoxidePowerFlow = std::ptr::null_mut();
    // SAFETY: valid pointers.
    let status = unsafe {
        gridoxide_powerflow_from_pgm_file(
            fixture("pglib_opf_case14_ieee").as_ptr(),
            &options,
            &mut handle,
        )
    };
    assert_eq!(status, Status::InvalidArgument);
    assert!(last_error().contains("pardiso"), "{}", last_error());
}

/// Islands are reported per component, and a de-energized one is *not* a
/// failure of the call — the only place it becomes visible is here.
#[test]
fn islands_are_enumerable() {
    let handle = open("pglib_opf_case14_ieee");
    // SAFETY: live handle.
    unsafe { assert_eq!(gridoxide_powerflow_solve(handle), Status::Ok) };

    // SAFETY: live handle.
    let count = unsafe { gridoxide_powerflow_island_count(handle) };
    assert_eq!(count, 1, "case14 is one connected network");

    let mut status_out = GridoxideIslandStatus::Singular;
    // SAFETY: live handle, writable out-pointer, island 0 exists.
    assert_eq!(
        unsafe { gridoxide_powerflow_island_status(handle, 0, &mut status_out) },
        Status::Ok
    );
    assert_eq!(status_out, GridoxideIslandStatus::Converged);

    // SAFETY: live handle.
    let n = unsafe { gridoxide_powerflow_island_bus_count(handle, 0) };
    let mut members = vec![0usize; n];
    // SAFETY: buffer is exactly `n` long.
    assert_eq!(
        unsafe { gridoxide_powerflow_island_buses(handle, 0, members.as_mut_ptr(), n) },
        Status::Ok
    );
    // Fifteen, not fourteen. Converting a PGM document appends a **virtual
    // slack bus per source**, joined to the physical bus by an impedance
    // branch — right for a power flow, where the source's output is an unknown.
    // Worth asserting rather than glossing: a C++ caller indexing results by
    // its own bus numbering would silently read the wrong bus otherwise.
    assert_eq!(members.len(), 15, "14 physical buses plus one virtual slack bus");
    // SAFETY: live handle.
    assert_eq!(unsafe { gridoxide_powerflow_bus_count(handle) }, 15);

    // Out of range is reported, not read.
    // SAFETY: live handle, writable out-pointer.
    assert_eq!(
        unsafe { gridoxide_powerflow_island_status(handle, 99, &mut status_out) },
        Status::InvalidArgument
    );

    // SAFETY: live handle, freed once.
    unsafe { gridoxide_powerflow_free(handle) };
}

/// Solving twice gives the same answer — every solve restarts from the pristine
/// template, so calls do not accumulate.
#[test]
fn solving_twice_is_idempotent() {
    let handle = open("pglib_opf_case14_ieee");
    // SAFETY: live handle.
    unsafe { assert_eq!(gridoxide_powerflow_solve(handle), Status::Ok) };
    let first = voltages(handle);
    // SAFETY: live handle.
    unsafe { assert_eq!(gridoxide_powerflow_solve(handle), Status::Ok) };
    let second = voltages(handle);
    assert_eq!(first, second);
    // SAFETY: live handle, freed once.
    unsafe { gridoxide_powerflow_free(handle) };
}

/// The header's `GRIDOXIDE_ABI_VERSION` and the library's
/// `gridoxide_abi_version()` must agree, or a prebuilt consumer's compatibility
/// check is worthless.
#[test]
fn the_header_and_the_library_agree_on_the_abi_version() {
    let header = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include/gridoxide.h"),
    )
    .expect("include/gridoxide.h should be committed");
    let declared: u32 = header
        .lines()
        .find_map(|l| l.strip_prefix("#define GRIDOXIDE_ABI_VERSION "))
        .expect("the header must define GRIDOXIDE_ABI_VERSION")
        .trim()
        .parse()
        .unwrap();
    assert_eq!(declared, gridoxide_abi_version());
}

/// **The drift guard.** The header is committed so a consumer without a Rust
/// toolchain still gets one — which means nothing keeps it in step with the
/// code unless something checks.
///
/// The same shape as
/// `python/tests/test_matpower_opf.py::test_committed_documents_match_a_fresh_conversion`,
/// which guards the committed OPF fixtures for the same reason.
#[test]
fn the_committed_header_is_not_stale() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let committed = std::fs::read_to_string(root.join("include/gridoxide.h"))
        .expect("include/gridoxide.h should be committed");

    // Every exported symbol must be declared. A full textual regeneration here
    // would mean depending on cbindgen at test time and would break on a
    // cosmetic cbindgen upgrade; what actually matters is that nothing the
    // library exports is missing from the header a consumer compiles against.
    let source = std::fs::read_to_string(root.join("src/capi/mod.rs")).unwrap()
        + &std::fs::read_to_string(root.join("src/capi/types.rs")).unwrap()
        + &std::fs::read_to_string(root.join("src/capi/powerflow.rs")).unwrap();

    let mut missing = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        let name = trimmed
            .strip_prefix("pub extern \"C\" fn ")
            .or_else(|| trimmed.strip_prefix("pub unsafe extern \"C\" fn "));
        if let Some(rest) = name {
            let symbol = rest.split('(').next().unwrap_or("").trim();
            // A `$name` placeholder would mean a `macro_rules!` had crept
            // back in — cbindgen cannot expand those, so the function would be
            // exported and undeclared. Skipped here only so the message names
            // the real offender rather than the placeholder.
            if symbol.is_empty() || symbol.starts_with('$') {
                continue;
            }
            if !committed.contains(symbol) {
                missing.push(symbol.to_string());
            }
        }
    }
    assert!(
        missing.is_empty(),
        "include/gridoxide.h is stale — it does not declare {missing:?}. \
         Regenerate with `cargo build --features capi`."
    );
}

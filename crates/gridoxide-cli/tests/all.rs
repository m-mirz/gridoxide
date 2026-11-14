//! The `gridox` command line's integration tests, compiled as one binary as at
//! the root (see `tests/all.rs` there). They run the built binary through
//! `CARGO_BIN_EXE_gridox`, which only this package gets, and read the fixtures
//! and helpers (`tests/pglib`, `tests/openrao`) of the repository root.

/// The repository root, where `tests/data` is.
const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

#[path = "cli_continuation_test.rs"]
mod cli_continuation_test;
#[cfg(feature = "dynamics")]
#[path = "cli_dynamics_test.rs"]
mod cli_dynamics_test;
#[path = "cli_estimate_test.rs"]
mod cli_estimate_test;
#[cfg(feature = "opf")]
#[path = "cli_opf_test.rs"]
mod cli_opf_test;
#[path = "cli_qv_test.rs"]
mod cli_qv_test;
#[cfg(all(feature = "rao", feature = "ucte", feature = "iidm"))]
#[path = "cli_rao_test.rs"]
mod cli_rao_test;
#[cfg(all(feature = "rao", feature = "ucte", feature = "iidm"))]
#[path = "cli_security_test.rs"]
mod cli_security_test;
#[path = "cli_sensitivity_test.rs"]
mod cli_sensitivity_test;
#[path = "cli_short_circuit_test.rs"]
mod cli_short_circuit_test;
#[cfg(all(feature = "cgmes", feature = "ucte"))]
#[path = "cli_solve_test.rs"]
mod cli_solve_test;
#[cfg(feature = "cgmes")]
#[path = "cli_switches_test.rs"]
mod cli_switches_test;

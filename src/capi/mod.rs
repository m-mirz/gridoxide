//! A C ABI, so a C++ project can embed gridoxide.
//!
//! The **mirror image** of the rest of this crate's FFI. `sparse_klu`,
//! `sparse_pardiso`, `opf::highs` and `opf::ipopt` all consume a C header to
//! reach a library; this module produces one so a library can reach us.
//! `include/gridoxide.h` is generated from what is here.
//!
//! [`python`](crate::python) is the other binding, and its curation decisions
//! are followed rather than re-litigated — in particular **an opaque handle
//! exists only where a factorization survives between calls**, everything else
//! being a plain function. That is why there is one handle type here and not
//! several.
//!
//! # Three things a C boundary needs that PyO3 handled for us
//!
//! **1. Panics must not cross it.** A `panic!` unwinding out of an
//! `extern "C"` function aborts the process — it would take the host C++
//! program down with it, from inside a library it merely linked. That is not
//! hypothetical: [`newton_raphson_distributing_slack`] asserts on a
//! wrong-length participation vector, `YBus::finish` expects its triplets to
//! be well formed, and the PGM conversion unwraps in places. Every exported
//! function therefore runs its body inside [`catch_unwind`], turning an abort
//! into [`GRIDOXIDE_ERR_PANIC`](Status::Panic).
//!
//! **2. A third answer, beside success and failure.** The Rust API uses
//! `Option::None` to mean *"no such answer exists"* — a radial branch has no
//! outage distribution factors, an island with no source has no reference bus.
//! Neither is an error. Collapsing them into one would force a caller to parse
//! message strings to tell "your input was wrong" from "the question does not
//! apply here", so [`Status::NoAnswer`] is its own code.
//!
//! **3. Thread affinity, stated rather than enforced.** A handle owns a
//! [`PersistentSolver`](crate::solver::PersistentSolver), and under the `klu`
//! or `pardiso` features that holds raw pointers into SuiteSparse or MKL state
//! that is not safe for concurrent use. PyO3 encodes this as
//! `#[pyclass(unsendable)]`; C has no equivalent, so the header says it and
//! this module repeats it: **one handle, one thread**. Separate handles on
//! separate threads are fine.
//!
//! # Memory
//!
//! **Nothing allocated here is ever freed by the caller, and vice versa.**
//! Results are written into caller-owned buffers whose length the caller
//! learns from a `_count` function first. That removes the whole category of
//! allocator-mismatch bugs that `malloc`-in-Rust/`free`-in-C++ invites, and it
//! costs nothing: every result this API returns is a flat array indexed by bus
//! or by branch, which is exactly the shape a caller wants anyway.
//!
//! The one exception is the handle itself, which is opaque and freed by
//! [`gridoxide_powerflow_free`](powerflow::gridoxide_powerflow_free).

pub mod powerflow;
pub mod types;

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The ABI this build speaks.
///
/// Bumped on any change that alters the meaning or layout of anything crossing
/// the boundary. A prebuilt-artifact consumer — one who did not compile the
/// header and the library together — can call this and refuse to continue if
/// it disagrees with `GRIDOXIDE_ABI_VERSION` in the header it compiled
/// against, which turns a silent struct-layout mismatch into an early, legible
/// failure.
pub const ABI_VERSION: u32 = 1;

/// How a call ended.
///
/// The split between [`InvalidArgument`](Self::InvalidArgument) and the
/// numerical codes mirrors the `PyValueError`/`PyRuntimeError` distinction
/// [`python`](crate::python) already draws: the first says *you* passed
/// something wrong, the rest say the computation or its environment did not
/// work out.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The call succeeded.
    Ok = 0,
    /// A caller-supplied argument was wrong: a null pointer, an out-of-range
    /// index, a buffer of the wrong length, an unknown enum value.
    InvalidArgument = 1,
    /// A file could not be read.
    Io = 2,
    /// A document could not be parsed.
    Parse = 3,
    /// The solve ran out of iterations. **The state is still readable** — the
    /// voltages are whatever the last iteration produced, and the per-island
    /// detail says which components failed to settle.
    NotConverged = 4,
    /// A matrix was singular. As above, the state is readable but meaningless.
    Singular = 5,
    /// The question does not apply here — see the module docs. Not a failure.
    NoAnswer = 6,
    /// A panic was caught at the boundary. Always a bug in gridoxide; the
    /// message from [`gridoxide_last_error_message`] should go in the report.
    Panic = 7,
    /// Anything else.
    Internal = 8,
}

thread_local! {
    /// The last error message on *this* thread.
    ///
    /// Per-thread rather than global so two threads driving two handles cannot
    /// overwrite each other's diagnosis — which would be at its worst exactly
    /// when it matters, with several solves in flight.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub(crate) fn set_error(message: impl Into<Vec<u8>>) {
    // A message containing an interior NUL cannot become a C string; say so
    // rather than dropping the diagnosis entirely.
    let text = CString::new(message)
        .unwrap_or_else(|_| CString::new("error message contained a NUL byte").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(text));
}

pub(crate) fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// The last error message on this thread, or `NULL` if the last call succeeded.
///
/// # Safety
///
/// The returned pointer is owned by gridoxide and stays valid **only until the
/// next gridoxide call on this thread**. Copy it before calling anything else.
/// Never free it.
#[unsafe(no_mangle)]
pub extern "C" fn gridoxide_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| match &*slot.borrow() {
        Some(text) => text.as_ptr(),
        None => std::ptr::null(),
    })
}

/// The ABI version this library was built with — see [`ABI_VERSION`].
#[unsafe(no_mangle)]
pub extern "C" fn gridoxide_abi_version() -> u32 {
    ABI_VERSION
}

/// The crate version, as a NUL-terminated string. Static; never freed.
#[unsafe(no_mangle)]
pub extern "C" fn gridoxide_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Runs `body`, converting a panic into [`Status::Panic`] instead of letting
/// it unwind out of an `extern "C"` function and abort the process.
///
/// `AssertUnwindSafe` is sound here for a specific reason rather than by
/// convenience: a panic leaves the handle's cached solver and bus state
/// possibly half-updated, but every entry point re-clones its pristine
/// `buses_template` before solving, so the next call starts from a known state
/// regardless. Nothing observes the torn intermediate.
pub(crate) fn guard(body: impl FnOnce() -> Status) -> Status {
    clear_error();
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(status) => status,
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panicked with a non-string payload".to_string());
            set_error(format!(
                "gridoxide panicked internally: {what}. This is a bug — please report it \
                 with the network that triggered it."
            ));
            Status::Panic
        }
    }
}

/// Reports a caller-fault with a message, in one expression.
pub(crate) fn invalid(message: impl Into<Vec<u8>>) -> Status {
    set_error(message);
    Status::InvalidArgument
}

/// Copies `values` into a caller-owned buffer, refusing a length mismatch.
///
/// The length is checked rather than trusted. A caller who allocated for the
/// bus count and passed it to a per-branch accessor would otherwise get a
/// buffer overrun — silent, memory-corrupting, and attributed to the wrong
/// library.
///
/// # Safety
///
/// `out` must point to `len` writable `f64`s.
pub(crate) unsafe fn write_slice(values: &[f64], out: *mut f64, len: usize, what: &str) -> Status {
    if out.is_null() {
        return invalid(format!("{what}: output pointer is null"));
    }
    if len != values.len() {
        return invalid(format!(
            "{what}: buffer holds {len} entries, but this network needs {}",
            values.len()
        ));
    }
    // SAFETY: the caller guarantees `len` writable f64s, and `len` was just
    // checked to equal the source length.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), out, len) };
    Status::Ok
}

/// How one connected component of the network turned out.
///
/// The last two are **not failures**. A real network routinely contains a
/// de-energized pocket, and a solve containing one still returns
/// [`Status::Ok`] — so this is the only place such a pocket becomes visible.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridoxideIslandStatus {
    /// This component's own mismatch is below tolerance.
    Converged = 0,
    /// The solve ran out of iterations with this component still unsettled.
    MaxIterationsReached = 1,
    /// The factorization was singular. Attribution is best-effort: one
    /// combined matrix is factorized, so every unconverged component is marked
    /// this way whether or not it was the cause.
    Singular = 2,
    /// No source in this component, so nothing pins its voltage. Reported, not
    /// solved.
    NoReferenceBus = 3,
    /// More than one source, so the reference is ambiguous.
    AmbiguousReferenceBus = 4,
}

/// Maps an [`IslandStatus`](crate::solver::IslandStatus) onto the ABI's own
/// enum.
///
/// Written out rather than derived from the Rust enum's discriminant, so that
/// reordering that enum — a change with no other consequence — cannot silently
/// renumber a value a compiled C++ binary is comparing against.
pub(crate) fn island_status_code(status: crate::solver::IslandStatus) -> GridoxideIslandStatus {
    use crate::solver::IslandStatus::*;
    match status {
        Converged => GridoxideIslandStatus::Converged,
        MaxIterationsReached => GridoxideIslandStatus::MaxIterationsReached,
        Singular => GridoxideIslandStatus::Singular,
        NoReferenceBus => GridoxideIslandStatus::NoReferenceBus,
        AmbiguousReferenceBus => GridoxideIslandStatus::AmbiguousReferenceBus,
    }
}

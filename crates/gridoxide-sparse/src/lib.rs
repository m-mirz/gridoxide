//! The sparse linear algebra under gridoxide's solvers.
//!
//! Every Newton loop in gridoxide factors a real sparse system whose pattern is
//! fixed and whose values change every iteration. [`sparse::LinearSolver`] is
//! that contract, and each backend here implements it:
//!
//! - [`sparse::RealSparseSystem`], on `faer` — the default;
//! - [`klu_native::KluNativeSystem`], a pure-Rust port of SuiteSparse KLU;
//! - `sparse_klu::KluRealSystem`, the vendored KLU over FFI (feature `klu`);
//! - `sparse_pardiso::PardisoRealSystem`, oneMKL PARDISO (feature `pardiso`).
//!
//! [`lu::SparseLu`] picks one of them at run time. [`block_sparse`] is the
//! 2×2-block LU power flow's `Block` backend uses, and [`sparse`] also holds
//! the complex and fully-fixed factorizations the linear methods need.

pub mod block_sparse;
pub mod klu_native;
pub mod lu;
pub mod sparse;
#[cfg(feature = "klu")]
pub mod sparse_klu;
#[cfg(feature = "pardiso")]
pub mod sparse_pardiso;

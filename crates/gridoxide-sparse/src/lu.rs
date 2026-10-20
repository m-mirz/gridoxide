//! The sparse LU backend, chosen at run time.
//!
//! Every Newton loop in the crate — power flow, state estimation, continuation,
//! dynamics — factors a real system whose sparsity pattern is fixed and whose
//! values change every iteration, which is the contract
//! [`LinearSolver`] states. Each backend implements that trait; [`SparseLu`]
//! picks one of them from a [`LuBackend`] value, so a caller holds one field
//! and writes no `match` over the backends, and the `klu`/`pardiso` feature
//! gates live here and nowhere else.
//!
//! The cost is one `match` per factorization, against the factorization
//! itself.

use crate::klu_native::KluNativeSystem;
use crate::sparse::{LinearSolver, RealSparseSystem};
#[cfg(feature = "klu")]
use crate::sparse_klu::KluRealSystem;
#[cfg(feature = "pardiso")]
use crate::sparse_pardiso::PardisoRealSystem;

/// Which sparse LU factors the system.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LuBackend {
    /// `faer`'s sparse LU ([`RealSparseSystem`]). The default.
    #[default]
    Faer,
    /// The pure-Rust port of KLU ([`KluNativeSystem`]). Always compiled.
    KluNative,
    /// The vendored SuiteSparse KLU over FFI. Needs the `klu` feature.
    #[cfg(feature = "klu")]
    Klu,
    /// Intel oneMKL's PARDISO over FFI. Needs the `pardiso` feature.
    #[cfg(feature = "pardiso")]
    Pardiso,
}

enum System {
    Faer(RealSparseSystem),
    KluNative(KluNativeSystem),
    #[cfg(feature = "klu")]
    Klu(KluRealSystem),
    #[cfg(feature = "pardiso")]
    Pardiso(PardisoRealSystem),
}

/// A sparse LU that analyzes a pattern once and refactors it on every solve,
/// with its backend picked by a [`LuBackend`].
///
/// Starts empty. [`analyze`](Self::analyze) builds the symbolic factorization;
/// the solves reuse it until [`reset`](Self::reset). The same
/// precondition as [`LinearSolver`] holds: every solve must present the
/// pattern that was analyzed, in the same order.
pub struct SparseLu {
    backend: LuBackend,
    system: Option<System>,
}

impl SparseLu {
    pub fn new(backend: LuBackend) -> Self {
        Self {
            backend,
            system: None,
        }
    }

    pub fn backend(&self) -> LuBackend {
        self.backend
    }

    /// Whether a symbolic factorization is cached.
    pub fn is_analyzed(&self) -> bool {
        self.system.is_some()
    }

    /// Analyzes `entries`' pattern and factors it, replacing anything cached.
    /// `false` if the pattern is structurally invalid or the matrix singular,
    /// in which case nothing is cached.
    pub fn analyze(&mut self, n: usize, entries: &[(usize, usize, f64)]) -> bool {
        self.system = match self.backend {
            LuBackend::Faer => RealSparseSystem::new(n, entries).map(System::Faer),
            LuBackend::KluNative => KluNativeSystem::new(n, entries).map(System::KluNative),
            #[cfg(feature = "klu")]
            LuBackend::Klu => KluRealSystem::new(n, entries).map(System::Klu),
            #[cfg(feature = "pardiso")]
            LuBackend::Pardiso => PardisoRealSystem::new(n, entries).map(System::Pardiso),
        };
        self.system.is_some()
    }

    /// Discards the cached factorization, so the next solve needs a fresh
    /// [`analyze`](Self::analyze).
    pub fn reset(&mut self) {
        self.system = None;
    }

    /// See [`LinearSolver::factor_and_solve`]. `None` if nothing is analyzed.
    pub fn factor_and_solve(
        &mut self,
        entries: &[(usize, usize, f64)],
        rhs: &[f64],
    ) -> Option<Vec<f64>> {
        match self.system.as_mut()? {
            System::Faer(s) => s.factor_and_solve(entries, rhs),
            System::KluNative(s) => s.factor_and_solve(entries, rhs),
            #[cfg(feature = "klu")]
            System::Klu(s) => s.factor_and_solve(entries, rhs),
            #[cfg(feature = "pardiso")]
            System::Pardiso(s) => s.factor_and_solve(entries, rhs),
        }
    }

    /// See [`LinearSolver::factor_and_solve_values`]. `None` if nothing is
    /// analyzed.
    pub fn factor_and_solve_values(&mut self, values: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
        match self.system.as_mut()? {
            System::Faer(s) => s.factor_and_solve_values(values, rhs),
            System::KluNative(s) => s.factor_and_solve_values(values, rhs),
            #[cfg(feature = "klu")]
            System::Klu(s) => s.factor_and_solve_values(values, rhs),
            #[cfg(feature = "pardiso")]
            System::Pardiso(s) => s.factor_and_solve_values(values, rhs),
        }
    }
}

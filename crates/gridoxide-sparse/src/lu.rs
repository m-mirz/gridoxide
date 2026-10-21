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
    /// What [`adopt_storage_order`](Self::adopt_storage_order) returned, for
    /// a caller that rebuilds its values' source while keeping this
    /// factorization.
    order: Option<Vec<usize>>,
}

impl SparseLu {
    pub fn new(backend: LuBackend) -> Self {
        Self {
            backend,
            system: None,
            order: None,
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
        self.order = None;
        self.system.is_some()
    }

    /// Asks the analyzed backend to take values in its own storage order
    /// from now on, so a caller that can write them that way spares it a
    /// copy per solve. Returns `order`, where `order[k]` is the index, in the
    /// entries [`analyze`](Self::analyze) was given, of the value position
    /// `k` takes; the caller reorders its values to match, once, and every
    /// later [`factor_and_solve_values`](Self::factor_and_solve_values) and
    /// [`factor_and_solve`](Self::factor_and_solve) presents them so.
    ///
    /// `None`, changing nothing, for a backend without one fixed layout
    /// (faer, PARDISO), for duplicate positions, or before `analyze`. The
    /// two KLUs say yes: copying the values into their column-compressed
    /// layout cost a tenth of a warm Newton solve.
    pub fn adopt_storage_order(&mut self) -> Option<Vec<usize>> {
        let order = match self.system.as_mut()? {
            System::KluNative(s) => s.adopt_storage_order(),
            #[cfg(feature = "klu")]
            System::Klu(s) => s.adopt_storage_order(),
            _ => None,
        }?;
        self.order = Some(order.clone());
        Some(order)
    }

    /// The order adopted by [`adopt_storage_order`](Self::adopt_storage_order),
    /// if one is in force.
    pub fn storage_order(&self) -> Option<&[usize]> {
        self.order.as_deref()
    }

    /// Discards the cached factorization, so the next solve needs a fresh
    /// [`analyze`](Self::analyze).
    pub fn reset(&mut self) {
        self.system = None;
        self.order = None;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3×3 system given in an order that is not column-major, so the
    /// adopted order is a real permutation.
    const ENTRIES: [(usize, usize, f64); 7] = [
        (0, 0, 4.0),
        (0, 1, 1.0),
        (1, 1, 5.0),
        (2, 1, 2.0),
        (1, 0, 1.0),
        (2, 2, 3.0),
        (1, 2, 1.0),
    ];
    const RHS: [f64; 3] = [1.0, 2.0, 3.0];

    fn adopted_order_solves_the_same_system(backend: LuBackend) {
        let values: Vec<f64> = ENTRIES.iter().map(|e| e.2).collect();
        let mut lu = SparseLu::new(backend);
        assert!(lu.analyze(3, &ENTRIES));
        let before = lu.factor_and_solve_values(&values, &RHS).unwrap();

        let order = lu.adopt_storage_order().expect("KLU adopts its order");
        assert_ne!(
            order,
            (0..ENTRIES.len()).collect::<Vec<_>>(),
            "not a real permutation"
        );
        assert_eq!(lu.storage_order(), Some(&order[..]));
        let reordered: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        let after = lu.factor_and_solve_values(&reordered, &RHS).unwrap();
        assert_eq!(before, after);

        lu.reset();
        assert_eq!(lu.storage_order(), None);
    }

    #[test]
    fn klu_native_adopts_its_storage_order() {
        adopted_order_solves_the_same_system(LuBackend::KluNative);
    }

    #[cfg(feature = "klu")]
    #[test]
    fn klu_adopts_its_storage_order() {
        adopted_order_solves_the_same_system(LuBackend::Klu);
    }

    #[test]
    fn faer_keeps_the_callers_order() {
        let mut lu = SparseLu::new(LuBackend::Faer);
        assert!(lu.analyze(3, &ENTRIES));
        assert_eq!(lu.adopt_storage_order(), None);
    }

    #[test]
    fn duplicate_positions_keep_the_callers_order() {
        let mut entries = ENTRIES.to_vec();
        entries.push((0, 0, 1.0));
        let mut lu = SparseLu::new(LuBackend::KluNative);
        assert!(lu.analyze(3, &entries));
        assert_eq!(lu.adopt_storage_order(), None);
    }
}

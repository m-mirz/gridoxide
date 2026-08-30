//! The shifted DAE operator: `(A − σI)⁻¹`, without ever forming `A`.
//!
//! [`smallsignal::analyze`](super::smallsignal::analyze) eliminates the
//! algebraic block by brute force — one solve per state column, then a dense
//! `n_x × n_x` matrix and a dense eigen-decomposition. That is right up to a
//! couple of thousand states and hopeless past it: the reduction alone is
//! `O(n_x²·n_net)`, and a four-thousand-bus case has twenty thousand states.
//!
//! A sparse eigensolver does not need `A`. It needs to *apply* something, and
//! for the modes anyone asks about — the poorly damped ones, in a named
//! frequency band — the thing to apply is the shift-invert operator
//! `(A − σI)⁻¹`, whose largest eigenvalues are the eigenvalues of `A` nearest
//! `σ`. This module provides exactly that, matrix-free.
//!
//! # The pencil
//!
//! Write the linearized DAE as a generalized eigenproblem rather than a reduced
//! one:
//!
//! ```text
//!  ⎡ A_x  A_v ⎤ ⎡x⎤       ⎡ I  0 ⎤ ⎡x⎤
//!  ⎣ C_x  C_v ⎦ ⎣v⎦ = λ   ⎣ 0  0 ⎦ ⎣v⎦
//!         J                    E
//! ```
//!
//! Its finite eigenvalues are exactly those of the reduced `A`. The bottom
//! block-row carries no `λ` — `E`'s lower block is zero — so it reads
//! `C_x x + C_v v = 0` whatever `λ` is, giving `v = −C_v⁻¹C_x x`, and the top
//! row then becomes `A x = λ x`. Nothing is approximated by working with the
//! pencil; it is the same problem written without performing the elimination.
//!
//! # The operator
//!
//! For `b` in the state space, define
//!
//! ```text
//!  S(b) = the leading n_x entries of (J − σE)⁻¹ [b; 0]
//! ```
//!
//! By the Schur-complement identity `(K⁻¹)₁₁ = (P − Q T⁻¹ R)⁻¹` applied to
//! `K = J − σE`, this **is** `(A − σI)⁻¹ b`, exactly. The Krylov space lives in
//! `C^{n_x}`; the cost per application is one sparse solve on the bordered
//! `(n_x + 2·n_bus)` system, against a factorization computed once per shift.
//!
//! # Why the shift is free
//!
//! [`DaePattern::fill`](super::dae::DaePattern::fill) at `h·a = 1` writes
//!
//! ```text
//!  M = ⎡ I − A_x   −A_v ⎤
//!      ⎣   C_x      C_v ⎦
//! ```
//!
//! and `J − σE` is that with the top `n_x` rows **negated** and `(1 − σ)` added
//! on the state diagonal: `−(I − A_x) + (1 − σ)I = A_x − σI`, `−(−A_v) = A_v`,
//! and the bottom rows are already what they should be. The state diagonal is
//! structurally present — `DaePattern::analyze` emits a *dense* `∂f/∂x` block
//! per device — so a shift adds no fill-in and changes no pattern. The same
//! property that makes every event in a run value-only makes every shift
//! value-only here.
//!
//! # Arithmetic
//!
//! Complex, because the question is. A real shift can only aim at a point on
//! the real axis; naming a frequency band — *the inter-area modes near 0.5 Hz*
//! — needs `σ = α + jβ`, and targeting a region of the complex plane is the
//! whole reason to use this method rather than the dense one.
//!
//! The price is that the five pluggable
//! [`LinearSolver`](crate::solver::LinearSolver) backends are real-only and do
//! not apply here; `sparse::ComplexSparseSystem` serves, whose
//! factorize-once/solve-many contract is exactly what an Arnoldi iteration
//! wants. That is the same position the Y-bus's own complex solves are already
//! in, and it is recorded rather than worked around.

use num_complex::Complex;

use crate::sparse::ComplexSparseSystem;

use super::smallsignal::fill_at_equilibrium;
use super::DynamicSystem;

/// A factorization of `J − σE` for one shift, and the operator it provides.
///
/// Built once per shift and applied many times — a hundred or more, across an
/// Arnoldi run and its adjoint. Both directions share this one factorization.
pub struct ShiftedDae {
    n_x: usize,
    n: usize,
    sigma: Complex<f64>,
    lu: ComplexSparseSystem,
}

/// Why a shifted operator could not be built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShiftError {
    /// The system has no differential states, so there is nothing to shift.
    NoStates,
    /// `J − σE` is singular at this shift.
    ///
    /// In exact arithmetic that means σ *is* an eigenvalue, which is the one
    /// value a shift may not take. In practice it also catches a network block
    /// with no voltage reference, which is singular at every shift.
    Singular { sigma: Complex<f64> },
}

impl std::fmt::Display for ShiftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShiftError::NoStates => write!(f, "the system has no differential states"),
            ShiftError::Singular { sigma } => write!(
                f,
                "the shifted system is singular at σ = {:+.6}{:+.6}j — either the shift sits \
                 exactly on an eigenvalue, or the network has no voltage reference",
                sigma.re, sigma.im
            ),
        }
    }
}

impl std::error::Error for ShiftError {}

impl ShiftedDae {
    /// Assembles and factorizes `J − σE` at the system's current point.
    ///
    /// The point is taken as given: this does *not* check that it is an
    /// equilibrium, because the operator is well-defined either way and the
    /// caller that cares — small-signal analysis — makes that check itself and
    /// reports it as its own refusal.
    pub fn new(system: &DynamicSystem, sigma: Complex<f64>) -> Result<Self, ShiftError> {
        let layout = system.pattern.layout();
        let n_x = layout.n_diff;
        let n = layout.n();
        if n_x == 0 {
            return Err(ShiftError::NoStates);
        }

        let values = fill_at_equilibrium(system);
        let triplets = system.pattern.to_triplets(&values);

        // The top block-rows negated, the bottom ones as they stand. See the
        // module doc for why that is `J − σE` up to the diagonal term below.
        let mut entries: Vec<(usize, usize, Complex<f64>)> = Vec::with_capacity(triplets.len() + n_x);
        for (r, c, value) in triplets {
            let value = if r < n_x { -value } else { value };
            entries.push((r, c, Complex::new(value, 0.0)));
        }

        // `+ (1 − σ)` on the state diagonal, pushed as its own triplet rather
        // than hunted for in the pattern: duplicate `(row, col)` pairs sum, and
        // every one of these positions is structurally present anyway, so this
        // adds arithmetic and no fill-in.
        let shift = Complex::new(1.0, 0.0) - sigma;
        for r in 0..n_x {
            entries.push((r, r, shift));
        }

        let lu = ComplexSparseSystem::new(n, &entries)
            .ok_or(ShiftError::Singular { sigma })?;
        Ok(Self { n_x, n, sigma, lu })
    }

    /// The shift this operator was built at.
    pub fn sigma(&self) -> Complex<f64> {
        self.sigma
    }

    /// How many differential states — the dimension the operator acts on.
    pub fn n_states(&self) -> usize {
        self.n_x
    }

    /// `(A − σI)⁻¹ b`, exactly, without forming `A`.
    pub fn apply(&self, b: &[Complex<f64>]) -> Option<Vec<Complex<f64>>> {
        let solved = self.lu.solve(&self.bordered(b))?;
        Some(solved[..self.n_x].to_vec())
    }

    /// `((A − σI)⁻¹)ᴴ b` — the same operator adjointed, against the *same*
    /// factorization.
    ///
    /// This is what left eigenvectors need, and having it for the price of a
    /// triangular solve in the other direction is why participation factors and
    /// eigenvalue sensitivities stay affordable at scale.
    ///
    /// It is the adjoint of `S` and not of the whole bordered system: the
    /// conjugate transpose of `K = [[P, Q], [R, T]]` is `[[Pᴴ, Rᴴ], [Qᴴ, Tᴴ]]`,
    /// whose own leading inverse block is `(Pᴴ − Rᴴ(Tᴴ)⁻¹Qᴴ)⁻¹`, i.e.
    /// `((P − QT⁻¹R)ᴴ)⁻¹`. Padding with zeros and reading the leading entries
    /// therefore gives `Sᴴ` for the same reason it gives `S`.
    pub fn apply_adjoint(&self, b: &[Complex<f64>]) -> Option<Vec<Complex<f64>>> {
        let solved = self.lu.solve_adjoint(&self.bordered(b))?;
        Some(solved[..self.n_x].to_vec())
    }

    /// `[b; 0]` — a state-space vector padded out to the bordered system.
    fn bordered(&self, b: &[Complex<f64>]) -> Vec<Complex<f64>> {
        debug_assert_eq!(b.len(), self.n_x);
        let mut rhs = vec![Complex::new(0.0, 0.0); self.n];
        rhs[..self.n_x].copy_from_slice(b);
        rhs
    }
}

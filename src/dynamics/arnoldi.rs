//! Shift-invert Arnoldi: a few eigenvalues of a large operator, near a point.
//!
//! A dense eigen-decomposition computes every mode, at `O(n³)`. Past a few
//! thousand states that is not slow, it is hours — and almost none of what it
//! computes is wanted. What a stability study asks is narrower: *the modes near
//! this frequency*, or *the least damped ones*, six or ten of them.
//!
//! Arnoldi answers that. It builds a Krylov space `span{v, Sv, S²v, …}`,
//! projects the operator onto it, and takes the eigenvalues of the small
//! projected matrix — which converge first to the **largest** eigenvalues of
//! `S`. Applied to [`ShiftedDae`](super::shift::ShiftedDae), whose eigenvalues
//! are `1/(λ − σ)`, "largest" means "nearest σ", and the shift is the aim.
//!
//! Nothing here knows what the operator is. It takes a closure, which is what
//! lets the same code serve the forward pass and the adjoint one that produces
//! left eigenvectors.
//!
//! # Orthogonalization
//!
//! Modified Gram–Schmidt with one reorthogonalization pass. The classical
//! failure of Arnoldi at scale is a basis that stops being orthogonal, which
//! shows up not as an error but as spurious eigenvalues — duplicated copies of
//! genuine ones, indistinguishable from real multiplicity. Reorthogonalizing
//! doubles the inner products of an operation whose cost is dominated by the
//! sparse solve either way, and buys away the whole failure mode.
//!
//! # Restarting, and why it means *growing*
//!
//! With shift-invert the wanted eigenvalues are usually dominant by a wide
//! margin and a single cycle converges. The case where they are not is the one
//! power systems actually produce: a ring of two thousand machines has two
//! thousand electromechanical modes packed into a tenth of a hertz, so the
//! eighth-nearest eigenvalue to the shift and the ninth are almost the same
//! distance away, and no Krylov space of a few dozen vectors can separate them.
//!
//! Measured on exactly that case, restarting from a combination of the wanted
//! Ritz vectors at a fixed dimension barely helps — five restarts at dimension
//! 32 moved the residual from 6e-2 to 7e-3, still useless — while **raising the
//! dimension** fixed it outright: dimension 128 converged to 2e-19 in one
//! cycle, faster in wall time than the five restarts that failed.
//!
//! So a restart here doubles the Krylov dimension and re-aims the start vector
//! at the wanted Ritz vectors, rather than only doing the second. It stops at a
//! cap set by memory — the basis is `m` vectors of length `n` — and reports
//! what it has, with residuals, rather than pretending.
//!
//! Krylov–Schur, which keeps the converged subspace across a restart instead of
//! collapsing it, would reach the same place in fewer applications. It needs a
//! Schur decomposition with eigenvalue reordering, which `faer` does not expose
//! and which is a numerical project of its own; growth costs triangular solves
//! against a factorization already computed, which is the cheap thing here. If
//! a case ever needs the better method, this is the module to change and
//! nothing above it.
//!
//! # What is reported, and in what units
//!
//! A Ritz pair's residual `‖S ŷ − θ ŷ‖` measures the *transformed* problem.
//! What a caller wants to know is the error in `λ`, and since `λ = σ + 1/θ` a
//! perturbation `ε` in `θ` moves `λ` by about `ε/|θ|²`. So the residual is
//! reported scaled that way: in reciprocal seconds, the same units as the
//! eigenvalue it qualifies.
//!
//! It is computed **explicitly**, by applying the operator once more to each
//! Ritz vector that is going to be returned, rather than by the textbook
//! estimate `|h_{m+1,m}|·|eₘᵀy|`. That estimate is free and, on these problems,
//! wrong: measured on the ring, it reported `8e-52` where the true residual was
//! `1.3e-14`, because the last Krylov component underflows once a Ritz vector
//! is essentially contained in the early part of the space. A residual is only
//! worth reporting if it can be trusted, and `count` extra triangular solves
//! against a factorization that already exists — against the hundreds the
//! iteration itself spent — is not a cost worth trading trust for.
//!
//! # A residual is not an error bar
//!
//! Worth stating plainly, because the ring makes it visible. A tiny residual
//! does **not** by itself mean a tiny error in `λ`: the two are related by the
//! eigenvalue's condition, `1/|wᴴu|`, and a densely clustered non-normal
//! spectrum can be badly conditioned. On the 4 096-bus ring the residuals are
//! `3e-14` and the forward and adjoint passes still disagree about the
//! eigenvalues at the `1e-4` level — not a bug in either, but the honest
//! accuracy available for modes that ill-conditioned. What the residual
//! certifies is that the pair solves the problem it claims to; how far the
//! eigenvalue itself could still move is a separate question the condition
//! answers.

use num_complex::Complex;

use faer::Mat;

/// How hard to look, and for how many.
#[derive(Clone, Copy, Debug)]
pub struct ArnoldiOptions {
    /// How many eigenvalues are wanted, nearest the shift.
    pub count: usize,
    /// Krylov dimension of the *first* cycle. Zero picks
    /// `max(2·count + 20, 30)`, clamped to the problem size. Each restart
    /// doubles it, up to the cap described in the module doc.
    pub max_dim: usize,
    /// Relative convergence tolerance on a Ritz pair's residual.
    pub tol: f64,
    /// How many times to grow and retry before reporting what converged.
    ///
    /// Small on purpose: every restart doubles the dimension, so this is a
    /// count of doublings and the cap bites long before a large value would.
    pub max_restarts: usize,
}

impl Default for ArnoldiOptions {
    fn default() -> Self {
        Self { count: 6, max_dim: 0, tol: 1e-10, max_restarts: 8 }
    }
}

/// How much memory the Krylov basis may occupy, in bytes.
///
/// The basis is `m` vectors of `n` complex numbers, so this is what bounds the
/// dimension on a large system. Generous — a twenty-thousand-state case reaches
/// dimension 780 inside it — but not unbounded, since the alternative to a cap
/// is a machine swapping instead of an answer.
const BASIS_BUDGET_BYTES: usize = 256 << 20;

/// An absolute ceiling on the Krylov dimension, whatever the memory allows.
///
/// The projected eigenproblem is dense and `O(m³)`; past about a thousand it
/// stops being the small problem the method assumes.
const MAX_KRYLOV_DIM: usize = 1024;

/// One eigenvalue of the operator, and the vector that goes with it.
#[derive(Clone, Debug)]
pub struct RitzPair {
    /// The operator's eigenvalue — `1/(λ − σ)` for a shift-invert operator, not
    /// `λ` itself. Undoing the transform is the caller's job, because only the
    /// caller knows the shift.
    pub theta: Complex<f64>,
    /// The Ritz vector, normalized to unit length.
    pub vector: Vec<Complex<f64>>,
    /// `‖S ŷ − θ ŷ‖ / |θ|²` — the residual expressed in the units of the
    /// *untransformed* eigenvalue, computed by applying the operator rather
    /// than estimated. See the module doc, including what it does and does not
    /// certify.
    pub residual: f64,
    /// Whether this pair met the tolerance. An unconverged pair is still
    /// returned, with its residual, rather than silently dropped or silently
    /// reported as an answer.
    pub converged: bool,
}

/// What a run found.
#[derive(Clone, Debug)]
pub struct ArnoldiResult {
    /// The wanted pairs, largest `|θ|` first — nearest the shift first.
    pub pairs: Vec<RitzPair>,
    /// Restarts actually taken. Zero means the first cycle converged.
    pub restarts: usize,
    /// The Krylov dimension the last cycle reached. Below `max_dim` means the
    /// iteration broke down happily — it found an invariant subspace.
    pub dim: usize,
}

impl ArnoldiResult {
    /// Whether every wanted pair met the tolerance.
    pub fn converged(&self) -> bool {
        self.pairs.iter().all(|p| p.converged)
    }
}

/// Why a run could not produce anything.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArnoldiError {
    /// The operator refused a vector — for a shifted DAE, a singular
    /// factorization.
    OperatorFailed,
    /// The projected eigenproblem did not decompose.
    Eigen,
    /// The operator has dimension zero, or none was asked for.
    Empty,
}

impl std::fmt::Display for ArnoldiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArnoldiError::OperatorFailed => {
                write!(f, "the operator could not be applied — a singular factorization")
            }
            ArnoldiError::Eigen => write!(f, "the projected eigenproblem did not decompose"),
            ArnoldiError::Empty => write!(f, "nothing to iterate on"),
        }
    }
}

impl std::error::Error for ArnoldiError {}

/// `⟨a, b⟩ = Σ conj(aᵢ)·bᵢ` — conjugate-linear in the first argument, which is
/// the convention that makes `⟨v, v⟩` the squared norm.
fn dot(a: &[Complex<f64>], b: &[Complex<f64>]) -> Complex<f64> {
    a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
}

fn norm(a: &[Complex<f64>]) -> f64 {
    a.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

/// A deterministic start vector.
///
/// Pseudo-random rather than `[1, 0, 0, …]` or all-ones, because a structured
/// start vector can be exactly orthogonal to the modes being looked for — a
/// symmetric system's antisymmetric modes are invisible to a symmetric start,
/// and a ring of identical machines is precisely that case. Deterministic
/// rather than random, because a gate that passes on some runs is not a gate.
fn start_vector(n: usize) -> Vec<Complex<f64>> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut v: Vec<Complex<f64>> = (0..n).map(|_| Complex::new(next(), next())).collect();
    let scale = norm(&v);
    for value in &mut v {
        *value /= scale;
    }
    v
}

/// Runs shift-invert Arnoldi on `apply`, returning the eigenvalues of the
/// operator with the largest magnitude.
///
/// `apply` returns `None` if the operator cannot be applied, which is reported
/// rather than treated as a zero vector.
pub fn eigenpairs<F>(
    n: usize,
    apply: F,
    opts: &ArnoldiOptions,
) -> Result<ArnoldiResult, ArnoldiError>
where
    F: Fn(&[Complex<f64>]) -> Option<Vec<Complex<f64>>>,
{
    if n == 0 || opts.count == 0 {
        return Err(ArnoldiError::Empty);
    }
    let count = opts.count.min(n);
    let requested = if opts.max_dim == 0 { (2 * count + 20).max(30) } else { opts.max_dim };
    let cap = dimension_cap(n, count);
    let mut m = requested.max(count + 1).min(cap);

    let mut v0 = start_vector(n);
    let mut restarts = 0usize;
    let mut best: Option<(Vec<RitzPair>, usize)> = None;

    loop {
        let (pairs, dim) = cycle(n, &apply, &v0, m, count, opts.tol)?;
        let converged = pairs.iter().all(|p| p.converged);
        let improved = match &best {
            None => true,
            Some((previous, _)) => worst_residual(&pairs) < worst_residual(previous),
        };
        if improved {
            best = Some((pairs.clone(), dim));
        }

        // A short cycle means a happy breakdown: the Krylov space closed on an
        // invariant subspace, and neither growing it nor restarting can add to
        // it. Reaching the cap without converging is the other terminal case —
        // one attempt at the largest affordable dimension is what there is, and
        // a second at the same dimension would only re-aim a start vector,
        // which the module doc records as barely worth the solves.
        if converged || dim < m || m >= cap || restarts >= opts.max_restarts {
            let (pairs, dim) = best.expect("at least one cycle ran");
            return Ok(ArnoldiResult { pairs, restarts, dim });
        }

        // Grow, and re-aim. The growth is what actually converges a clustered
        // spectrum; the re-aiming is nearly free and does no harm.
        m = (2 * m).min(cap);
        let mut next = vec![Complex::new(0.0, 0.0); n];
        for pair in &pairs {
            // Weighted towards the pairs that have not converged: repeating
            // work on one already at tolerance buys nothing.
            let weight = if pair.converged { 0.1 } else { 1.0 };
            for (acc, value) in next.iter_mut().zip(&pair.vector) {
                *acc += weight * value;
            }
        }
        let scale = norm(&next);
        if scale > f64::EPSILON {
            for value in &mut next {
                *value /= scale;
            }
            v0 = next;
        }
        restarts += 1;
    }
}

/// The largest Krylov dimension this problem may use.
///
/// Bounded by the basis's memory and by the cost of the dense projected
/// eigenproblem. It clamps an explicit `max_dim` too: a request that cannot be
/// afforded is better answered at the affordable dimension, with the residuals
/// saying how well, than by exhausting memory.
fn dimension_cap(n: usize, count: usize) -> usize {
    let per_vector = n * std::mem::size_of::<Complex<f64>>();
    let by_memory = (BASIS_BUDGET_BYTES / per_vector.max(1)).max(count + 1);
    by_memory.min(MAX_KRYLOV_DIM).min(n)
}

fn worst_residual(pairs: &[RitzPair]) -> f64 {
    pairs.iter().fold(0.0f64, |m, p| m.max(p.residual))
}

/// One Arnoldi cycle from a given start vector.
///
/// Returns the wanted Ritz pairs and the dimension actually reached.
fn cycle<F>(
    n: usize,
    apply: &F,
    v0: &[Complex<f64>],
    m_max: usize,
    count: usize,
    tol: f64,
) -> Result<(Vec<RitzPair>, usize), ArnoldiError>
where
    F: Fn(&[Complex<f64>]) -> Option<Vec<Complex<f64>>>,
{
    let zero = Complex::new(0.0, 0.0);
    let mut basis: Vec<Vec<Complex<f64>>> = Vec::with_capacity(m_max + 1);
    basis.push(v0.to_vec());

    // `h[j]` is column `j` of the Hessenberg matrix, holding `j + 2` entries.
    let mut h: Vec<Vec<Complex<f64>>> = Vec::with_capacity(m_max);
    let mut m = m_max;

    for j in 0..m_max {
        let mut w = apply(&basis[j]).ok_or(ArnoldiError::OperatorFailed)?;
        let w_norm_before = norm(&w);
        let mut column = vec![zero; j + 2];

        // Modified Gram-Schmidt, then one correction pass. See the module doc
        // for why the second pass is not optional.
        for pass in 0..2 {
            for (i, vi) in basis.iter().enumerate().take(j + 1) {
                let coeff = dot(vi, &w);
                for (wk, vk) in w.iter_mut().zip(vi) {
                    *wk -= coeff * vk;
                }
                if pass == 0 {
                    column[i] = coeff;
                } else {
                    column[i] += coeff;
                }
            }
        }

        let beta = norm(&w);
        column[j + 1] = Complex::new(beta, 0.0);
        h.push(column);

        // A happy breakdown: the Krylov space is invariant, so the Ritz values
        // computed from it are exact eigenvalues and there is nothing further
        // to build.
        if beta <= 1e-14 * w_norm_before.max(1.0) {
            m = j + 1;
            break;
        }
        for value in &mut w {
            *value /= beta;
        }
        basis.push(w);
    }

    // The projected operator, `m × m`.
    let hm = Mat::<Complex<f64>>::from_fn(m, m, |i, j| {
        if i < h[j].len() { h[j][i] } else { zero }
    });
    let eigen = hm.eigen().map_err(|_| ArnoldiError::Eigen)?;
    let s = eigen.S();
    let y = eigen.U();

    // Rank by |θ| — largest first is nearest the shift — and keep the wanted
    // ones. Only those get a Ritz vector built and a residual measured, which
    // is what keeps the explicit residual affordable.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        let (fa, fb) = (s[b].norm(), s[a].norm());
        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(count);

    let mut pairs: Vec<RitzPair> = Vec::with_capacity(order.len());
    for i in order {
        let theta = Complex::new(s[i].re, s[i].im);
        // The Ritz vector, in the state space: ŷ = V·y.
        let mut vector = vec![zero; n];
        for k in 0..m {
            let coeff = y[(k, i)];
            for (acc, value) in vector.iter_mut().zip(&basis[k]) {
                *acc += coeff * value;
            }
        }
        let scale = norm(&vector);
        if scale > 0.0 {
            for value in &mut vector {
                *value /= scale;
            }
        }

        // ‖S ŷ − θ ŷ‖, applied rather than estimated. See the module doc for
        // the measurement that made this worth an extra solve.
        let image = apply(&vector).ok_or(ArnoldiError::OperatorFailed)?;
        let raw = image
            .iter()
            .zip(&vector)
            .map(|(a, b)| (a - theta * b).norm_sqr())
            .sum::<f64>()
            .sqrt();

        let magnitude = theta.norm();
        let converged = magnitude > 0.0 && raw <= tol * magnitude;
        let residual = if magnitude > 0.0 { raw / (magnitude * magnitude) } else { f64::INFINITY };
        pairs.push(RitzPair { theta, vector, residual, converged });
    }

    Ok((pairs, m))
}

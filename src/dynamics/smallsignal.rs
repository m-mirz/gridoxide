//! Small-signal analysis: the modes of the linearized system.
//!
//! A time-domain run answers "what happens after *this* disturbance". This
//! answers a different question, and one no single run can: **what is this
//! system's dynamic character at all?** Which oscillations exist, how fast they
//! decay, and — the part that makes it actionable — which machines are taking
//! part in each.
//!
//! It is also the only way to find a *negatively* damped mode reliably. A
//! time-domain run finds one if the disturbance happens to excite it; an
//! eigenvalue finds it whether anything excited it or not.
//!
//! # The reduction
//!
//! Linearize the DAE about an equilibrium:
//!
//! ```text
//! Δẋ = A_x·Δx + A_v·ΔV
//! 0   = C_x·Δx + C_v·ΔV
//! ```
//!
//! The algebraic half has no derivative, so it is a constraint on `ΔV` rather
//! than a state of its own — which means it can be eliminated outright:
//!
//! \\[ \Delta V = -C_v^{-1} C_x \Delta x, \qquad
//!    A = A_x - A_v C_v^{-1} C_x \\]
//!
//! That reduced `A` is the whole dynamic system, of dimension the differential
//! states alone. The network contributes to every mode without contributing a
//! single one of its own.
//!
//! # Where the four blocks come from
//!
//! **Not from new code.** They are the very blocks
//! [`DaePattern::fill`](super::dae::DaePattern::fill) already assembles for
//! every Newton iteration of every step, read out at `h·a = 1`:
//!
//! ```text
//! ⎡ I − A_x   −A_v ⎤     so   A_x = I − (top-left),   A_v = −(top-right)
//! ⎣   C_x      C_v ⎦          C_x, C_v as they stand
//! ```
//!
//! Reusing them is what makes this analysis cheap, and — more importantly —
//! what makes it impossible for the linearization to drift away from the
//! simulation. Two hand-written derivations of the same Jacobian would be two
//! things to keep in step; there is only one, and it is the one already checked
//! against a finite-difference oracle for every model.
//!
//! # Participation factors
//!
//! An eigenvalue says a mode exists. A participation factor says *whose* it is:
//!
//! \\[ p_{ki} = \frac{|u_{ki}\,w_{ik}|}{\sum_j |u_{ji}\,w_{ij}|} \\]
//!
//! with `u` the right eigenvectors and `w = u^{-1}` the left ones. The product
//! is dimensionless and scale-free, which is what lets a rotor angle in radians
//! and a field voltage in per unit be compared at all.
//!
//! In practice this is the output that gets used: a poorly damped mode at
//! 0.8 Hz with 90% participation from two machines' speeds is an inter-area
//! oscillation between them, and it names the machines to tune.
//!
//! # Scope
//!
//! Dense. The reduced `A` is `n_states × n_states` and is eigen-decomposed
//! whole, which is `O(n³)` and the normal approach for this analysis; a system
//! of a few thousand states wants a sparse Arnoldi method targeting a region of
//! the complex plane instead, and that is not implemented.

use faer::linalg::solvers::Solve;
use faer::Mat;
use num_complex::Complex;

use crate::solver::LinearSolver;
use crate::sparse::RealSparseSystem;

use super::DynamicSystem;

/// One mode of the linearized system.
#[derive(Clone, Debug, PartialEq)]
pub struct Mode {
    /// `σ ± jω`, in reciprocal seconds.
    pub eigenvalue: Complex<f64>,
    /// Damping ratio `ζ = −σ/|λ|`. **Negative means growing** — the number to
    /// look at first. Below about 0.03 is conventionally "poorly damped".
    pub damping: f64,
    /// Oscillation frequency in Hz. Zero for a real mode, which decays without
    /// oscillating.
    pub frequency: f64,
    /// How long the amplitude takes to fall by `1/e`, in seconds. Infinite for
    /// an undamped mode, negative for a growing one.
    pub time_constant: f64,
    /// Which states take part, largest first, as `(state index, factor)`.
    /// Factors are normalized to sum to one across the mode.
    pub participation: Vec<(usize, f64)>,
    /// The **mode shape** at the rotor angles: `(state index, component)`, with
    /// the components rotated and scaled so the largest is `1∠0`.
    ///
    /// Participation says *whose* a mode is; the shape says *how they move*.
    /// Two machines with components at nearly opposite phase are swinging
    /// against each other, which is what distinguishes an inter-area mode from
    /// a local one — and a local mode from a machine simply riding along.
    ///
    /// Empty for a non-oscillatory mode, where a relative phase means nothing.
    pub shape: Vec<(usize, Complex<f64>)>,
}

impl Mode {
    /// Whether this mode grows rather than decays.
    pub fn is_unstable(&self) -> bool {
        self.eigenvalue.re > 0.0
    }

    /// Whether this mode oscillates at all.
    pub fn is_oscillatory(&self) -> bool {
        self.frequency > 0.0
    }
}

/// The modes of a system, and the names to read their participations against.
#[derive(Clone, Debug)]
pub struct SmallSignal {
    /// Sorted by damping ratio, least damped first — which is the order the
    /// question is usually asked in.
    pub modes: Vec<Mode>,
    /// One name per differential state, parallel to the participation indices.
    pub state_names: Vec<String>,
}

impl SmallSignal {
    /// The least-damped mode, which is the one a stability study is about.
    pub fn critical(&self) -> Option<&Mode> {
        self.modes.first()
    }

    /// Modes that grow rather than decay.
    pub fn unstable(&self) -> Vec<&Mode> {
        self.modes.iter().filter(|m| m.is_unstable()).collect()
    }

    /// The named states taking part in a mode, largest first.
    pub fn participants(&self, mode: &Mode) -> Vec<(String, f64)> {
        mode.participation
            .iter()
            .map(|&(k, p)| (self.state_names[k].clone(), p))
            .collect()
    }

    /// The named mode shape: each rotor's magnitude and phase in degrees.
    pub fn shape(&self, mode: &Mode) -> Vec<(String, f64, f64)> {
        mode.shape
            .iter()
            .map(|&(k, c)| {
                (self.state_names[k].clone(), c.norm(), c.arg().to_degrees())
            })
            .collect()
    }
}

/// Why a system could not be analyzed.
#[derive(Clone, Debug, PartialEq)]
pub enum SmallSignalError {
    /// The system has no differential states, so it has no modes.
    NoStates,
    /// The current point is not an equilibrium.
    ///
    /// Refused rather than analyzed: a linearization about a point the system
    /// is not sitting at describes the dynamics of nothing in particular, and
    /// the resulting eigenvalues would look entirely plausible.
    NotAnEquilibrium { max_derivative: f64 },
    /// The algebraic block is singular, so `ΔV` cannot be eliminated. An island
    /// with no voltage reference does this.
    SingularNetwork,
    /// The eigen-decomposition did not converge.
    Eigen,
    /// Too many states for the dense method.
    ///
    /// Refused rather than attempted: the reduction alone is `O(n²·n_net)` and
    /// the decomposition `O(n³)`, so a system past this size does not run
    /// slowly, it runs for hours. A sparse Arnoldi method targeting a region of
    /// the complex plane is what such a system wants, and is not implemented.
    TooLarge { states: usize, limit: usize },
}

impl std::fmt::Display for SmallSignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SmallSignalError::NoStates => write!(f, "the system has no differential states"),
            SmallSignalError::NotAnEquilibrium { max_derivative } => write!(
                f,
                "not at an equilibrium (largest state derivative {max_derivative:.3e}); \
                 a linearization about a point the system is not sitting at describes nothing"
            ),
            SmallSignalError::SingularNetwork => {
                write!(f, "the algebraic block is singular, so the network cannot be eliminated")
            }
            SmallSignalError::Eigen => write!(f, "the eigen-decomposition did not converge"),
            SmallSignalError::TooLarge { states, limit } => write!(
                f,
                "{states} states is past the {limit} this dense method is honest about; \
                 a system this size wants sparse Arnoldi targeting a region of the complex \
                 plane, which is not implemented"
            ),
        }
    }
}

impl std::error::Error for SmallSignalError {}

/// How far from zero a derivative may be and still count as an equilibrium.
///
/// Generous against what a correct build achieves (`~1e-15`), because the point
/// is to catch a caller analyzing a mid-transient state, not to police
/// arithmetic.
const EQUILIBRIUM_TOL: f64 = 1e-6;

/// The largest system the dense method will attempt.
///
/// Not a hard numerical limit — an honesty one. At two thousand states the
/// decomposition is minutes and the reduction is worse; past that a caller
/// deserves to be told the method is wrong for the problem rather than left
/// waiting.
const MAX_DENSE_STATES: usize = 2000;

/// Linearizes about the system's current state and returns its modes.
pub fn analyze(system: &DynamicSystem) -> Result<SmallSignal, SmallSignalError> {
    let layout = system.pattern.layout().clone();
    let (n_x, n_net) = (layout.n_diff, 2 * layout.n_bus);
    if n_x == 0 {
        return Err(SmallSignalError::NoStates);
    }
    if n_x > MAX_DENSE_STATES {
        return Err(SmallSignalError::TooLarge { states: n_x, limit: MAX_DENSE_STATES });
    }
    let drift = system.max_derivative();
    if drift > EQUILIBRIUM_TOL {
        return Err(SmallSignalError::NotAnEquilibrium { max_derivative: drift });
    }

    // One fill at h·a = 1 gives all four blocks — the same assembly every step
    // of every run uses, so the linearization cannot drift from the simulation.
    let mut scratch: Vec<super::models::ModelJacobian> = (0..layout.n_devices())
        .map(|d| super::models::ModelJacobian::zeros(layout.dev_len[d]))
        .collect();
    let mut values: Vec<f64> = Vec::with_capacity(system.pattern.nnz());
    system.latch(&system.x0, &system.v0);
    system.pattern.fill(
        &system.models,
        &system.x0,
        &system.v0,
        1.0,
        &system.ybus,
        &mut scratch,
        &mut values,
    );
    let triplets = system.pattern.to_triplets(&values);

    let mut a_x = vec![0.0; n_x * n_x];
    let mut a_v = vec![0.0; n_x * n_net];
    let mut c_x = vec![0.0; n_net * n_x];
    let mut c_v: Vec<(usize, usize, f64)> = Vec::new();
    for (r, c, value) in triplets {
        match (r < n_x, c < n_x) {
            // `I − A_x`, so the identity comes back off.
            (true, true) => {
                let identity = if r == c { 1.0 } else { 0.0 };
                a_x[r * n_x + c] = identity - value;
            }
            (true, false) => a_v[r * n_net + (c - n_x)] = -value,
            (false, true) => c_x[(r - n_x) * n_x + c] = value,
            (false, false) => c_v.push((r - n_x, c - n_x, value)),
        }
    }

    // Eliminate ΔV. One factorization of the network block serves every column
    // of C_x, which is what keeps this cheap: the expensive part of a
    // small-signal analysis should be the eigen-decomposition, not the
    // reduction.
    let mut network =
        RealSparseSystem::new(n_net, &c_v).ok_or(SmallSignalError::SingularNetwork)?;
    let mut z = vec![0.0; n_net * n_x];
    let mut column = vec![0.0; n_net];
    for k in 0..n_x {
        for row in 0..n_net {
            column[row] = c_x[row * n_x + k];
        }
        let solved = LinearSolver::factor_and_solve_values(&mut network, &values_of(&c_v), &column)
            .ok_or(SmallSignalError::SingularNetwork)?;
        for row in 0..n_net {
            z[row * n_x + k] = solved[row];
        }
    }

    // A = A_x − A_v·Z.
    let mut a = Mat::<f64>::zeros(n_x, n_x);
    for i in 0..n_x {
        for j in 0..n_x {
            let mut acc = a_x[i * n_x + j];
            for m in 0..n_net {
                acc -= a_v[i * n_net + m] * z[m * n_x + j];
            }
            a[(i, j)] = acc;
        }
    }

    // Which states are rotor angles, for the mode shapes below.
    let angle_states: Vec<usize> = (0..layout.n_devices())
        .filter_map(|d| {
            let offset = layout.dev_offset[d];
            system.models[d].angle_index().map(|k| offset + k)
        })
        .collect();

    let eigen = a.eigen().map_err(|_| SmallSignalError::Eigen)?;
    let u = eigen.U();
    let s = eigen.S();

    // Left eigenvectors are the rows of U⁻¹.
    let mut identity = Mat::<faer::c64>::zeros(n_x, n_x);
    for i in 0..n_x {
        identity[(i, i)] = faer::c64::new(1.0, 0.0);
    }
    let w = u.partial_piv_lu().solve(&identity);

    let mut modes = Vec::with_capacity(n_x);
    for i in 0..n_x {
        let lambda = Complex::new(s[i].re, s[i].im);
        let magnitude = lambda.norm();
        let damping = if magnitude > 0.0 { -lambda.re / magnitude } else { 1.0 };
        let frequency = lambda.im.abs() / std::f64::consts::TAU;
        let time_constant = if lambda.re == 0.0 { f64::INFINITY } else { -1.0 / lambda.re };

        let raw: Vec<f64> =
            (0..n_x).map(|k| (u[(k, i)] * w[(i, k)]).norm()).collect();
        let total: f64 = raw.iter().sum();
        let mut participation: Vec<(usize, f64)> = raw
            .iter()
            .enumerate()
            .map(|(k, p)| (k, if total > 0.0 { p / total } else { 0.0 }))
            .collect();
        participation.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // A long tail of numerically-zero participations tells nobody anything.
        participation.retain(|&(_, p)| p > 1e-6);

        // The mode shape, read off the rotor angles and normalized so the
        // largest component is 1∠0. An eigenvector is defined only up to a
        // complex scale, so only the *relative* magnitudes and phases mean
        // anything — normalizing is what makes two runs comparable.
        let mut shape: Vec<(usize, Complex<f64>)> = Vec::new();
        if lambda.im.abs() > 0.0 {
            let raw: Vec<(usize, Complex<f64>)> = angle_states
                .iter()
                .map(|&k| (k, Complex::new(u[(k, i)].re, u[(k, i)].im)))
                .collect();
            if let Some(&(_, pivot)) = raw
                .iter()
                .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
                .filter(|(_, c)| c.norm() > 0.0)
            {
                shape = raw.into_iter().map(|(k, c)| (k, c / pivot)).collect();
            }
        }

        modes.push(Mode {
            eigenvalue: lambda,
            damping,
            frequency,
            time_constant,
            participation,
            shape,
        });
    }
    // Least damped first: that is the order the question is asked in, and a
    // conjugate pair sorts adjacently because both halves share a damping.
    modes.sort_by(|a, b| a.damping.partial_cmp(&b.damping).unwrap_or(std::cmp::Ordering::Equal));

    Ok(SmallSignal { modes, state_names: system.differential_state_names() })
}

fn values_of(entries: &[(usize, usize, f64)]) -> Vec<f64> {
    entries.iter().map(|&(_, _, v)| v).collect()
}

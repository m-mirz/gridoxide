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

use crate::sparse::RealFactorization;

use super::arnoldi::{self, ArnoldiOptions, RitzPair};
use super::shift::ShiftedDae;

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
    /// How far this eigenvalue may be from the true one, in reciprocal seconds
    /// — the same units as the eigenvalue itself.
    ///
    /// Zero from the dense method, which does not iterate and so has no
    /// residual to report. From the sparse method it is the Ritz residual
    /// scaled into λ's units, and it is the number that says whether an answer
    /// is an answer. See [`arnoldi::RitzPair`].
    pub residual: f64,
    /// The right eigenvector, one entry per differential state, normalized to
    /// unit length.
    ///
    /// [`shape`](Self::shape) is this vector read at the rotor angles and
    /// re-normalized, which is the part worth *printing*. This is the part
    /// worth *using*: displacing the state along its real part excites this
    /// mode and predominantly this mode, which is how a mode is confirmed
    /// against a trajectory, and how a study seeds one deliberately.
    ///
    /// Defined only up to a complex scale, like any eigenvector — two runs
    /// agree on relative magnitudes and phases, not on an overall rotation.
    pub eigenvector: Vec<Complex<f64>>,
    /// The **left** eigenvector `w`, satisfying `wᴴA = λwᴴ`, normalized to unit
    /// length.
    ///
    /// The right eigenvector says how the states move in this mode; the left
    /// one says how strongly each state *excites* it, and the product of the
    /// two is the participation factor above. It is also the half an
    /// eigenvalue sensitivity cannot do without — see [`sensitivities`].
    ///
    /// Empty when the sparse method could not pair this mode with a left
    /// eigenvector unambiguously, which is the same condition that leaves
    /// [`participation`](Self::participation) empty.
    pub left_eigenvector: Vec<Complex<f64>>,
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

/// Which method produced a result.
///
/// Reported rather than inferred, because the two answer different questions:
/// the dense method returns *every* mode, the sparse one returns the `k` modes
/// nearest a shift. A caller reading a mode list needs to know which of those
/// it is holding — "no unstable modes" means something very different in each.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// Every mode, by a dense eigen-decomposition of the reduced `A`.
    Dense,
    /// The modes nearest `shift`, by shift-invert Arnoldi.
    Sparse { shift: Complex<f64>, restarts: usize },
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Method::Dense => write!(f, "dense (every mode)"),
            Method::Sparse { shift, restarts } => write!(
                f,
                "sparse Arnoldi near {:+.4}{:+.4}j, {restarts} restart(s)",
                shift.re, shift.im
            ),
        }
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
    /// Which method answered, and — if the sparse one — where it was aimed.
    pub method: Method,
    /// Whether every mode reported met its tolerance. Always true of the dense
    /// method; from the sparse one, a `false` means at least one eigenvalue
    /// below is an estimate rather than an answer, and its `residual` says how
    /// rough.
    pub converged: bool,
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
    /// Too many states for the *dense* method.
    ///
    /// The reduction alone is `O(n²·n_net)` and the decomposition `O(n³)`, so a
    /// system past this size does not run slowly, it runs for hours. This is
    /// what [`state_matrix`] and the dense half of [`analyze`] refuse with —
    /// but `analyze` itself does not surface it, because at that size it hands
    /// the problem to [`analyze_near`] instead. A caller sees this only by
    /// asking for the dense object specifically.
    TooLarge { states: usize, limit: usize },
    /// The sparse method could not build or apply its shifted operator.
    Shift(super::shift::ShiftError),
    /// The Arnoldi iteration could not produce anything.
    Arnoldi(super::arnoldi::ArnoldiError),
    /// A sensitivity was asked for a mode carrying no left eigenvector.
    ///
    /// Only the sparse method produces such a mode, and only when it could not
    /// pair the mode unambiguously — the same condition that leaves its
    /// participation factors empty.
    NoLeftEigenvector,
    /// The mode is defective, or close enough that its eigenvalue derivative is
    /// genuinely unbounded: `w_xᴴu` is zero, and a sensitivity is not a
    /// meaningful thing to report about it.
    DefectiveMode,
    /// No device has that parameter.
    NoSuchParameter(ParameterRef),
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
                "{states} states is past the {limit} the dense method is honest about; \
                 a system this size wants sparse Arnoldi targeting a region of the complex \
                 plane, which is what `analyze_near` does"
            ),
            SmallSignalError::Shift(e) => write!(f, "{e}"),
            SmallSignalError::Arnoldi(e) => write!(f, "{e}"),
            SmallSignalError::NoLeftEigenvector => write!(
                f,
                "this mode carries no left eigenvector, so it has no sensitivity — the sparse \
                 method could not pair it unambiguously"
            ),
            SmallSignalError::DefectiveMode => write!(
                f,
                "the mode is defective (wᴴu = 0), so its eigenvalue derivative is unbounded \
                 rather than large"
            ),
            SmallSignalError::NoSuchParameter(p) => write!(
                f,
                "device {} has no tunable parameter {:?}",
                p.device, p.name
            ),
        }
    }
}

impl From<super::shift::ShiftError> for SmallSignalError {
    fn from(e: super::shift::ShiftError) -> Self {
        SmallSignalError::Shift(e)
    }
}

impl From<super::arnoldi::ArnoldiError> for SmallSignalError {
    fn from(e: super::arnoldi::ArnoldiError) -> Self {
        SmallSignalError::Arnoldi(e)
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

/// Where to aim the sparse method, and how hard to look.
///
/// The shift is the whole interface. Shift-invert finds the eigenvalues nearest
/// `shift`, so naming one is naming the question — and because the modes that
/// matter are oscillatory, the useful shifts are complex.
/// [`near_frequency`](Self::near_frequency) is the form to reach for: it takes
/// the question as it is actually asked, a frequency and a damping guess.
#[derive(Clone, Copy, Debug)]
pub struct SmallSignalOptions {
    /// The point in the complex plane to search around, in reciprocal seconds.
    pub shift: Complex<f64>,
    /// How many modes to return.
    pub count: usize,
    /// Krylov dimension per cycle; zero picks a default from `count`.
    pub max_dim: usize,
    /// Relative convergence tolerance on a Ritz pair.
    pub tol: f64,
    /// How many restarts before reporting what converged and what did not.
    pub max_restarts: usize,
}

impl Default for SmallSignalOptions {
    /// Aimed at the electromechanical band: 1 Hz, lightly damped.
    ///
    /// Not a neutral choice, and there is no neutral choice to make — a shift
    /// has to be somewhere. This is where a stability study looks: inter-area
    /// oscillations sit between 0.1 and 1 Hz and local modes between 1 and 2,
    /// and both are what a poorly damped mode usually turns out to be.
    fn default() -> Self {
        Self::near_frequency(1.0, 0.05)
    }
}

impl SmallSignalOptions {
    /// A shift aimed at an oscillation of `hz` with damping ratio `zeta`.
    ///
    /// A mode at damped frequency `ω_d` with damping `ζ` sits at
    /// `−ζω_d/√(1−ζ²) ± jω_d`. The guess need not be good: shift-invert finds
    /// what is nearest, so an aim within a factor of two of the right frequency
    /// lands on the right modes.
    pub fn near_frequency(hz: f64, zeta: f64) -> Self {
        let omega_d = std::f64::consts::TAU * hz;
        let zeta = zeta.clamp(0.0, 0.999);
        let sigma = -zeta * omega_d / (1.0 - zeta * zeta).sqrt();
        Self {
            shift: Complex::new(sigma, omega_d),
            count: 6,
            max_dim: 0,
            tol: 1e-10,
            max_restarts: 20,
        }
    }

    /// How many modes to return.
    pub fn count(mut self, count: usize) -> Self {
        self.count = count;
        self
    }
}

/// Assembles the DAE Jacobian at the system's current point, at `h·a = 1`.
///
/// The one fill both methods here read their blocks out of — the dense
/// reduction below, and the shifted operator `super::shift` factorizes. Sharing
/// it is not merely tidy: it is what guarantees the sparse path and the dense
/// path are linearizing the *same* system, which is what phase 1's gate
/// compares.
pub(crate) fn fill_at_equilibrium(system: &DynamicSystem) -> Vec<f64> {
    let layout = system.pattern.layout();
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
    values
}

/// The reduced state matrix `A = A_x − A_v·C_v⁻¹·C_x`, row-major, with its
/// dimension.
///
/// Dense, and so subject to the same size limit [`analyze`] is: the `A_v` and
/// `C_x` blocks alone are `n_x × n_net`, which is what makes this the expensive
/// half at scale and why `analyze_near` never forms it.
pub(crate) fn reduced_state_matrix(
    system: &DynamicSystem,
) -> Result<(usize, Vec<f64>), SmallSignalError> {
    let layout = system.pattern.layout().clone();
    let (n_x, n_net) = (layout.n_diff, 2 * layout.n_bus);
    if n_x == 0 {
        return Err(SmallSignalError::NoStates);
    }
    // The equilibrium check comes first deliberately. Both refusals are
    // honest, but "you are not at an equilibrium" is the actionable one, and a
    // large system analyzed mid-transient has both problems.
    let drift = system.max_derivative();
    if drift > EQUILIBRIUM_TOL {
        return Err(SmallSignalError::NotAnEquilibrium { max_derivative: drift });
    }
    if n_x > MAX_DENSE_STATES {
        return Err(SmallSignalError::TooLarge { states: n_x, limit: MAX_DENSE_STATES });
    }

    // One fill at h·a = 1 gives all four blocks — the same assembly every step
    // of every run uses, so the linearization cannot drift from the simulation.
    let values = fill_at_equilibrium(system);
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
    //
    // `RealFactorization` is the type that delivers that, and the reason the
    // `LinearSolver` backends are not used here: their contract is "pattern
    // fixed, values change every call", so each column would pay its own
    // numeric refactorization of a matrix that never moves. This is the same
    // one-factorization-many-columns shape `linear::sensitivity`'s PTDF and
    // `ac_sensitivity`'s forward mode already have.
    let network =
        RealFactorization::new(n_net, &c_v).ok_or(SmallSignalError::SingularNetwork)?;
    let mut z = vec![0.0; n_net * n_x];
    let mut column = vec![0.0; n_net];
    for k in 0..n_x {
        for row in 0..n_net {
            column[row] = c_x[row * n_x + k];
        }
        let solved = network.solve(&column).ok_or(SmallSignalError::SingularNetwork)?;
        for row in 0..n_net {
            z[row * n_x + k] = solved[row];
        }
    }

    // A = A_x − A_v·Z.
    let mut a = vec![0.0; n_x * n_x];
    for i in 0..n_x {
        for j in 0..n_x {
            let mut acc = a_x[i * n_x + j];
            for m in 0..n_net {
                acc -= a_v[i * n_net + m] * z[m * n_x + j];
            }
            a[i * n_x + j] = acc;
        }
    }
    Ok((n_x, a))
}

/// The linearized system's **state matrix**, one row per differential state.
///
/// The same `A` [`analyze`] decomposes, handed over rather than consumed — for
/// a caller who wants to do something this module does not, and because it is
/// the concrete object the reduction's formula names. Row `i` column `j` is
/// `∂ẋ_i/∂x_j` with the network eliminated, in the state order
/// [`DynamicSystem::differential_state_names`] gives.
///
/// Subject to the same size limit as `analyze`, and for the same reason: `A` is
/// dense, and forming it is the half of the work the sparse method exists to
/// avoid.
pub fn state_matrix(system: &DynamicSystem) -> Result<Vec<Vec<f64>>, SmallSignalError> {
    let (n_x, flat) = reduced_state_matrix(system)?;
    Ok(flat.chunks(n_x).map(|row| row.to_vec()).collect())
}

/// Linearizes about the system's current state and returns its modes.
pub fn analyze(system: &DynamicSystem) -> Result<SmallSignal, SmallSignalError> {
    // Past the dense limit the question is answered by the other method rather
    // than refused. The two are not interchangeable — one returns every mode,
    // the other the ones near a shift — which is why the result says which ran.
    if system.pattern.layout().n_diff > MAX_DENSE_STATES {
        return analyze_near(system, &SmallSignalOptions::default());
    }

    let (n_x, flat) = reduced_state_matrix(system)?;
    let a = Mat::<f64>::from_fn(n_x, n_x, |i, j| flat[i * n_x + j]);
    let angle_states = angle_states(system);

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
        let right: Vec<Complex<f64>> =
            (0..n_x).map(|k| Complex::new(u[(k, i)].re, u[(k, i)].im)).collect();
        // Row `i` of `U⁻¹` — already a row vector, so it needs no conjugating.
        let left_row: Vec<Complex<f64>> =
            (0..n_x).map(|k| Complex::new(w[(i, k)].re, w[(i, k)].im)).collect();

        modes.push(mode_from(
            lambda,
            participation_of(&right, &left_row),
            shape_of(is_oscillatory(lambda), &right, &angle_states),
            0.0,
            normalized(right),
            // `left_row` is `wᴴ`; the vector `w` itself is its conjugate.
            normalized(left_row.iter().map(|c| c.conj()).collect()),
        ));
    }
    sort_by_damping(&mut modes);

    Ok(SmallSignal {
        modes,
        state_names: system.differential_state_names(),
        method: Method::Dense,
        converged: true,
    })
}

/// Linearizes about the system's current state and returns the modes **nearest
/// a shift**, by shift-invert Arnoldi.
///
/// The method [`analyze`] cannot be at scale. It never forms the reduced `A`:
/// see [`shift`](super::shift) for the operator and
/// [`arnoldi`] for the iteration. What it costs is
/// completeness — this returns `count` modes and says nothing about the rest,
/// so "no unstable modes here" means *near this shift*, and a caller has to aim
/// it. What it buys is the twenty-thousand-state case, where the dense method
/// cannot start.
///
/// Participation factors need left eigenvectors, which the dense method gets by
/// inverting `U`. There is no `U` here, so they come from a second Arnoldi pass
/// on the adjoint operator, against the *same* factorization. The two passes
/// converge to the same eigenvalues from different subspaces and are paired by
/// eigenvalue; where that pairing is ambiguous — two modes closer to each other
/// than the pairing tolerance — the mode is reported **without** participation
/// rather than with a plausible wrong one.
pub fn analyze_near(
    system: &DynamicSystem,
    opts: &SmallSignalOptions,
) -> Result<SmallSignal, SmallSignalError> {
    let n_x = system.pattern.layout().n_diff;
    if n_x == 0 {
        return Err(SmallSignalError::NoStates);
    }
    let drift = system.max_derivative();
    if drift > EQUILIBRIUM_TOL {
        return Err(SmallSignalError::NotAnEquilibrium { max_derivative: drift });
    }

    let sigma = opts.shift;
    let op = ShiftedDae::new(system, sigma)?;
    let arnoldi_opts = ArnoldiOptions {
        count: opts.count,
        max_dim: opts.max_dim,
        tol: opts.tol,
        max_restarts: opts.max_restarts,
    };

    let right = arnoldi::eigenpairs(n_x, |b| op.apply(b), &arnoldi_opts)?;
    // The adjoint pass sees `Sᴴ`, whose eigenvalues are the conjugates of `S`'s
    // — so a left eigenvector for λ sits at `θ̄`, and that is what the pairing
    // below matches on.
    let left = arnoldi::eigenpairs(n_x, |b| op.apply_adjoint(b), &arnoldi_opts)?;

    let angle_states = angle_states(system);
    let mut modes = Vec::with_capacity(right.pairs.len());
    for pair in &right.pairs {
        let lambda = sigma + 1.0 / pair.theta;

        // Pair with a left eigenvector by eigenvalue. Ambiguity is refused
        // rather than resolved: with two candidates equally close there is no
        // evidence for either, and a participation vector attributed to the
        // wrong mode is worse than none — it reads exactly like an answer.
        let (participation, left_vector) = match matching_left(&left.pairs, pair.theta) {
            Some(z) => {
                // `z` satisfies `Aᴴz = λ̄z`, so the *row* vector `wᴴ = zᴴ`.
                let left_row: Vec<Complex<f64>> = z.iter().map(|c| c.conj()).collect();
                (participation_of(&pair.vector, &left_row), normalized(z.to_vec()))
            }
            None => (Vec::new(), Vec::new()),
        };

        modes.push(mode_from(
            lambda,
            participation,
            shape_of(is_oscillatory(lambda), &pair.vector, &angle_states),
            pair.residual,
            normalized(pair.vector.clone()),
            left_vector,
        ));
    }
    sort_by_damping(&mut modes);

    Ok(SmallSignal {
        modes,
        state_names: system.differential_state_names(),
        method: Method::Sparse { shift: sigma, restarts: right.restarts },
        converged: right.converged(),
    })
}

/// How far apart a left/right match may be, relative to `|θ|`, before there is
/// no match at all.
///
/// Loose, deliberately. The two passes build different Krylov spaces, and on an
/// ill-conditioned cluster their eigenvalue estimates genuinely differ — `1e-4`
/// on the 4 096-bus ring, at residuals of `3e-14` — so a tight threshold here
/// rejects correct pairings rather than wrong ones. What separates a match from
/// a non-match is not an absolute distance but whether one candidate stands
/// clear of the rest, which [`PAIRING_MARGIN`] is for.
const PAIRING_BOUND: f64 = 1e-2;

/// How much nearer the best candidate must be than the second.
///
/// This is the rule that actually decides. On that same ring the correct match
/// sits at `2e-4` and the next candidate at `2e-2` — a hundredfold margin, and
/// unambiguous — while two genuinely degenerate modes would produce two
/// candidates at the same distance and no evidence for either. A participation
/// vector attributed to the wrong mode is worse than none, because it reads
/// exactly like an answer.
const PAIRING_MARGIN: f64 = 10.0;

/// The left eigenvector matching an operator eigenvalue `θ`, if one candidate
/// is unambiguously the match.
fn matching_left(pairs: &[RitzPair], theta: Complex<f64>) -> Option<&[Complex<f64>]> {
    // The adjoint operator's spectrum is the conjugate of the forward one, so a
    // left eigenvector for this mode sits at `θ̄`.
    let target = theta.conj();
    let scale = theta.norm().max(f64::MIN_POSITIVE);
    let mut ranked: Vec<(f64, &RitzPair)> =
        pairs.iter().map(|p| ((p.theta - target).norm() / scale, p)).collect();
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let (best, pair) = *ranked.first()?;
    if best > PAIRING_BOUND {
        // The adjoint pass did not find this mode at all — it converged on a
        // different subset of a cluster, which is possible and is not an error.
        return None;
    }
    if let Some(&(second, _)) = ranked.get(1)
        && second < best * PAIRING_MARGIN
    {
        return None;
    }
    Some(&pair.vector)
}

/// Which states are rotor angles — the ones a mode shape is read at.
fn angle_states(system: &DynamicSystem) -> Vec<usize> {
    let layout = system.pattern.layout();
    (0..layout.n_devices())
        .filter_map(|d| {
            let offset = layout.dev_offset[d];
            system.models[d].angle_index().map(|k| offset + k)
        })
        .collect()
}

/// `p_ki = |u_ki·w_ik| / Σ_j |u_ji·w_ij|`, sorted largest first.
///
/// Shared by both methods deliberately: a participation factor from the sparse
/// path and one from the dense path must be the same number computed the same
/// way, or the gate comparing them is comparing two conventions rather than two
/// eigensolvers.
fn participation_of(right: &[Complex<f64>], left_row: &[Complex<f64>]) -> Vec<(usize, f64)> {
    let raw: Vec<f64> = right.iter().zip(left_row).map(|(u, w)| (u * w).norm()).collect();
    let total: f64 = raw.iter().sum();
    let mut participation: Vec<(usize, f64)> = raw
        .iter()
        .enumerate()
        .map(|(k, p)| (k, if total > 0.0 { p / total } else { 0.0 }))
        .collect();
    participation.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // A long tail of numerically-zero participations tells nobody anything.
    participation.retain(|&(_, p)| p > 1e-6);
    participation
}

/// When an eigenvalue's real or imaginary part counts as zero, relative to its
/// own magnitude.
///
/// There has to be a tolerance, and this is where the two methods would
/// otherwise disagree. A real eigenvalue of a real matrix comes out of the
/// dense decomposition with an imaginary part of *exactly* zero; the same
/// eigenvalue reached through `λ = σ + 1/θ` in complex arithmetic comes out at
/// `1e-17j`. Testing `im == 0.0` therefore made the sparse path report a mode
/// shape — a set of relative phases — for a mode that does not oscillate,
/// which is meaningless and looks like a finding.
///
/// Relative rather than absolute, because that is what the question means: a
/// mode at `−100 + 0.001j` decays sixty thousand times over before it completes
/// one cycle, and calling it oscillatory would be a technicality.
const NEGLIGIBLE: f64 = 1e-9;

/// The mode shape at the rotor angles, normalized so the largest component is
/// `1∠0`.
///
/// An eigenvector is defined only up to a complex scale, so only the *relative*
/// magnitudes and phases mean anything — normalizing is what makes two runs, or
/// two methods, comparable. Empty for a non-oscillatory mode, where a relative
/// phase between things that are not oscillating means nothing.
fn shape_of(
    oscillatory: bool,
    right: &[Complex<f64>],
    angle_states: &[usize],
) -> Vec<(usize, Complex<f64>)> {
    if !oscillatory {
        return Vec::new();
    }
    let raw: Vec<(usize, Complex<f64>)> = angle_states.iter().map(|&k| (k, right[k])).collect();
    match raw
        .iter()
        .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap_or(std::cmp::Ordering::Equal))
        .filter(|(_, c)| c.norm() > 0.0)
    {
        Some(&(_, pivot)) => raw.into_iter().map(|(k, c)| (k, c / pivot)).collect(),
        None => Vec::new(),
    }
}

/// An eigenvector scaled to unit length.
///
/// The sparse method's Ritz vectors arrive normalized and the dense method's do
/// not, and a caller comparing the two should not have to know which. Only the
/// magnitude is fixed — the phase is not, since an eigenvector has no
/// distinguished one.
fn normalized(mut vector: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let scale = vector.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if scale > 0.0 {
        for value in &mut vector {
            *value /= scale;
        }
    }
    vector
}

/// Whether an eigenvalue oscillates at all — see [`NEGLIGIBLE`].
fn is_oscillatory(lambda: Complex<f64>) -> bool {
    lambda.im.abs() > NEGLIGIBLE * lambda.norm()
}

/// The derived numbers every mode carries, from its eigenvalue.
///
/// Shared by both methods, which is what makes them comparable at all: every
/// rounding convention below is a convention, and two of them would be two
/// answers.
fn mode_from(
    lambda: Complex<f64>,
    participation: Vec<(usize, f64)>,
    shape: Vec<(usize, Complex<f64>)>,
    residual: f64,
    eigenvector: Vec<Complex<f64>>,
    left_eigenvector: Vec<Complex<f64>>,
) -> Mode {
    let magnitude = lambda.norm();
    let decaying = lambda.re.abs() > NEGLIGIBLE * magnitude;
    Mode {
        eigenvalue: lambda,
        damping: if magnitude > 0.0 && decaying { -lambda.re / magnitude } else { 0.0 },
        frequency: if is_oscillatory(lambda) {
            lambda.im.abs() / std::f64::consts::TAU
        } else {
            0.0
        },
        time_constant: if decaying { -1.0 / lambda.re } else { f64::INFINITY },
        participation,
        shape,
        residual,
        eigenvector,
        left_eigenvector,
    }
}

/// Least damped first: that is the order the question is asked in, and a
/// conjugate pair sorts adjacently because both halves share a damping.
fn sort_by_damping(modes: &mut [Mode]) {
    modes.sort_by(|a, b| a.damping.partial_cmp(&b.damping).unwrap_or(std::cmp::Ordering::Equal));
}

// ---------------------------------------------------------------------------
// Eigenvalue sensitivities
// ---------------------------------------------------------------------------

/// One parameter of one device.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterRef {
    /// Index into the system's devices, in the order `differential_state_names`
    /// runs.
    pub device: usize,
    /// The parameter's name — one of the device's
    /// [`tunable`](super::models::DynamicModel::tunable) set.
    pub name: String,
}

/// How one mode moves when one parameter does.
#[derive(Clone, Debug)]
pub struct ParameterSensitivity {
    pub parameter: ParameterRef,
    /// The parameter's current value, so a *relative* sensitivity — how much a
    /// 1% change buys — is one multiplication away.
    pub value: f64,
    /// `dλ/dp`, in reciprocal seconds per unit of the parameter.
    pub d_eigenvalue: Complex<f64>,
    /// `dζ/dp`. **This is usually the one that matters**: a study asks which
    /// knob best damps a mode, and damping ratio is the answer's units.
    pub d_damping: f64,
    /// `df/dp`, in hertz per unit of the parameter.
    pub d_frequency: f64,
}

/// The relative step a central difference takes.
///
/// A central difference is second-order, so its error is `O(δ²)` while the
/// subtraction loses digits as `O(ε/δ)`; the balance sits near `ε^(1/3)`, which
/// for a double is about `6e-6`. Rounded to `1e-5`, and relative to the
/// parameter so that an inertia of 5 s and one of 0.05 get the same treatment.
const DIFFERENCE_STEP: f64 = 1e-5;

/// How one mode's eigenvalue moves when a device parameter moves.
///
/// \\[ \frac{d\lambda}{dp} =
///    \frac{W^H (\partial J/\partial p)\, U}{w_x^H u} \\]
///
/// # Why the pencil, and not `A`
///
/// The reduced `A` depends on `p` through all four blocks *and* through
/// `C_v⁻¹`, so differentiating it directly means differentiating a matrix
/// inverse. The pencil `(J, E)` of [`shift`](super::shift) has no such problem:
/// `J` is the assembled Jacobian, `∂J/∂p` touches only the rows of the one
/// device that owns `p`, and the eigenvalue's derivative is a quadratic form in
/// the pencil's own left and right eigenvectors. Both of those are one solve
/// away from the eigenvectors [`Mode`] already carries — `v = −C_v⁻¹C_x u` and
/// `w_vᴴ = −w_xᴴA_v C_v⁻¹` — against the network block that
/// [`analyze`] factorizes anyway.
///
/// `∂J/∂p` itself is a **central difference of the assembly**: set the
/// parameter to `p ± δ`, refill the pattern, subtract. The pattern does not
/// move, because sparsity depends on structure and not on values, so the
/// difference is a value array over the very same `(row, col)` pairs. Nothing
/// is re-derived, and nothing can fall out of step with the models as they
/// change.
///
/// # What this can and cannot answer
///
/// Only parameters the *equilibrium* does not depend on — which is why
/// [`DynamicModel::tunable`](super::models::DynamicModel::tunable) lists so few.
/// The formula above holds the operating point fixed; for a parameter that
/// moves it, the true derivative carries a `∂J/∂x · dx/dp` term that would need
/// the initialization differentiated too. A reactance is such a parameter, and
/// is deliberately not offered rather than answered by two thirds.
///
/// # The system is borrowed mutably, and given back
///
/// The difference sets the parameter, refills, and sets it back. The system is
/// in its original state on return, including when a parameter cannot be set.
pub fn sensitivities(
    system: &mut DynamicSystem,
    mode: &Mode,
    parameters: &[ParameterRef],
) -> Result<Vec<ParameterSensitivity>, SmallSignalError> {
    let layout = system.pattern.layout().clone();
    let (n_x, n_net) = (layout.n_diff, 2 * layout.n_bus);
    if n_x == 0 {
        return Err(SmallSignalError::NoStates);
    }
    if mode.eigenvector.len() != n_x || mode.left_eigenvector.len() != n_x {
        // A sparse mode whose left eigenvector could not be paired has no
        // sensitivity to report, and saying so beats returning a number built
        // from an empty vector.
        return Err(SmallSignalError::NoLeftEigenvector);
    }

    // The unperturbed assembly, and the network factorization both halves of
    // the pencil's eigenvectors are completed against.
    let base = fill_at_equilibrium(system);
    let triplets = system.pattern.to_triplets(&base);
    let c_v: Vec<(usize, usize, f64)> = triplets
        .iter()
        .filter(|&&(r, c, _)| r >= n_x && c >= n_x)
        .map(|&(r, c, value)| (r - n_x, c - n_x, value))
        .collect();
    let network =
        RealFactorization::new(n_net, &c_v).ok_or(SmallSignalError::SingularNetwork)?;

    let u = &mode.eigenvector;
    let w_x = &mode.left_eigenvector;

    // v = −C_v⁻¹·(C_x u), as a sparse product then one solve per component.
    let mut c_x_u = vec![Complex::new(0.0, 0.0); n_net];
    for &(r, c, value) in &triplets {
        if r >= n_x && c < n_x {
            c_x_u[r - n_x] += value * u[c];
        }
    }
    let v = solve_complex_real(&network, &c_x_u, false)
        .ok_or(SmallSignalError::SingularNetwork)?
        .into_iter()
        .map(|value| -value)
        .collect::<Vec<_>>();

    // w_v = −(C_vᵀ)⁻¹·(A_vᵀ w_x). `A_v = −(top-right of the fill)`, and `A_v`
    // is real, so its conjugate transpose is its transpose.
    let mut a_v_t_w = vec![Complex::new(0.0, 0.0); n_net];
    for &(r, c, value) in &triplets {
        if r < n_x && c >= n_x {
            a_v_t_w[c - n_x] += -value * w_x[r];
        }
    }
    let w_v = solve_complex_real(&network, &a_v_t_w, true)
        .ok_or(SmallSignalError::SingularNetwork)?
        .into_iter()
        .map(|value| -value)
        .collect::<Vec<_>>();

    // Denominator: WᴴEU = w_xᴴu, the eigenvalue's own condition. Near zero
    // means a defective or near-defective mode, whose eigenvalue derivative is
    // genuinely unbounded rather than merely large.
    let denominator: Complex<f64> = w_x.iter().zip(u).map(|(w, u)| w.conj() * u).sum();
    if denominator.norm() < 1e-12 {
        return Err(SmallSignalError::DefectiveMode);
    }

    let mut out = Vec::with_capacity(parameters.len());
    let mut perturbed = Vec::with_capacity(base.len());
    for reference in parameters {
        let device = system
            .models
            .get(reference.device)
            .ok_or_else(|| SmallSignalError::NoSuchParameter(reference.clone()))?;
        let value = device
            .parameter(&reference.name)
            .ok_or_else(|| SmallSignalError::NoSuchParameter(reference.clone()))?;

        let step = DIFFERENCE_STEP * value.abs().max(1.0);
        let mut difference = vec![0.0; base.len()];
        for (sign, weight) in [(1.0, 0.5), (-1.0, -0.5)] {
            let model = &mut system.models[reference.device];
            let previous = model
                .set_parameter(&reference.name, value + sign * step)
                .ok_or_else(|| SmallSignalError::NoSuchParameter(reference.clone()))?;
            perturbed.clear();
            perturbed.extend_from_slice(&fill_at_equilibrium(system));
            system.models[reference.device].set_parameter(&reference.name, previous);
            for (slot, value) in difference.iter_mut().zip(&perturbed) {
                *slot += weight * value / step;
            }
        }

        // Wᴴ(∂J/∂p)U over the pattern. `J` is the fill with its top block-rows
        // negated — the same transform `ShiftedDae` applies, and for the same
        // reason.
        let mut numerator = Complex::new(0.0, 0.0);
        for (k, &(r, c, _)) in triplets.iter().enumerate() {
            let entry = if r < n_x { -difference[k] } else { difference[k] };
            if entry == 0.0 {
                continue;
            }
            let left = if r < n_x { w_x[r] } else { w_v[r - n_x] };
            let right = if c < n_x { u[c] } else { v[c - n_x] };
            numerator += left.conj() * entry * right;
        }

        let d_eigenvalue = numerator / denominator;
        out.push(ParameterSensitivity {
            parameter: reference.clone(),
            value,
            d_eigenvalue,
            d_damping: damping_derivative(mode.eigenvalue, d_eigenvalue),
            d_frequency: frequency_derivative(mode.eigenvalue, d_eigenvalue),
        });
    }
    Ok(out)
}

/// Every tunable parameter of every device, in device order.
///
/// The list to hand [`sensitivities`] when the question is "which knob", rather
/// than "how much does this one".
pub fn tunable_parameters(system: &DynamicSystem) -> Vec<ParameterRef> {
    system
        .models
        .iter()
        .enumerate()
        .flat_map(|(device, model)| {
            model
                .tunable()
                .iter()
                .map(move |name| ParameterRef { device, name: (*name).to_string() })
        })
        .collect()
}

/// Solves a real system against a complex right-hand side, one component at a
/// time.
///
/// `C_v` is real — it is the constant real form of the Y-bus — while the
/// eigenvectors are not, so the two halves go through the same factorization
/// separately. Cheaper than a complex factorization of a real matrix, and it
/// reuses the one [`analyze`]'s reduction already builds.
fn solve_complex_real(
    network: &RealFactorization,
    rhs: &[Complex<f64>],
    transpose: bool,
) -> Option<Vec<Complex<f64>>> {
    let real: Vec<f64> = rhs.iter().map(|c| c.re).collect();
    let imaginary: Vec<f64> = rhs.iter().map(|c| c.im).collect();
    let (real, imaginary) = if transpose {
        (network.solve_transpose(&real)?, network.solve_transpose(&imaginary)?)
    } else {
        (network.solve(&real)?, network.solve(&imaginary)?)
    };
    Some(real.into_iter().zip(imaginary).map(|(re, im)| Complex::new(re, im)).collect())
}

/// `dζ/dp` from `dλ/dp`, with `ζ = −σ/|λ|`.
fn damping_derivative(lambda: Complex<f64>, d_lambda: Complex<f64>) -> f64 {
    let magnitude = lambda.norm();
    if magnitude == 0.0 {
        return 0.0;
    }
    let cube = magnitude * magnitude * magnitude;
    let d_sigma = -(lambda.im * lambda.im) / cube;
    let d_omega = (lambda.re * lambda.im) / cube;
    d_sigma * d_lambda.re + d_omega * d_lambda.im
}

/// `df/dp` from `dλ/dp`, with `f = |ω|/2π`.
fn frequency_derivative(lambda: Complex<f64>, d_lambda: Complex<f64>) -> f64 {
    if !is_oscillatory(lambda) {
        return 0.0;
    }
    let sign = if lambda.im >= 0.0 { 1.0 } else { -1.0 };
    sign * d_lambda.im / std::f64::consts::TAU
}

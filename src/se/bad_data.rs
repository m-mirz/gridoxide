//! Detecting and identifying measurements that disagree with the rest.
//!
//! A converged estimate is not a correct one. Weighted least squares will
//! happily absorb a broken sensor by bending the state toward it, and the
//! result looks exactly like a successful solve — the fixture
//! `node-injection-sensor-and-zero-injection` is a worked example of how far
//! that can go. Bad-data analysis is the step that asks whether the
//! measurements were mutually consistent in the first place.
//!
//! Two questions, answered separately:
//!
//! 1. **Is there bad data at all?** The chi-squared test on the objective. Under
//!    the assumption that each measurement error is independent and normal with
//!    the declared sigma, `J = rᵀWr` is chi-squared distributed with `m − n`
//!    degrees of freedom. A `J` far out in the tail says the residuals are too
//!    large to be noise.
//! 2. **Which measurement?** The largest normalized residual. Raw residuals are
//!    not comparable — a residual of 0.1 is enormous for a sigma of 0.001 and
//!    negligible for a sigma of 1 — and dividing by sigma alone is not enough
//!    either, because a redundantly measured quantity spreads its error across
//!    its neighbours. The right scale is the residual's own standard deviation,
//!    `Ω = R − H G⁻¹ Hᵀ`.
//!
//! # Cost, and why only the top candidates
//!
//! `Ωᵢᵢ` needs `hᵢ G⁻¹ hᵢᵀ`, i.e. one linear solve per measurement. Computing
//! the whole diagonal costs `m` solves, which for a real grid is far more than
//! the estimate itself. [`analyze`] therefore ranks candidates by the cheap
//! proxy `|rᵢ|/σᵢ`, computes `Ωᵢᵢ` for the worst [`Candidates::limit`] of them,
//! and reports those.
//!
//! That is an approximation and worth being precise about: the true largest
//! normalized residual can in principle sit outside the shortlist, because
//! `Ωᵢᵢ ≤ Rᵢᵢ` varies per measurement. In practice the two orderings agree
//! closely — a measurement with a large normalized residual almost always has a
//! large raw one — but "almost always" is not "always", and a caller that needs
//! certainty can raise the limit to `m`.

use crate::measurement::Measurement;
use crate::solver::LinearSolver;
use crate::sparse::RealSparseSystem;
use crate::types::Bus;

use super::constraints::{augment, Constraints};
use super::jacobian::{gain_and_rhs, mask_untouched, measurement_jacobian, Row, StateLayout};
use super::SeNetwork;

/// How many measurements to examine in detail.
#[derive(Clone, Copy, Debug)]
pub struct Candidates {
    /// Maximum number of measurements to compute `Ωᵢᵢ` for.
    pub limit: usize,
}

impl Default for Candidates {
    fn default() -> Self {
        Self { limit: 20 }
    }
}

/// A measurement flagged as suspect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Suspect {
    /// Index into the measurement list the estimate was run on.
    pub measurement: usize,
    /// `|rᵢ| / √Ωᵢᵢ`. Conventionally compared against 3.
    pub normalized_residual: f64,
}

/// The outcome of a bad-data analysis.
#[derive(Clone, Debug)]
pub struct BadDataReport {
    /// `J = rᵀWr`, the chi-squared statistic. Twice
    /// [`SeReport::objective`](super::nr::SeReport::objective), which carries
    /// the conventional ½.
    pub chi_squared: f64,
    pub degrees_of_freedom: usize,
    /// Probability of seeing a `J` at least this large if every measurement
    /// were merely noisy. Small means the data is not merely noisy.
    pub p_value: f64,
    /// Measurements examined, worst first.
    pub suspects: Vec<Suspect>,
}

impl BadDataReport {
    /// Whether the chi-squared test rejects at the given significance (0.05 is
    /// conventional).
    ///
    /// Rejection says *something* is wrong, not which measurement — that is
    /// what [`suspects`](Self::suspects) is for. A test that does not reject is
    /// also not a clean bill of health: a single moderate error, or several
    /// that partly cancel, can sit inside the threshold.
    pub fn rejects_at(&self, significance: f64) -> bool {
        self.p_value < significance
    }

    /// The worst measurement, if any were examined.
    pub fn worst(&self) -> Option<Suspect> {
        self.suspects.first().copied()
    }
}

/// Regularized lower incomplete gamma `P(a, x)`, by series expansion.
///
/// Converges quickly for `x < a + 1`; [`gamma_q`] picks between this and the
/// continued fraction accordingly.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..500 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Regularized upper incomplete gamma `Q(a, x)`, by continued fraction
/// (modified Lentz).
fn gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Lanczos approximation to `ln Γ(x)`, accurate to ~1e-15 for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];
    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000_000_000_190_015;
    for c in COF {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Regularized upper incomplete gamma `Q(a, x) = 1 − P(a, x)`.
fn gamma_q(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_continued_fraction(a, x)
    }
}

/// Upper-tail probability of the chi-squared distribution: the chance of
/// observing at least `x` with `dof` degrees of freedom.
///
/// Implemented here rather than pulled from a statistics crate because it is
/// the only special function this crate needs, and the candidates bring a
/// second linear-algebra stack alongside `faer`. `tests` checks it against
/// published chi-squared table values.
pub fn chi_squared_upper_tail(x: f64, dof: usize) -> f64 {
    if dof == 0 {
        return if x > 0.0 { 0.0 } else { 1.0 };
    }
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(dof as f64 / 2.0, x / 2.0).clamp(0.0, 1.0)
}

/// Runs bad-data analysis at a converged state.
///
/// `residuals` must be the ones the estimate finished with
/// ([`SeReport::residuals`](super::nr::SeReport::residuals)), and `buses` the
/// state it finished at — the analysis linearizes there.
pub fn analyze(
    measurements: &[Measurement],
    residuals: &[f64],
    buses: &[Bus],
    net: &SeNetwork,
    layout: &StateLayout,
    constraints: &Constraints,
    candidates: Candidates,
) -> BadDataReport {
    let weighted: Vec<(usize, &Measurement, f64)> = measurements
        .iter()
        .zip(residuals)
        .enumerate()
        .filter(|(_, (m, _))| m.weight().is_finite() && m.weight() > 0.0)
        .map(|(i, (m, &r))| (i, m, r))
        .collect();

    let chi_squared: f64 = weighted.iter().map(|(_, m, r)| m.weight() * r * r).sum();

    let rows = measurement_jacobian(measurements, buses, net, layout);
    let (c_values, c_rows) = constraints.evaluate(buses, net, layout);
    let n = layout.n_unknowns();

    let (mut triplets, mut rhs, _) = gain_and_rhs(&rows, measurements, residuals);
    rhs.resize(n, 0.0);
    let untouched = mask_untouched(&mut triplets, &mut rhs, &[&rows, &c_rows], n);
    let (triplets, _) = augment(triplets, rhs, n, &c_values, &c_rows);
    let n_aug = n + constraints.len();

    // Degrees of freedom: measurements, less the state variables they had to
    // determine. Variables that were pinned rather than estimated do not
    // consume a degree of freedom, and each equality constraint gives one back.
    let estimated = n.saturating_sub(untouched.len());
    let degrees_of_freedom = weighted
        .len()
        .saturating_sub(estimated)
        .saturating_add(constraints.len());

    let p_value = chi_squared_upper_tail(chi_squared, degrees_of_freedom);

    // Shortlist by the cheap proxy, then pay for the real thing.
    let mut shortlist: Vec<(usize, &Measurement, f64)> = weighted.clone();
    shortlist.sort_by(|a, b| {
        let key = |t: &(usize, &Measurement, f64)| (t.2 / t.1.sigma).abs();
        key(b).partial_cmp(&key(a)).unwrap_or(std::cmp::Ordering::Equal)
    });
    shortlist.truncate(candidates.limit);

    let mut system = match RealSparseSystem::new(n_aug, &triplets) {
        Some(s) => s,
        // Without a factorizable system there is no residual covariance to
        // compute; the chi-squared half of the report still stands.
        None => {
            return BadDataReport { chi_squared, degrees_of_freedom, p_value, suspects: Vec::new() }
        }
    };

    // The backend caches its symbolic factorization but wants the values on
    // every call, so they are hoisted out of the candidate loop. Each candidate
    // still costs one numeric refactorization — the documented price of the
    // exact residual covariance.
    let values: Vec<f64> = triplets.iter().map(|&(_, _, v)| v).collect();

    let mut suspects = Vec::new();
    for (index, m, r) in shortlist {
        let Some(omega) = residual_variance(&mut system, &values, &rows[index], m, n_aug) else {
            continue;
        };
        if omega <= 0.0 {
            // A measurement whose residual has no variance is one the estimate
            // is forced to reproduce exactly — critical, in the usual
            // terminology. Its error cannot be detected at all, so reporting a
            // normalized residual for it would be meaningless.
            continue;
        }
        suspects.push(Suspect { measurement: index, normalized_residual: r.abs() / omega.sqrt() });
    }
    suspects.sort_by(|a, b| {
        b.normalized_residual
            .partial_cmp(&a.normalized_residual)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    BadDataReport { chi_squared, degrees_of_freedom, p_value, suspects }
}

/// `Ωᵢᵢ = Rᵢᵢ − hᵢ G⁻¹ hᵢᵀ`, the variance of one measurement's residual.
///
/// Solved through the *augmented* system rather than `G` alone, so the
/// zero-injection constraints are accounted for: a constrained estimate has
/// less freedom to move, which changes how much of an error shows up in the
/// residual rather than in the state.
fn residual_variance<S: LinearSolver>(
    system: &mut S,
    values: &[f64],
    row: &Row,
    m: &Measurement,
    n_aug: usize,
) -> Option<f64> {
    let mut rhs = vec![0.0; n_aug];
    for &(c, v) in row {
        rhs[c] += v;
    }
    let y = system.factor_and_solve_values(values, &rhs)?;
    let quadratic: f64 = row.iter().map(|&(c, v)| v * y[c]).sum();
    Some(m.sigma * m.sigma - quadratic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch_flow::Terminal;
    use crate::measurement::{MeasurementKind, Target};
    use crate::se::measurement_functions;
    use crate::se::nr::{estimate, flat_start, SeOptions};

    /// Published chi-squared critical values: `P(X > x) = 0.05`.
    ///
    /// These are the numbers in the back of every statistics textbook, and they
    /// are the reason this hand-written incomplete gamma is defensible — it is
    /// checkable against an external source rather than against itself.
    #[test]
    fn chi_squared_matches_published_critical_values() {
        for (dof, critical) in [(1, 3.841), (2, 5.991), (5, 11.070), (10, 18.307), (20, 31.410)] {
            let p = chi_squared_upper_tail(critical, dof);
            assert!(
                (p - 0.05).abs() < 1e-3,
                "dof {dof}: P(X > {critical}) = {p}, expected 0.05"
            );
        }
    }

    /// The other tail, and the median, from the same tables.
    #[test]
    fn chi_squared_matches_other_quantiles() {
        assert!((chi_squared_upper_tail(0.004, 1) - 0.95).abs() < 1e-3);
        assert!((chi_squared_upper_tail(9.342, 10) - 0.50).abs() < 1e-3);
        assert!((chi_squared_upper_tail(23.209, 10) - 0.01).abs() < 1e-3);
    }

    #[test]
    fn chi_squared_edges_are_sane() {
        assert_eq!(chi_squared_upper_tail(-1.0, 5), 1.0);
        assert_eq!(chi_squared_upper_tail(0.0, 5), 1.0);
        assert!(chi_squared_upper_tail(1e6, 5) < 1e-12);
    }

    /// Builds a measurement set read exactly off a known state, optionally with
    /// one reading corrupted.
    fn measurements_from(
        truth: &[Bus],
        net: &SeNetwork,
        corrupt: Option<(usize, f64)>,
    ) -> Vec<Measurement> {
        let probe: Vec<Measurement> = [
            (MeasurementKind::VoltageMagnitude, Target::Bus(0)),
            (MeasurementKind::VoltageMagnitude, Target::Bus(1)),
            (MeasurementKind::ActivePower, Target::Bus(1)),
            (MeasurementKind::ReactivePower, Target::Bus(1)),
            (
                MeasurementKind::ActivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
            ),
            (
                MeasurementKind::ReactivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
            ),
            (
                MeasurementKind::ActivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::To },
            ),
        ]
        .into_iter()
        .map(|(kind, target)| Measurement { kind, target, value: 0.0, sigma: 0.01 })
        .collect();

        let exact = measurement_functions(&probe, truth, net);
        probe
            .iter()
            .zip(&exact)
            .enumerate()
            .map(|(i, (p, &v))| {
                let value = match corrupt {
                    Some((j, delta)) if j == i => v + delta,
                    _ => v,
                };
                Measurement { value, ..*p }
            })
            .collect()
    }

    /// Consistent data must not be flagged. A test that only detects bad data
    /// proves nothing if it also flags good data.
    #[test]
    fn consistent_measurements_are_not_flagged() {
        let (net, truth) = crate::se::tests::two_bus_net();
        let measurements = measurements_from(&truth, &net, None);

        let mut buses = truth.clone();
        flat_start(&mut buses, &measurements);
        let report = estimate(&measurements, &mut buses, &net, &SeOptions::default());

        let layout = StateLayout::new(&buses, &measurements, &net);
        let constraints = Constraints::new(&net.zero_injection);
        let bad = analyze(
            &measurements,
            &report.residuals,
            &buses,
            &net,
            &layout,
            &constraints,
            Candidates::default(),
        );

        assert!(bad.chi_squared < 1e-12, "noise-free data: {bad:?}");
        assert!(!bad.rejects_at(0.05), "clean data must not be rejected: {bad:?}");
    }

    /// One grossly wrong reading, twenty sigma out, must be both detected and
    /// identified.
    #[test]
    fn a_gross_error_is_detected_and_identified() {
        let (net, truth) = crate::se::tests::two_bus_net();
        let corrupted = 4; // the branch active-power measurement
        let measurements = measurements_from(&truth, &net, Some((corrupted, 0.2)));

        let mut buses = truth.clone();
        flat_start(&mut buses, &measurements);
        let report = estimate(&measurements, &mut buses, &net, &SeOptions::default());

        let layout = StateLayout::new(&buses, &measurements, &net);
        let constraints = Constraints::new(&net.zero_injection);
        let bad = analyze(
            &measurements,
            &report.residuals,
            &buses,
            &net,
            &layout,
            &constraints,
            Candidates::default(),
        );

        assert!(
            bad.rejects_at(0.05),
            "a 20-sigma error must fail the chi-squared test: {bad:?}"
        );
        let worst = bad.worst().expect("some measurement should be examined");
        assert_eq!(
            worst.measurement, corrupted,
            "the corrupted measurement should be the worst: {bad:?}"
        );
        assert!(
            worst.normalized_residual > 3.0,
            "and should exceed the conventional threshold: {worst:?}"
        );
    }
}

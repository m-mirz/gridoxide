//! The cost and limit data an OPF needs and a power flow has no place for.
//!
//! Nothing in gridoxide's network model carries a generator's cost curve, its
//! active-power limits, or a branch's MVA rating: `Bus` has `q_min`/`q_max`
//! and that is all. This module is where that data lives, read from the
//! companion document `python/gridoxide/matpower.py` writes beside the PGM
//! network file.
//!
//! # Why generators are listed independently of the network document
//!
//! Because the network document deliberately loses them. The MATPOWER
//! converter **aggregates generators per bus** — several units at one bus
//! become a single `sym_gen` with their powers summed — and turns the
//! reference bus's generator into a `source` with no generator record at all.
//! For a power flow that is exactly right: only the bus's net injection
//! matters. For an OPF it is fatal, because an OPF dispatches each unit
//! against *its own* cost curve and *its own* limits, and two different
//! quadratics cannot be summed into one.
//!
//! So a [`Generator`] here carries its own `index` and the `node` it sits on,
//! and the OPF builds its dispatchable set from this document rather than from
//! the network's components. Branch limits are different: branches *are*
//! one-to-one with network components, so a [`BranchLimit`] keys straight into
//! them by id.
//!
//! # Units
//!
//! The document is in **MATPOWER's own units** — MW, MVAr, MVA, and dollars
//! per MWh — not the network document's watts.
//!
//! That is deliberate. Every published cost curve is quoted per MW, so
//! converting the power without converting the curve would be a silent factor
//! of `1e6` on the linear term and `1e12` on the quadratic. Keeping the
//! original units also means a value can be read straight against the `.m`
//! file it came from. The conversion to per-unit happens here, once, in
//! [`CostCurve::to_per_unit`] and the limit accessors — where it is visible
//! and tested rather than spread across a formulation.

use serde::Deserialize;

/// How a generator's cost varies with its active-power output.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum CostCurve {
    /// `coefficients[k]` multiplies `p^k` — **ascending** order, so index is
    /// degree.
    ///
    /// MATPOWER stores these highest-degree first; the converter reverses them
    /// on the way in. Ascending is self-describing, and the reversal is the
    /// single most likely thing to get backwards, so it lives in one place
    /// with a test rather than at every point of use.
    Polynomial { coefficients: Vec<f64> },
    /// Breakpoints as `(output, cost)` pairs, in ascending output order.
    PiecewiseLinear { points: Vec<(f64, f64)> },
}

impl CostCurve {
    /// Cost of producing `p`, in the curve's own units.
    ///
    /// Outside a piecewise curve's breakpoints the nearest segment is
    /// extended rather than the value clamped: a clamp would make the curve
    /// flat there, which reads to an optimizer as free production beyond the
    /// last breakpoint.
    pub fn evaluate(&self, p: f64) -> f64 {
        match self {
            Self::Polynomial { coefficients } => {
                // Horner, from the highest degree down.
                coefficients.iter().rev().fold(0.0, |acc, c| acc * p + c)
            }
            Self::PiecewiseLinear { points } => match points.len() {
                0 => 0.0,
                1 => points[0].1,
                _ => {
                    let segment = points
                        .windows(2)
                        .find(|w| p <= w[1].0)
                        .unwrap_or_else(|| &points[points.len() - 2..]);
                    let ((x0, y0), (x1, y1)) = (segment[0], segment[1]);
                    if x1 == x0 {
                        y0
                    } else {
                        y0 + (y1 - y0) * (p - x0) / (x1 - x0)
                    }
                }
            },
        }
    }

    /// Marginal cost — `d(cost)/dp` — at `p`.
    ///
    /// This is the quantity a locational marginal price is compared against,
    /// so it is worth having directly rather than differencing `evaluate`.
    pub fn marginal(&self, p: f64) -> f64 {
        match self {
            Self::Polynomial { coefficients } => coefficients
                .iter()
                .enumerate()
                .skip(1)
                .map(|(k, c)| c * (k as f64) * p.powi(k as i32 - 1))
                .sum(),
            Self::PiecewiseLinear { points } => {
                if points.len() < 2 {
                    return 0.0;
                }
                let segment = points
                    .windows(2)
                    .find(|w| p <= w[1].0)
                    .unwrap_or_else(|| &points[points.len() - 2..]);
                let ((x0, y0), (x1, y1)) = (segment[0], segment[1]);
                if x1 == x0 { 0.0 } else { (y1 - y0) / (x1 - x0) }
            }
        }
    }

    /// Rewrites the curve so its argument is per-unit on `base_mva` rather
    /// than MW, leaving the cost itself in dollars per hour.
    ///
    /// With `p_mw = base · p_pu`, the term `c_k · p_mw^k` becomes
    /// `c_k · base^k · p_pu^k`, so **coefficient `k` scales by `base^k`** —
    /// the constant is untouched, the linear term scales once, the quadratic
    /// twice. Getting this wrong produces a cost that is off by orders of
    /// magnitude on the quadratic term alone while still looking like a cost,
    /// which is why it is a method with a test rather than an expression
    /// inlined into the objective assembly.
    pub fn to_per_unit(&self, base_mva: f64) -> Self {
        match self {
            Self::Polynomial { coefficients } => Self::Polynomial {
                coefficients: coefficients
                    .iter()
                    .enumerate()
                    .map(|(k, c)| c * base_mva.powi(k as i32))
                    .collect(),
            },
            Self::PiecewiseLinear { points } => Self::PiecewiseLinear {
                points: points.iter().map(|&(x, y)| (x / base_mva, y)).collect(),
            },
        }
    }

    /// Whether the curve is convex, which decides whether the OPF built on it
    /// is.
    ///
    /// A quadratic is convex when its leading coefficient is non-negative; a
    /// piecewise-linear curve when its segment slopes are non-decreasing. A
    /// non-convex curve is not an error here — it is data — but it does mean
    /// the "optimal is provable" argument in `plans/OPF_PLAN.md` no longer
    /// applies, so the formulation should refuse it rather than return a
    /// number nobody can certify.
    pub fn is_convex(&self) -> bool {
        match self {
            Self::Polynomial { coefficients } => match coefficients.len() {
                0..=2 => true,
                3 => coefficients[2] >= 0.0,
                // A cubic or higher is not convex over an unbounded domain
                // whatever its coefficients, and MATPOWER's own OPF rejects
                // them for the same reason.
                _ => false,
            },
            Self::PiecewiseLinear { points } => points
                .windows(3)
                .all(|w| slope(w[0], w[1]) <= slope(w[1], w[2]) + 1e-12),
        }
    }
}

fn slope(a: (f64, f64), b: (f64, f64)) -> f64 {
    if b.0 == a.0 { 0.0 } else { (b.1 - a.1) / (b.0 - a.0) }
}

/// One dispatchable unit.
#[derive(Clone, Debug, Deserialize)]
pub struct Generator {
    /// Its row in the source case's generator table. Identity within this
    /// document — deliberately not a network-component id, since several
    /// generators may share one.
    pub index: usize,
    /// The node id it injects at, in the network document's own numbering.
    pub node: u64,
    /// Active-power limits, MW.
    pub p_min: f64,
    pub p_max: f64,
    /// Reactive-power limits, MVAr. Absent when the source case declared none
    /// — MATPOWER writes literal infinities there, which are not valid JSON.
    #[serde(default)]
    pub q_min: Option<f64>,
    #[serde(default)]
    pub q_max: Option<f64>,
    /// Absent for a case with no `gencost`, in which case this unit's output
    /// is free and the OPF has nothing to trade it against.
    #[serde(default)]
    pub cost: Option<CostCurve>,
}

impl Generator {
    /// Active-power limits in per-unit on `base_mva`.
    pub fn p_limits_pu(&self, base_mva: f64) -> (f64, f64) {
        (self.p_min / base_mva, self.p_max / base_mva)
    }

    /// Reactive-power limits in per-unit on `base_mva`, with an absent bound
    /// becoming an infinity.
    ///
    /// Unbounded rather than zero is the only safe reading: MATPOWER writes
    /// literal infinities for an unlimited unit, which are not valid JSON and
    /// so arrive here as `None`. Defaulting those to zero would silently forbid
    /// the unit from producing *any* reactive power — turning a machine with no
    /// stated limit into the most constrained one on the network, and doing it
    /// in a way that looks like a converged answer rather than an error.
    pub fn q_limits_pu(&self, base_mva: f64) -> (f64, f64) {
        (
            self.q_min.map_or(f64::NEG_INFINITY, |q| q / base_mva),
            self.q_max.map_or(f64::INFINITY, |q| q / base_mva),
        )
    }
}

/// One bus's voltage magnitude limits, per-unit.
#[derive(Clone, Debug, Deserialize)]
pub struct BusVoltage {
    /// The network document's node id.
    pub node: u64,
    pub v_min: f64,
    pub v_max: f64,
}

/// A branch's thermal rating.
#[derive(Clone, Debug, Deserialize)]
pub struct BranchLimit {
    /// The network document's own component id — branches are one-to-one, so
    /// this keys directly.
    pub id: u64,
    /// `rateA`, MVA.
    pub rate_a: f64,
    /// **MATPOWER's convention: a rate of zero means unlimited**, not a
    /// binding zero. Carried explicitly so nothing downstream has to
    /// rediscover it, and so a genuinely unrated branch is distinguishable
    /// from one that was simply omitted.
    #[serde(default)]
    pub unlimited: bool,
}

impl BranchLimit {
    /// The rating in per-unit on `base_mva`, or `None` when unlimited.
    pub fn rate_pu(&self, base_mva: f64) -> Option<f64> {
        (!self.unlimited).then(|| self.rate_a / base_mva)
    }
}

/// A load that could be shed, if the formulation allows it.
#[derive(Clone, Debug, Deserialize)]
pub struct Load {
    pub id: u64,
    pub node: u64,
}

/// Everything the companion OPF document carries.
#[derive(Clone, Debug, Deserialize)]
pub struct OpfData {
    /// System base, MVA — the divisor for every per-unit accessor here.
    pub base_mva: f64,
    #[serde(default)]
    pub generator: Vec<Generator>,
    /// Per-bus voltage magnitude limits, per-unit. Only AC-OPF uses these —
    /// DC holds `|V| = 1`.
    ///
    /// Empty for a document written before they were emitted, which
    /// [`AcOpfNetwork`](super::ac::AcOpfNetwork) treats as "fall back to the
    /// option defaults" rather than as "unbounded". Silently unbounded
    /// voltages would make every case solve, and solve wrongly.
    #[serde(default)]
    pub bus_voltage: Vec<BusVoltage>,
    #[serde(default)]
    pub branch_limit: Vec<BranchLimit>,
    #[serde(default)]
    pub load: Vec<Load>,
}

impl OpfData {
    /// Reads a companion document.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Whether every cost curve present is convex — see
    /// [`CostCurve::is_convex`].
    pub fn costs_are_convex(&self) -> bool {
        self.generator
            .iter()
            .filter_map(|g| g.cost.as_ref())
            .all(CostCurve::is_convex)
    }

    /// The branches that actually constrain anything.
    ///
    /// Worth having as its own accessor because an all-unlimited case is a
    /// real and easily-missed situation: it makes every congestion constraint
    /// vacuous and every locational marginal price identical, which looks like
    /// a working OPF right up until someone reads the prices. See
    /// `tests/data/pglib-opf/README.md`.
    pub fn binding_capable_branches(&self) -> usize {
        self.branch_limit.iter().filter(|b| !b.unlimited).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic() -> CostCurve {
        // 0.043 p^2 + 20 p + 0, the shape of a real `gencost` row.
        CostCurve::Polynomial { coefficients: vec![0.0, 20.0, 0.043] }
    }

    #[test]
    fn a_polynomial_is_evaluated_in_ascending_order() {
        let c = quadratic();
        // At p = 10: 0.043*100 + 20*10 = 4.3 + 200.
        assert!((c.evaluate(10.0) - 204.3).abs() < 1e-12);
        // Marginal: 2*0.043*10 + 20 = 20.86.
        assert!((c.marginal(10.0) - 20.86).abs() < 1e-12);
    }

    /// The conversion that is easy to get wrong and impossible to spot
    /// afterwards: coefficient `k` scales by `base^k`, so the quadratic term
    /// moves by the *square* of the base.
    #[test]
    fn per_unit_conversion_scales_each_coefficient_by_its_own_power() {
        let pu = quadratic().to_per_unit(100.0);
        let CostCurve::Polynomial { coefficients } = &pu else { panic!("still polynomial") };
        assert!((coefficients[0] - 0.0).abs() < 1e-12);
        assert!((coefficients[1] - 2000.0).abs() < 1e-12, "{coefficients:?}");
        assert!((coefficients[2] - 430.0).abs() < 1e-12, "{coefficients:?}");

        // The point of the exercise: the same physical dispatch costs the same
        // whichever units it is expressed in.
        let mw = quadratic().evaluate(10.0);
        let per_unit = pu.evaluate(10.0 / 100.0);
        assert!((mw - per_unit).abs() < 1e-9, "{mw} vs {per_unit}");
    }

    #[test]
    fn a_piecewise_curve_interpolates_and_extends_its_end_segments() {
        let c = CostCurve::PiecewiseLinear {
            points: vec![(0.0, 0.0), (10.0, 100.0), (20.0, 300.0)],
        };
        assert!((c.evaluate(5.0) - 50.0).abs() < 1e-12);
        assert!((c.evaluate(15.0) - 200.0).abs() < 1e-12);
        assert!((c.marginal(5.0) - 10.0).abs() < 1e-12);
        assert!((c.marginal(15.0) - 20.0).abs() < 1e-12);

        // Beyond the last breakpoint the final segment continues, rather than
        // the curve going flat and reading as free production.
        assert!((c.evaluate(25.0) - 400.0).abs() < 1e-12);
    }

    #[test]
    fn convexity_is_judged_on_the_shape_that_matters() {
        assert!(quadratic().is_convex());
        assert!(CostCurve::Polynomial { coefficients: vec![1.0, 2.0] }.is_convex());
        assert!(!CostCurve::Polynomial { coefficients: vec![0.0, 20.0, -0.5] }.is_convex());
        // A cubic is non-convex whatever its coefficients.
        assert!(!CostCurve::Polynomial { coefficients: vec![0.0, 1.0, 1.0, 1.0] }.is_convex());

        // Rising slopes are convex; falling ones are not.
        assert!(CostCurve::PiecewiseLinear {
            points: vec![(0.0, 0.0), (10.0, 100.0), (20.0, 300.0)]
        }
        .is_convex());
        assert!(!CostCurve::PiecewiseLinear {
            points: vec![(0.0, 0.0), (10.0, 300.0), (20.0, 400.0)]
        }
        .is_convex());
    }

    #[test]
    fn a_zero_rate_reads_as_unlimited_not_as_a_binding_zero() {
        let unlimited = BranchLimit { id: 1, rate_a: 0.0, unlimited: true };
        assert_eq!(unlimited.rate_pu(100.0), None);

        let rated = BranchLimit { id: 2, rate_a: 250.0, unlimited: false };
        assert_eq!(rated.rate_pu(100.0), Some(2.5));
    }

    #[test]
    fn the_document_round_trips_from_json() {
        let text = r#"{
            "version": "1.0", "type": "opf_input", "base_mva": 100.0,
            "generator": [
                {"index": 0, "node": 1, "p_min": 0.0, "p_max": 332.4,
                 "q_min": -16.9, "q_max": 10.0,
                 "cost": {"model": "polynomial", "coefficients": [0.0, 20.0, 0.043]}},
                {"index": 1, "node": 2, "p_min": 0.0, "p_max": 140.0}
            ],
            "branch_limit": [
                {"id": 700, "rate_a": 300.0, "unlimited": false},
                {"id": 701, "rate_a": 0.0, "unlimited": true}
            ],
            "load": [{"id": 800, "node": 3}]
        }"#;
        let data = OpfData::from_json(text).unwrap();

        assert_eq!(data.base_mva, 100.0);
        assert_eq!(data.generator.len(), 2);
        assert_eq!(data.generator[0].p_limits_pu(100.0), (0.0, 3.324));
        assert_eq!(data.generator[0].q_min, Some(-16.9));
        // A generator with no cost curve is legal — its output is simply free.
        assert!(data.generator[1].cost.is_none());
        assert!(data.generator[1].q_max.is_none());

        assert_eq!(data.binding_capable_branches(), 1);
        assert!(data.costs_are_convex());
        assert_eq!(data.load[0].node, 3);
    }
}

use serde::{Deserialize, Serialize};
use num_complex::Complex;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BusType {
    Slack,
    PV,
    PQ,
}

/// A voltage-dependent (ZIP-model) power term: constant power, constant
/// current (S ∝ |V|), or constant impedance (S ∝ |V|²), evaluated at |V|=1.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ZipKind {
    ConstPower,
    ConstCurrent,
    ConstImpedance,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ZipTerm {
    pub s_const: Complex<f64>,
    pub kind: ZipKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bus {
    pub idx: usize,          // index in arrays (0-based)
    pub bus_type: BusType,
    pub voltage_mag: f64,    // Vm (p.u.)
    pub voltage_ang: f64,    // Va (rad)
    pub p_spec: f64,         // P specified (generation - load) in p.u., constant-power part
    pub q_spec: f64,         // Q specified (generation - load) in p.u., constant-power part
    pub q_min: f64,          // PV bus reactive limits, enforced only by
    pub q_max: f64,          // solver::newton_raphson_enforcing_q_limits
    #[serde(default)]
    pub u_rated: f64,        // rated line-to-line voltage in V (0 = not set)
    /// Additional voltage-dependent (constant-current/-impedance) injection terms,
    /// summed on top of `p_spec`/`q_spec` at the current voltage estimate.
    #[serde(default)]
    pub zip_terms: Vec<ZipTerm>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Line {
    pub from: usize,
    pub to: usize,
    pub r: f64,
    pub x: f64,
    pub b_shunt: f64, // total line charging
    #[serde(default)]
    pub g_shunt: f64, // total shunt conductance (CGMES ACLineSegment.gch; usually 0)
}

/// Two-winding transformer parameters in per-unit (system base, to-side voltage base).
///
/// `tap` = k · exp(j · clock · π/6) where k is the off-nominal voltage-magnitude ratio
/// and the argument encodes the vector-group phase shift. `tap.norm()` gives k.
#[derive(Clone, Debug)]
pub struct Transformer {
    pub from: usize,
    pub to: usize,
    pub from_status: u8,
    pub to_status: u8,
    pub y_series: Complex<f64>,
    pub y_shunt: Complex<f64>,
    pub tap: Complex<f64>,
}

/// Three-phase line parameters in per-unit.
/// Positive- and zero-sequence values are stored separately;
/// `build_ybus_3ph` converts them to the phase-domain 3×3 admittance matrix.
/// `b1`/`b0` are the *total* shunt susceptances (ω·c·Z_base); the π-model
/// splits them equally to both terminals, analogous to `Line::b_shunt`.
/// Sequence-domain (0, 1, 2) branch admittance parameters for an asymmetric
/// (3-phase) transformer branch. Each `[Complex<f64>; 4]` holds the four
/// branch entries `[yff, yft, ytf, ytt]` for that sequence; `network::
/// stamp_transformers_3ph` converts them to phase-domain 3×3 blocks via the
/// Fortescue transform and stamps them into the Y-bus.
#[derive(Clone, Debug)]
pub struct Transformer3PhSeq {
    pub from: usize,
    pub to: usize,
    pub y0: [Complex<f64>; 4],
    pub y1: [Complex<f64>; 4],
    pub y2: [Complex<f64>; 4],
}

#[derive(Clone, Debug)]
pub struct Line3Ph {
    pub from: usize, // physical node index
    pub to: usize,
    pub r1: f64,
    pub x1: f64,
    pub b1: f64, // total positive-sequence shunt susceptance (p.u.)
    pub r0: f64,
    pub x0: f64,
    pub b0: f64, // total zero-sequence shunt susceptance (p.u.)
    /// Total positive-sequence shunt *conductance* (p.u.), the dielectric-loss
    /// term power-grid-model writes as `tan δ`: its line shunt is
    /// `2πf·c·(tan δ + j)`, so `g = b · tan δ`.
    ///
    /// Defaulted to zero throughout the symmetric fixtures (which all specify
    /// `tan1 = tan0 = 0`) and therefore invisible to them — but not to a real
    /// cable, and not to `dummy-test-line-into-itself`, which is where its
    /// absence first showed up. Kept as a separate field rather than folding
    /// the shunt into one complex so that `Line3Ph` stays parallel to
    /// [`Line`]'s own `b_shunt`/`g_shunt` pair.
    #[doc(alias = "tan1")]
    pub g1: f64,
    /// Total zero-sequence shunt conductance (p.u.) — `b0 · tan δ₀`.
    #[doc(alias = "tan0")]
    pub g0: f64,
}

/// One machine that holds a voltage, retained.
///
/// [`Bus`] carries only what the Newton system needs: a `PV` bus's magnitude,
/// and the *summed* reactive capability of everything holding it. That is the
/// right input to a power flow and it is all any importer used to keep — a bus
/// held by six machines arrives as one `PV` bus with one `q_min`/`q_max` pair,
/// and which machine produces what is gone.
///
/// Anything that has to *attribute* reactive output needs the discarded half
/// back, exactly as [`TapChanger`] does for tap positions. The bus-level clamp
/// can say "this bus is out of reactive capability"; only the per-machine split
/// can say which machine ran out while the others still had headroom.
///
/// Note that `at_bus` and `controls_bus` may differ — a machine regulating the
/// far side of its own step-up transformer is ordinary. Where they differ the
/// reactive power is produced at `at_bus`, which is *not* where gridoxide's
/// solver currently puts it; see `plans/REACTIVE_DISPATCH_PLAN.md` §5.
#[derive(Clone, Debug, PartialEq)]
pub struct RegulatingMachine {
    /// The source document's own identifier, so a caller can match this back to
    /// the machine it came from rather than to a bus index.
    pub id: String,
    /// Where the machine injects.
    pub at_bus: usize,
    /// The bus whose voltage it holds. Equal to `at_bus` for local control.
    pub controls_bus: usize,
    /// The voltage it holds at `controls_bus`, per-unit on that bus's base.
    pub target_pu: f64,
    /// This machine's own reactive capability, per-unit — not the bus's sum.
    pub q_min: f64,
    pub q_max: f64,
    /// The reactive output the source document scheduled, per-unit. Not a
    /// constraint: at a voltage-controlled bus the solver decides Q. Kept
    /// because it is what has to be subtracted from the bus's solved injection
    /// to leave the part the machines are responsible for.
    pub q_scheduled: f64,
    /// An explicit share, where the document states one. `None` falls back to
    /// capability-proportional and then to uniform — see
    /// `dispatch::reactive_keys`.
    pub key: Option<f64>,
}

/// The discrete tap positions of a transformer, retained.
///
/// [`Transformer::tap`] is one complex number — the ratio and phase shift the
/// Y-bus needs, and nothing else. That is all a power flow wants, and it is
/// why every importer until now computed the position it was told to use and
/// threw the rest away: `network::transformer_tap` takes
/// `(tap_pos, tap_min, tap_max, tap_nom, tap_size, clock)` and returns a single
/// `Complex<f64>`.
///
/// Anything that *moves* a tap needs the discarded half back. A phase-shifter
/// remedial action decides in taps, not degrees; the map between them is
/// nonlinear (and, for an asymmetric changer, not even monotone in the angle),
/// so it cannot be reconstructed from the current value plus a step size.
///
/// `steps[i]` is the value to assign to [`Transformer::tap`] at position
/// `low + i as i32` — the *whole* ratio, nominal turns ratio included, not a
/// multiplier to be applied to something else. That makes
/// [`set_position`](Self::set_position) a straight assignment and leaves no
/// room for a caller to combine the two halves differently from the importer.
#[derive(Clone, Debug, PartialEq)]
pub struct TapChanger {
    /// Lowest valid position (inclusive). Often negative.
    pub low: i32,
    /// The position the network is currently at.
    pub position: i32,
    /// Neutral position — the one whose ratio is nominal. Reported rather than
    /// derived: an asymmetric changer's neutral need not be `(low + high) / 2`.
    pub neutral: i32,
    /// One entry per position, from `low` upwards.
    pub steps: Vec<Complex<f64>>,
    /// The series admittance to assign at each position, when it varies with
    /// the tap. `None` — the common case — means the branch's own `y_series`
    /// holds at every position.
    ///
    /// A CGMES `PhaseTapChangerLinear`/`Symmetrical`/`Asymmetrical` carrying
    /// `xMin`/`xMax` genuinely changes reactance as it moves: the reactance
    /// follows its own trig curve in the tap angle, and IEC's own model says
    /// so. Writing only the complex ratio for such a changer would move the
    /// phase shift while leaving the impedance at whatever position the
    /// document happened to be exported at — a wrong answer that looks
    /// entirely plausible, since the flow still changes in the right
    /// direction.
    pub series: Option<Vec<Complex<f64>>>,
}

impl TapChanger {
    /// Highest valid position (inclusive).
    pub fn high(&self) -> i32 {
        self.low + self.steps.len() as i32 - 1
    }

    /// Number of positions.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The complex tap at `position`, or `None` when out of range.
    pub fn at(&self, position: i32) -> Option<Complex<f64>> {
        if position < self.low {
            return None;
        }
        self.steps.get((position - self.low) as usize).copied()
    }

    /// The complex tap at the current position.
    ///
    /// `None` only if `position` is out of range, which a well-formed changer
    /// never is — but importers read positions from files, so it is reachable.
    pub fn current(&self) -> Option<Complex<f64>> {
        self.at(self.position)
    }

    /// Phase shift at `position`, in degrees. The quantity a phase-shifter
    /// range action's set-point is expressed in.
    pub fn angle_deg(&self, position: i32) -> Option<f64> {
        self.at(position).map(|t| t.arg().to_degrees())
    }

    /// Voltage-magnitude ratio at `position`.
    pub fn ratio(&self, position: i32) -> Option<f64> {
        self.at(position).map(|t| t.norm())
    }

    /// Move to `position` and write the corresponding tap into `transformer`.
    ///
    /// Returns `false` — leaving both the changer and the transformer
    /// untouched — when the position is out of range, so a caller sweeping a
    /// range never silently pins at an endpoint. This mirrors
    /// `NodeBreakerNetwork::set_switch_open`, which reports the same way for
    /// the same reason.
    ///
    /// Writes [`series`](Self::series) too where the changer has one, so a
    /// phase changer whose reactance varies with position stays consistent —
    /// moving the ratio and leaving the impedance behind is a wrong answer
    /// that still moves the flow in the right direction, which is the worst
    /// kind.
    ///
    /// Note what this does *not* do: the Y-bus is not rebuilt. A tap change
    /// alters admittance values but not the sparsity pattern, so a
    /// `PersistentSolver` keeps its symbolic factorization and needs only
    /// `invalidate_admittances` — the same position a switch flip is in.
    /// `outerloop::SolveContext::restamp_ybus` is what does the rebuilding.
    pub fn set_position(&mut self, transformer: &mut Transformer, position: i32) -> bool {
        let Some(tap) = self.at(position) else { return false };
        if let Some(series) = &self.series {
            // Checked before either write, so a malformed changer leaves both
            // halves untouched rather than one.
            let Some(y) = series.get((position - self.low) as usize) else { return false };
            transformer.y_series = *y;
        }
        self.position = position;
        transformer.tap = tap;
        true
    }

    /// The position whose voltage-magnitude ratio is closest to `ratio`.
    ///
    /// The sibling of [`nearest_to_angle`](Self::nearest_to_angle), and what a
    /// voltage-regulating tap changer needs to turn a continuous
    /// \\(\\rho + \\Delta\\rho\\) into a position it can actually take.
    /// Searches the step table rather than dividing by a step size, for the
    /// same reason: the map is not linear, and for a `RatioTapChangerTable` it
    /// is not even regular.
    pub fn nearest_to_ratio(&self, ratio: f64) -> Option<i32> {
        (0..self.steps.len())
            .min_by(|&a, &b| {
                let da = (self.steps[a].norm() - ratio).abs();
                let db = (self.steps[b].norm() - ratio).abs();
                da.total_cmp(&db)
            })
            .map(|i| self.low + i as i32)
    }

    /// The position whose phase shift is closest to `angle_deg`.
    ///
    /// The rounding step every continuous-relaxation answer needs before
    /// anyone can act on it. Searches rather than dividing by a step size
    /// because the tap-to-angle map is not linear.
    pub fn nearest_to_angle(&self, angle_deg: f64) -> Option<i32> {
        (0..self.steps.len())
            .min_by(|&a, &b| {
                let da = (self.steps[a].arg().to_degrees() - angle_deg).abs();
                let db = (self.steps[b].arg().to_degrees() - angle_deg).abs();
                da.total_cmp(&db)
            })
            .map(|i| self.low + i as i32)
    }
}

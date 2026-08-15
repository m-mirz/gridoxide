//! Short-circuit calculation, IEC 60909 style.
//!
//! Answers the question a protection study asks: *if a fault happens here, how
//! much current flows, and what does the rest of the network see while it
//! does?* The initial symmetrical short-circuit current \\(I_k''\\) that comes
//! out is what breaker ratings, conductor sizing and relay coordination are
//! all derived from.
//!
//! # What makes this not a power flow
//!
//! Three things, all of them deliberate simplifications the standard makes so
//! that the answer does not depend on the operating point:
//!
//! - **Loads and generation are ignored entirely.** Not approximated —
//!   ignored. A short-circuit current is a property of the network's
//!   impedances, and IEC 60909 isolates it from whatever the dispatch happened
//!   to be. Shunt admittances *are* kept: a shunt is a passive element, not a
//!   dispatch decision.
//! - **Sources sit behind a fixed EMF**, scaled by the voltage factor `c`
//!   ([`VoltageScaling`]) rather than by their own setpoint.
//! - **The problem is linear.** No iteration, no initial guess, no convergence
//!   to fail — one sparse factorization, exactly like
//!   [`crate::linear::impedance`]. A short-circuit calculation either has a
//!   solvable network or it does not.
//!
//! # Formulation
//!
//! Phase domain (abc), following power-grid-model's `ShortCircuitSolver`. The
//! fault's boundary conditions are stamped directly into the 3N×3N admittance
//! matrix — rows and columns at the faulted bus are rewritten — and the whole
//! thing is solved in one shot:
//!
//! \\[ I_N = Y_{bus} U_N \\]
//!
//! The alternative formulation, and the one IEC 60909 itself is written in, is
//! the sequence domain: build \\(Z_0\\), \\(Z_1\\), \\(Z_2\\) Thevenin
//! equivalents at the fault point and connect them per fault type. The two
//! agree; the phase domain was chosen because it extends to unbalanced faults
//! without special-casing each one, and because power-grid-model's fixtures
//! then cross-validate this implementation exactly.
//!
//! It inherits one limitation from that choice, which power-grid-model
//! documents for itself too: **the network needs a path to ground**. An
//! ungrounded network has a singular zero-sequence system in the phase domain
//! (it does not in the sequence domain), and comes back as
//! [`ShortCircuitError::Singular`].
//!
//! Results are reported in both bases: phase quantities, which is what
//! power-grid-model emits and what the fixtures check, and symmetrical
//! components via [`fortescue`], which is what powsybl's short-circuit API
//! models and what the standard's own vocabulary is written in.

pub mod fortescue;
pub mod solver;

use num_complex::Complex;

pub use fortescue::SequenceValue;
pub use solver::{
    short_circuit, BranchResult, FaultResult, NodeResult, ShortCircuitError,
    ShortCircuitReport, SourceResult,
};

/// The nominal phase angles, in the order power-grid-model's own
/// `ComplexValue<asymmetric_t>` scalar constructor produces: `[x, x·a², x·a]`
/// with `a = exp(j2π/3)`. Phase b lags by 120°, phase c leads by 120°.
pub(crate) const PHASE_ANG: [f64; 3] =
    [0.0, -2.0 * std::f64::consts::PI / 3.0, 2.0 * std::f64::consts::PI / 3.0];

/// Which of the four fault situations IEC 60909 distinguishes.
///
/// The discriminants are power-grid-model's own `FaultType` codes, as they
/// appear on `pgm::PgmFault::fault_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultType {
    /// All three phases shorted together (and to ground). The balanced case,
    /// and usually — though not always — the largest current.
    ThreePhase,
    /// One phase to ground. The most common fault in practice by a wide
    /// margin.
    SinglePhaseToGround,
    /// Two phases shorted to each other, clear of ground.
    TwoPhase,
    /// Two phases shorted to each other *and* to ground.
    TwoPhaseToGround,
}

impl FaultType {
    /// Reads power-grid-model's integer code.
    pub fn from_code(code: i8) -> Result<Self, ShortCircuitError> {
        match code {
            0 => Ok(Self::ThreePhase),
            1 => Ok(Self::SinglePhaseToGround),
            2 => Ok(Self::TwoPhase),
            3 => Ok(Self::TwoPhaseToGround),
            other => Err(ShortCircuitError::UnknownFaultType(other)),
        }
    }

    /// The phase combination this fault type assumes when the document leaves
    /// `fault_phase` at its default sentinel.
    ///
    /// Mirrors power-grid-model's `Fault::get_fault_phase`. **Note that the
    /// two two-phase types default to `bc`, not `ab`** — most of the reference
    /// fixtures rely on this, and assuming `ab` would put the current on the
    /// wrong pair of phases while still producing an entirely plausible
    /// magnitude.
    pub fn default_phase(self) -> FaultPhase {
        match self {
            Self::ThreePhase => FaultPhase::Abc,
            Self::SinglePhaseToGround => FaultPhase::A,
            Self::TwoPhase | Self::TwoPhaseToGround => FaultPhase::Bc,
        }
    }
}

/// Which phase or phase pair a fault involves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultPhase {
    Abc,
    A,
    B,
    C,
    Ab,
    Ac,
    Bc,
}

impl FaultPhase {
    /// Reads power-grid-model's integer code. `-1` is its `default_value`
    /// sentinel, which is not a phase — it is resolved against the fault type
    /// by [`FaultType::default_phase`], so it is returned as `None` here
    /// rather than guessed at.
    pub fn from_code(code: i8) -> Result<Option<Self>, ShortCircuitError> {
        match code {
            -1 => Ok(None),
            0 => Ok(Some(Self::Abc)),
            1 => Ok(Some(Self::A)),
            2 => Ok(Some(Self::B)),
            3 => Ok(Some(Self::C)),
            4 => Ok(Some(Self::Ab)),
            5 => Ok(Some(Self::Ac)),
            6 => Ok(Some(Self::Bc)),
            other => Err(ShortCircuitError::UnknownFaultPhase(other)),
        }
    }

    /// The zero-based phase indices this involves, as
    /// power-grid-model's `set_phase_index` computes them. `Abc` involves all
    /// three and so names none individually.
    pub(crate) fn indices(self) -> (Option<usize>, Option<usize>) {
        match self {
            Self::Abc => (None, None),
            Self::A => (Some(0), None),
            Self::B => (Some(1), None),
            Self::C => (Some(2), None),
            Self::Ab => (Some(0), Some(1)),
            Self::Ac => (Some(0), Some(2)),
            Self::Bc => (Some(1), Some(2)),
        }
    }
}

/// Whether to compute the maximum or the minimum short-circuit current.
///
/// The two are different studies with different purposes: the maximum sizes
/// equipment for the worst mechanical and thermal stress it must survive, the
/// minimum checks that protection will still reliably *detect* a fault under
/// the least favourable conditions. IEC 60909 expresses both through a single
/// voltage factor `c` applied to the source EMF — see [`c_factor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VoltageScaling {
    #[default]
    Maximum,
    Minimum,
}

/// The IEC 60909 voltage scaling factor `c`, for a source on a network of the
/// given rated line-to-line voltage.
///
/// | | `c_max` | `c_min` |
/// |---|---|---|
/// | `U_nom` ≤ 1 kV | 1.10 | 0.95 |
/// | `U_nom` > 1 kV | 1.10 | 1.00 |
///
/// The standard distinguishes a 6% and a 10% voltage tolerance for the
/// low-voltage `c_min`; like power-grid-model, this assumes 10%.
pub fn c_factor(u_rated: f64, scaling: VoltageScaling) -> f64 {
    match scaling {
        VoltageScaling::Maximum => 1.10,
        VoltageScaling::Minimum if u_rated <= 1000.0 => 0.95,
        VoltageScaling::Minimum => 1.00,
    }
}

/// A fault's admittance to the fault point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FaultAdmittance {
    /// A dead short — zero impedance, therefore infinite admittance.
    ///
    /// Kept as its own variant rather than an actual `f64::INFINITY` because
    /// the two cases are solved by genuinely different matrix surgery, not by
    /// the same formula evaluated at a limit: a bolted fault *replaces* the
    /// faulted bus's equations rather than adding to them. Letting an infinity
    /// into the matrix would produce `NaN`, not a large current.
    ///
    /// This is the default for a `fault` entry that specifies no `r_f`/`x_f`.
    Bolted,
    /// A fault through a finite admittance, per unit.
    Finite(Complex<f64>),
}

/// One fault, resolved against the network it applies to.
#[derive(Clone, Copy, Debug)]
pub struct Fault {
    /// The document's own `fault` id, carried through to the result.
    pub id: u64,
    /// Physical node index of the faulted bus.
    pub bus: usize,
    pub fault_type: FaultType,
    pub fault_phase: FaultPhase,
    pub y_fault: FaultAdmittance,
}

/// Options for a short-circuit calculation.
///
/// Deliberately small, and for the same reason [`crate::linear::DcOptions`] is:
/// a direct solve has no tolerance and no iteration limit. The only real
/// choice IEC 60909 leaves open at this level is maximum versus minimum
/// current.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShortCircuitOptions {
    pub scaling: VoltageScaling,
}

/// Resolves a document's `fault` entries against the network they apply to.
///
/// Inactive faults are dropped. `fault_phase` left at its default sentinel is
/// resolved through [`FaultType::default_phase`], and a zero `r_f`/`x_f` pair
/// becomes [`FaultAdmittance::Bolted`] rather than a zero admittance — see
/// [`crate::pgm::PgmFault`] for why that reading is the right one.
///
/// The fault admittance is per-unit against the **faulted node's** own base,
/// `z_base = u_rated² / s_base`, which is why this needs the network and not
/// just the document.
pub fn faults_from_pgm(
    input: &crate::pgm::PgmInput,
    net: &crate::pgm::ScNetwork3Ph,
) -> Result<Vec<Fault>, ShortCircuitError> {
    let mut out = Vec::new();
    for f in input.data.fault.iter().filter(|f| f.status != 0) {
        let bus = *net
            .node_idx
            .get(&f.fault_object)
            .ok_or(ShortCircuitError::UnknownFaultObject(f.fault_object))?;
        let fault_type = FaultType::from_code(f.fault_type)?;
        let fault_phase =
            FaultPhase::from_code(f.fault_phase)?.unwrap_or_else(|| fault_type.default_phase());

        let y_fault = if f.r_f == 0.0 && f.x_f == 0.0 {
            FaultAdmittance::Bolted
        } else {
            let u_rated = net.u_rated[bus];
            let z_base = u_rated * u_rated / net.s_base_va;
            FaultAdmittance::Finite(z_base / Complex::new(f.r_f, f.x_f))
        };

        out.push(Fault { id: f.id, bus, fault_type, fault_phase, y_fault });
    }
    Ok(out)
}

/// Runs a short-circuit calculation straight from a PGM-format document.
///
/// The shortest path from a file to an answer; [`short_circuit`] is the entry
/// point for callers that already hold a network.
pub fn short_circuit_from_pgm(
    input: &crate::pgm::PgmInput,
    s_base_va: f64,
    freq_hz: f64,
    opts: ShortCircuitOptions,
) -> Result<(crate::pgm::ScNetwork3Ph, ShortCircuitReport), ShortCircuitError> {
    let net = crate::pgm::pgm_to_3ph_sc_network(input, s_base_va, freq_hz);
    let faults = faults_from_pgm(input, &net)?;
    let report = short_circuit(&net, &faults, opts)?;
    Ok((net, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `c` table, at every branch including the 1 kV boundary itself.
    /// No fixture exercises the low-voltage `c_min`, so this is the only thing
    /// pinning it.
    #[test]
    fn the_voltage_scaling_factor_follows_the_iec_table() {
        assert_eq!(c_factor(400.0, VoltageScaling::Maximum), 1.10);
        assert_eq!(c_factor(150_000.0, VoltageScaling::Maximum), 1.10);

        assert_eq!(c_factor(400.0, VoltageScaling::Minimum), 0.95);
        // The boundary is inclusive on the low-voltage side.
        assert_eq!(c_factor(1000.0, VoltageScaling::Minimum), 0.95);
        assert_eq!(c_factor(1000.1, VoltageScaling::Minimum), 1.00);
        assert_eq!(c_factor(150_000.0, VoltageScaling::Minimum), 1.00);
    }

    /// The default phase for the two two-phase fault types is `bc`. Getting
    /// this wrong puts a correct-looking magnitude on the wrong conductors.
    #[test]
    fn two_phase_faults_default_to_bc() {
        assert_eq!(FaultType::TwoPhase.default_phase(), FaultPhase::Bc);
        assert_eq!(FaultType::TwoPhaseToGround.default_phase(), FaultPhase::Bc);
        assert_eq!(FaultType::ThreePhase.default_phase(), FaultPhase::Abc);
        assert_eq!(FaultType::SinglePhaseToGround.default_phase(), FaultPhase::A);
    }

    #[test]
    fn the_default_phase_sentinel_is_not_mistaken_for_a_phase() {
        assert_eq!(FaultPhase::from_code(-1).unwrap(), None);
        assert_eq!(FaultPhase::from_code(0).unwrap(), Some(FaultPhase::Abc));
        assert!(FaultPhase::from_code(9).is_err());
    }

    #[test]
    fn phase_indices_match_the_reference_mapping() {
        assert_eq!(FaultPhase::Abc.indices(), (None, None));
        assert_eq!(FaultPhase::A.indices(), (Some(0), None));
        assert_eq!(FaultPhase::C.indices(), (Some(2), None));
        assert_eq!(FaultPhase::Ab.indices(), (Some(0), Some(1)));
        assert_eq!(FaultPhase::Ac.indices(), (Some(0), Some(2)));
        assert_eq!(FaultPhase::Bc.indices(), (Some(1), Some(2)));
    }
}

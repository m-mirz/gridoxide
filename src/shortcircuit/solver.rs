//! Matrix assembly, fault stamping and the solve.
//!
//! Transcribed from power-grid-model's
//! `math_solver/short_circuit_solver.hpp`, whose structure this follows
//! closely on purpose: the fault boundary conditions are fiddly, every one of
//! its cases is exercised by a fixture in `tests/data/pgm/short_circuit/`, and
//! a formulation that merely looks reasonable will produce plausible numbers
//! that are wrong.

use std::collections::HashMap;

use num_complex::Complex;

use crate::network::{
    build_ybus_3ph, connected_components, line3ph_blocks, seq_to_phase_shunt,
    stamp_shunts_3ph, stamp_transformers_3ph, transformer3ph_blocks,
};
use crate::pgm::ScNetwork3Ph;
use crate::sparse;

use super::{
    c_factor, Fault, FaultAdmittance, FaultType, ShortCircuitOptions, PHASE_ANG,
};

const ZERO: Complex<f64> = Complex::new(0.0, 0.0);

/// Why a short-circuit calculation could not be carried out.
#[derive(Clone, Debug, PartialEq)]
pub enum ShortCircuitError {
    /// The faults in one calculation disagree about type or phase.
    ///
    /// power-grid-model imposes the same restriction, and it is not
    /// arbitrary: the boundary conditions of different fault types rewrite the
    /// same matrix entries in incompatible ways, so "one of each, at once" is
    /// not a well-posed single linear system. Several faults of the *same*
    /// type, at the same or different buses, are fine.
    MixedFaultTypes,
    /// A `fault` names a `fault_object` that is not a node in this document.
    UnknownFaultObject(u64),
    UnknownFaultType(i8),
    UnknownFaultPhase(i8),
    /// The assembled system has no unique solution. In the phase domain the
    /// usual cause is a network with no path to ground — see the module docs.
    Singular,
}

impl std::fmt::Display for ShortCircuitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MixedFaultTypes => write!(
                f,
                "all faults in one short-circuit calculation must share a fault type and phase"
            ),
            Self::UnknownFaultObject(id) => {
                write!(f, "fault_object {id} is not a node in this document")
            }
            Self::UnknownFaultType(c) => write!(f, "unknown fault_type {c}"),
            Self::UnknownFaultPhase(c) => write!(f, "unknown fault_phase {c}"),
            Self::Singular => write!(
                f,
                "the short-circuit system is singular — in the phase domain this usually \
                 means the network has no path to ground"
            ),
        }
    }
}

impl std::error::Error for ShortCircuitError {}

/// One node's short-circuit voltages.
#[derive(Clone, Debug)]
pub struct NodeResult {
    /// Per-phase voltage magnitude, per unit.
    pub u_pu: [f64; 3],
    /// Per-phase voltage angle, radians.
    pub u_angle: [f64; 3],
    /// Per-phase voltage magnitude in volts, **line-to-neutral** — the
    /// asymmetric convention, `u_pu · u_rated / √3`.
    pub u: [f64; 3],
    /// Whether this node has a path to any active source.
    pub energized: bool,
}

/// One fault's current.
#[derive(Clone, Debug)]
pub struct FaultResult {
    pub id: u64,
    /// Per-phase fault-current magnitude, amperes.
    pub i_f: [f64; 3],
    /// Per-phase fault-current angle, radians.
    pub i_f_angle: [f64; 3],
}

/// One branch's terminal currents.
#[derive(Clone, Debug)]
pub struct BranchResult {
    pub i_from: [f64; 3],
    pub i_from_angle: [f64; 3],
    pub i_to: [f64; 3],
    pub i_to_angle: [f64; 3],
}

/// One source's current contribution.
#[derive(Clone, Debug)]
pub struct SourceResult {
    pub id: u64,
    pub i: [f64; 3],
    pub i_angle: [f64; 3],
}

/// Everything a short-circuit calculation produces.
#[derive(Clone, Debug)]
pub struct ShortCircuitReport {
    /// Indexed by physical node.
    pub nodes: Vec<NodeResult>,
    /// In the order the faults were supplied.
    pub faults: Vec<FaultResult>,
    /// Lines first, then transformers — the flat branch order the rest of the
    /// crate uses.
    pub branches: Vec<BranchResult>,
    pub sources: Vec<SourceResult>,
    /// Complex per-phase node voltages, per unit, kept so callers can derive
    /// their own quantities (and so [`super::fortescue`] can project them)
    /// without re-deriving them from magnitude and angle.
    pub u_bus: Vec<[Complex<f64>; 3]>,
}

/// Runs a short-circuit calculation.
///
/// `faults` may be empty, in which case this is simply the network's
/// pre-fault state under the scaled source EMFs.
pub fn short_circuit(
    net: &ScNetwork3Ph,
    faults: &[Fault],
    opts: ShortCircuitOptions,
) -> Result<ShortCircuitReport, ShortCircuitError> {
    check_faults_agree(faults)?;

    let n = net.n_nodes;
    let dim = 3 * n;

    // ── Assembly ─────────────────────────────────────────────────────────
    // The passive network first, reusing the same Y-bus builders the
    // asymmetric power flow uses — a short circuit sees the same lines,
    // transformers and shunts, which is the whole reason this is cheap.
    let mut ybus = build_ybus_3ph(n, &net.lines);
    stamp_transformers_3ph(&mut ybus, &net.transformers);
    stamp_shunts_3ph(&mut ybus, &net.shunts);
    let mut entries = ybus.into_entries();
    let mut rhs = vec![ZERO; dim];

    // Then the sources, as Norton equivalents: internal admittance onto the
    // diagonal, EMF into the right-hand side.
    //
    // **Before the faults, not after.** A bolted fault zeroes the faulted
    // bus's whole column, and it must zero the source's contribution to that
    // column along with everything else. Reversing these two loops would
    // leave a source stamped into a column that is supposed to be empty.
    let source_y: Vec<[[Complex<f64>; 3]; 3]> = net
        .sources
        .iter()
        .map(|src| seq_to_phase_shunt(src.y1, src.y0))
        .collect();
    let source_emf: Vec<[Complex<f64>; 3]> = net
        .sources
        .iter()
        .map(|src| {
            // IEC 60909's source EMF is `c · e^{jθ}`. It is **not** scaled by
            // the source's own `u_ref`: the voltage factor replaces the
            // setpoint rather than multiplying it, which is what isolates the
            // result from the operating point. Every reference fixture has
            // `u_ref = 1.0`, so this distinction is invisible to them and
            // shows up only on real data — see this module's tests.
            let c = c_factor(src.u_rated, opts.scaling);
            [0, 1, 2].map(|p| Complex::from_polar(c, src.u_ref_angle + PHASE_ANG[p]))
        })
        .collect();

    for (s, src) in net.sources.iter().enumerate() {
        let y = &source_y[s];
        let u = &source_emf[s];
        for p in 0..3 {
            for q in 0..3 {
                entries.push((3 * src.node + p, 3 * src.node + q, y[p][q]));
                rhs[3 * src.node + p] += y[p][q] * u[q];
            }
        }
    }

    // ── Fault stamping ───────────────────────────────────────────────────
    //
    // A fault on a de-energized bus is not a fault: with no source feeding it
    // there is nothing to drive a current, and power-grid-model says the same
    // thing by handing such a fault a zero admittance
    // (`Fault::calc_param`'s `energized` guard). Skipping it here keeps it out
    // of the matrix entirely, and it comes back with zero current below.
    let energized = energized_nodes(net);
    let mut faults_by_bus: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, f) in faults.iter().enumerate() {
        if energized[f.bus] {
            faults_by_bus.entry(f.bus).or_default().push(i);
        }
    }
    // How many bolted faults sit on each bus. A bolted fault short-circuits
    // the bus outright, so any further fault there — bolted or not — is
    // shorted out by it; the count is what splits the resulting current
    // between several bolted faults that share a bus.
    let mut bolted_count: HashMap<usize, usize> = HashMap::new();

    for (&bus, fault_idx) in &faults_by_bus {
        for &i in fault_idx {
            let f = &faults[i];
            match f.y_fault {
                FaultAdmittance::Bolted => {
                    *bolted_count.entry(bus).or_insert(0) += 1;
                    stamp_bolted(&mut entries, &mut rhs, bus, f);
                    // Once a bus is dead-shorted there is nothing further to
                    // stamp there — every other fault on it is bypassed.
                    break;
                }
                FaultAdmittance::Finite(y) => {
                    stamp_finite(&mut entries, &mut rhs, bus, f, y);
                }
            }
        }
    }

    // ── Solve ────────────────────────────────────────────────────────────
    //
    // Only over the energized rows. A component with no source has neither a
    // source admittance nor any other reference to ground, so including it
    // would make the whole system singular and lose the answer for the parts
    // that *are* solvable — the same reason every other solver in this crate
    // partitions before solving (see `network::mark_unreferenced_islands`).
    // De-energized buses are pinned at zero rather than invented, which is
    // this crate's established contract for them.
    //
    // Energization follows the same connectivity the Y-bus does, so the
    // matrix is block-diagonal across the split and dropping the other block
    // changes nothing about this one.
    let mut reduced_pos: Vec<Option<usize>> = vec![None; dim];
    let mut n_reduced = 0;
    for k in 0..n {
        if energized[k] {
            for p in 0..3 {
                reduced_pos[3 * k + p] = Some(n_reduced);
                n_reduced += 1;
            }
        }
    }

    let reduced_entries: Vec<(usize, usize, Complex<f64>)> = entries
        .iter()
        .filter_map(|&(r, c, v)| Some((reduced_pos[r]?, reduced_pos[c]?, v)))
        .collect();
    let reduced_rhs: Vec<Complex<f64>> =
        (0..dim).filter(|&i| reduced_pos[i].is_some()).map(|i| rhs[i]).collect();

    let mut u_bus = vec![[ZERO; 3]; n];
    if n_reduced > 0 {
        let sol = sparse::solve_complex(n_reduced, &reduced_entries, &reduced_rhs)
            .ok_or(ShortCircuitError::Singular)?;
        for k in 0..n {
            for p in 0..3 {
                if let Some(r) = reduced_pos[3 * k + p] {
                    u_bus[k][p] = sol[r];
                }
            }
        }
    }

    // ── Results ──────────────────────────────────────────────────────────
    let mut fault_current = vec![[ZERO; 3]; faults.len()];
    resolve_fault_currents(
        net, faults, &faults_by_bus, &bolted_count, &source_y, &source_emf,
        &mut u_bus, &mut fault_current,
    );

    Ok(assemble_report(net, faults, &fault_current, &source_y, &source_emf, u_bus, &energized))
}

/// power-grid-model's `check_input_valid`: one calculation, one fault type and
/// phase.
fn check_faults_agree(faults: &[Fault]) -> Result<(), ShortCircuitError> {
    let Some(first) = faults.first() else { return Ok(()) };
    if faults
        .iter()
        .any(|f| f.fault_type != first.fault_type || f.fault_phase != first.fault_phase)
    {
        return Err(ShortCircuitError::MixedFaultTypes);
    }
    Ok(())
}

/// Removes every triplet in global column `col`.
///
/// This is why the matrix is carried as triplets rather than as a finished
/// `YBusSparse`: zeroing a column is a *filter*, and it has to be one.
/// Duplicate `(row, col)` triplets are summed downstream, so "adding a
/// compensating negative entry" would cancel only the entries it was computed
/// against and silently leave any other contribution to that column standing.
fn zero_column(entries: &mut Vec<(usize, usize, Complex<f64>)>, col: usize) {
    entries.retain(|&(_, c, _)| c != col);
}

/// Adds column `from` into column `into`, then empties column `from`.
fn fold_column(
    entries: &mut Vec<(usize, usize, Complex<f64>)>,
    into: usize,
    from: usize,
) {
    let folded: Vec<(usize, usize, Complex<f64>)> = entries
        .iter()
        .filter(|&&(_, c, _)| c == from)
        .map(|&(r, _, v)| (r, into, v))
        .collect();
    entries.extend(folded);
    zero_column(entries, from);
}

/// Stamps a bolted (zero-impedance) fault.
///
/// The faulted bus's own equations are *replaced* rather than added to: its
/// voltage is known to be zero, so the unknown at that position becomes the
/// injected current instead, which is what the `-1` diagonal expresses.
fn stamp_bolted(
    entries: &mut Vec<(usize, usize, Complex<f64>)>,
    rhs: &mut [Complex<f64>],
    bus: usize,
    f: &Fault,
) {
    let (p1, p2) = f.fault_phase.indices();
    match f.fault_type {
        FaultType::ThreePhase => {
            for p in 0..3 {
                zero_column(entries, 3 * bus + p);
                entries.push((3 * bus + p, 3 * bus + p, -Complex::new(1.0, 0.0)));
                rhs[3 * bus + p] = ZERO;
            }
        }
        FaultType::SinglePhaseToGround => {
            let p1 = p1.expect("a single-phase fault names its phase");
            zero_column(entries, 3 * bus + p1);
            entries.push((3 * bus + p1, 3 * bus + p1, -Complex::new(1.0, 0.0)));
            rhs[3 * bus + p1] = ZERO;
        }
        FaultType::TwoPhase => {
            // The two phases are tied together but not to ground, so the
            // constraint is `u_p1 = u_p2` rather than either being zero —
            // hence a fold rather than two independent column erasures.
            let (p1, p2) = two_phase_indices(p1, p2);
            fold_column(entries, 3 * bus + p1, 3 * bus + p2);
            entries.push((3 * bus + p1, 3 * bus + p2, -Complex::new(1.0, 0.0)));
            entries.push((3 * bus + p2, 3 * bus + p2, Complex::new(1.0, 0.0)));
            let carried = rhs[3 * bus + p1];
            rhs[3 * bus + p2] += carried;
            rhs[3 * bus + p1] = ZERO;
        }
        FaultType::TwoPhaseToGround => {
            let (p1, p2) = two_phase_indices(p1, p2);
            for p in [p1, p2] {
                zero_column(entries, 3 * bus + p);
                entries.push((3 * bus + p, 3 * bus + p, -Complex::new(1.0, 0.0)));
                rhs[3 * bus + p] = ZERO;
            }
        }
    }
}

/// Stamps a fault through a finite admittance. Except for the two-phase-to-
/// ground case these are pure diagonal-block additions — the fault is just
/// another admittance to ground.
fn stamp_finite(
    entries: &mut Vec<(usize, usize, Complex<f64>)>,
    rhs: &mut [Complex<f64>],
    bus: usize,
    f: &Fault,
    y: Complex<f64>,
) {
    let (p1, p2) = f.fault_phase.indices();
    match f.fault_type {
        FaultType::ThreePhase => {
            for p in 0..3 {
                entries.push((3 * bus + p, 3 * bus + p, y));
            }
        }
        FaultType::SinglePhaseToGround => {
            let p1 = p1.expect("a single-phase fault names its phase");
            entries.push((3 * bus + p1, 3 * bus + p1, y));
        }
        FaultType::TwoPhase => {
            let (p1, p2) = two_phase_indices(p1, p2);
            entries.push((3 * bus + p1, 3 * bus + p1, y));
            entries.push((3 * bus + p2, 3 * bus + p2, y));
            entries.push((3 * bus + p1, 3 * bus + p2, -y));
            entries.push((3 * bus + p2, 3 * bus + p1, -y));
        }
        FaultType::TwoPhaseToGround => {
            // Two constraints at once — the phases are tied to each other
            // (the fold) *and* the pair is tied to ground through `y` — so
            // this one needs the column surgery even with finite impedance.
            let (p1, p2) = two_phase_indices(p1, p2);
            fold_column(entries, 3 * bus + p1, 3 * bus + p2);
            entries.push((3 * bus + p1, 3 * bus + p2, -Complex::new(1.0, 0.0)));
            entries.push((3 * bus + p2, 3 * bus + p2, Complex::new(1.0, 0.0)));
            // Column p1 was *not* zeroed, so this is a genuine addition.
            entries.push((3 * bus + p2, 3 * bus + p1, y));
            let carried = rhs[3 * bus + p1];
            rhs[3 * bus + p2] += carried;
            rhs[3 * bus + p1] = ZERO;
        }
    }
}

fn two_phase_indices(p1: Option<usize>, p2: Option<usize>) -> (usize, usize) {
    (
        p1.expect("a two-phase fault names its first phase"),
        p2.expect("a two-phase fault names its second phase"),
    )
}

/// Recovers fault currents and restores the true voltages at faulted buses.
///
/// At a faulted bus the solved vector does not hold a voltage in the phases
/// the fault constrains — the bolted stamping swapped that unknown for an
/// injected current. This reads the current back out and writes the known
/// voltage (zero, or the tied partner's) into its place, which is why
/// `u_bus` is taken by `&mut`.
#[allow(clippy::too_many_arguments)]
fn resolve_fault_currents(
    net: &ScNetwork3Ph,
    faults: &[Fault],
    faults_by_bus: &HashMap<usize, Vec<usize>>,
    bolted_count: &HashMap<usize, usize>,
    source_y: &[[[Complex<f64>; 3]; 3]],
    source_emf: &[[Complex<f64>; 3]],
    u_bus: &mut [[Complex<f64>; 3]],
    fault_current: &mut [[Complex<f64>; 3]],
) {
    for (&bus, fault_idx) in faults_by_bus {
        // Snapshot before the loop below starts rewriting the bus, exactly as
        // the reference does: every fault at this bus reads the *same*
        // pre-restoration values.
        let subtotal = u_bus[bus];
        let n_bolted = *bolted_count.get(&bus).unwrap_or(&0);
        let split = n_bolted as f64;

        for &i in fault_idx {
            let f = &faults[i];
            let (p1, p2) = f.fault_phase.indices();
            let i_f = &mut fault_current[i];

            match f.y_fault {
                FaultAdmittance::Bolted => match f.fault_type {
                    FaultType::ThreePhase => {
                        for p in 0..3 {
                            i_f[p] = -subtotal[p] / split;
                            u_bus[bus][p] = ZERO;
                        }
                    }
                    FaultType::SinglePhaseToGround => {
                        let p1 = p1.unwrap();
                        i_f[p1] = -subtotal[p1] / split;
                        u_bus[bus][p1] = ZERO;
                    }
                    FaultType::TwoPhase => {
                        let (p1, p2) = two_phase_indices(p1, p2);
                        i_f[p1] = -subtotal[p2] / split;
                        i_f[p2] = -i_f[p1];
                        u_bus[bus][p2] = u_bus[bus][p1];
                    }
                    FaultType::TwoPhaseToGround => {
                        let (p1, p2) = two_phase_indices(p1, p2);
                        i_f[p1] = -subtotal[p1] / split;
                        i_f[p2] = -subtotal[p2] / split;
                        u_bus[bus][p1] = ZERO;
                        u_bus[bus][p2] = ZERO;
                    }
                },
                FaultAdmittance::Finite(y) => {
                    // A bolted fault on the same bus shorts this one out
                    // entirely: it sits across zero volts and carries nothing.
                    if n_bolted > 0 {
                        continue;
                    }
                    match f.fault_type {
                        FaultType::ThreePhase => {
                            for p in 0..3 {
                                i_f[p] = y * subtotal[p];
                            }
                        }
                        FaultType::SinglePhaseToGround => {
                            let p1 = p1.unwrap();
                            i_f[p1] = y * subtotal[p1];
                        }
                        FaultType::TwoPhase => {
                            let (p1, p2) = two_phase_indices(p1, p2);
                            i_f[p1] = y * subtotal[p1] - y * subtotal[p2];
                            i_f[p2] = y * subtotal[p2] - y * subtotal[p1];
                        }
                        FaultType::TwoPhaseToGround => {
                            let (p1, p2) = two_phase_indices(p1, p2);
                            i_f[p1] = -subtotal[p2];
                            i_f[p2] = -i_f[p1] + y * subtotal[p1];
                            u_bus[bus][p2] = u_bus[bus][p1];
                        }
                    }
                }
            }
        }

        // Source currents at this bus, read against the *restored* voltages.
        let mut i_source_bus = [ZERO; 3];
        let mut i_source_inject = [ZERO; 3];
        for (s, src) in net.sources.iter().enumerate() {
            if src.node != bus {
                continue;
            }
            let (inject, out) = source_currents(&source_y[s], &source_emf[s], &u_bus[bus]);
            for p in 0..3 {
                i_source_bus[p] += out[p];
                i_source_inject[p] += inject[p];
            }
        }

        // A source sitting on the faulted bus feeds the fault directly,
        // without passing through any branch — so its contribution never
        // appeared in the solve above and has to be added back here.
        for &i in fault_idx {
            let f = &faults[i];
            let (p1, p2) = f.fault_phase.indices();
            let i_f = &mut fault_current[i];
            match f.y_fault {
                FaultAdmittance::Bolted => match f.fault_type {
                    FaultType::ThreePhase => {
                        for p in 0..3 {
                            i_f[p] += i_source_bus[p] / split;
                        }
                    }
                    FaultType::SinglePhaseToGround => {
                        let p1 = p1.unwrap();
                        i_f[p1] += i_source_bus[p1] / split;
                    }
                    FaultType::TwoPhase => {
                        let (p1, p2) = two_phase_indices(p1, p2);
                        i_f[p1] += i_source_inject[p1] / split;
                        i_f[p2] -= i_source_inject[p1] / split;
                    }
                    FaultType::TwoPhaseToGround => {
                        let (p1, p2) = two_phase_indices(p1, p2);
                        i_f[p1] += i_source_bus[p1];
                        i_f[p2] += i_source_bus[p2];
                    }
                },
                FaultAdmittance::Finite(_) => {
                    if f.fault_type == FaultType::TwoPhaseToGround && n_bolted == 0 {
                        let (p1, p2) = two_phase_indices(p1, p2);
                        let finite = fault_idx.len() as f64;
                        i_f[p1] += i_source_inject[p1] / finite;
                        i_f[p2] -= i_source_inject[p1] / finite;
                    }
                }
            }
        }
    }

    // Sources on unfaulted buses still have a current worth reporting; those
    // are computed in `assemble_report`, which sees every source.
}

/// A source's raw Norton injection and its net current into the bus.
fn source_currents(
    y: &[[Complex<f64>; 3]; 3],
    emf: &[Complex<f64>; 3],
    u: &[Complex<f64>; 3],
) -> ([Complex<f64>; 3], [Complex<f64>; 3]) {
    let mut inject = [ZERO; 3];
    let mut out = [ZERO; 3];
    for p in 0..3 {
        for q in 0..3 {
            inject[p] += y[p][q] * emf[q];
            out[p] += y[p][q] * (emf[q] - u[q]);
        }
    }
    (inject, out)
}

fn assemble_report(
    net: &ScNetwork3Ph,
    faults: &[Fault],
    fault_current: &[[Complex<f64>; 3]],
    source_y: &[[[Complex<f64>; 3]; 3]],
    source_emf: &[[Complex<f64>; 3]],
    u_bus: Vec<[Complex<f64>; 3]>,
    energized: &[bool],
) -> ShortCircuitReport {
    let sqrt3 = 3.0_f64.sqrt();

    let nodes = (0..net.n_nodes)
        .map(|k| {
            let u_rated = net.u_rated[k];
            NodeResult {
                u_pu: [0, 1, 2].map(|p| u_bus[k][p].norm()),
                u_angle: [0, 1, 2].map(|p| u_bus[k][p].arg()),
                u: [0, 1, 2].map(|p| u_bus[k][p].norm() * u_rated / sqrt3),
                energized: energized[k],
            }
        })
        .collect();

    // Per-unit currents become amperes against the faulted node's own base.
    // The asymmetric base power is `s_base/3` against a line-to-neutral
    // voltage base of `u_rated/√3`, which is the same as `s_base/(√3·u_rated)`.
    let i_base = |u_rated: f64| net.s_base_va / (sqrt3 * u_rated);

    let fault_results = faults
        .iter()
        .zip(fault_current)
        .map(|(f, i)| {
            let base = i_base(net.u_rated[f.bus]);
            FaultResult {
                id: f.id,
                i_f: [0, 1, 2].map(|p| i[p].norm() * base),
                i_f_angle: [0, 1, 2].map(|p| i[p].arg()),
            }
        })
        .collect();

    let sources = net
        .sources
        .iter()
        .enumerate()
        .map(|(s, src)| {
            let (_, out) = source_currents(&source_y[s], &source_emf[s], &u_bus[src.node]);
            let base = i_base(src.u_rated);
            SourceResult {
                id: src.id,
                i: [0, 1, 2].map(|p| out[p].norm() * base),
                i_angle: [0, 1, 2].map(|p| out[p].arg()),
            }
        })
        .collect();

    let branches = branch_currents(net, &u_bus, sqrt3);

    ShortCircuitReport { nodes, faults: fault_results, branches, sources, u_bus }
}

/// Terminal currents for every branch, lines first then transformers — the
/// same flat order `branch_flow::branch_params` uses.
fn branch_currents(
    net: &ScNetwork3Ph,
    u_bus: &[[Complex<f64>; 3]],
    sqrt3: f64,
) -> Vec<BranchResult> {
    let mut out = Vec::with_capacity(net.lines.len() + net.transformers.len());

    let push = |from: usize,
                    to: usize,
                    blocks: [[[Complex<f64>; 3]; 3]; 4],
                    out: &mut Vec<BranchResult>| {
        let [yff, yft, ytf, ytt] = blocks;
        let mut i_from = [ZERO; 3];
        let mut i_to = [ZERO; 3];
        for p in 0..3 {
            for q in 0..3 {
                i_from[p] += yff[p][q] * u_bus[from][q] + yft[p][q] * u_bus[to][q];
                i_to[p] += ytf[p][q] * u_bus[from][q] + ytt[p][q] * u_bus[to][q];
            }
        }
        // Each terminal is referred to its own node's base.
        let base_from = net.s_base_va / (sqrt3 * net.u_rated[from]);
        let base_to = net.s_base_va / (sqrt3 * net.u_rated[to]);
        out.push(BranchResult {
            i_from: [0, 1, 2].map(|p| i_from[p].norm() * base_from),
            i_from_angle: [0, 1, 2].map(|p| i_from[p].arg()),
            i_to: [0, 1, 2].map(|p| i_to[p].norm() * base_to),
            i_to_angle: [0, 1, 2].map(|p| i_to[p].arg()),
        });
    };

    for ln in &net.lines {
        push(ln.from, ln.to, line3ph_blocks(ln), &mut out);
    }
    for t in &net.transformers {
        push(t.from, t.to, transformer3ph_blocks(t), &mut out);
    }
    out
}

/// Which nodes have a path to an active source, over the passive network's own
/// connectivity.
fn energized_nodes(net: &ScNetwork3Ph) -> Vec<bool> {
    let mut ybus = build_ybus_3ph(net.n_nodes, &net.lines);
    stamp_transformers_3ph(&mut ybus, &net.transformers);
    let finished = ybus.finish();
    let components = connected_components(&finished);

    let mut energized = vec![false; net.n_nodes];
    for comp in &components {
        // A component is a set of *phase* rows; a physical node is energized
        // if any of its phases sits in a component holding a source.
        let has_source = comp
            .iter()
            .any(|&row| net.sources.iter().any(|s| s.node == row / 3));
        if has_source {
            for &row in comp {
                energized[row / 3] = true;
            }
        }
    }
    energized
}

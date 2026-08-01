//! The measurement Jacobian `H` and the gain matrix `G = HᵀWH`.
//!
//! # State layout
//!
//! State estimation's unknowns are not power flow's. There are no PV buses
//! here — a generator's voltage is something to be estimated, not asserted — so
//! *every* bus contributes a magnitude, and every bus but the angle reference
//! contributes an angle:
//!
//! ```text
//! x = [ θ₀ … θ_{N-1} except the reference ,  V₀ … V_{N-1} ]
//! ```
//!
//! giving `n = 2N − 1`, against power flow's `n_angle + n_pq`
//! (`solver::newton_raphson_cached`). Confusing the two layouts is the easiest
//! way to produce a Jacobian that is subtly, consistently wrong, so
//! [`StateLayout`] owns the mapping and nothing else indexes by hand.
//!
//! # Why the normal equations
//!
//! `H` is rectangular (`m × n`), and gridoxide's [`LinearSolver`] backends are
//! all square — they are power flow's Jacobian solvers. Forming the normal
//! equations `G Δx = HᵀW r` makes the system square *and* symmetric, so all
//! five backends carry state estimation with no new solver abstraction. The
//! cost is conditioning: `G` squares `H`'s condition number, which is why the
//! plan keeps the option of an orthogonal method open and why phase 4 prefers
//! equality constraints over huge weights.

use crate::branch_flow::{terminal_flow_derivs, Terminal};
use crate::measurement::{Measurement, MeasurementKind, Target};
use crate::types::Bus;

use super::SeNetwork;

/// Maps buses to positions in the state vector.
pub struct StateLayout {
    pub n_buses: usize,
    /// The bus whose angle is held at zero, if one is pinned.
    ///
    /// A network measured only in magnitudes and powers is invariant under a
    /// global phase shift, so without a reference `G` is singular by
    /// construction no matter how many measurements there are. But that
    /// invariance disappears the moment a *phasor* measurement supplies an
    /// absolute angle — and pinning a reference on top of one would then be a
    /// false constraint, rotating the whole estimate away from the angles that
    /// were actually measured. So this is `None` exactly when the measurement
    /// set determines the phase by itself.
    pub angle_ref: Option<usize>,
    /// Position of each bus's angle unknown, or `None` for the reference.
    theta_pos: Vec<Option<usize>>,
    /// Where the magnitude block starts, i.e. the number of angle unknowns.
    vmag_offset: usize,
}

impl StateLayout {
    /// Builds the layout for a measurement set.
    ///
    /// A reference is pinned only when the measurements contain no voltage
    /// angle; see [`angle_ref`](Self::angle_ref).
    ///
    /// When one is pinned it is the *physical node a source feeds*, not the
    /// virtual slack bus behind it. power-grid-model normalizes by setting its
    /// slack bus's angle to zero, and its slack bus is that physical node,
    /// since it has no virtual bus at all. Pinning the virtual bus instead
    /// leaves every angle offset by the drop across the source impedance — a
    /// constant, invisible in magnitudes, and exactly the 6.5e-3 rad uniform
    /// shift the transmission-case fixture showed before this was fixed. Stiff
    /// sources hide it, which is why the smaller fixtures agreed either way.
    pub fn new(buses: &[Bus], measurements: &[Measurement], net: &SeNetwork) -> Self {
        let phase_is_measured = measurements
            .iter()
            .any(|m| m.kind == MeasurementKind::VoltageAngle && m.weight() > 0.0);

        let angle_ref = (!phase_is_measured).then(|| {
            net.source_branches
                .iter()
                .position(|feeding| !feeding.is_empty())
                .or_else(|| {
                    buses
                        .iter()
                        .position(|b| matches!(b.bus_type, crate::types::BusType::Slack))
                })
                .unwrap_or(0)
        });

        let mut theta_pos = vec![None; buses.len()];
        let mut next = 0;
        for i in 0..buses.len() {
            if Some(i) != angle_ref {
                theta_pos[i] = Some(next);
                next += 1;
            }
        }
        Self { n_buses: buses.len(), angle_ref, theta_pos, vmag_offset: next }
    }

    /// Number of unknowns: `2N` when the phase is measured, `2N − 1` when a
    /// reference has to be pinned.
    pub fn n_unknowns(&self) -> usize {
        self.vmag_offset + self.n_buses
    }

    /// Column of a bus's angle, or `None` for the reference bus.
    pub fn theta(&self, bus: usize) -> Option<usize> {
        self.theta_pos[bus]
    }

    /// Column of a bus's voltage magnitude.
    pub fn vmag(&self, bus: usize) -> usize {
        self.vmag_offset + bus
    }
}

/// One row of `H`: the nonzero partials of a measurement, as `(column, value)`.
///
/// Rows are short — a voltage measurement touches one column, a branch flow
/// four, a bus injection two per adjacent bus — which is what keeps `G` sparse.
pub type Row = Vec<(usize, f64)>;

/// Accumulates a partial into a row.
///
/// Only the reference bus's angle is dropped, because it is not an unknown at
/// all. A numerically zero partial is *kept*: `H`'s pattern has to depend on the
/// topology alone, not on the current state. Dropping zeros would shrink the row
/// at a flat start — where many `sin(θ_i − θ_k)` terms vanish exactly — and grow
/// it again on the next iteration, changing the gain matrix's sparsity pattern
/// between iterations and invalidating the cached symbolic factorization
/// (`sparse::RealSparseSystem` documents that its argsort assumes positional
/// correspondence, and faer asserts on the mismatch).
fn push(row: &mut Row, col: Option<usize>, value: f64) {
    if let Some(col) = col {
        row.push((col, value));
    }
}

/// Partials of the bus injections `P_i`, `Q_i` with respect to a neighbour's
/// angle and magnitude.
///
/// These are the classic power-flow Jacobian expressions, but written for
/// arbitrary `(i, k)` rather than for power flow's reduced unknown set, since a
/// state estimator needs the full `2N − 1` columns.
fn injection_partials(
    net: &SeNetwork,
    buses: &[Bus],
    p_inj: &[f64],
    q_inj: &[f64],
    i: usize,
    k: usize,
) -> (f64, f64, f64, f64) {
    let y_ik = net.ybus.get(i, k);
    let (g, b) = (y_ik.re, y_ik.im);
    let (v_i, v_k) = (buses[i].voltage_mag, buses[k].voltage_mag);

    if i == k {
        let v2 = v_i * v_i;
        (
            // dP/dθ, dP/dV, dQ/dθ, dQ/dV
            -q_inj[i] - b * v2,
            p_inj[i] / v_i + g * v_i,
            p_inj[i] - g * v2,
            q_inj[i] / v_i - b * v_i,
        )
    } else {
        let (s, c) = (buses[i].voltage_ang - buses[k].voltage_ang).sin_cos();
        (
            v_i * v_k * (g * s - b * c),
            v_i * (g * c + b * s),
            -v_i * v_k * (g * c + b * s),
            v_i * (g * s - b * c),
        )
    }
}

/// Adds a bus injection's partials into `row`, for whichever of `P` or `Q` the
/// measurement is about.
fn add_injection_row(
    row: &mut Row,
    layout: &StateLayout,
    net: &SeNetwork,
    buses: &[Bus],
    p_inj: &[f64],
    q_inj: &[f64],
    bus: usize,
    active: bool,
    scale: f64,
) {
    // Only buses adjacent in the Y-bus (plus the bus itself) contribute, which
    // is exactly `ybus.row`.
    for &(k, _) in net.ybus.row(bus) {
        let (dp_dth, dp_dv, dq_dth, dq_dv) =
            injection_partials(net, buses, p_inj, q_inj, bus, k);
        let (d_th, d_v) = if active { (dp_dth, dp_dv) } else { (dq_dth, dq_dv) };
        push(row, layout.theta(k), scale * d_th);
        push(row, Some(layout.vmag(k)), scale * d_v);
    }
}

/// Adds a branch terminal flow's partials into `row`, scaled by `scale` (which
/// is `-1` for a source, whose injection is the negated terminal flow).
fn add_branch_row(
    row: &mut Row,
    layout: &StateLayout,
    net: &SeNetwork,
    v: &[num_complex::Complex<f64>],
    branch: usize,
    terminal: Terminal,
    active: bool,
    scale: f64,
) {
    let params = &net.branches[branch];
    let d = terminal_flow_derivs(params, terminal, v);
    let (near, far) = match terminal {
        Terminal::From => (params.from, params.to),
        Terminal::To => (params.to, params.from),
    };
    let (dth_near, dth_far, dv_near, dv_far) = if active {
        (d.dp_dtheta_near, d.dp_dtheta_far, d.dp_dv_near, d.dp_dv_far)
    } else {
        (d.dq_dtheta_near, d.dq_dtheta_far, d.dq_dv_near, d.dq_dv_far)
    };
    push(row, layout.theta(near), scale * dth_near);
    push(row, layout.theta(far), scale * dth_far);
    push(row, Some(layout.vmag(near)), scale * dv_near);
    push(row, Some(layout.vmag(far)), scale * dv_far);

    // A self-loop branch (gridoxide's half-open representation) has near == far,
    // so the two contributions land in the same columns and must sum rather
    // than overwrite. `Row` is an association list, so duplicates are summed
    // when the gain matrix is assembled — see `gain_triplets`.
}

/// Adds a shunt injection's partials. A shunt depends only on its own bus's
/// magnitude: `-|V|²g` and `+|V|²b` have no angle dependence at all.
fn add_shunt_row(row: &mut Row, layout: &StateLayout, net: &SeNetwork, buses: &[Bus], bus: usize, active: bool) {
    let y = net.shunt_y[bus];
    let v = buses[bus].voltage_mag;
    let d = if active { -2.0 * v * y.re } else { 2.0 * v * y.im };
    push(row, Some(layout.vmag(bus)), d);
}

/// One bus-injection row on its own, for callers that need the same partials
/// outside the measurement set — `se::constraints` builds its constraint rows
/// from exactly this, since a zero-injection constraint and an injection
/// measurement differ only in how the system consumes the row.
pub fn injection_row(
    layout: &StateLayout,
    net: &SeNetwork,
    buses: &[Bus],
    p_inj: &[f64],
    q_inj: &[f64],
    bus: usize,
    active: bool,
) -> Row {
    let mut row = Row::new();
    add_injection_row(&mut row, layout, net, buses, p_inj, q_inj, bus, active, 1.0);
    row
}

/// Builds `H`, one row per measurement, in the same order.
pub fn measurement_jacobian(
    measurements: &[Measurement],
    buses: &[Bus],
    net: &SeNetwork,
    layout: &StateLayout,
) -> Vec<Row> {
    let v: Vec<num_complex::Complex<f64>> = buses
        .iter()
        .map(|b| num_complex::Complex::from_polar(b.voltage_mag, b.voltage_ang))
        .collect();
    let (p_inj, q_inj) = crate::network::power_injections(buses, &net.ybus);

    measurements
        .iter()
        .map(|m| {
            let mut row = Row::new();
            let active = m.kind == MeasurementKind::ActivePower;
            match m.target {
                Target::Bus(b) => match m.kind {
                    MeasurementKind::VoltageMagnitude => push(&mut row, Some(layout.vmag(b)), 1.0),
                    MeasurementKind::VoltageAngle => push(&mut row, layout.theta(b), 1.0),
                    _ => add_injection_row(
                        &mut row, layout, net, buses, &p_inj, &q_inj, b, active, 1.0,
                    ),
                },
                Target::BranchTerminal { branch, terminal } => {
                    add_branch_row(&mut row, layout, net, &v, branch, terminal, active, 1.0)
                }
                Target::SourceInjection { branch } => {
                    add_branch_row(&mut row, layout, net, &v, branch, Terminal::To, active, -1.0)
                }
                Target::ShuntInjection { bus } => {
                    add_shunt_row(&mut row, layout, net, buses, bus, active)
                }
                Target::NodeInjection(bus) => {
                    add_injection_row(
                        &mut row, layout, net, buses, &p_inj, &q_inj, bus, active, 1.0,
                    );
                    for &branch in &net.source_branches[bus] {
                        add_branch_row(&mut row, layout, net, &v, branch, Terminal::To, active, -1.0);
                    }
                    add_shunt_row(&mut row, layout, net, buses, bus, active);
                }
            }
            row
        })
        .collect()
}

/// Pins state variables that nothing reaches, and reports which.
///
/// A column no row touches leaves an all-zero row and column in `G`, making the
/// whole system singular even when the rest of it is perfectly determined.
/// Writing an identity there with a zero right-hand side holds that variable at
/// its current value and lets the observable part solve — the same masking
/// `bde::mask_scenario` uses for a converged block, and for the same reason:
/// dropping the row instead would change the sparsity pattern and invalidate a
/// cached symbolic factorization.
///
/// "Touched" is structural — any appearance in a row, regardless of value — so
/// that the mask, and hence the sparsity pattern, is identical across
/// iterations. Whether a present-but-degenerate column is *usefully* determined
/// is a different question, and
/// [`observability`](super::observability)'s.
pub fn mask_untouched(
    triplets: &mut Vec<(usize, usize, f64)>,
    rhs: &mut [f64],
    row_sets: &[&[Row]],
    n: usize,
) -> Vec<usize> {
    let mut touched = vec![false; n];
    for rows in row_sets {
        for row in rows.iter() {
            for &(c, _) in row {
                touched[c] = true;
            }
        }
    }
    let mut untouched = Vec::new();
    for (c, &seen) in touched.iter().enumerate() {
        if !seen {
            triplets.push((c, c, 1.0));
            rhs[c] = 0.0;
            untouched.push(c);
        }
    }
    untouched
}

/// Assembles `G = HᵀWH` as triplets, and `HᵀW r` as a dense vector.
///
/// `G = Σ_m w_m·hₘhₘᵀ`, so it is built as a sum of per-row outer products —
/// cheap because rows are short. Duplicate `(row, col)` triplets are summed by
/// the sparse backends, which is also what makes a self-loop branch's repeated
/// columns come out right.
pub fn gain_and_rhs(
    rows: &[Row],
    measurements: &[Measurement],
    residuals: &[f64],
) -> (Vec<(usize, usize, f64)>, Vec<f64>, usize) {
    let n = rows
        .iter()
        .flat_map(|r| r.iter().map(|&(c, _)| c + 1))
        .max()
        .unwrap_or(0);
    let mut triplets = Vec::new();
    let mut rhs = vec![0.0; n];

    for ((row, m), &r) in rows.iter().zip(measurements).zip(residuals) {
        let w = m.weight();
        if !w.is_finite() || w == 0.0 {
            // An infinite sigma contributes nothing, by design.
            continue;
        }
        for &(i, hi) in row {
            rhs[i] += w * hi * r;
            for &(j, hj) in row {
                triplets.push((i, j, w * hi * hj));
            }
        }
    }
    (triplets, rhs, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::Target;
    use crate::se::measurement_functions;

    /// Every row of `H` against central differences of `h(x)`.
    ///
    /// This is the real guard on this module: the injection partials in
    /// particular are eight sign-sensitive expressions that are easy to write
    /// plausibly and wrongly, and a consistently wrong Jacobian still converges
    /// — to the wrong answer, slowly, in a way that looks like a bad fixture.
    pub(super) fn assert_jacobian_matches_finite_differences(
        measurements: &[Measurement],
        buses: &[Bus],
        net: &SeNetwork,
        tol: f64,
    ) {
        let layout = StateLayout::new(buses, measurements, net);
        let rows = measurement_jacobian(measurements, buses, net, &layout);
        let n = layout.n_unknowns();
        let h = 1e-7;

        // Dense the rows out for easy comparison, summing duplicates.
        let mut dense = vec![vec![0.0; n]; measurements.len()];
        for (r, row) in rows.iter().enumerate() {
            for &(c, val) in row {
                dense[r][c] += val;
            }
        }

        for col in 0..n {
            let bump = |delta: f64| {
                let mut b = buses.to_vec();
                // Columns below `vmag(0)` are angles, the rest magnitudes.
                if col < layout.vmag(0) {
                    let bus = (0..buses.len())
                        .find(|&i| layout.theta(i) == Some(col))
                        .expect("angle column maps to a bus");
                    b[bus].voltage_ang += delta;
                } else {
                    b[col - layout.vmag(0)].voltage_mag += delta;
                }
                measurement_functions(measurements, &b, net)
            };
            let plus = bump(h);
            let minus = bump(-h);

            for (r, m) in measurements.iter().enumerate() {
                let numeric = (plus[r] - minus[r]) / (2.0 * h);
                let analytic = dense[r][col];
                assert!(
                    (numeric - analytic).abs() < tol,
                    "row {r} ({:?} on {:?}), column {col}: analytic {analytic}, numeric {numeric}",
                    m.kind,
                    m.target,
                );
            }
        }
    }

    fn measurement(kind: MeasurementKind, target: Target) -> Measurement {
        Measurement { kind, target, value: 0.0, sigma: 1.0 }
    }

    /// Covers every target kind at once, on a network where no partial is zero
    /// by symmetry.
    #[test]
    fn jacobian_matches_finite_differences_for_every_target_kind() {
        let (net, buses) = super::super::tests::two_bus_net();
        let measurements = vec![
            measurement(MeasurementKind::VoltageMagnitude, Target::Bus(1)),
            measurement(MeasurementKind::VoltageAngle, Target::Bus(1)),
            measurement(MeasurementKind::ActivePower, Target::Bus(1)),
            measurement(MeasurementKind::ReactivePower, Target::Bus(1)),
            measurement(
                MeasurementKind::ActivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
            ),
            measurement(
                MeasurementKind::ReactivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::To },
            ),
            measurement(MeasurementKind::ActivePower, Target::SourceInjection { branch: 0 }),
            measurement(MeasurementKind::ReactivePower, Target::ShuntInjection { bus: 1 }),
            measurement(MeasurementKind::ActivePower, Target::NodeInjection(1)),
            measurement(MeasurementKind::ReactivePower, Target::NodeInjection(1)),
        ];
        assert_jacobian_matches_finite_differences(&measurements, &buses, &net, 1e-5);
    }

    /// The layout is `2N − 1`, and the reference bus has no angle column.
    #[test]
    fn state_layout_excludes_only_the_reference_angle() {
        let (net, buses) = super::super::tests::two_bus_net();
        let layout = StateLayout::new(&buses, &[], &net);
        assert_eq!(layout.n_unknowns(), 2 * buses.len() - 1);
        let reference = layout.angle_ref.expect("no angle measurements, so a reference is pinned");
        assert_eq!(layout.theta(reference), None);
        for i in 0..buses.len() {
            if i != reference {
                assert!(layout.theta(i).is_some(), "bus {i} should carry an angle unknown");
            }
        }
    }

    /// `G` must be symmetric — it is `HᵀWH` — and its diagonal non-negative.
    #[test]
    fn gain_matrix_is_symmetric() {
        let (net, buses) = super::super::tests::two_bus_net();
        let measurements = vec![
            measurement(MeasurementKind::VoltageMagnitude, Target::Bus(1)),
            measurement(MeasurementKind::ActivePower, Target::Bus(1)),
            measurement(
                MeasurementKind::ReactivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
            ),
        ];
        let layout = StateLayout::new(&buses, &measurements, &net);
        let rows = measurement_jacobian(&measurements, &buses, &net, &layout);
        let (triplets, _, n) = gain_and_rhs(&rows, &measurements, &[0.1, 0.2, 0.3]);

        let mut dense = vec![vec![0.0; n]; n];
        for (i, j, v) in triplets {
            dense[i][j] += v;
        }
        for i in 0..n {
            assert!(dense[i][i] >= -1e-12, "diagonal {i} is negative: {}", dense[i][i]);
            for j in 0..n {
                assert!(
                    (dense[i][j] - dense[j][i]).abs() < 1e-9,
                    "G is not symmetric at ({i},{j}): {} vs {}",
                    dense[i][j],
                    dense[j][i]
                );
            }
        }
    }
}

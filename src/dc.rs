//! Real-valued DC network model and Newton-Raphson solver.
//!
//! A DC network has no angle and no reactive power — only a voltage
//! magnitude and a real power/current balance at each node — so this is
//! deliberately *not* built on [`crate::types::Bus`]/[`crate::types::Line`]:
//! reusing those would drag in dead fields (`voltage_ang`, `q_spec`,
//! `q_min`/`q_max`) and make AC-only machinery (`BusType::PV` Q-limit
//! switching, the complex Y-bus) nonsensically reachable for a DC bus.
//!
//! Used for CGMES HVDC subsystems: a [`DcBus`] per `DCTopologicalNode`, a
//! [`DcLine`] per resistive DC branch (`DCLineSegment`, a switch/breaker
//! stamped as a near-zero resistance, etc.). These networks are small (a
//! single HVDC link's switchyard fixture tops out around a dozen nodes), so
//! `solve_dc_network` uses a plain dense Newton-Raphson solve with Gaussian
//! elimination rather than pulling in `solver.rs`'s sparse-LU machinery,
//! which exists to scale to real AC networks with thousands of buses.

use super::network::{connected_components, YBus};
use num_complex::Complex;

/// What fixes a DC bus's voltage/current/power, mirroring the CGMES
/// `pPccControl`/`ACDCConverter`/`DCGround` semantics that produce each role.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DcBusRole {
    /// Voltage held fixed at `DcBus::udc_fixed` — the DC-voltage-regulating
    /// converter of an HVDC link (`pPccControl=udc`/`dcVoltage`).
    UdcSlack,
    /// Voltage held fixed at 0 — a `DCGround` connection. Structurally
    /// identical to `UdcSlack` (both remove the bus from the unknowns), kept
    /// as its own variant so callers/reports can tell the two apart.
    Ground,
    /// Net power injection into the DC network held fixed
    /// (`pPccControl=pPcc`) — the nonlinear row: `p_spec = V·ΣG·V`. Same
    /// injection sign convention as `types::Bus::p_spec`: positive means
    /// power flows OUT of this bus into the network (generation-like),
    /// negative means it's drawn FROM the network (load-like) — the
    /// opposite of CGMES's own SSH "load sign convention" on
    /// `ACDCConverter.p`, so callers wiring in CGMES data negate it, exactly
    /// as `cgmes.rs` already does for every AC injection.
    FixedP { p_spec: f64 },
    /// Net current injection into the DC network held fixed
    /// (`pPccControl=dcCurrent`) — a row that's linear in the unknown
    /// voltages: `idc_spec = ΣG·V`. Same injection-sign convention as
    /// `FixedP::p_spec` above.
    FixedIdc { idc_spec: f64 },
    /// No injection at all — a plain junction/switchyard node
    /// (`DCBusbar`, a closed switch's synthesized node, etc.). Same linear
    /// form as `FixedIdc` with `idc_spec = 0`.
    Passive,
}

#[derive(Clone, Debug)]
pub struct DcBus {
    pub idx: usize,
    pub role: DcBusRole,
    /// Fixed voltage for `UdcSlack`/`Ground` roles; ignored otherwise.
    pub udc_fixed: f64,
    /// Working/solved voltage. For `UdcSlack`/`Ground` this is overwritten
    /// with `udc_fixed` at the start of `solve_dc_network`; for other roles
    /// it's both the initial guess (0.0 means "let the solver pick one") and,
    /// after solving, the result.
    pub voltage: f64,
    /// Extra conductance to ground at this node (e.g. a `DCShunt`, which has
    /// only one terminal so can't be represented as a `DcLine`).
    pub shunt_g: f64,
}

/// A resistive two-terminal DC branch: `DCLineSegment`, `DCSeriesDevice`, or
/// a switch/breaker/disconnector stamped as a near-zero resistance.
#[derive(Clone, Copy, Debug)]
pub struct DcLine {
    pub from: usize,
    pub to: usize,
    pub r: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DcSolveStatus {
    pub converged: bool,
    pub iterations: usize,
    /// Bus indices belonging to a connected component with no `UdcSlack`/
    /// `Ground` reference at all (e.g. a disconnected spare switchyard
    /// branch) — left unsolved (`voltage` untouched) rather than blocking
    /// the rest of the network, mirroring how AC's multi-island support
    /// isolates a no-reference component instead of failing the whole solve.
    pub isolated_buses: Vec<usize>,
}

/// Builds the dense conductance ("real Y-bus") matrix for `lines`/`shunt_g`,
/// and, via [`connected_components`], the connectivity graph used to isolate
/// reference-less components. Reuses `network::YBus`'s triplet accumulation
/// (with zero imaginary parts) rather than a separate graph implementation.
fn build_conductance(buses: &[DcBus], lines: &[DcLine]) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let n = buses.len();
    let mut g = vec![vec![0.0_f64; n]; n];
    let mut ybus = YBus::new(n);
    for line in lines {
        let gij = 1.0 / line.r;
        g[line.from][line.from] += gij;
        g[line.to][line.to] += gij;
        g[line.from][line.to] -= gij;
        g[line.to][line.from] -= gij;
        let c = Complex::new(gij, 0.0);
        ybus.add(line.from, line.from, c);
        ybus.add(line.to, line.to, c);
        ybus.add(line.from, line.to, -c);
        ybus.add(line.to, line.from, -c);
    }
    for (i, b) in buses.iter().enumerate() {
        if b.shunt_g != 0.0 {
            g[i][i] += b.shunt_g;
            ybus.add(i, i, Complex::new(b.shunt_g, 0.0));
        }
    }
    let components = connected_components(&ybus.finish());
    (g, components)
}

/// `S[i] = Σ_j G[i][j]·V[j]` for every bus — the net current injected into
/// the network at each node, given the buses' current `voltage` values.
fn nodal_injections(g: &[Vec<f64>], buses: &[DcBus]) -> Vec<f64> {
    (0..buses.len())
        .map(|i| (0..buses.len()).map(|j| g[i][j] * buses[j].voltage).sum())
        .collect()
}

/// The net current injected into the network at each bus (`S[i]` above),
/// computed from `buses`' current (typically post-solve) voltages — the
/// per-bus DC current `Idc`, in the same units as `DcLine::r`/`DcBus::voltage`
/// combine to produce (kA, given resistance in Ω and voltage in kV). Used by
/// CGMES conversion to translate a solved DC bus's state back into a
/// converter's own AC-side power via its loss curve.
pub fn injected_currents(buses: &[DcBus], lines: &[DcLine]) -> Vec<f64> {
    let (g, _) = build_conductance(buses, lines);
    nodal_injections(&g, buses)
}

/// Solves in-place Ax = b via Gaussian elimination with partial pivoting.
/// Returns `None` if `a` is singular (no pivot above a small threshold).
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot_row][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let sum: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - sum) / a[row][row];
    }
    Some(x)
}

/// Solves a small resistive DC network for the unknown bus voltages, given
/// each bus's role (fixed voltage, fixed power, fixed current, or passive).
///
/// Newton-Raphson over the non-`UdcSlack`/`Ground` buses (`FixedIdc`/
/// `Passive` rows are already linear — `ΣG·V - injection = 0` — but are
/// solved in the same loop as `FixedP`'s genuinely nonlinear `V·ΣG·V -
/// p_spec = 0` rather than special-cased, exactly as AC's P- and Q-mismatch
/// rows differ in structure but share one Jacobian). Components with no
/// `UdcSlack`/`Ground` reference at all are excluded from the unknowns
/// entirely and reported via `DcSolveStatus::isolated_buses`, rather than
/// producing a singular (indeterminate) system.
pub fn solve_dc_network(buses: &mut [DcBus], lines: &[DcLine], tol: f64, max_iter: usize) -> DcSolveStatus {
    let (g, components) = build_conductance(buses, lines);

    for b in buses.iter_mut() {
        if matches!(b.role, DcBusRole::UdcSlack | DcBusRole::Ground) {
            b.voltage = b.udc_fixed;
        }
    }

    let mut isolated_buses = Vec::new();
    let mut unknown_idx = Vec::new();
    for component in &components {
        let has_reference = component.iter().any(|&i| matches!(buses[i].role, DcBusRole::UdcSlack | DcBusRole::Ground));
        if has_reference {
            for &i in component {
                if !matches!(buses[i].role, DcBusRole::UdcSlack | DcBusRole::Ground) {
                    unknown_idx.push(i);
                }
            }
        } else {
            isolated_buses.extend(component.iter().copied());
        }
    }
    unknown_idx.sort_unstable();

    // Seed any not-yet-initialized (voltage == 0.0) unknown with the average
    // fixed voltage in the network, so Newton starts from a physically
    // sensible point rather than 0 V.
    let fixed_avg = {
        let fixed: Vec<f64> = buses.iter()
            .filter(|b| matches!(b.role, DcBusRole::UdcSlack))
            .map(|b| b.udc_fixed)
            .collect();
        if fixed.is_empty() { 1.0 } else { fixed.iter().sum::<f64>() / fixed.len() as f64 }
    };
    for &i in &unknown_idx {
        if buses[i].voltage == 0.0 {
            buses[i].voltage = fixed_avg;
        }
    }

    let m = unknown_idx.len();
    if m == 0 {
        return DcSolveStatus { converged: true, iterations: 0, isolated_buses };
    }

    for iter in 0..max_iter {
        let s = nodal_injections(&g, buses);

        let mut mismatch = vec![0.0; m];
        let mut converged = true;
        for (row, &i) in unknown_idx.iter().enumerate() {
            let vi = buses[i].voltage;
            mismatch[row] = match buses[i].role {
                DcBusRole::FixedP { p_spec } => p_spec - vi * s[i],
                DcBusRole::FixedIdc { idc_spec } => idc_spec - s[i],
                DcBusRole::Passive => -s[i],
                DcBusRole::UdcSlack | DcBusRole::Ground => unreachable!("fixed-voltage buses are excluded from unknown_idx"),
            };
            if mismatch[row].abs() > tol {
                converged = false;
            }
        }
        if converged {
            return DcSolveStatus { converged: true, iterations: iter, isolated_buses };
        }

        let mut jac = vec![vec![0.0; m]; m];
        for (row, &i) in unknown_idx.iter().enumerate() {
            let vi = buses[i].voltage;
            for (col, &k) in unknown_idx.iter().enumerate() {
                jac[row][col] = match buses[i].role {
                    // d(p_spec - V_i·S_i)/dV_i = -(S_i + V_i·G_ii), since S_i
                    // itself already contains the G_ii·V_i term.
                    DcBusRole::FixedP { .. } if k == i => -(s[i] + g[i][i] * vi),
                    DcBusRole::FixedP { .. } => -vi * g[i][k],
                    DcBusRole::FixedIdc { .. } | DcBusRole::Passive => -g[i][k],
                    DcBusRole::UdcSlack | DcBusRole::Ground => unreachable!(),
                };
            }
        }

        let Some(delta) = solve_linear(jac, mismatch) else {
            return DcSolveStatus { converged: false, iterations: iter, isolated_buses };
        };
        for (row, &i) in unknown_idx.iter().enumerate() {
            buses[i].voltage -= delta[row];
        }
    }

    DcSolveStatus { converged: false, iterations: max_iter, isolated_buses }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passive(idx: usize) -> DcBus {
        DcBus { idx, role: DcBusRole::Passive, udc_fixed: 0.0, voltage: 0.0, shunt_g: 0.0 }
    }

    #[test]
    fn two_node_fixed_p_matches_analytic_solution() {
        // bus0 UdcSlack @ 100, bus1 FixedP, r=2.0. bus1 is load-like (draws
        // power from the network), so p_spec is negative — the physical
        // root is V1=90: S1 = (90-100)/2 = -5, P1 = 90*(-5) = -450.
        let mut buses = vec![
            DcBus { idx: 0, role: DcBusRole::UdcSlack, udc_fixed: 100.0, voltage: 100.0, shunt_g: 0.0 },
            DcBus { idx: 1, role: DcBusRole::FixedP { p_spec: -450.0 }, udc_fixed: 0.0, voltage: 0.0, shunt_g: 0.0 },
        ];
        let lines = vec![DcLine { from: 0, to: 1, r: 2.0 }];
        let status = solve_dc_network(&mut buses, &lines, 1e-9, 50);
        assert!(status.converged);
        assert!(status.isolated_buses.is_empty());
        assert!((buses[1].voltage - 90.0).abs() < 1e-6, "V1 = {}", buses[1].voltage);
    }

    #[test]
    fn three_node_passive_junction_splits_series_resistance() {
        // bus0 UdcSlack @ 100 -- r=1 -- bus1 Passive -- r=1 -- bus2
        // FixedIdc=-5 (load-like, draws current from the network).
        // Series resistance: idc = (V2 - 100) / 2 = -5 => V2 = 90, V1 = 95.
        let mut buses = vec![
            DcBus { idx: 0, role: DcBusRole::UdcSlack, udc_fixed: 100.0, voltage: 100.0, shunt_g: 0.0 },
            passive(1),
            DcBus { idx: 2, role: DcBusRole::FixedIdc { idc_spec: -5.0 }, udc_fixed: 0.0, voltage: 0.0, shunt_g: 0.0 },
        ];
        let lines = vec![
            DcLine { from: 0, to: 1, r: 1.0 },
            DcLine { from: 1, to: 2, r: 1.0 },
        ];
        let status = solve_dc_network(&mut buses, &lines, 1e-9, 50);
        assert!(status.converged);
        assert!((buses[1].voltage - 95.0).abs() < 1e-6, "V1 = {}", buses[1].voltage);
        assert!((buses[2].voltage - 90.0).abs() < 1e-6, "V2 = {}", buses[2].voltage);
    }

    #[test]
    fn fixed_idc_bus_converges_immediately_since_linear() {
        let mut buses = vec![
            DcBus { idx: 0, role: DcBusRole::UdcSlack, udc_fixed: 100.0, voltage: 100.0, shunt_g: 0.0 },
            DcBus { idx: 1, role: DcBusRole::FixedIdc { idc_spec: -10.0 }, udc_fixed: 0.0, voltage: 0.0, shunt_g: 0.0 },
        ];
        let lines = vec![DcLine { from: 0, to: 1, r: 1.0 }];
        let status = solve_dc_network(&mut buses, &lines, 1e-9, 50);
        assert!(status.converged);
        assert!(status.iterations <= 2, "expected near-immediate convergence for a linear row, got {} iterations", status.iterations);
        assert!((buses[1].voltage - 90.0).abs() < 1e-6, "V1 = {}", buses[1].voltage);
    }

    #[test]
    fn disconnected_dead_subgraph_does_not_perturb_live_solve() {
        // Live subsystem: bus0 UdcSlack @ 100 -- r=2 -- bus1 FixedP=-450
        // (same as the two-node test, so V1 should still be exactly 90).
        // Dead subsystem: bus2 -- bus3, both Passive, no reference at all —
        // must be isolated, not solved, and must not perturb the live pair.
        let mut buses = vec![
            DcBus { idx: 0, role: DcBusRole::UdcSlack, udc_fixed: 100.0, voltage: 100.0, shunt_g: 0.0 },
            DcBus { idx: 1, role: DcBusRole::FixedP { p_spec: -450.0 }, udc_fixed: 0.0, voltage: 0.0, shunt_g: 0.0 },
            passive(2),
            passive(3),
        ];
        let lines = vec![
            DcLine { from: 0, to: 1, r: 2.0 },
            DcLine { from: 2, to: 3, r: 1.0 },
        ];
        let status = solve_dc_network(&mut buses, &lines, 1e-9, 50);
        assert!(status.converged);
        assert!((buses[1].voltage - 90.0).abs() < 1e-6, "V1 = {}", buses[1].voltage);
        let mut isolated = status.isolated_buses.clone();
        isolated.sort_unstable();
        assert_eq!(isolated, vec![2, 3]);
        // Untouched: no fixed reference means no sensible value to compute.
        assert_eq!(buses[2].voltage, 0.0);
        assert_eq!(buses[3].voltage, 0.0);
    }
}

use super::types::{Bus, BusType};
use super::network::{effective_injection, power_injections, YBusSparse};
use super::sparse::RealSparseSystem;

pub fn newton_raphson(buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize) {

    // Identify PV and PQ indices (exclude slack)
    let mut pv_idx: Vec<usize> = Vec::new();
    let mut pq_idx: Vec<usize> = Vec::new();
    for b in buses.iter() {
        match b.bus_type {
            BusType::Slack => (),
            BusType::PV => pv_idx.push(b.idx),
            BusType::PQ => pq_idx.push(b.idx),
        }
    }

    let non_slack_idx: Vec<usize> = buses
        .iter()
        .filter(|b| !matches!(b.bus_type, BusType::Slack))
        .map(|b| b.idx)
        .collect();

    let n_angle = non_slack_idx.len();
    let n_vmag = pq_idx.len();
    let n_unknowns = n_angle + n_vmag;

    // Physical bus index -> position within non_slack_idx / pq_idx, for
    // O(1) lookup while walking each bus's actual Y-bus neighbors (as
    // opposed to the full non_slack_idx × non_slack_idx / pq_idx × pq_idx
    // cross product the dense implementation used).
    let mut non_slack_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in non_slack_idx.iter().enumerate() {
        non_slack_pos[i] = Some(pos);
    }
    let mut pq_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in pq_idx.iter().enumerate() {
        pq_pos[i] = Some(pos);
    }

    // The Jacobian's sparsity *pattern* is fixed across iterations (same
    // bus topology every time, only numeric values change), so the
    // symbolic factorization (ordering + fill-in) is computed once here and
    // reused via `factor_and_solve` for every iteration's numeric-only
    // refactorization, mirroring PGM's own prefactorization-reuse approach.
    let mut sparse_system: Option<RealSparseSystem> = None;

    for iter in 0..max_iter {
        // compute injections
        let (p_calc, q_calc) = power_injections(buses, ybus);

        // Build mismatch vector
        let mut mismatch = vec![0.0; n_unknowns];
        let mut mis_idx = 0;
        for &i in &non_slack_idx {
            // P mismatch for PV and PQ buses
            let (p_eff, _) = effective_injection(&buses[i]);
            mismatch[mis_idx] = p_eff - p_calc[i];
            mis_idx += 1;
        }
        for &i in &pq_idx {
            // Q mismatch for PQ buses
            let (_, q_eff) = effective_injection(&buses[i]);
            mismatch[mis_idx] = q_eff - q_calc[i];
            mis_idx += 1;
        }

        let max_mis = mismatch.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        println!("iter {}: max mismatch = {:.6e}", iter + 1, max_mis);
        if max_mis < tol {
            println!("Converged in {} iterations", iter + 1);
            return;
        }

        // Build Jacobian (sparse triplets)
        let triplets = build_jacobian_triplets(
            buses, ybus, &non_slack_idx, &pq_idx, &non_slack_pos, &pq_pos, n_angle, &p_calc, &q_calc,
        );

        if sparse_system.is_none() {
            sparse_system = RealSparseSystem::new(n_unknowns, &triplets);
        }
        let Some(system) = sparse_system.as_ref() else {
            println!("Jacobian is singular. Failed to solve.");
            return;
        };
        let dx = match system.factor_and_solve(&triplets, &mismatch) {
            Some(sol) => sol,
            None => {
                println!("Jacobian is singular. Failed to solve.");
                return;
            }
        };

        // Update state
        let mut dx_idx = 0;
        for &i in &non_slack_idx {
            // update voltage angles
            buses[i].voltage_ang += dx[dx_idx];
            dx_idx += 1;
        }
        for &i in &pq_idx {
            // update voltage magnitudes
            buses[i].voltage_mag += dx[dx_idx];
            dx_idx += 1;
        }
    }

    println!("Failed to converge in {} iterations", max_iter);
}

/// Assembles the Newton-Raphson Jacobian's nonzero entries as `(row, col,
/// value)` triplets, walking only each unknown bus's actual Y-bus neighbors
/// (`ybus.row(i)`) instead of the full cross product of unknown-bus indices.
/// Every topological neighbor is always included regardless of its computed
/// value (even if it happens to evaluate to exactly zero at some iteration),
/// so the resulting sparsity *pattern* is identical across calls — required
/// for `RealSparseSystem`'s cached symbolic factorization to stay valid.
///
/// Block structure (H/N/M/L), matching the original dense Jacobian:
/// ```text
/// J = [ H  N ]   H = dP/d_ang, N = dP/d_vmag
///     [ M  L ]   M = dQ/d_ang, L = dQ/d_vmag
/// ```
#[allow(clippy::too_many_arguments)]
fn build_jacobian_triplets(
    buses: &[Bus],
    ybus: &YBusSparse,
    non_slack_idx: &[usize],
    pq_idx: &[usize],
    non_slack_pos: &[Option<usize>],
    pq_pos: &[Option<usize>],
    n_angle: usize,
    p_calc: &[f64],
    q_calc: &[f64],
) -> Vec<(usize, usize, f64)> {
    let vm: Vec<f64> = buses.iter().map(|b| b.voltage_mag).collect();
    let va: Vec<f64> = buses.iter().map(|b| b.voltage_ang).collect();
    let mut triplets = Vec::new();

    // H and N blocks: rows = non_slack_idx (P-mismatch equations).
    for (row_idx, &i) in non_slack_idx.iter().enumerate() {
        for &(k, y_ik) in ybus.row(i) {
            if k == i {
                // H_ii = -Q_i - V_i^2 * B_ii
                triplets.push((row_idx, row_idx, -q_calc[i] - vm[i].powi(2) * y_ik.im));
                // N_ii = P_i/V_i + V_i * G_ii (only if i has a magnitude unknown)
                if let Some(col_idx) = pq_pos[i] {
                    triplets.push((row_idx, n_angle + col_idx, p_calc[i] / vm[i] + vm[i] * y_ik.re));
                }
                continue;
            }
            let angle_ik = va[i] - va[k];
            if let Some(col_idx) = non_slack_pos[k] {
                // H_ik = V_i * V_k * (G_ik * sin(d_ik) - B_ik * cos(d_ik))
                let h_ik = vm[i] * vm[k] * (y_ik.re * angle_ik.sin() - y_ik.im * angle_ik.cos());
                triplets.push((row_idx, col_idx, h_ik));
            }
            if let Some(col_idx) = pq_pos[k] {
                // N_ik = V_i * (G_ik * cos(d_ik) + B_ik * sin(d_ik))
                let n_ik = vm[i] * (y_ik.re * angle_ik.cos() + y_ik.im * angle_ik.sin());
                triplets.push((row_idx, n_angle + col_idx, n_ik));
            }
        }
    }

    // M and L blocks: rows = pq_idx (Q-mismatch equations).
    for (row_idx, &i) in pq_idx.iter().enumerate() {
        for &(k, y_ik) in ybus.row(i) {
            if k == i {
                // M_ii = P_i - V_i^2 * G_ii; column is i's position among
                // *all* non-slack buses, not just PQ ones (may differ from
                // row_idx once PV buses exist).
                let col_idx = non_slack_pos[i].expect("a PQ bus is always non-slack");
                triplets.push((n_angle + row_idx, col_idx, p_calc[i] - vm[i].powi(2) * y_ik.re));
                // L_ii = Q_i/V_i - V_i * B_ii
                triplets.push((n_angle + row_idx, n_angle + row_idx, q_calc[i] / vm[i] - vm[i] * y_ik.im));
                continue;
            }
            let angle_ik = va[i] - va[k];
            if let Some(col_idx) = non_slack_pos[k] {
                // M_ik = -V_i * V_k * (G_ik * cos(d_ik) + B_ik * sin(d_ik))
                let m_ik = -vm[i] * vm[k] * (y_ik.re * angle_ik.cos() + y_ik.im * angle_ik.sin());
                triplets.push((n_angle + row_idx, col_idx, m_ik));
            }
            if let Some(col_idx) = pq_pos[k] {
                // L_ik = V_i * (G_ik * sin(d_ik) - B_ik * cos(d_ik))
                let l_ik = vm[i] * (y_ik.re * angle_ik.sin() - y_ik.im * angle_ik.cos());
                triplets.push((n_angle + row_idx, n_angle + col_idx, l_ik));
            }
        }
    }

    triplets
}

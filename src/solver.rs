use nalgebra::{DMatrix, DVector};
use nalgebra::Complex;
use super::types::{Bus, BusType};
use super::network::power_injections;

pub fn newton_raphson(buses: &mut [Bus], ybus: &DMatrix<Complex<f64>>, tol: f64, max_iter: usize) {

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

    for iter in 0..max_iter {
        // compute injections
        let (p_calc, q_calc) = power_injections(buses, ybus);

        // Build mismatch vector
        let mut mismatch = DVector::from_element(n_unknowns, 0.0);
        let mut mis_idx = 0;
        for &i in &non_slack_idx {
            // P mismatch for PV and PQ buses
            mismatch[mis_idx] = buses[i].p_spec - p_calc[i];
            mis_idx += 1;
        }
        for &i in &pq_idx {
            // Q mismatch for PQ buses
            mismatch[mis_idx] = buses[i].q_spec - q_calc[i];
            mis_idx += 1;
        }

        let max_mis = mismatch.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        println!("iter {}: max mismatch = {:.6e}", iter + 1, max_mis);
        if max_mis < tol {
            println!("Converged in {} iterations", iter + 1);
            return;
        }

        // Build Jacobian
        let j = build_jacobian(buses, ybus, &non_slack_idx, &pq_idx, &p_calc, &q_calc);

        // Solve
        let lu = j.lu();
        let dx = match lu.solve(&mismatch) {
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

fn build_jacobian(
    buses: &[Bus],
    ybus: &DMatrix<Complex<f64>>,
    non_slack_idx: &[usize],
    pq_idx: &[usize],
    p_calc: &[f64],
    q_calc: &[f64],
) -> DMatrix<f64> {
    // J = [ H  N ]
    //     [ M  L ]
    // H = dP/d_ang, N = dP/d_vmag
    // M = dQ/d_ang, L = dQ/d_vmag
    let n_angle = non_slack_idx.len();
    let n_vmag = pq_idx.len();
    let n_unknowns = n_angle + n_vmag;
    let mut j = DMatrix::from_element(n_unknowns, n_unknowns, 0.0);

    // Extract voltage magnitudes and angles for each bus
    let vm: Vec<f64> = buses.iter().map(|b| b.voltage_mag).collect();
    let va: Vec<f64> = buses.iter().map(|b| b.voltage_ang).collect();

    // Jacobian structure:
    // J = [ H  N ]
    //     [ M  L ]
    // H = dP/d_ang, N = dP/d_vmag
    // M = dQ/d_ang, L = dQ/d_vmag

    // Loop over non-slack buses for rows
    // First n_angle rows: P equations
    for (row_idx, &i) in non_slack_idx.iter().enumerate() {
        // H block (dP/d_ang)
        for (col_idx, &k) in non_slack_idx.iter().enumerate() {
            if i == k { // H_ii = dP_i/d_ang_i
                // H_ii = -Q_i - V_i^2 * B_ii
                j[(row_idx, col_idx)] = -q_calc[i] - vm[i].powi(2) * ybus[(i, i)].im;
            } else { // H_ik = dP_i/d_ang_k
                // H_ik = V_i * V_k * (G_ik * sin(d_i - d_k) - B_ik * cos(d_i - d_k))
                let y_ik = ybus[(i, k)];
                let angle_ik = va[i] - va[k];
                j[(row_idx, col_idx)] = vm[i] * vm[k] * (y_ik.re * angle_ik.sin() - y_ik.im * angle_ik.cos());
            }
        }
        // N block (dP/d_vmag)
        for (col_idx, &k) in pq_idx.iter().enumerate() {
            if i == k { // N_ii = dP_i/d_vmag_i
                // N_ii = P_i/V_i + V_i * G_ii
                j[(row_idx, n_angle + col_idx)] = p_calc[i] / vm[i] + vm[i] * ybus[(i, i)].re;
            } else { // N_ik = dP_i/d_vmag_k
                // N_ik = V_i * (G_ik * cos(d_i - d_k) + B_ik * sin(d_i - d_k))
                let y_ik = ybus[(i, k)];
                let angle_ik = va[i] - va[k];
                j[(row_idx, n_angle + col_idx)] = vm[i] * (y_ik.re * angle_ik.cos() + y_ik.im * angle_ik.sin());
            }
        }
    }

    // Loop over PQ buses for rows
    // Next n_vmag rows: Q equations
    for (row_idx, &i) in pq_idx.iter().enumerate() {
        // M block (dQ/d_ang)
        for (col_idx, &k) in non_slack_idx.iter().enumerate() {
            if i == k { // M_ii = dQ_i/d_ang_i
                // M_ii = P_i - V_i^2 * G_ii
                j[(n_angle + row_idx, col_idx)] = p_calc[i] - vm[i].powi(2) * ybus[(i, i)].re;
            } else { // M_ik = dQ_i/d_ang_k
                // M_ik = -V_i * V_k * (G_ik * cos(d_i - d_k) + B_ik * sin(d_i - d_k))
                let y_ik = ybus[(i, k)];
                let angle_ik = va[i] - va[k];
                j[(n_angle + row_idx, col_idx)] = -vm[i] * vm[k] * (y_ik.re * angle_ik.cos() + y_ik.im * angle_ik.sin());
            }
        }
        // L block (dQ/d_vmag)
        for (col_idx, &k) in pq_idx.iter().enumerate() {
             if i == k { // L_ii = dQ_i/d_vmag_i
                // L_ii = Q_i/V_i - V_i * B_ii
                j[(n_angle + row_idx, n_angle + col_idx)] = q_calc[i] / vm[i] - vm[i] * ybus[(i, i)].im;
            } else { // L_ik = dQ_i/d_vmag_k
                // L_ik = V_i * (G_ik * sin(d_i - d_k) - B_ik * cos(d_i - d_k))
                let y_ik = ybus[(i, k)];
                let angle_ik = va[i] - va[k];
                j[(n_angle + row_idx, n_angle + col_idx)] = vm[i] * (y_ik.re * angle_ik.sin() - y_ik.im * angle_ik.cos());
            }
        }
    }

    j
}

use nalgebra::{DMatrix, DVector};
use nalgebra::Complex;
use super::types::{Bus, Line, Line3Ph, Transformer};

pub fn build_ybus(n: usize, lines: &[Line], transformers: &[Transformer]) -> DMatrix<Complex<f64>> {
    let mut y = DMatrix::from_element(n, n, Complex::new(0.0, 0.0));
    for ln in lines {
        // Self-loop: pure shunt element (no series branch).
        if ln.from == ln.to {
            y[(ln.from, ln.from)] += Complex::new(0.0, ln.b_shunt);
            continue;
        }
        let z = Complex::new(ln.r, ln.x);
        // series admittance
        let y_line = Complex::new(1.0, 0.0) / z;
        // split shunt susceptance equally to both ends of line
        let b2 = Complex::new(0.0, ln.b_shunt / 2.0);
        // diagonal elements
        y[(ln.from, ln.from)] += y_line + b2;
        y[(ln.to, ln.to)] += y_line + b2;
        // off-diagonal elements
        y[(ln.from, ln.to)] -= y_line;
        y[(ln.to, ln.from)] -= y_line;
    }
    stamp_transformers(&mut y, transformers);
    y
}

/// Builds a 3N×3N phase-domain Y-bus from a list of three-phase lines.
///
/// Physical node `k` maps to rows/columns `3k`, `3k+1`, `3k+2` (phases a, b, c).
/// Sequence parameters are converted to the 3×3 primitive admittance matrix via
/// the symmetrical-components transform; off-diagonal terms couple phases when
/// r0≠r1 or x0≠x1.
pub fn build_ybus_3ph(n: usize, lines: &[Line3Ph]) -> DMatrix<Complex<f64>> {
    let zero = Complex::new(0.0, 0.0);
    let mut y = DMatrix::from_element(3 * n, 3 * n, zero);

    for ln in lines {
        let y_c1 = Complex::new(0.0, ln.b1);
        let y_c0 = Complex::new(0.0, ln.b0);

        if ln.from == ln.to {
            // Pure shunt: add full 3×3 shunt matrix to the diagonal block.
            let d = (y_c0 + 2.0 * y_c1) / 3.0;
            let o = (y_c0 - y_c1) / 3.0;
            let fi = ln.from;
            for p in 0..3 {
                for q in 0..3 {
                    let val = if p == q { d } else { o };
                    y[(3 * fi + p, 3 * fi + q)] += val;
                }
            }
            continue;
        }

        let y1 = Complex::new(1.0, 0.0) / Complex::new(ln.r1, ln.x1);
        let y0 = Complex::new(1.0, 0.0) / Complex::new(ln.r0, ln.x0);

        // 3×3 series admittance: diagonal (y0+2y1)/3, off-diagonal (y0-y1)/3.
        let d_s = (y0 + 2.0 * y1) / 3.0;
        let o_s = (y0 - y1) / 3.0;
        // Half-shunt per terminal.
        let d_sh = (y_c0 + 2.0 * y_c1) / 6.0;
        let o_sh = (y_c0 - y_c1) / 6.0;

        let fi = ln.from;
        let ti = ln.to;
        for p in 0..3 {
            for q in 0..3 {
                let ys = if p == q { d_s } else { o_s };
                let ysh = if p == q { d_sh } else { o_sh };
                y[(3 * fi + p, 3 * fi + q)] += ys + ysh;
                y[(3 * ti + p, 3 * ti + q)] += ys + ysh;
                y[(3 * fi + p, 3 * ti + q)] -= ys;
                y[(3 * ti + p, 3 * fi + q)] -= ys;
            }
        }
    }
    y
}

/// Stamps two-winding transformer contributions into an existing Y-bus.
///
/// Implements PGM's π-equivalent model: y_shunt is split equally between both
/// terminals. The complex tap ratio `t.tap = k·exp(jθ)` carries both the
/// off-nominal magnitude k and the vector-group phase shift θ.
///
/// Status rules (mirrors PGM's `calc_param_y_sym`):
///   (1,1): Y[ff] += (y_s+y_sh/2)/k², Y[tt] += y_s+y_sh/2,
///          Y[ft] -= y_s/conj(a), Y[tf] -= y_s/a
///   (1,0)/(0,1): effective shunt = y_sh/2 + 1/(1/y_s + 2/y_sh) at connected end
///   (0,0): no contribution
fn stamp_transformers(ybus: &mut DMatrix<Complex<f64>>, transformers: &[Transformer]) {
    let one = Complex::new(1.0, 0.0);
    for t in transformers {
        let k = t.tap.norm();
        match (t.from_status, t.to_status) {
            (1, 1) => {
                let y_diag = t.y_series + t.y_shunt * 0.5;
                ybus[(t.from, t.from)] += y_diag / (k * k);
                ybus[(t.to, t.to)] += y_diag;
                ybus[(t.from, t.to)] -= t.y_series / t.tap.conj();
                ybus[(t.to, t.from)] -= t.y_series / t.tap;
            }
            (1, 0) | (0, 1) => {
                let branch_shunt = t.y_shunt * 0.5
                    + one / (one / t.y_series + Complex::new(2.0, 0.0) / t.y_shunt);
                if t.from_status == 1 {
                    ybus[(t.from, t.from)] += branch_shunt / (k * k);
                } else {
                    ybus[(t.to, t.to)] += branch_shunt;
                }
            }
            _ => {}
        }
    }
}

/// Computes per-unit source impedance (r, x) from short-circuit power and R/X ratio.
pub fn source_impedance_pu(u_ref: f64, sk: f64, rx_ratio: f64, s_base_va: f64) -> (f64, f64) {
    let z_s_pu = u_ref * u_ref * s_base_va / sk;
    let x_s = z_s_pu / (rx_ratio * rx_ratio + 1.0_f64).sqrt();
    (rx_ratio * x_s, x_s)
}

/// Computes the complex off-nominal tap ratio k·exp(j·clock·π/6) from transformer nameplate data.
pub fn transformer_tap(
    u1: f64, u2: f64, tap_side: u8,
    tap_pos: i32, tap_nom: i32, tap_size: f64, clock: i32,
) -> Complex<f64> {
    let k = if tap_side == 0 {
        (u1 + (tap_pos - tap_nom) as f64 * tap_size) / u1
    } else {
        u2 / (u2 + (tap_pos - tap_nom) as f64 * tap_size)
    };
    Complex::from_polar(k, clock as f64 * std::f64::consts::PI / 6.0)
}

/// Computes per-unit series and shunt admittances from transformer nameplate data.
/// Both are referenced to the to-side (u2) voltage base.
pub fn transformer_admittances(
    u2: f64, sn: f64, uk: f64, pk: f64, i0: f64, p0: f64, s_base_va: f64,
) -> (Complex<f64>, Complex<f64>) {
    let base_y_to = s_base_va / (u2 * u2);
    let r_ohm = pk * u2 * u2 / (sn * sn);
    let x_ohm = ((uk * u2 * u2 / sn).powi(2) - r_ohm * r_ohm).sqrt();
    let y_series = Complex::new(1.0, 0.0) / Complex::new(r_ohm, x_ohm) / base_y_to;
    let g_fe = p0 / (u2 * u2);
    let y_sh_abs = i0 * sn / (u2 * u2);
    let b_m = -(y_sh_abs * y_sh_abs - g_fe * g_fe).sqrt();
    let y_shunt = Complex::new(g_fe, b_m) / base_y_to;
    (y_series, y_shunt)
}

pub fn power_injections(
    buses: &[Bus],
    ybus: &DMatrix<Complex<f64>>,
) -> (Vec<f64>, Vec<f64>) {
    // Calculates the complex power injection into each bus.
    // S = V .* conj(I) where I = Ybus * V
    // S_k = V_k * I_k^*
    let n = buses.len();
    let mut p = vec![0.0; n];
    let mut q = vec![0.0; n];

    let v = DVector::from_iterator(
        n,
        buses.iter().map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang)),
    );

    let i = ybus * v.clone();
    let s = v.component_mul(&i.conjugate());

    for k in 0..n {
        p[k] = s[k].re;
        q[k] = s[k].im;
    }

    (p, q)
}

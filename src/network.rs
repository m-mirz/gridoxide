use nalgebra::{DMatrix, DVector};
use nalgebra::Complex;
use super::types::{Bus, Line, Line3Ph};

pub fn build_ybus(n: usize, lines: &[Line]) -> DMatrix<Complex<f64>> {
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

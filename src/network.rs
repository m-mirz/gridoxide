use nalgebra::{DMatrix, DVector};
use nalgebra::Complex;
use super::types::{Bus, Line};

pub fn build_ybus(n: usize, lines: &[Line]) -> DMatrix<Complex<f64>> {
    let mut y = DMatrix::from_element(n, n, Complex::new(0.0, 0.0));
    for ln in lines {
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

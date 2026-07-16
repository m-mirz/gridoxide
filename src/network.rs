use nalgebra::{DMatrix, DVector};
use nalgebra::Complex;
use super::types::{Bus, BusType, Line, Line3Ph, Transformer, Transformer3PhSeq, ZipKind};

/// A lumped shunt admittance to be added to the Y-bus diagonal at bus `at`.
pub struct ShuntAdm {
    pub at: usize,
    pub y: Complex<f64>,
}

/// A three-phase shunt admittance, given as positive- and zero-sequence values.
pub struct ShuntAdm3Ph {
    pub at: usize,
    pub y1: Complex<f64>,
    pub y0: Complex<f64>,
}

/// Converts sequence-domain admittance (y1, y0) to the 3×3 phase-domain shunt
/// tensor: diagonal `(2·y1+y0)/3`, off-diagonal `(y0−y1)/3`.
fn seq_to_phase_shunt(y1: Complex<f64>, y0: Complex<f64>) -> [[Complex<f64>; 3]; 3] {
    let d = (y0 + 2.0 * y1) / 3.0;
    let o = (y0 - y1) / 3.0;
    let mut m = [[o; 3]; 3];
    for i in 0..3 {
        m[i][i] = d;
    }
    m
}

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
            let m = seq_to_phase_shunt(y_c1, y_c0);
            let fi = ln.from;
            for p in 0..3 {
                for q in 0..3 {
                    y[(3 * fi + p, 3 * fi + q)] += m[p][q];
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

/// Computes the complex off-nominal tap ratio `k · exp(j·clock·π/6)` from a
/// numerator/denominator voltage pair. `u_num`/`u_denom` are the (possibly
/// tap-adjusted) nameplate voltages on the two sides; the magnitude ratio
/// `u_num/u_denom` gives `k`. Used both by `transformer_tap` (2-winding) and
/// the three-winding/asymmetric-transformer paths.
pub fn tap_ratio_from_voltages(u_num: f64, u_denom: f64, clock: i32) -> Complex<f64> {
    Complex::from_polar(u_num / u_denom, clock as f64 * std::f64::consts::PI / 6.0)
}

/// Computes the four branch admittance entries `[yff, yft, ytf, ytt]` for a
/// two-winding transformer branch. Implements PGM's π-equivalent model:
/// `y_shunt` is split equally between both terminals. The complex tap ratio
/// `tap = k·exp(jθ)` carries both the off-nominal magnitude `k` and the
/// vector-group phase shift `θ`.
///
/// Status rules (mirrors PGM's `calc_param_y_sym`):
///   (1,1): yff = (y_s+y_sh/2)/k², ytt = y_s+y_sh/2,
///          yft = -y_s/conj(a), ytf = -y_s/a
///   (1,0)/(0,1): effective shunt = y_sh/2 + 1/(1/y_s + 2/y_sh) at connected end
///   (0,0): all zero
pub fn branch_calc_param(
    y_series: Complex<f64>, y_shunt: Complex<f64>, tap: Complex<f64>,
    from_status: u8, to_status: u8,
) -> [Complex<f64>; 4] {
    let zero = Complex::new(0.0, 0.0);
    let one = Complex::new(1.0, 0.0);
    let k = tap.norm();
    match (from_status, to_status) {
        (1, 1) => {
            let y_diag = y_series + y_shunt * 0.5;
            [y_diag / (k * k), -y_series / tap.conj(), -y_series / tap, y_diag]
        }
        (1, 0) => {
            let branch_shunt = y_shunt * 0.5
                + one / (one / y_series + Complex::new(2.0, 0.0) / y_shunt);
            [branch_shunt / (k * k), zero, zero, zero]
        }
        (0, 1) => {
            let branch_shunt = y_shunt * 0.5
                + one / (one / y_series + Complex::new(2.0, 0.0) / y_shunt);
            [zero, zero, zero, branch_shunt]
        }
        _ => [zero, zero, zero, zero],
    }
}

/// Stamps two-winding transformer contributions into an existing Y-bus, via
/// `branch_calc_param` for the per-branch [yff, yft, ytf, ytt] entries.
fn stamp_transformers(ybus: &mut DMatrix<Complex<f64>>, transformers: &[Transformer]) {
    for t in transformers {
        let [yff, yft, ytf, ytt] =
            branch_calc_param(t.y_series, t.y_shunt, t.tap, t.from_status, t.to_status);
        ybus[(t.from, t.from)] += yff;
        ybus[(t.from, t.to)] += yft;
        ybus[(t.to, t.from)] += ytf;
        ybus[(t.to, t.to)] += ytt;
    }
}

/// Converts sequence-domain admittances `(y0, y1, y2)` to the 3×3 phase-domain
/// tensor via the Fortescue transform `Yabc = A · diag(y0,y1,y2) · A⁻¹`, where
/// `A = [[1,1,1],[1,a²,a],[1,a,a²]]`, `a = exp(j·2π/3)`. Degenerates to
/// `seq_to_phase_shunt`'s `(2y1+y0)/3` / `(y0−y1)/3` formula when `y1 == y2`.
fn fortescue_to_phase(y0: Complex<f64>, y1: Complex<f64>, y2: Complex<f64>) -> [[Complex<f64>; 3]; 3] {
    let one = Complex::new(1.0, 0.0);
    let a = Complex::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
    let a2 = a * a;
    let big_a = [[one, one, one], [one, a2, a], [one, a, a2]];
    let a_inv = [[one, one, one], [one, a, a2], [one, a2, a]];
    let y_seq = [y0, y1, y2];
    let mut out = [[Complex::new(0.0, 0.0); 3]; 3];
    for p in 0..3 {
        for q in 0..3 {
            let mut sum = Complex::new(0.0, 0.0);
            for k in 0..3 {
                sum += big_a[p][k] * y_seq[k] * a_inv[k][q] / 3.0;
            }
            out[p][q] = sum;
        }
    }
    out
}

/// Computes sequence-domain `(y0, y1, y2)` branch admittance parameters
/// (each `[yff, yft, ytf, ytt]`) for a two-winding transformer, needed for
/// asymmetric (3-phase) power flow.
///
/// Positive/negative sequence reuse `branch_calc_param` with the tap rotated
/// by ±clock (mirrors PGM's `asym_calc_param`). Zero sequence supports exactly
/// two winding combinations, matching gridoxide's test fixtures — any other
/// combination panics rather than silently computing physically wrong results:
///
/// - Dyn (`winding_from` = delta = 2, `winding_to` = wye_n = 1): PGM's "*yn"
///   branch — a ground path via the magnetizing branch plus, since the "from"
///   side is a delta winding, an additional path via the series impedance —
///   plus the artificial low-susceptance term PGM adds on the delta side
///   (which has no zero-sequence path of its own).
/// - YNyn (`winding_from` = `winding_to` = wye_n = 1): PGM's full two-port
///   zero-sequence branch, `z0_series = 1/y_series + 3·(z_grounding_to +
///   z_grounding_from/k²)`. gridoxide has no grounding-impedance fields on
///   `PgmTransformer` (none of its fixtures specify them, and PGM itself
///   defaults unset grounding r/x to 0), so both terms are taken as zero,
///   collapsing `z0_series` to `1/y_series` — i.e. `y0_series = y_series`.
///   The zero-sequence tap picks up a 180° flip for clock ∈ {2, 6, 10}
///   (reverse-connected variants); otherwise it matches the untransformed
///   magnitude `k`.
pub fn transformer_seq_params(
    y_series: Complex<f64>, y_shunt: Complex<f64>, tap: Complex<f64>,
    from_status: u8, to_status: u8,
    winding_from: u8, winding_to: u8,
    sn: f64, uk: f64, s_base_va: f64, clock: i32,
) -> ([Complex<f64>; 4], [Complex<f64>; 4], [Complex<f64>; 4]) {
    let y1 = branch_calc_param(y_series, y_shunt, tap, from_status, to_status);
    let y2 = branch_calc_param(y_series, y_shunt, tap.conj(), from_status, to_status);

    let y0 = match (winding_from, winding_to) {
        (2, 1) => {
            let mut y0 = [Complex::new(0.0, 0.0); 4];
            if to_status == 1 {
                y0[3] = y_shunt + y_series;
            }
            if from_status == 1 {
                let low_susceptance = -1e-8 * sn / s_base_va / uk;
                y0[0] += Complex::new(0.0, low_susceptance);
            }
            y0
        }
        (1, 1) => {
            let phase_shift_0 = if matches!(clock, 2 | 6 | 10) { std::f64::consts::PI } else { 0.0 };
            let k = tap.norm();
            let tap0 = Complex::from_polar(k, phase_shift_0);
            branch_calc_param(y_series, y_shunt, tap0, from_status, to_status)
        }
        _ => panic!(
            "transformer_seq_params only supports Dyn (winding_from=2, winding_to=1) or YNyn (winding_from=1, winding_to=1); got ({winding_from}, {winding_to})"
        ),
    };

    (y0, y1, y2)
}

/// Stamps asymmetric transformer contributions into a 3N×3N phase-domain Y-bus.
pub fn stamp_transformers_3ph(ybus: &mut DMatrix<Complex<f64>>, transformers: &[Transformer3PhSeq]) {
    for t in transformers {
        let blocks = [(t.from, t.from), (t.from, t.to), (t.to, t.from), (t.to, t.to)];
        for (i, &(bi, bj)) in blocks.iter().enumerate() {
            let m = fortescue_to_phase(t.y0[i], t.y1[i], t.y2[i]);
            for p in 0..3 {
                for q in 0..3 {
                    ybus[(3 * bi + p, 3 * bj + q)] += m[p][q];
                }
            }
        }
    }
}

/// Adds a set of lumped shunt admittances to the Y-bus diagonal.
pub fn stamp_shunts(ybus: &mut DMatrix<Complex<f64>>, shunts: &[ShuntAdm]) {
    for s in shunts {
        ybus[(s.at, s.at)] += s.y;
    }
}

/// Adds a set of three-phase shunt admittances to the phase-domain Y-bus diagonal blocks.
pub fn stamp_shunts_3ph(ybus: &mut DMatrix<Complex<f64>>, shunts: &[ShuntAdm3Ph]) {
    for s in shunts {
        let m = seq_to_phase_shunt(s.y1, s.y0);
        for p in 0..3 {
            for q in 0..3 {
                ybus[(3 * s.at + p, 3 * s.at + q)] += m[p][q];
            }
        }
    }
}

/// Computes per-unit source impedance (r, x) from short-circuit power and R/X
/// ratio. Matches PGM's `Source::math_param` (`z_abs = base_power_3p / sk`,
/// with no dependence on `u_ref` — the source's reference voltage only sets
/// the fixed voltage at the virtual slack bus, not the branch impedance).
pub fn source_impedance_pu(sk: f64, rx_ratio: f64, s_base_va: f64) -> (f64, f64) {
    let z_s_pu = s_base_va / sk;
    let x_s = z_s_pu / (rx_ratio * rx_ratio + 1.0_f64).sqrt();
    (rx_ratio * x_s, x_s)
}

/// Computes per-unit positive- and zero-sequence source impedance
/// `(r1, x1, r0, x0)`, where `z01_ratio = z0 / z1` (PGM convention), so the
/// zero-sequence impedance is `z1 * z01_ratio`.
pub fn source_impedance_pu_seq(
    sk: f64, rx_ratio: f64, z01_ratio: f64, s_base_va: f64,
) -> (f64, f64, f64, f64) {
    let (r1, x1) = source_impedance_pu(sk, rx_ratio, s_base_va);
    let (r0, x0) = (r1 * z01_ratio, x1 * z01_ratio);
    (r1, x1, r0, x0)
}

/// Computes the complex off-nominal tap ratio k·exp(j·clock·π/6) from transformer nameplate data.
pub fn transformer_tap(
    u1: f64, u2: f64, tap_side: u8,
    tap_pos: i32, tap_nom: i32, tap_size: f64, clock: i32,
) -> Complex<f64> {
    let delta = (tap_pos - tap_nom) as f64 * tap_size;
    if tap_side == 0 {
        tap_ratio_from_voltages(u1 + delta, u1, clock)
    } else {
        tap_ratio_from_voltages(u2, u2 + delta, clock)
    }
}

/// Computes per-unit series and shunt admittances from transformer nameplate data.
/// Both are referenced to the to-side (u2) voltage base.
/// Like `transformer_admittances`, but with separately specified nameplate
/// ("to"-field, used for absolute impedance/shunt magnitude scaling) and rated
/// ("to"-base, used for the per-unit base) to-side voltages — needed when the
/// two differ, as for a three-winding transformer's internal star leg, whose
/// nameplate voltage tracks the tapped side while its per-unit base stays
/// pinned to the physical node's rated voltage. `uk`/`pk` may be negative (as
/// PGM's three-winding delta→wye conversion can produce); the sign of `uk` is
/// carried onto the series reactance, matching PGM's `transformer_params()`.
pub fn transformer_admittances_ex(
    u_field: f64, u_base: f64, sn: f64, uk: f64, pk: f64, i0: f64, p0: f64, s_base_va: f64,
) -> (Complex<f64>, Complex<f64>) {
    let base_y_to = s_base_va / (u_base * u_base);
    let uk_sign = if uk >= 0.0 { 1.0 } else { -1.0 };
    let z_abs = uk.abs() * u_field * u_field / sn;
    let r_ohm = pk * u_field * u_field / (sn * sn);
    let x_sq = z_abs * z_abs - r_ohm * r_ohm;
    let x_ohm = uk_sign * if x_sq > 0.0 { x_sq.sqrt() } else { 0.0 };
    let y_series = Complex::new(1.0, 0.0) / Complex::new(r_ohm, x_ohm) / base_y_to;
    let g_fe = p0 / (u_field * u_field);
    let y_sh_abs = i0 * sn / (u_field * u_field);
    let b_sq = y_sh_abs * y_sh_abs - g_fe * g_fe;
    let b_m = if b_sq > 0.0 { -b_sq.sqrt() } else { 0.0 };
    let y_shunt = Complex::new(g_fe, b_m) / base_y_to;
    (y_series, y_shunt)
}

pub fn transformer_admittances(
    u2: f64, sn: f64, uk: f64, pk: f64, i0: f64, p0: f64, s_base_va: f64,
) -> (Complex<f64>, Complex<f64>) {
    transformer_admittances_ex(u2, u2, sn, uk, pk, i0, p0, s_base_va)
}

/// Computes the three-winding transformer's delta→wye (star-equivalent)
/// short-circuit voltage and loss parameters `(uk_T1, uk_T2, uk_T3)` and
/// `(pk_T1, pk_T2, pk_T3)`, referenced to side 1's power base. Mirrors PGM's
/// `ThreeWindingTransformer::calculate_uk`/`calculate_pk` (tap-dependent
/// `uk_min`/`uk_max` adjustment is not implemented — this three-winding
/// support only covers fixtures with no such tap-dependent impedance range).
pub fn three_winding_star_params(
    sn_1: f64, sn_2: f64, sn_3: f64,
    uk_12: f64, uk_13: f64, uk_23: f64,
    pk_12: f64, pk_13: f64, pk_23: f64,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let uk_12r = uk_12 * sn_1 / sn_1.min(sn_2);
    let uk_13r = uk_13 * sn_1 / sn_1.min(sn_3);
    let uk_23r = uk_23 * sn_1 / sn_2.min(sn_3);
    let uk_t1p = 0.5 * (uk_12r + uk_13r - uk_23r);
    let uk_t2p = 0.5 * (uk_12r + uk_23r - uk_13r);
    let uk_t3p = 0.5 * (uk_13r + uk_23r - uk_12r);
    let uk = (uk_t1p, uk_t2p * (sn_2 / sn_1), uk_t3p * (sn_3 / sn_1));

    let pk_12r = pk_12 * (sn_1 / sn_1.min(sn_2)).powi(2);
    let pk_13r = pk_13 * (sn_1 / sn_1.min(sn_3)).powi(2);
    let pk_23r = pk_23 * (sn_1 / sn_2.min(sn_3)).powi(2);
    let pk_t1p = 0.5 * (pk_12r + pk_13r - pk_23r);
    let pk_t2p = 0.5 * (pk_12r + pk_23r - pk_13r);
    let pk_t3p = 0.5 * (pk_13r + pk_23r - pk_12r);
    let pk = (pk_t1p, pk_t2p * (sn_2 / sn_1).powi(2), pk_t3p * (sn_3 / sn_1).powi(2));

    (uk, pk)
}

/// Computes the effective (P, Q) injection at a bus at its current voltage
/// magnitude: `p_spec`/`q_spec` (constant-power) plus any ZIP terms evaluated
/// at `bus.voltage_mag`. For a bus with no ZIP terms this is exactly
/// `(bus.p_spec, bus.q_spec)`.
pub fn effective_injection(bus: &Bus) -> (f64, f64) {
    let mut s = Complex::new(bus.p_spec, bus.q_spec);
    let vmag = bus.voltage_mag;
    for zt in &bus.zip_terms {
        s += match zt.kind {
            ZipKind::ConstPower => zt.s_const,
            ZipKind::ConstImpedance => zt.s_const * vmag * vmag,
            ZipKind::ConstCurrent => zt.s_const * vmag,
        };
    }
    (s.re, s.im)
}

/// Computes a smarter-than-flat-start initial voltage guess for Newton-Raphson
/// by solving one linearized constant-admittance system, mirroring PGM's
/// `NewtonRaphsonPFSolver::initialize_derived_solver`. Each energized (`PQ`)
/// bus's injection is approximated as a constant admittance `y = -conj(S)`
/// evaluated at the flat-start `|V|=1` assumption and added to the Y-bus
/// diagonal; `Slack` buses (both real sources and de-energized nodes, which
/// gridoxide models as fixed `Slack` buses at V=0) keep their already-set
/// voltage. The resulting linear system `Y'_nn·U_n = -Y'_ns·U_s` is solved for
/// the unknown `PQ` bus voltages via LU decomposition, and `voltage_mag`/
/// `voltage_ang` are overwritten with the result. If the reduced system is
/// singular, buses are left at their prior (flat-start) values. PGM applies no
/// step damping or voltage clamping beyond this — a better initial guess is
/// its only robustness mechanism, needed on networks combining weak sources
/// with large transformer phase shifts where plain flat-start NR diverges.
pub fn linear_initial_guess(buses: &mut [Bus], ybus: &DMatrix<Complex<f64>>) {
    let n = buses.len();
    let mut yp = ybus.clone();
    for (i, b) in buses.iter().enumerate() {
        if !matches!(b.bus_type, BusType::PQ) {
            continue;
        }
        let (p, q) = effective_injection(b);
        yp[(i, i)] += -Complex::new(p, q).conj();
    }

    let unknown_idx: Vec<usize> = (0..n).filter(|&i| matches!(buses[i].bus_type, BusType::PQ)).collect();
    if unknown_idx.is_empty() {
        return;
    }
    let m = unknown_idx.len();

    let mut a = DMatrix::from_element(m, m, Complex::new(0.0, 0.0));
    let mut rhs = DVector::from_element(m, Complex::new(0.0, 0.0));
    for (r, &i) in unknown_idx.iter().enumerate() {
        for (c, &j) in unknown_idx.iter().enumerate() {
            a[(r, c)] = yp[(i, j)];
        }
        let mut b = Complex::new(0.0, 0.0);
        for k in 0..n {
            if unknown_idx.binary_search(&k).is_ok() {
                continue;
            }
            let u_k = Complex::from_polar(buses[k].voltage_mag, buses[k].voltage_ang);
            b -= yp[(i, k)] * u_k;
        }
        rhs[r] = b;
    }

    if let Some(sol) = a.lu().solve(&rhs) {
        for (r, &i) in unknown_idx.iter().enumerate() {
            buses[i].voltage_mag = sol[r].norm();
            buses[i].voltage_ang = sol[r].arg();
        }
    }
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

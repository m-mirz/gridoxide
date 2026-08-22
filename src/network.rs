use num_complex::Complex;
use super::types::{Bus, BusType, Line, Line3Ph, Transformer, Transformer3PhSeq, ZipKind};
use super::sparse;

/// A lumped shunt admittance to be added to the Y-bus diagonal at bus `at`.
#[derive(Clone, Copy, Debug, PartialEq)]
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
pub(crate) fn seq_to_phase_shunt(y1: Complex<f64>, y0: Complex<f64>) -> [[Complex<f64>; 3]; 3] {
    let d = (y0 + 2.0 * y1) / 3.0;
    let o = (y0 - y1) / 3.0;
    let mut m = [[o; 3]; 3];
    for i in 0..3 {
        m[i][i] = d;
    }
    m
}

/// A mutable Y-bus under construction: a COO (triplet) accumulator. Entries
/// at the same `(row, col)` are summed once finalized via `finish()` — this
/// gives the same accumulation semantics as the old dense `ybus[(i,j)] +=
/// val`, including for parallel branches between the same two buses (e.g.
/// `symmetric/distribution-case`'s two parallel transformers).
pub struct YBus {
    n: usize,
    entries: Vec<(usize, usize, Complex<f64>)>,
}

impl YBus {
    pub fn new(n: usize) -> Self {
        Self { n, entries: Vec::new() }
    }

    pub fn add(&mut self, i: usize, j: usize, val: Complex<f64>) {
        self.entries.push((i, j, val));
    }

    /// The raw, unconsolidated COO triplets, for callers that need to modify
    /// the matrix *by column* before solving.
    ///
    /// [`finish`](Self::finish) is the normal exit and gives a row-major
    /// structure ([`YBusSparse::row`]) that cannot serve a column-wise
    /// rewrite. `shortcircuit` needs exactly that: a bolted fault zeroes the
    /// faulted bus's whole column, which is a filter over these triplets.
    ///
    /// **Zeroing a column means removing its entries, not adding a
    /// compensating one.** Duplicate `(row, col)` triplets are *summed*, both
    /// here in `finish` and in `sparse::solve_complex`'s own
    /// `try_new_from_triplets`, so an added negative would cancel only the
    /// entries it was computed against and silently leave anything else.
    pub fn into_entries(self) -> Vec<(usize, usize, Complex<f64>)> {
        self.entries
    }

    /// Consolidates the accumulated triplets into the frozen, sparse form
    /// used for the actual power-flow solve. Call once all `build_ybus*`/
    /// `stamp_*` contributions have been added.
    pub fn finish(self) -> YBusSparse {
        // Consolidate duplicate (i, j) entries and group by row, giving each
        // row's actual admittance neighbors (including its own diagonal) —
        // used by `linear_initial_guess` and `build_jacobian` to walk only
        // real neighbors instead of every other bus.
        let mut merged: std::collections::HashMap<(usize, usize), Complex<f64>> = std::collections::HashMap::new();
        for &(i, j, v) in &self.entries {
            *merged.entry((i, j)).or_insert(Complex::new(0.0, 0.0)) += v;
        }
        let mut adjacency: Vec<Vec<(usize, Complex<f64>)>> = vec![Vec::new(); self.n];
        for (&(i, j), &v) in &merged {
            adjacency[i].push((j, v));
        }
        for row in &mut adjacency {
            row.sort_unstable_by_key(|&(j, _)| j);
        }
        let matrix = sparse::SparseMatrix::build(self.n, &self.entries)
            .expect("Y-bus triplet set should always form a valid sparse matrix");
        YBusSparse { n: self.n, adjacency, matrix }
    }
}

/// A finalized, frozen Y-bus: consolidated per-row admittance neighbors (for
/// sparse-aware assembly of the linear initial guess and the Jacobian) plus
/// a ready-to-use sparse matrix (for `power_injections`'s mat-vec, needed
/// every Newton-Raphson iteration). Built once via `YBus::finish`.
#[derive(Clone)]
pub struct YBusSparse {
    n: usize,
    adjacency: Vec<Vec<(usize, Complex<f64>)>>,
    matrix: sparse::SparseMatrix,
}

impl YBusSparse {
    pub fn n(&self) -> usize {
        self.n
    }

    /// The `(col, value)` pairs for row `i`'s actual admittance neighbors
    /// (including the diagonal), sorted by column index.
    pub fn row(&self, i: usize) -> &[(usize, Complex<f64>)] {
        &self.adjacency[i]
    }

    /// The value at `(i, j)`, or zero if there's no entry there.
    pub fn get(&self, i: usize, j: usize) -> Complex<f64> {
        self.adjacency[i]
            .binary_search_by_key(&j, |&(col, _)| col)
            .map(|idx| self.adjacency[i][idx].1)
            .unwrap_or(Complex::new(0.0, 0.0))
    }

    pub fn mul_vec(&self, v: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.matrix.mul_vec(v)
    }
}

/// Groups buses into connected components via the Y-bus's actual admittance
/// graph — two buses are in the same component iff there's a path of
/// nonzero off-diagonal Y-bus entries between them (`row(i)`'s own diagonal
/// self-entry, `j == i`, is skipped, since it's a shunt term, not a branch
/// to another bus). Each returned `Vec<usize>` is one component's member
/// bus indices, sorted ascending; components are returned in ascending
/// order of their first (lowest-index) member.
///
/// Generic over the *finished* Y-bus rather than any particular input
/// format's own `Line`/`Transformer`/branch-status representation, so it
/// applies uniformly to native JSON, PGM-JSON, and CGMES input with no
/// format-specific code — see `classify`/`mark_unreferenced_islands` for
/// what this partition is used for.
pub fn connected_components(ybus: &YBusSparse) -> Vec<Vec<usize>> {
    let n = ybus.n();
    let mut visited = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut members = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(i) = stack.pop() {
            members.push(i);
            for &(j, y) in ybus.row(i) {
                // A structurally-present but numerically *zero* entry does not
                // connect anything. `build_ybus` stamps one for every
                // transformer regardless of terminal status, and
                // `build_ybus_with_outages` does the same deliberately, so
                // ignoring the value here would call an out-of-service branch a
                // connection — and leave the bus behind it with an all-zero
                // Jacobian row inside somebody else's island.
                if j != i && y != Complex::new(0.0, 0.0) && !visited[j] {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
        members.sort_unstable();
        components.push(members);
    }
    components
}

/// A single connected component's slack-bus classification, computed once
/// up front and trusted unconditionally thereafter — see
/// `mark_unreferenced_islands`'s doc comment for why an `AmbiguousReferenceBus`
/// verdict is never later overwritten by a numerically-convergent-looking
/// post-hoc mismatch check. `pub(crate)`, not part of the public API: it's
/// an intermediate value `lib.rs`'s orchestration passes to
/// `solver::finish_island_reports`, not something downstream users are
/// meant to construct or match on directly — they get `solver::IslandReport`
/// instead.
pub(crate) enum Verdict {
    NoReferenceBus,
    AmbiguousReferenceBus,
    Solvable,
}

pub(crate) struct Classified {
    pub(crate) bus_indices: Vec<usize>,
    /// The `Slack` bus(es) found in this component, captured *before*
    /// `mark_unreferenced_islands` runs: empty for `NoReferenceBus`, exactly
    /// one for `Solvable`, two-or-more for `AmbiguousReferenceBus`.
    pub(crate) slack_indices: Vec<usize>,
    pub(crate) verdict: Verdict,
}

/// Classifies each connected component (from `connected_components`) by how
/// many `Slack` buses it already contains. Exactly one is the normal,
/// solvable case; the other two counts are the situations
/// `mark_unreferenced_islands`/`solver::finish_island_reports` need to
/// handle specially.
pub(crate) fn classify(buses: &[Bus], components: &[Vec<usize>]) -> Vec<Classified> {
    components
        .iter()
        .map(|members| {
            // A de-energized bus is `Slack` at `V = 0` — a placeholder, not a
            // reference (see `mark_unreferenced_islands`). Counting one as a
            // reference leaves the component "solvable" with nothing to solve
            // against, and a neighbouring `PQ` bus then gets an identically
            // zero angle equation: `H_ii = −Q_i − V_i²B_ii`, whose two terms
            // cancel exactly when the only neighbour sits at zero volts. That
            // is a structurally singular row, and it is how this surfaced —
            // Svedala's node-breaker import produces such pairs, because the
            // finer partition stops merging a dead node into a live one.
            let slack_indices: Vec<usize> = members
                .iter()
                .copied()
                .filter(|&i| buses[i].bus_type == BusType::Slack && buses[i].voltage_mag != 0.0)
                .collect();
            let verdict = match slack_indices.len() {
                0 => Verdict::NoReferenceBus,
                1 => Verdict::Solvable,
                _ => Verdict::AmbiguousReferenceBus,
            };
            Classified { bus_indices: members.clone(), slack_indices, verdict }
        })
        .collect()
}

/// For every component with no `Slack` bus of its own, pins every member
/// bus to a fixed, zero-injection placeholder (`V = 0`, no P/Q) rather than
/// solving it as an ordinary PQ region — mirrors `cgmes.rs`'s existing
/// de-energized-bus handling exactly. Deliberately does **not** auto-promote
/// any PV/PQ bus in such a component to `Slack`: there is no principled way
/// to fabricate a reference voltage/angle for a genuinely sourceless
/// island, and every unit/sign-convention bug this project has actually
/// fixed got fixed by matching verified physical or reference-implementation
/// behavior, never by guessing — inventing a slack here would repeat that
/// same mistake class.
///
/// `AmbiguousReferenceBus` components are deliberately left untouched here
/// (not mutated at all): their non-slack buses stay fully live in the
/// shared Newton-Raphson system, since there's no safe placeholder value for
/// them either. See `solver::IslandStatus::AmbiguousReferenceBus`'s doc
/// comment for the resulting caveat.
pub(crate) fn mark_unreferenced_islands(buses: &mut [Bus], classified: &[Classified]) {
    for c in classified {
        if matches!(c.verdict, Verdict::NoReferenceBus) {
            for &i in &c.bus_indices {
                buses[i].bus_type = BusType::Slack;
                buses[i].voltage_mag = 0.0;
                buses[i].voltage_ang = 0.0;
                buses[i].p_spec = 0.0;
                buses[i].q_spec = 0.0;
            }
        }
    }
}

/// Builds a Y-bus with the branches flagged in `outaged` taken out of service,
/// **keeping their structural entries** so the sparsity pattern is identical to
/// the intact network's.
///
/// `outaged` is indexed by the crate's flat branch index — lines first, then
/// transformers, the same space `branch_flow::branch_params` uses — and may be
/// shorter than the branch list, in which case the missing tail is in service.
///
/// The preserved zeros are the whole point. `jacobian::JacobianPattern`
/// derives its pattern from `YBusSparse::row`, and `YBus::finish` keeps every
/// `(i, j)` that was ever added regardless of value, so a contingency built
/// this way factorizes against the *base* symbolic factorization — no
/// re-analysis, which is most of what an N-1 sweep would otherwise pay per
/// scenario. It is also exactly how gridoxide already represents an open
/// transformer terminal: `branch_calc_param` returns zeros for a `(1,0)`
/// status and `stamp_transformers` stamps them anyway.
///
/// **This is only safe when the outage does not disconnect anything.**
/// `network::connected_components` walks the same structural entries and
/// cannot see that a zero-valued one carries nothing, so an outage that splits
/// the network would be classified as still-connected and its now-referenceless
/// buses would come back as a singular solve rather than an honest
/// `NoReferenceBus`. Callers must check first —
/// [`structural_component_count`] is the cheap test, and
/// `batch::BatchSolver::solve_contingencies` uses it to fall back to a full
/// rebuild for the scenarios that need one.
pub fn build_ybus_with_outages(
    n: usize,
    lines: &[Line],
    transformers: &[Transformer],
    outaged: &[bool],
) -> YBus {
    let out = |idx: usize| outaged.get(idx).copied().unwrap_or(false);

    // Stamp the in-service network through the ordinary builder, so the
    // π-model, tap and open-terminal rules stay defined in exactly one place.
    let in_service_lines: Vec<Line> =
        lines.iter().enumerate().filter(|(i, _)| !out(*i)).map(|(_, l)| l.clone()).collect();
    let in_service_transformers: Vec<Transformer> = transformers
        .iter()
        .enumerate()
        .filter(|(j, _)| !out(lines.len() + j))
        .map(|(_, t)| t.clone())
        .collect();
    let mut y = build_ybus(n, &in_service_lines, &in_service_transformers);

    // Then re-add what the outaged branches *would* have touched, at zero.
    let zero = Complex::new(0.0, 0.0);
    let mut stamp_pattern = |from: usize, to: usize| {
        y.add(from, from, zero);
        if from != to {
            y.add(from, to, zero);
            y.add(to, from, zero);
            y.add(to, to, zero);
        }
    };
    for (i, ln) in lines.iter().enumerate() {
        if out(i) {
            stamp_pattern(ln.from, ln.to);
        }
    }
    for (j, t) in transformers.iter().enumerate() {
        if out(lines.len() + j) {
            stamp_pattern(t.from, t.to);
        }
    }
    y
}

/// How many components the branch list forms, counting a branch as connecting
/// exactly when [`build_ybus`] would give it a structural off-diagonal entry.
///
/// Deliberately mirrors what [`connected_components`] sees rather than what is
/// physically connected: a half-open transformer counts here, because it
/// counts there too. That makes the comparison between an intact count and an
/// outaged one a sound test of "did this outage split anything", which is what
/// [`build_ybus_with_outages`]'s fast path needs.
pub fn structural_component_count(
    n: usize,
    lines: &[Line],
    transformers: &[Transformer],
    outaged: &[bool],
) -> usize {
    let out = |idx: usize| outaged.get(idx).copied().unwrap_or(false);
    let mut uf = crate::topology::UnionFind::new(n);
    for (i, ln) in lines.iter().enumerate() {
        if !out(i) && ln.from != ln.to {
            uf.union(ln.from, ln.to);
        }
    }
    for (j, t) in transformers.iter().enumerate() {
        if !out(lines.len() + j) && t.from != t.to {
            uf.union(t.from, t.to);
        }
    }
    let mut roots: Vec<usize> = (0..n).map(|i| uf.find(i)).collect();
    roots.sort_unstable();
    roots.dedup();
    roots.len()
}

pub fn build_ybus(n: usize, lines: &[Line], transformers: &[Transformer]) -> YBus {
    let mut y = YBus::new(n);
    for ln in lines {
        // Self-loop: pure shunt element (no series branch).
        if ln.from == ln.to {
            y.add(ln.from, ln.from, Complex::new(ln.g_shunt, ln.b_shunt));
            continue;
        }
        let z = Complex::new(ln.r, ln.x);
        // series admittance
        let y_line = Complex::new(1.0, 0.0) / z;
        // split shunt admittance (conductance + susceptance) equally to both ends of line
        let y_shunt_half = Complex::new(ln.g_shunt / 2.0, ln.b_shunt / 2.0);
        // diagonal elements
        y.add(ln.from, ln.from, y_line + y_shunt_half);
        y.add(ln.to, ln.to, y_line + y_shunt_half);
        // off-diagonal elements
        y.add(ln.from, ln.to, -y_line);
        y.add(ln.to, ln.from, -y_line);
    }
    stamp_transformers(&mut y, transformers);
    y
}

/// The four 3×3 phase-domain π-model blocks of a three-phase line,
/// `[Yff, Yft, Ytf, Ytt]` — the same order and meaning as
/// [`branch_calc_param`]'s four scalars.
///
/// Extracted so that [`build_ybus_3ph`] and the state estimator's own branch
/// functionals read one description rather than two. A Y-bus and a measurement
/// model that disagree about a branch produce an estimate that converges
/// confidently to the wrong answer, which is the failure mode
/// `tests/measurement_residual_test.rs` exists to catch on the symmetric side.
pub fn line3ph_blocks(ln: &Line3Ph) -> [[[Complex<f64>; 3]; 3]; 4] {
    let y_c1 = Complex::new(ln.g1, ln.b1);
    let y_c0 = Complex::new(ln.g0, ln.b0);
    let zero = [[Complex::new(0.0, 0.0); 3]; 3];

    // A `from == to` line is gridoxide's half-open branch: the whole equivalent
    // admittance sits on one diagonal block and there is no mutual term.
    if ln.from == ln.to {
        let m = seq_to_phase_shunt(y_c1, y_c0);
        return [m, zero, zero, zero];
    }

    let y1 = Complex::new(1.0, 0.0) / Complex::new(ln.r1, ln.x1);
    let y0 = Complex::new(1.0, 0.0) / Complex::new(ln.r0, ln.x0);
    let d_s = (y0 + 2.0 * y1) / 3.0;
    let o_s = (y0 - y1) / 3.0;
    let d_sh = (y_c0 + 2.0 * y_c1) / 6.0;
    let o_sh = (y_c0 - y_c1) / 6.0;

    let mut diag = zero;
    let mut mutual = zero;
    for p in 0..3 {
        for q in 0..3 {
            let ys = if p == q { d_s } else { o_s };
            let ysh = if p == q { d_sh } else { o_sh };
            diag[p][q] = ys + ysh;
            mutual[p][q] = -ys;
        }
    }
    [diag, mutual, mutual, diag]
}

/// The four 3×3 phase-domain blocks of a three-phase transformer, in the same
/// order as [`line3ph_blocks`].
///
/// Unlike a line, a transformer's positive and negative sequences differ once
/// its clock phase shift is in, so this needs the full Fortescue product rather
/// than the closed form a passive branch admits.
pub fn transformer3ph_blocks(t: &Transformer3PhSeq) -> [[[Complex<f64>; 3]; 3]; 4] {
    [0, 1, 2, 3].map(|i| fortescue_to_phase(t.y0[i], t.y1[i], t.y2[i]))
}

/// Builds a 3N×3N phase-domain Y-bus from a list of three-phase lines.
///
/// Physical node `k` maps to rows/columns `3k`, `3k+1`, `3k+2` (phases a, b, c).
/// Sequence parameters are converted to the 3×3 primitive admittance matrix via
/// the symmetrical-components transform; off-diagonal terms couple phases when
/// r0≠r1 or x0≠x1.
pub fn build_ybus_3ph(n: usize, lines: &[Line3Ph]) -> YBus {
    let mut y = YBus::new(3 * n);
    for ln in lines {
        let [yff, yft, ytf, ytt] = line3ph_blocks(ln);
        // A half-open branch collapses to `from == to`, where all four blocks
        // land on one diagonal and the three zero ones add nothing.
        for (block, (bi, bj)) in [yff, yft, ytf, ytt]
            .iter()
            .zip([(ln.from, ln.from), (ln.from, ln.to), (ln.to, ln.from), (ln.to, ln.to)])
        {
            for p in 0..3 {
                for q in 0..3 {
                    y.add(3 * bi + p, 3 * bj + q, block[p][q]);
                }
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

/// The effective shunt admittance seen at a half-open transformer's
/// connected end: `y_shunt/2 + 1/(1/y_series + 2/y_shunt)`. Mathematically,
/// this expression's limit as `y_shunt → 0` is exactly `0` (with no
/// magnetizing branch at all, opening one end truly isolates the connected
/// side — there's no other path for current to flow), but literal complex
/// division by `y_shunt = 0+0j` hits a `0/0` pattern and evaluates to `NaN`
/// instead of that limit. `y_shunt == 0` exactly is common in practice
/// (magnetizing admittance is often left unspecified/zero, e.g. in real
/// CGMES transformer data far more often than gridoxide's own PGM test
/// fixtures happen to combine with a half-open status), so this needs an
/// explicit guard, not just trusting the formula.
pub fn half_open_branch_shunt(y_series: Complex<f64>, y_shunt: Complex<f64>) -> Complex<f64> {
    if y_shunt == Complex::new(0.0, 0.0) {
        return Complex::new(0.0, 0.0);
    }
    let one = Complex::new(1.0, 0.0);
    y_shunt * 0.5 + one / (one / y_series + Complex::new(2.0, 0.0) / y_shunt)
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
    let k = tap.norm();
    match (from_status, to_status) {
        (1, 1) => {
            let y_diag = y_series + y_shunt * 0.5;
            [y_diag / (k * k), -y_series / tap.conj(), -y_series / tap, y_diag]
        }
        (1, 0) => {
            let branch_shunt = half_open_branch_shunt(y_series, y_shunt);
            [branch_shunt / (k * k), zero, zero, zero]
        }
        (0, 1) => {
            let branch_shunt = half_open_branch_shunt(y_series, y_shunt);
            [zero, zero, zero, branch_shunt]
        }
        _ => [zero, zero, zero, zero],
    }
}

/// Stamps two-winding transformer contributions into an existing Y-bus, via
/// `branch_calc_param` for the per-branch [yff, yft, ytf, ytt] entries.
fn stamp_transformers(ybus: &mut YBus, transformers: &[Transformer]) {
    for t in transformers {
        let [yff, yft, ytf, ytt] =
            branch_calc_param(t.y_series, t.y_shunt, t.tap, t.from_status, t.to_status);
        ybus.add(t.from, t.from, yff);
        ybus.add(t.from, t.to, yft);
        ybus.add(t.to, t.from, ytf);
        ybus.add(t.to, t.to, ytt);
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

/// PGM's `WindingType` codes, as they appear on `PgmTransformer`'s
/// `winding_from`/`winding_to` (`common/enum.hpp`).
pub const WYE: u8 = 0;
pub const WYE_N: u8 = 1;
pub const DELTA: u8 = 2;
pub const ZIGZAG: u8 = 3;
pub const ZIGZAG_N: u8 = 4;

/// Whether a winding offers the zero sequence a path of its own, given what
/// sits on the other side — PGM's `zero_seq_available` lambda
/// (`component/transformer.hpp`). A wye_n winding needs the other side to be
/// wye_n or delta to have somewhere for zero-sequence current to go; a
/// zigzag_n winding always does; nothing else ever does.
///
/// The sides where this is false are exactly the ones that pick up the
/// artificial low-susceptance term, which exists to keep such a side from
/// floating free of ground in the zero-sequence network (an all-zero row is
/// a singular matrix, not a physical answer).
fn zero_seq_available(this_side: u8, other_side: u8) -> bool {
    match this_side {
        WYE_N => other_side == WYE_N || other_side == DELTA,
        ZIGZAG_N => true,
        _ => false,
    }
}

/// Computes sequence-domain `(y0, y1, y2)` branch admittance parameters
/// (each `[yff, yft, ytf, ytt]`) for a two-winding transformer, needed for
/// asymmetric (3-phase) power flow and for short-circuit calculation.
///
/// Positive/negative sequence reuse `branch_calc_param` with the tap rotated
/// by ±clock (mirrors PGM's `asym_calc_param`).
///
/// Zero sequence follows PGM's own general algorithm
/// (`component/transformer.hpp`, the `sc_calc_param`/`calc_param_asym` zero-
/// sequence block) rather than enumerating winding pairs. Four cases, in
/// PGM's own order — the first three are mutually exclusive, the zigzag pair
/// is additive on top:
///
/// - **YNyn** — the full two-port zero-sequence branch, `z0_series =
///   1/y_series + 3·(z_grounding_to + z_grounding_from/k²)`, with the tap's
///   phase picking up a 180° flip for clock ∈ {2, 6, 10} (reverse-connected
///   variants).
/// - **YN\*** (from side wye_n, in service) — a ground path via the
///   magnetizing branch, plus a path via the series impedance when the *other*
///   side is delta; a one-port term on `yff`, scaled by `1/k²`.
/// - **\*yn** (to side wye_n, in service) — the mirror, a one-port term on
///   `ytt`. Dyn is this case, and was the only form of it gridoxide
///   previously implemented.
/// - **ZN\*/\*zn** — a zigzag_n winding's zero-sequence impedance is taken as
///   10% of its positive-sequence value.
///
/// Then every side without a zero-sequence path of its own
/// ([`zero_seq_available`]) and in service picks up
/// `−j·1e-8·sn/s_base/uk`.
///
/// # Two modelling substitutions
///
/// gridoxide has no `i0_zero_sequence`/`p0_zero_sequence` fields on
/// `PgmTransformer`, so the zero-sequence magnetizing admittance is taken as
/// the positive-sequence one, `y0_shunt = y_shunt`. That is not an
/// approximation this crate invents: it is exactly what PGM computes when
/// those two fields are left unset, since it defaults `i0_zero_sequence` to
/// `i0` and `p0_zero_sequence` to `p0` in that case.
///
/// Likewise there are no grounding-impedance fields, so `z_grounding_from =
/// z_grounding_to = 0` throughout (PGM defaults them to zero too). That is
/// what collapses YNyn's `z0_series` to `1/y_series` — i.e. `y0_series =
/// y_series` — and drops the `3·z_grounding` term from the one-port cases,
/// where it would otherwise sit in series with `1/y0`.
pub fn transformer_seq_params(
    y_series: Complex<f64>, y_shunt: Complex<f64>, tap: Complex<f64>,
    from_status: u8, to_status: u8,
    winding_from: u8, winding_to: u8,
    sn: f64, uk: f64, s_base_va: f64, clock: i32,
) -> ([Complex<f64>; 4], [Complex<f64>; 4], [Complex<f64>; 4]) {
    let zero = Complex::new(0.0, 0.0);
    let y1 = branch_calc_param(y_series, y_shunt, tap, from_status, to_status);
    let y2 = branch_calc_param(y_series, y_shunt, tap.conj(), from_status, to_status);

    let k = tap.norm();
    // See the doc comment: PGM's own default when the zero-sequence
    // magnetizing fields are unset, which is the only case gridoxide models.
    let y0_shunt = y_shunt;

    let mut y0 = [zero; 4];
    if winding_from == WYE_N && winding_to == WYE_N {
        // YNyn. With both grounding impedances zero, `z0_series` collapses to
        // `1/y_series`, so `y0_series` is just `y_series`.
        let phase_shift_0 = if matches!(clock, 2 | 6 | 10) { std::f64::consts::PI } else { 0.0 };
        let tap0 = Complex::from_polar(k, phase_shift_0);
        y0 = branch_calc_param(y_series, y0_shunt, tap0, from_status, to_status);
    } else if winding_from == WYE_N && from_status == 1 {
        // YN*: ground path via the magnetizing branch, plus one via zk when
        // the other side is delta.
        let mut y = y0_shunt;
        if winding_to == DELTA {
            y += y_series;
        }
        // Guarded exactly as PGM guards it: with no magnetizing branch and no
        // delta on the far side there is no zero-sequence path at all, and
        // `1/y` would be a division by zero rather than the zero it means.
        if y != zero {
            y0[0] = y / (k * k);
        }
    } else if winding_to == WYE_N && to_status == 1 {
        // *yn — the mirror of the above. Dyn lands here.
        let mut y = y0_shunt;
        if winding_from == DELTA {
            y += y_series;
        }
        if y != zero {
            y0[3] = y;
        }
    }

    // Zigzag windings are additive on top of the above, not an alternative to
    // it — PGM writes these as separate `if`s, not `else if`s. A zigzag_n
    // winding's zero-sequence series impedance is approximated as 10% of its
    // positive-sequence value, so its admittance is 10x.
    if winding_from == ZIGZAG_N && from_status == 1 {
        y0[0] = y_series * 10.0 / (k * k);
    }
    if winding_to == ZIGZAG_N && to_status == 1 {
        y0[3] = y_series * 10.0;
    }

    let low_susceptance = -1e-8 * sn / s_base_va / uk;
    if !zero_seq_available(winding_from, winding_to) && from_status == 1 {
        y0[0] += Complex::new(0.0, low_susceptance);
    }
    if !zero_seq_available(winding_to, winding_from) && to_status == 1 {
        y0[3] += Complex::new(0.0, low_susceptance);
    }

    (y0, y1, y2)
}

/// Stamps asymmetric transformer contributions into a 3N×3N phase-domain Y-bus.
pub fn stamp_transformers_3ph(ybus: &mut YBus, transformers: &[Transformer3PhSeq]) {
    for t in transformers {
        let blocks = transformer3ph_blocks(t);
        let ends = [(t.from, t.from), (t.from, t.to), (t.to, t.from), (t.to, t.to)];
        for (block, (bi, bj)) in blocks.iter().zip(ends) {
            for p in 0..3 {
                for q in 0..3 {
                    ybus.add(3 * bi + p, 3 * bj + q, block[p][q]);
                }
            }
        }
    }
}

/// Adds a set of lumped shunt admittances to the Y-bus diagonal.
pub fn stamp_shunts(ybus: &mut YBus, shunts: &[ShuntAdm]) {
    for s in shunts {
        ybus.add(s.at, s.at, s.y);
    }
}

/// Adds a set of three-phase shunt admittances to the phase-domain Y-bus diagonal blocks.
pub fn stamp_shunts_3ph(ybus: &mut YBus, shunts: &[ShuntAdm3Ph]) {
    for s in shunts {
        let m = seq_to_phase_shunt(s.y1, s.y0);
        for p in 0..3 {
            for q in 0..3 {
                ybus.add(3 * s.at + p, 3 * s.at + q, m[p][q]);
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
///
/// `tap_pos` is clamped to `[min(tap_min, tap_max), max(tap_min, tap_max)]` before use, mirroring
/// PGM's own `Transformer::tap_limit` — real-world converted data (e.g. pandapower's MATPOWER-derived
/// test cases) can carry a `tap_pos` outside that range for transformers with no adjustable tap
/// changer (`tap_min == tap_max`), and PGM's reference implementation silently clamps rather than
/// applying the out-of-range offset.
pub fn transformer_tap(
    u1: f64, u2: f64, tap_side: u8,
    tap_pos: i32, tap_min: i32, tap_max: i32, tap_nom: i32, tap_size: f64, clock: i32,
    u1_rated: f64, u2_rated: f64,
) -> Complex<f64> {
    let tap_pos = tap_pos.clamp(tap_min.min(tap_max), tap_min.max(tap_max));
    let delta = (tap_pos - tap_nom) as f64 * tap_size;
    let (u1_tapped, u2_tapped) = if tap_side == 0 {
        (u1 + delta, u2)
    } else {
        (u1, u2 + delta)
    };
    // `k = (u1/u2) / (u1_rated/u2_rated)`, matching power-grid-model's
    // `transformer.hpp:194` exactly. Both halves matter:
    //
    // - the *nameplate* ratio `u1/u2` is what the windings actually do, tap
    //   included;
    // - the *node* ratings are the per-unit bases the Y-bus is expressed in.
    //
    // Their quotient is the off-nominal ratio. Taking only `(u1 + delta)/u1` —
    // as this did until the `vision-validation-network` fixture exposed it —
    // cancels the nameplate against itself and drops the node bases entirely,
    // so a transformer whose nameplate ratio differs from its nodes' rating
    // ratio is modelled as though it did not. That is ordinary data, not an
    // edge case: 20 of that fixture's 22 transformers are 10750/420 units on a
    // 10500 V node, a 2.4% ratio silently lost.
    tap_ratio_from_voltages(u1_tapped * u2_rated, u2_tapped * u1_rated, clock)
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
///
/// The solve itself lives in
/// [`linear::impedance`](crate::linear::impedance), which offers the same
/// linearization as a standalone power-flow method
/// ([`linear::linear_power_flow`](crate::linear::linear_power_flow)). This
/// entry point stays deliberately island-unaware: it runs *before* the
/// solver's own `classify`/`mark_unreferenced_islands` pass, so it must not
/// mutate bus types, and a singular result here is simply a warm start that
/// did not happen rather than a failure to report.
pub fn linear_initial_guess(buses: &mut [Bus], ybus: &YBusSparse) {
    let n = buses.len();
    let unknown_idx: Vec<usize> =
        (0..n).filter(|&i| matches!(buses[i].bus_type, BusType::PQ)).collect();
    crate::linear::impedance::solve_constant_admittance(buses, ybus, &unknown_idx);
}

pub fn power_injections(
    buses: &[Bus],
    ybus: &YBusSparse,
) -> (Vec<f64>, Vec<f64>) {
    // Calculates the complex power injection into each bus.
    // S = V .* conj(I) where I = Ybus * V
    // S_k = V_k * I_k^*
    let n = buses.len();
    let mut p = vec![0.0; n];
    let mut q = vec![0.0; n];

    let v: Vec<Complex<f64>> = buses.iter().map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang)).collect();
    let i = ybus.mul_vec(&v);

    for k in 0..n {
        let s = v[k] * i[k].conj();
        p[k] = s.re;
        q[k] = s.im;
    }

    (p, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nameplate numbers shared by the `transformer_seq_params` cases below,
    /// picked so `y_shunt` is nonzero (the magnetizing branch is the only
    /// zero-sequence path a YN winding has when the far side is not delta,
    /// so a zero one would hide the difference between the arms).
    fn seq_fixture() -> (Complex<f64>, Complex<f64>, Complex<f64>) {
        let y_series = Complex::new(2.0, -6.0);
        let y_shunt = Complex::new(1e-4, -5e-4);
        let tap = Complex::from_polar(1.05, std::f64::consts::PI / 6.0);
        (y_series, y_shunt, tap)
    }

    fn seq_params_for(winding_from: u8, winding_to: u8, clock: i32) -> [Complex<f64>; 4] {
        let (y_series, y_shunt, tap) = seq_fixture();
        let (y0, _, _) = transformer_seq_params(
            y_series, y_shunt, tap, 1, 1, winding_from, winding_to,
            2e7, 0.05, 1e6, clock,
        );
        y0
    }

    /// YNd is the pair 11 of power-grid-model's 15 short-circuit fixtures use,
    /// and the one the old two-arm implementation panicked on. Its
    /// zero-sequence branch is a one-port term on the *from* side: the
    /// magnetizing branch in parallel with the series impedance (the delta on
    /// the far side gives zero-sequence current a circulating path), scaled by
    /// `1/k²`.
    #[test]
    fn ynd_gets_a_from_side_zero_sequence_path() {
        let (y_series, y_shunt, tap) = seq_fixture();
        let k = tap.norm();
        let y0 = seq_params_for(WYE_N, DELTA, 1);

        let expected_yff = (y_shunt + y_series) / (k * k);
        assert!((y0[0] - expected_yff).norm() < 1e-15, "yff = {}, want {expected_yff}", y0[0]);
        // A delta winding has no zero-sequence path of its own, so the to side
        // carries only the artificial low-susceptance term.
        let low = -1e-8 * 2e7 / 1e6 / 0.05;
        assert!((y0[3] - Complex::new(0.0, low)).norm() < 1e-18, "ytt = {}", y0[3]);
        // A one-port term couples nothing across the branch.
        assert_eq!(y0[1], Complex::new(0.0, 0.0));
        assert_eq!(y0[2], Complex::new(0.0, 0.0));
    }

    /// Dyn is YNd's mirror, and was the one arm the old implementation had.
    /// Pinning both together is what shows the generalization did not quietly
    /// transpose the sides.
    #[test]
    fn dyn_is_the_mirror_of_ynd() {
        let (y_series, y_shunt, _) = seq_fixture();
        let y0 = seq_params_for(DELTA, WYE_N, 5);

        // The `*yn` arm carries no `1/k²` — the to side is the per-unit base.
        assert!((y0[3] - (y_shunt + y_series)).norm() < 1e-15, "ytt = {}", y0[3]);
        let low = -1e-8 * 2e7 / 1e6 / 0.05;
        assert!((y0[0] - Complex::new(0.0, low)).norm() < 1e-18, "yff = {}", y0[0]);
    }

    /// The reverse-connected clock positions flip the zero-sequence tap by
    /// 180°, which the two-port YNyn arm is the only one to see.
    #[test]
    fn ynyn_flips_the_zero_sequence_tap_when_reverse_connected() {
        let straight = seq_params_for(WYE_N, WYE_N, 0);
        let reversed = seq_params_for(WYE_N, WYE_N, 6);

        // yff/ytt are magnitude-only and so unmoved; the coupling terms carry
        // the flip, and a 180° rotation is exactly a sign change.
        assert!((straight[0] - reversed[0]).norm() < 1e-15);
        assert!((straight[3] - reversed[3]).norm() < 1e-15);
        assert!((straight[1] + reversed[1]).norm() < 1e-15, "{} vs {}", straight[1], reversed[1]);
        assert!((straight[2] + reversed[2]).norm() < 1e-15, "{} vs {}", straight[2], reversed[2]);
        // Both sides are wye_n, so neither picks up a low-susceptance term.
        assert!(straight[1].norm() > 0.0);
    }

    /// A winding pair with no zero-sequence path at either end must not come
    /// back all-zero: an all-zero row is a singular matrix, not an answer.
    /// Both sides get the artificial low susceptance instead.
    #[test]
    fn a_pair_with_no_zero_sequence_path_is_not_left_floating() {
        let y0 = seq_params_for(WYE, DELTA, 1);
        let low = -1e-8 * 2e7 / 1e6 / 0.05;
        assert!((y0[0] - Complex::new(0.0, low)).norm() < 1e-18, "yff = {}", y0[0]);
        assert!((y0[3] - Complex::new(0.0, low)).norm() < 1e-18, "ytt = {}", y0[3]);
    }

    /// A zigzag_n winding's zero-sequence impedance is taken as 10% of its
    /// positive-sequence value, so its admittance is 10x — and it always has a
    /// path of its own, so it never picks up the low-susceptance term.
    #[test]
    fn zigzag_n_carries_ten_times_the_series_admittance() {
        let (y_series, _, _) = seq_fixture();
        let y0 = seq_params_for(WYE, ZIGZAG_N, 1);
        assert!((y0[3] - y_series * 10.0).norm() < 1e-15, "ytt = {}", y0[3]);
    }

    /// An out-of-service terminal contributes nothing at all — not even the
    /// low-susceptance term, which exists to ground a *connected* winding.
    #[test]
    fn an_open_terminal_contributes_no_zero_sequence_term() {
        let (y_series, y_shunt, tap) = seq_fixture();
        let (y0, _, _) = transformer_seq_params(
            y_series, y_shunt, tap, 1, 0, WYE_N, DELTA, 2e7, 0.05, 1e6, 1,
        );
        assert_eq!(y0[3], Complex::new(0.0, 0.0));
    }

    fn bus_at(idx: usize, bus_type: BusType, voltage_mag: f64) -> Bus {
        Bus {
            idx,
            bus_type,
            voltage_mag,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: 0.0,
            q_max: 0.0,
            u_rated: 0.0,
            zip_terms: Vec::new(),
        }
    }

    /// A de-energized bus is `Slack` at `V = 0` — a placeholder, not a
    /// reference. Counting one as a reference leaves a component nominally
    /// solvable with nothing to solve against, and the neighbouring `PQ` bus
    /// gets an identically zero angle row: `H_ii = −Q_i − V_i²B_ii`, whose
    /// terms cancel exactly when the only neighbour sits at zero volts.
    ///
    /// This is not hypothetical. It made Svedala's node-breaker import report
    /// `Singular`, and SmallGrid's fail under `RetainAll` — which looked like
    /// the switch-count ceiling `NODE_BREAKER_PLAN.md` §4.1 predicts, and was
    /// not.
    #[test]
    fn a_de_energized_placeholder_is_not_a_reference_bus() {
        let buses = vec![
            bus_at(0, BusType::Slack, 0.0), // de-energized placeholder
            bus_at(1, BusType::PQ, 1.0),
            bus_at(2, BusType::Slack, 1.05), // a real reference
            bus_at(3, BusType::PQ, 1.0),
        ];
        let components = vec![vec![0, 1], vec![2, 3]];
        let classified = classify(&buses, &components);

        assert!(
            matches!(classified[0].verdict, Verdict::NoReferenceBus),
            "a component whose only slack sits at V = 0 has no reference"
        );
        assert!(classified[0].slack_indices.is_empty());
        assert!(matches!(classified[1].verdict, Verdict::Solvable));
        assert_eq!(classified[1].slack_indices, vec![2]);

        // ...and the unreferenced one is then pinned, so its PQ bus stops
        // being an unknown at all.
        let mut buses = buses;
        mark_unreferenced_islands(&mut buses, &classified);
        assert_eq!(buses[1].bus_type, BusType::Slack);
        assert_eq!(buses[1].voltage_mag, 0.0);
        assert_eq!(buses[3].bus_type, BusType::PQ, "the referenced island is untouched");
    }

    /// A real reference alongside a placeholder still counts: the component is
    /// solvable, and only the placeholder is disregarded.
    #[test]
    fn a_real_reference_beside_a_placeholder_still_counts() {
        let buses = vec![
            bus_at(0, BusType::Slack, 0.0),
            bus_at(1, BusType::Slack, 1.05),
            bus_at(2, BusType::PQ, 1.0),
        ];
        let classified = classify(&buses, &[vec![0, 1, 2]]);
        assert!(matches!(classified[0].verdict, Verdict::Solvable));
        assert_eq!(classified[0].slack_indices, vec![1]);
    }

    /// A transformer whose nameplate ratio differs from its nodes' rating
    /// ratio has a real off-nominal ratio, even at nominal tap.
    ///
    /// This is the case that went wrong: the old formula computed
    /// `(u1 + delta)/u1`, which cancels the nameplate against itself and never
    /// looks at the node ratings, so a 10750/420 unit on a 10500 V node came out
    /// as 1:1. Twenty of `vision-validation-network`'s twenty-two transformers
    /// are exactly that, and the resulting node voltages were 10.7 V out against
    /// a 0.5 V tolerance.
    #[test]
    fn tap_ratio_accounts_for_nameplate_versus_node_rating() {
        // Nominal tap: pos == nom, so the tap contributes nothing and the whole
        // ratio comes from nameplate against rating.
        let k = transformer_tap(10750.0, 420.0, 0, 3, 5, 1, 3, 250.0, 0, 10500.0, 420.0);
        assert!(
            (k.norm() - 10750.0 / 10500.0).abs() < 1e-12,
            "expected the 2.4% off-nominal ratio, got {}",
            k.norm()
        );
    }

    /// The reassuring half: when nameplate and rating agree, nominal tap really
    /// is 1:1, so the fix does not disturb the ordinary case.
    #[test]
    fn a_matched_transformer_at_nominal_tap_is_unity() {
        let k = transformer_tap(110000.0, 10500.0, 0, 0, -12, 12, 0, 1320.0, 0, 110000.0, 10500.0);
        assert!((k.norm() - 1.0).abs() < 1e-12, "got {}", k.norm());
    }

    /// The tap itself still moves the ratio, on whichever side it sits.
    #[test]
    fn the_tap_position_shifts_the_tapped_side() {
        // tap_side 0: -3 steps of 1320 V off a 110 kV winding.
        let k = transformer_tap(110000.0, 10500.0, 0, -3, -12, 12, 0, 1320.0, 0, 110000.0, 10500.0);
        assert!((k.norm() - (110000.0 - 3960.0) / 110000.0).abs() < 1e-12, "got {}", k.norm());

        // tap_side 1 moves u2 instead, which inverts the direction.
        let k = transformer_tap(110000.0, 10500.0, 1, -3, -12, 12, 0, 100.0, 0, 110000.0, 10500.0);
        assert!((k.norm() - 10500.0 / (10500.0 - 300.0)).abs() < 1e-12, "got {}", k.norm());
    }

    /// `clock` is a vector-group phase shift of 30 degrees per unit, and must
    /// survive the ratio change untouched.
    #[test]
    fn the_clock_still_sets_the_phase() {
        let k = transformer_tap(10750.0, 420.0, 0, 3, 5, 1, 3, 250.0, 5, 10500.0, 420.0);
        assert!((k.arg() - 5.0 * std::f64::consts::PI / 6.0).abs() < 1e-12, "got {}", k.arg());
    }

    #[test]
    fn connected_components_single_cluster() {
        // 0-1-2 all tied together by ordinary series branches.
        let mut y = YBus::new(3);
        y.add(0, 1, Complex::new(1.0, -2.0));
        y.add(1, 0, Complex::new(1.0, -2.0));
        y.add(1, 2, Complex::new(1.0, -2.0));
        y.add(2, 1, Complex::new(1.0, -2.0));
        let ybus = y.finish();
        let components = connected_components(&ybus);
        assert_eq!(components, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn connected_components_multiple_clusters() {
        // {0,1} tied together, {2} a singleton (shunt-only diagonal entry,
        // no off-diagonal edge — must not be treated as connected to
        // anything), {3,4} tied together.
        let mut y = YBus::new(5);
        y.add(0, 1, Complex::new(1.0, -2.0));
        y.add(1, 0, Complex::new(1.0, -2.0));
        y.add(2, 2, Complex::new(0.0, 1e-6)); // pure shunt, no branch
        y.add(3, 4, Complex::new(1.0, -2.0));
        y.add(4, 3, Complex::new(1.0, -2.0));
        let ybus = y.finish();
        let components = connected_components(&ybus);
        assert_eq!(components, vec![vec![0, 1], vec![2], vec![3, 4]]);
    }

    #[test]
    fn connected_components_bus_with_no_ybus_entries_at_all() {
        // A bus that never got any `y.add(i, ...)` call at all (not even a
        // self-shunt) must still surface as its own singleton, not panic.
        let mut y = YBus::new(2);
        y.add(0, 0, Complex::new(0.0, 1e-6));
        let ybus = y.finish();
        let components = connected_components(&ybus);
        assert_eq!(components, vec![vec![0], vec![1]]);
    }

    fn test_bus(idx: usize, bus_type: BusType) -> Bus {
        Bus {
            idx, bus_type, voltage_mag: 1.0, voltage_ang: 0.0,
            p_spec: 0.3, q_spec: 0.1, q_min: -f64::INFINITY, q_max: f64::INFINITY,
            u_rated: 0.0, zip_terms: Vec::new(),
        }
    }

    #[test]
    fn classify_exactly_one_slack_is_solvable() {
        let buses = vec![test_bus(0, BusType::Slack), test_bus(1, BusType::PQ)];
        let classified = classify(&buses, &[vec![0, 1]]);
        assert_eq!(classified.len(), 1);
        assert!(matches!(classified[0].verdict, Verdict::Solvable));
        assert_eq!(classified[0].slack_indices, vec![0]);
    }

    #[test]
    fn classify_zero_slack_is_no_reference_bus() {
        let buses = vec![test_bus(0, BusType::PQ), test_bus(1, BusType::PV)];
        let classified = classify(&buses, &[vec![0, 1]]);
        assert!(matches!(classified[0].verdict, Verdict::NoReferenceBus));
        assert!(classified[0].slack_indices.is_empty());
    }

    #[test]
    fn classify_two_slack_is_ambiguous() {
        let buses = vec![test_bus(0, BusType::Slack), test_bus(1, BusType::Slack)];
        let classified = classify(&buses, &[vec![0, 1]]);
        assert!(matches!(classified[0].verdict, Verdict::AmbiguousReferenceBus));
        assert_eq!(classified[0].slack_indices, vec![0, 1]);
    }

    #[test]
    fn mark_unreferenced_islands_zeroes_no_reference_component_only() {
        let mut buses = vec![
            test_bus(0, BusType::PQ),  // component 0: no slack
            test_bus(1, BusType::PV),  // component 0: no slack
            test_bus(2, BusType::Slack), // component 1: solvable, untouched
            test_bus(3, BusType::PQ),    // component 1: solvable, untouched
        ];
        let components = vec![vec![0, 1], vec![2, 3]];
        let classified = classify(&buses, &components);
        mark_unreferenced_islands(&mut buses, &classified);

        for &i in &[0, 1] {
            assert_eq!(buses[i].bus_type, BusType::Slack);
            assert_eq!(buses[i].voltage_mag, 0.0);
            assert_eq!(buses[i].voltage_ang, 0.0);
            assert_eq!(buses[i].p_spec, 0.0);
            assert_eq!(buses[i].q_spec, 0.0);
        }
        // Component 1 (already solvable) must be untouched.
        assert_eq!(buses[2].bus_type, BusType::Slack);
        assert_eq!(buses[3].bus_type, BusType::PQ);
        assert_eq!(buses[3].p_spec, 0.3);
        assert_eq!(buses[3].q_spec, 0.1);
    }

    #[test]
    fn mark_unreferenced_islands_leaves_ambiguous_component_untouched() {
        let mut buses = vec![test_bus(0, BusType::Slack), test_bus(1, BusType::Slack), test_bus(2, BusType::PQ)];
        let components = vec![vec![0, 1, 2]];
        let classified = classify(&buses, &components);
        mark_unreferenced_islands(&mut buses, &classified);
        // Ambiguous components aren't mutated at all — bus 2 keeps its
        // original PQ spec rather than being zeroed like a NoReferenceBus one.
        assert_eq!(buses[2].bus_type, BusType::PQ);
        assert_eq!(buses[2].p_spec, 0.3);
    }
}

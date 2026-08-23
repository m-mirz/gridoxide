# Continuation power flow in gridoxide

Status: **implemented**, 2026-08-23, against `7360a58`. Phases 1–5 landed; §10 records where
this plan was wrong, which is the part worth keeping.

> **What this is.** A predictor–corrector continuation around the existing Newton solver, answering
> the one question nothing else in the crate answers: *how much further can this network be loaded
> before voltage collapse?* `docs/src/reference/feature_comparison.md:68` has carried a ❌ for this
> row since it was written, and lines 263–272 rank it first among the missing analysis types. This
> cashes that in.
>
> **Scope.** CPF only — the P-V curve, the nose (λ_max), the loading margin, the weakest-bus
> ranking. Q-V curves stay a follow-up, so the comparison row goes to ⚠️, not ✅.

## 1. Why an ordinary solve cannot answer this

At the collapse point the power-flow Jacobian is singular, so Newton does not converge — and from
the outside that is indistinguishable from a bad initial guess. `src/sparse.rs` detects singularity
only as a non-finite result vector (`solve_values`, :137); there is no pivot magnitude, determinant
or condition estimate anywhere in the tree to tell the two apart. Marching λ upward with ordinary
solves therefore brackets λ_max from below and never *finds* it.

Continuation reformulates the problem so the singularity is a **regular point of an augmented
system**, and walks the curve through it.

## 2. The formulation

The existing loop (`src/solver.rs:662`, `newton_raphson_cached`) builds

```
mismatch = s_spec − s_calc(x),   J = ∂s_calc/∂x,   J·dx = mismatch,   x += dx
```

with `x = [θ_i : i non-slack, bus order; |V|_i : i PQ, bus order]` and `J` from
`jacobian::JacobianPattern` (H = ∂P/∂θ, N = ∂P/∂|V|, M = ∂Q/∂θ, L = ∂Q/∂|V|, **not** normalized
by |V|).

Parameterize the specification by a scalar λ:

\\[ s_{spec}(\lambda) = s_{base} + \lambda\,\Delta s, \qquad
   g(x,\lambda) = s_{calc}(x) - s_{spec}(\lambda) = 0 \\]

\\[ \frac{\partial g}{\partial x} = J, \qquad \frac{\partial g}{\partial \lambda} = -\Delta s \\]

where `Δs` carries `Δp_i` in the P-row of every non-slack bus and `Δq_i` in the Q-row of every PQ
bus — the *same row layout* as `mismatch`. λ = 0 is the base case.

**Corrector.** One extended Newton step on `[g; p] = 0`:

\\[
\begin{bmatrix} J & -\Delta s \\\\ \partial p/\partial x & \partial p/\partial \lambda \end{bmatrix}
\begin{bmatrix} \Delta x \\\\ \Delta\lambda \end{bmatrix}
= \begin{bmatrix} \mathrm{mismatch}(x,\lambda) \\\\ -p(x,\lambda) \end{bmatrix}
\\]

With `Δλ = 0` this reduces **term for term** to the existing Newton step. That equivalence is the
first thing the tests assert.

**Predictor.** Same matrix, right-hand side `[0; ±1]`, normalized to unit 2-norm:

\\[
\begin{bmatrix} J & -\Delta s \\\\ b^{T} & d \end{bmatrix}
\begin{bmatrix} t_x \\\\ t_\lambda \end{bmatrix} = \begin{bmatrix} 0 \\\\ \pm 1 \end{bmatrix},
\qquad z^{p} = z_0 + \sigma\,t
\\]

**The parametrization row**, given the last accepted point `z₀`, unit tangent `t`, step `σ`:

| | `p(x, λ)` | `[bᵀ ∣ d]` |
|---|---|---|
| `Natural` | `λ − λ₀ − σ·t_λ` | `[0 … 0 ∣ 1]` |
| `Local`, k = argmax\|t_i\| | `z_k − z_k^p` | `e_kᵀ` |
| `PseudoArcLength` (**default**) | `tᵀ(z − z₀) − σ` | `[t_xᵀ ∣ t_λ]` |

### Why the augmented matrix is non-singular at the nose

At a simple fold `J` is singular with a one-dimensional null space `span(v)`, and transversality
gives `−Δs ∉ range(J)`. The bordered matrix is then non-singular **iff `bᵀv ≠ 0`**. For
pseudo-arclength `b = t_x`, and at the nose `t = (v, 0)`, so `bᵀv = ‖v‖² ≠ 0`. For local
parametrization `b = e_k` with `k = argmax|t_i| = argmax|v_i|`, so `bᵀv = v_k ≠ 0`. This is the
entire reason the augmented form exists, and it is what a bordered/Schur-complement solve cannot
have: that route needs two solves against `J` itself, which is exactly the matrix that goes
singular.

## 3. Assembly, and what the border costs

`AugmentedPattern` wraps a `JacobianPattern` and appends, in a fixed order:

1. every `JacobianPattern` entry unchanged, so `fill` is reused verbatim;
2. **column n**, one entry per row `0..n` — structurally present at *every* row even where
   `Δs_r == 0`, so the pattern does not depend on which buses carry load;
3. **row n**, one entry per column `0..=n` — structurally full.

That pattern is invariant across steps, across all three parametrizations and across step-size
changes. One `S::new(n+1, …)` per *segment* (a segment ends only when a Q-limit event changes bus
types), one `factor_and_solve_values` per corrector iteration and one per tangent — the same cost
model the Newton loop already has.

**Fill-in.** The reasoning below was written from faer's COLAMD source and is *half wrong*; §10
records what measurement found. Kept as written so the correction is legible.

> faer's sparse LU orders columns with COLAMD (`faer-0.24.4/src/sparse/linalg/colamd.rs`), whose
> defaults `dense_row: 0.5`, `dense_col: 0.5` give thresholds `max(16, 0.5·n)`. The full λ column
> trips `dense_col_count` and is ordered **last**; the full border row trips `dense_row_count` and
> is dropped from degree scoring. With the border ordered last the LU is exactly the bordered
> factorization, so the extra fill is exactly `2n` nonzeros — low single-digit percent against
> `nnz(LU)` on a case of that size. To be measured in Phase 6, not assumed.

Two consequences to state plainly:

- **`JacobianBackend::Block` is refused.** Its 2×2-per-bus structure has no home for a scalar λ
  unknown. `ContinuationError::BackendUnsupported`.
- **BTF is defeated** by the border, so `Klu`/`KluNative` lose any block-triangular split.
  Negligible on connected transmission networks, real on multi-island ones. `Scalar` is the CPF
  default.

## 4. The nose, and the two kinds of critical point

λ_max is where `t_λ = 0`, detected as a sign change between accepted points and refined by the same
locator the Q-limit events use, with event function `e(σ) = t_λ(σ)`. No condition number is needed
anywhere, which is fortunate, because the tree has none.

**There are two kinds of critical point, and missing the second gives a wrong answer:**

```rust
pub enum CriticalPointKind {
    /// t_λ passed through zero: J is singular. A fold.
    SaddleNode,
    /// A generator hit its Q limit and the re-seeded tangent had t_λ < 0.
    /// λ_max is the event point itself, and J is *not* singular there.
    LimitInduced { bus: usize },
}
```

A CPF that only watches for a smooth fold marches happily down the lower branch past a
limit-induced bifurcation and reports nonsense. So after every Q-limit event the re-seeded tangent
is checked, and `t_λ < 0` terminates with `LimitInduced`.

**Weakest bus.** At the located nose `t_x` spans `null(J)`, so the bus with the largest
`|dV_i/dσ|` is the one whose voltage collapses. Reported as a ranked participation vector, not just
an index.

**Margin.** The solver model carries no system base MVA, and `Bus::u_rated` is a *voltage* base.
So `margin_pu = λ_max · Σ(−Δp_i)` over net consumers, and `margin_mw = margin_pu · base_mva` with
`ContinuationOptions::base_mva` (default 100.0) **used for reporting only**. `CriticalPoint` also
carries `p_load_base_pu` and `p_load_nose_pu` so a caller on another base can recompute.

## 5. Q-limit events, and why the outer-loop driver is not reused

`outerloop::solve_with_loops` drives `PersistentSolver::solve`, an ordinary **n×n** Newton at fixed
injections; it cannot solve the augmented system. More to the point, its fixed-point driver would
switch a bus the moment a step overshoots a limit — which is exactly the imprecision event location
exists to remove.

The resolution, per loop:

- **`ReactiveLimits` — reused, but driven by the locator.** A Q limit is a discrete event; that is
  what event location is for. `ReactiveLimits::check` stays the *only* code that flips a bus type
  and writes `q_spec = limit`, and CPF calls it at the located crossing. No duplicated rule, and no
  double counting because it is called nowhere else during a step.
- **`DistributedSlack` / `AreaInterchange` — folded into `Δs` analytically.** This is a strict
  improvement, not a shortcut. Both are fixed-point iterations that move `p_spec` as a function of
  the *solved state*; that dependence is invisible to `∂g/∂λ`, so running them inside the corrector
  would silently corrupt the tangent — the one vector CPF needs exact. But their effect is exactly
  linear in λ: `Δp_gen,i = −ŵ_i · Σ_j Δp_load,j`, with `ŵ` normalized per island. Folding that into
  `Δp` makes the tangent exact and removes a nested loop.
- **`PhaseControl` / `TransformerVoltageControl` — refused.** A tap move is a discrete change to
  Y-bus *values*, needing its own event function, a `restamp_ybus` and an `invalidate_admittances`
  mid-curve. Refusing beats silently freezing taps at their base-case positions.
- **A safety net.** After every accepted point, run `ReactiveLimits::check` on the state. It must
  report `Stable`. If it ever reports `Unstable` outside a located event, the locator missed a
  crossing: record `ContinuationEvent::MissedQLimit`, accept the switch, continue. Every returned
  point is then outer-loop-consistent, verified by the loop's own code.

### The locator

Event functions from `network::power_injections`:

```
for each bus i still PV:  e_i(z) = max(Q_calc,i − q_max,i,  q_min,i − Q_calc,i)   // > 0 ⇔ violated
```

Per step: predict, correct with the loops frozen, evaluate. On a sign change, bracket in the
*arclength* `s ∈ [0, σ]` with `e*(s) = max_i e_i(z(s))` so the **earliest** crossing is found, and
run **regula falsi with Illinois damping and a bisection fallback** — each iteration costs a full
corrector, so superlinear convergence matters. At the located point:

1. push the event, call `ReactiveLimits::check` once;
2. **freeze the direction at the switched bus** — `Δq_i ← 0`, `q_base_i ← q_spec`. Mandatory:
   without it the next `q_spec ← q_base + λΔq` write would overwrite the limit clamp and silently
   un-enforce the limit;
3. re-analyze — `n_unknowns` grew by one, so rebuild `JacobianPattern`, `AugmentedPattern` and the
   `LinearSolver`. Same invalidation `ReactiveLimits::invalidates() == Invalidates::Pattern`
   already declares; CPF acts on it directly rather than through the driver;
4. re-converge at fixed λ_e (should take 0–1 iterations — asserted) and re-seed the tangent,
   checking for `LimitInduced`.

## 6. The loading direction

`Bus` stores only net `p_spec`/`q_spec`, so the direction is explicit, taken **once at the
converged base case and never regenerated** — `ReactiveLimits` rewrites `q_spec` in place, and
re-deriving would fold a generator's limit into the load direction.

The driver runs `connected_components` / `classify` / `mark_unreferenced_islands` up front, and the
base snapshot and direction are taken *after* it, so a sourceless island (zeroed to
`Slack, V = 0, P = Q = 0`) gets a zero direction for free.

**ZIP terms are refused by default.** `network::effective_injection` (:858) adds them at the current
|V|, but neither `JacobianPattern::fill` nor its oracle `solver::build_jacobian_triplets` carries
any `∂s_eff/∂|V|` term for them. For ordinary Newton that is a converged-answer-preserving
inexactness — only the step direction is off. For CPF it is not: a wrong `J` puts the nose in the
wrong place, and `J`'s singularity *is* the answer. So `run_continuation` returns
`ContinuationError::ZipTermsUnsupported` unless `allow_zip` is set, which attaches an
`ApproximateJacobian` warning instead. Adding the missing derivatives is a clean follow-up gated by
the existing bit-for-bit oracle test.

## 7. Validation

None of the six trees under `references/` implements CPF, so the gates must be self-supporting.

**The primary gate is analytic, and it holds including resistance.** Slack `|V₁| = E`, line
`(r, x)`, PQ bus with `s₀`, direction `scale_loads` so `s(λ) = (1+λ)s₀`. With
`a = rP + xQ`, `b = xP − rQ`, `u = |V₂|²`:

\\[ u^{2} - (2a + E^{2})u + (a^{2} + b^{2}) = 0, \qquad D = 4aE^{2} + E^{4} - 4b^{2} \\]

The nose is `D = 0`, which with `a = μa₀`, `b = μb₀`, `μ = 1 + λ` gives

\\[ \lambda_{max} = \frac{E^{2}\left(a_0 + \sqrt{a_0^{2} + b_0^{2}}\right)}{2\,b_0^{2}} - 1,
   \qquad |V_2|_{nose} = \sqrt{\tfrac{1}{2}\left(2\mu_{max}a_0 + E^{2}\right)} \\]

Verified numerically before implementation: reproduces the classic lossless unity-power-factor
result (`P_max = 1/2X`, `|V| = 1/√2`) and drives `D` to machine zero for `r/x = 0.3` and for
capacitive load alike.

The gate list:

| Gate | Property |
|---|---|
| G1 | λ_max and `|V|_nose` match the closed form above over a grid of `(r, x, pf)`, and `|t_λ| < 1e-6` at the located nose |
| G2 | Every returned point is an ordinary power flow at its own λ — rebuilt independently and re-checked with `power_injections`. Catches any sign error in the λ column |
| G3 | `run_power_flow` converges at `0.999·λ_max` and does not at `1.01·λ_max` — fully independent of the CPF code |
| G4 | `Local` and `PseudoArcLength`, and σ₀ ∈ {0.01, 0.05, 0.2}, agree on λ_max to 1e-6 relative |
| G5 | The tangent matches a central difference of two `Natural` correctors — the same gate `ac_sensitivity_test` already uses |
| G6 | Q-limit events: `Q_calc == limit` at λ_e; still inside its range just below; state continuous across the switch; post-switch re-convergence ≤ 1 iteration; `ReactiveLimits::check` reports `Stable` at every later point; λ_max strictly lower with limits than without; agreement with a brute-force bisection using `run_power_flow(enforce_q_limits)`; and a `LimitInduced` case |
| G7 | `pglib_opf_case14_ieee` λ_max in a *plausibility band*, with the reference cited — published values depend on direction and Q-limit treatment, so a tight comparison would be false precision |
| G8 | `case9241pegase` completes and every point passes G2; step/iteration/factorization counts recorded |
| G9 | `Scalar`/`KluNative`/`Klu`/`Pardiso` agree to 1e-9; `Block` is refused |
| G10 | Direction composition: pickup entirely on the slack is bit-identical to `scale_loads`; `Σ_gen Δp = −Σ_load Δp` per island |
| G11 | The lower branch is traced, its `|V|` at the critical bus is strictly below the upper branch's at a shared λ, and G2 holds there too |

## 8. Phases

0. *(optional prerequisite)* ZIP derivatives in `JacobianPattern` + the oracle. Not required, since
   v1 refuses ZIP.
1. **Skeleton, natural parametrization.** `direction.rs`, `augmented.rs`, `corrector.rs`, `mod.rs`;
   fixed-step march in λ; no tangent. Gates G2, G3, plus a unit test that `Δλ = 0` reproduces the
   existing Newton step bit-for-bit. Ships a usable P-V curve.
2. **Tangent and the nose.** Tangent solve, sign continuity, `PseudoArcLength` + `Local`, adaptive
   σ, nose refinement, `CriticalPoint`. Gates G1, G4, G5, G9.
3. **Events.** `events.rs`, the Illinois locator, `ReactiveLimits` reuse, pattern re-analysis,
   direction freezing, `LimitInduced`, the safety net. Gate G6.
4. **Directions.** Pickup and per-area variants. Gate G10.
5. **Surfaces.** CLI `gridoxide continuation`, Python, `docs/src/powerflow/continuation.md`,
   `SUMMARY.md`, the feature-comparison row.
6. **Scale and extras.** G8, benchmarking entry, lower branch (G11), left null vector, C API.

## 9. Risks and limitations

1. **The dense border** — mitigated by COLAMD's dense-row/column detection; `2n` extra fill when
   the border is ordered last. Measured in Phase 6, not assumed.
2. **`Block` unsupported**, by construction. **BTF defeated** for the KLU backends.
3. **ZIP refused by default** — a wrong `J` misplaces the nose rather than merely slowing
   convergence.
4. **Tap control refused during continuation.**
5. **PV→PQ is one-directional**, inherited from `ReactiveLimits`, so lower-branch results are
   conservative: a machine that comes back into range stays clamped.
6. **Limit-induced bifurcations are a distinct answer**, and easy to miss.
7. **Multi-island:** one λ and one parametrization row couple every island, so λ_max is *the first
   island to collapse* and the ranking describes only that island. Reported, not refused.
8. **A degenerate bifurcation** (transversality failure, pitchfork) makes even the augmented matrix
   singular; the backend returns `None` and CPF reports `Singular` — honest but uninformative.
9. **λ_max is a property of the direction, not of the network.** Stated in the chapter, in the CLI
   output header, and in the direction's doc comment.
10. **With `scale_loads` every incremental loss lands on the slack**, so at high λ its output can be
    physically absurd. Not an error; the pickup constructors are the realistic option.

---

## 10. What happened

All five phases landed. The plan above is left as written; this section records where it was wrong.

### Four bugs the plan did not anticipate, three of them silent

**The mirror solution.** In polar form, \\((-|V|, \theta + \pi)\\) satisfies the power-flow
equations exactly and carries the *same* λ. A step that overshoots can land there, and the corrector
converges to it happily — nothing in a mismatch check can see it. Observed on the two-bus case with a
`step_max`-sized step: the walk jumped to the mirror sheet at the sixth point, re-traversed the
entire curve three times reporting negative voltage magnitudes, and still found a nose at the right
λ, for entirely the wrong reason.

Two guards, both in `probe`:

- reject any corrected point with a non-positive voltage magnitude;
- reject any correction that moves further than the prediction did. This is the more fundamental
  one: the defining property of a *continuation* step is that the corrector lands near the
  predictor, so a larger correction means the step found some other solution, not this curve.

Neither guard is exotic, and the plan should have called for them. It said "a converged corrector"
where it should have said "a converged corrector *on this branch*".

**The `Natural` border can never report a falling λ.** The plan had the tangent seeded with the
`e_n` row throughout. That row makes the bottom equation read \\(t_\lambda = 1\\) *exactly*, so a
tangent seeded that way always comes back with \\(t_\lambda > 0\\) — and the nose criterion is a
sign change in \\(t_\lambda\\), which could therefore never fire. The fix is to border with the
**previous tangent**: the bottom equation becomes \\(t \cdot t_{prev} = 1\\), which guarantees
forward travel *and* leaves \\(t_\lambda\\) free to change sign. It also removes the sign-flip
heuristic the plan proposed, which was a workaround for a problem that only existed because of the
wrong border.

A consequence the plan missed entirely: a `PV → PQ` switch grows the unknown count, so the previous
tangent no longer has the right length. `Layout::embed` maps it bus by bus into the new layout — the
newly-freed magnitude gets a zero component — so the walk continues forward across an event instead
of doubling back.

**Base-case reactive clamps needed freezing too, and missing it was invisible.** §5 was careful that
a machine clamped *mid-walk* must have its reactive direction frozen, or the next
`q_spec ← q_base + λ·Δq` write ramps it straight back off its own clamp. It did not occur to the
plan that the *base solve* clamps machines too — `pglib_opf_case14_ieee` clamps two at λ = 0 — and
those were not frozen. The effect is a reactive limit that is enforced at λ = 0 and quietly
un-enforced everywhere after it. Every step still converged, every voltage looked reasonable, and
the reported λ_max was wrong by 5%. Nothing but an independent check of where each machine actually
saturates could have found it.

**The event path fell through into the accept path.** When the located point landed a hair *inside*
the limit, `ReactiveLimits` declined to switch, and control fell through to the ordinary accept path
— which advanced the arclength a second time and set λ to the full step's value while `buses` held
the located point. A λ/state desynchronization that survives every convergence check. Two changes:
the locator now targets the first point with \\(f \geq 0\\) (seeded with the step's own endpoint, so
such a point always exists and a switch always follows), and the two paths are mutually exclusive
with exactly one commit each.

### The plan's own verification was wrong first

§7 proposed checking event exactness by re-solving at the reported λ and comparing the machine's Q
against its limit. That check is wrong, and it failed loudly before the real bug was found: it
solves a network in which *every* generator still holds voltage, which is not the network the
continuation walks once anything has been clamped. The first three attempts at a diagnosis chased
the locator, which was working correctly the whole time.

The oracle that does work is brute force through code that knows nothing about continuation: bisect
λ, run an ordinary `run_power_flow` with `enforce_q_limits` at each trial, and find where the
ordinary outer loop first clamps that bus. It is what `tests/continuation_events_test.rs` uses, and
it is what caught the base-case-clamp bug.

### Measured

Located events match that oracle to **≤ 1e-6 in λ** across seven events on three fixtures.
Switching at step granularity instead is off by **3e-2 to 2.5e-1** — the test asserts the ratio, not
just the accuracy, because a gate that only checked the located value would pass just as happily if
the locator did nothing and the steps merely happened to be small.

The two-bus closed form is matched to **≤ 2.2e-7 in λ** and ≤ 4e-6 in the nose voltage, across
lossless, resistive and capacitive cases. The asymmetry is physics, not slack: at a fold
\\(d\lambda/d|V| \to 0\\), so a given λ error maps to a much larger voltage error.

| fixture | λ_max (free) | λ_max (Q limits) | kind | events |
|---|---|---|---|---|
| `case5_pjm` | 6.2194 | 3.2538 | saddle-node | 3 |
| `case14_ieee` | 2.5361 | 0.6018 | saddle-node | 2 |
| `case30_ieee` | 1.6945 | 0.4664 | saddle-node | 2 |
| `case118_ieee` | 0.3131 | 0.1903 | **limit-induced** (bus 115) | 10 |

`case118_ieee` justifies §4's insistence on distinguishing the two kinds of critical point: under
reactive limits its maximum is limit-induced, and a continuation watching only for a fold would have
walked past it.

### The dense border row was the one real design error

The plan's fill-in argument (§3) was right about the λ **column** and wrong about the border
**row**, and the difference is the whole performance story. Measured on `case1354pegase`
(n = 2449), against a plain Newton factor-and-solve:

| border row | symbolic | numeric + solve |
|---|---|---|
| sparse (`e_k`, one entry) | 1.1× | **1.4×** |
| dense (pseudo-arclength, `n+1` entries) | 1.0× | **92×** |

Symbolic analysis is unaffected, so this is numeric fill during factorization, not a bad ordering —
COLAMD's dense-row handling does not rescue it, and no amount of reading the source would have said
so. At 119 buses the same comparison is 1.7×, which is exactly why every committed fixture was
happy and the problem only appeared on a network large enough to matter.

Three consequences:

1. **`PseudoArcLength` is no longer the default; `Local` is.** Its bordering row is the single entry
   `e_k`, and §2's transversality argument covers it just as well — better, in fact, since `k` is
   chosen as the tangent's *largest* component, so `bᵀv = v_k` is not merely nonzero but as far from
   zero as any entry of `v` gets. The plan (and the design review that preceded it) preferred
   pseudo-arclength on robustness grounds without costing it.
2. **The continuation index is part of the sparsity pattern.** `AugmentedPattern` now takes the
   border row's columns, and `Segment::ensure_border` re-analyzes when they change. In practice `k`
   barely moves: far from the nose the tangent is dominated by λ, so `k = n`; near it one voltage
   component takes over. A full `case118` trace with reactive limits does **1** re-analysis against
   188 solves.
3. **The locator must not switch parametrization.** It originally forced pseudo-arclength for every
   trial — "a `Natural` trial cannot represent a point past the nose, and a `Local` one would
   re-pick its index mid-search". Both halves of that were addressed the wrong way: pinning `k` for
   the whole step solves the second, and the first never applied because the locator is handed the
   step's own parametrization. Forcing the dense row instead made every trial ~90× and dominated
   the run — 6.4 s of a 6.5 s trace.

**And the defaults must not be restated.** The CLI and the Python binding each named `arclength` as
*their* default rather than deferring to the library's, so both silently took the 92× path even
after the library default changed. `case1354pegase` went 91 s → 1.4 s, and peak memory 76 MB →
13 MB, on that one-line fix. `tests/continuation_test.rs` now pins the default's border row to a
single entry, because this is a cliff rather than a preference.

### Measured at scale, which §3 had deferred

| case | buses | wall | points | solves | segments | re-analyses |
|---|---|---|---|---|---|---|
| `case118_ieee` + Q limits | 119 | 17 ms | 32 | 188 | 11 | 1 |
| `case1354pegase` + Q limits | 1 355 | 1.4 s | 96 | 673 | 32 | — |
| `case9241pegase` | 9 241 | 10 s | 156 | 424 | 1 | — |

### Smaller things

- `solver::IslandReport` needed `Clone` for `ContinuationCurve` to derive it. One word.
- `lambda` is a Python keyword, so the event attribute is exposed as `lambda_`. The plural
  (`lambdas`, `lambda_max`) collides with nothing.
- The plan defaulted to `Local` parametrization; `PseudoArcLength` is the default instead. Its
  bordering row is guaranteed non-orthogonal to the null vector without a search, it needs no
  per-step index bookkeeping, and it is what MATPOWER defaults to. `Local` is kept, and the two
  agreeing on λ_max is a gate.
- `ContinuationCurve::q_limit_events()` exists because `events` also carries the walk's own
  bookkeeping (rejected steps), which is diagnostic rather than an answer.

### Still not done

Q-V curves; tap control during a trace; bidirectional `PQ → PV` switching; ZIP-load Jacobian
derivatives (refused rather than approximated); the left null vector and margin sensitivities; the
C API. The `case9241pegase` scaling measurement §3 promised **is** done — see above; it is what
found the dense-row problem.

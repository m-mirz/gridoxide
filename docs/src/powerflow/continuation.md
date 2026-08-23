# Continuation Power Flow

## Motivation

Every other method in this book answers *what is the state, given these injections*. Continuation
answers a different question: **how much further can this network be loaded before the voltage
collapses?**

An ordinary Newton solve cannot answer it, and the reason is worth being precise about. Approaching
the collapse point, the power-flow Jacobian becomes singular, so Newton stops converging. But
gridoxide — like every solver in its comparison set — detects singularity only as a non-finite
result vector (`sparse::RealSparseSystem::solve_values`); there is no pivot magnitude, determinant
or condition estimate anywhere in the tree. So "the Jacobian went singular" and "the initial guess
was bad" look identical from the outside. Ramping load with ordinary solves brackets the limit from
below and never actually finds it.

Continuation reformulates the problem so that the singular point is a **regular point of a larger
system**, and walks the solution curve straight through it.

## The formulation

The [Newton-Raphson](./index.md) solve drives

\\[ g(x) = s_{calc}(x) - s_{spec} = 0 \\]

with \\(x\\) holding every non-slack bus's angle followed by every PQ bus's voltage magnitude.
Continuation parameterizes the specification by a scalar \\(\lambda\\):

\\[ s_{spec}(\lambda) = s_{base} + \lambda\,\Delta s, \qquad
    g(x, \lambda) = s_{calc}(x) - s_{spec}(\lambda) = 0 \\]

so that

\\[ \frac{\partial g}{\partial x} = J, \qquad \frac{\partial g}{\partial \lambda} = -\Delta s \\]

\\(\lambda = 0\\) is the base case and \\(\lambda = 1\\) is the base case plus one full direction.
That is \\(n\\) equations in \\(n+1\\) unknowns; one more equation — the **parametrization**
\\(p(x,\lambda) = 0\\) — closes it.

### The corrector

One extended Newton step on \\([g;\ p] = 0\\):

\\[
\begin{bmatrix} J & -\Delta s \\\\ \partial p/\partial x & \partial p/\partial \lambda \end{bmatrix}
\begin{bmatrix} \Delta x \\\\ \Delta\lambda \end{bmatrix}
=
\begin{bmatrix} \mathrm{mismatch}(x,\lambda) \\\\ -p(x,\lambda) \end{bmatrix}
\\]

With \\(\Delta\lambda\\) pinned to zero this reduces, term for term, to the ordinary Newton step —
same unknown ordering, same mismatch, same convergence test. That equivalence is asserted directly
in the test suite.

### The predictor

The same matrix, with right-hand side \\([0;\ 1]\\), normalized to unit length:

\\[
\begin{bmatrix} J & -\Delta s \\\\ b^{T} & d \end{bmatrix}
\begin{bmatrix} t_x \\\\ t_\lambda \end{bmatrix} = \begin{bmatrix} 0 \\\\ 1 \end{bmatrix},
\qquad z^{p} = z_0 + \sigma\, t
\\]

### The parametrization row

| Variant | \\(p(x,\lambda)\\) | \\([b^{T} \mid d]\\) |
|---|---|---|
| `Natural` | \\(\lambda - \lambda^{p}\\) | \\(e_n^{T}\\) |
| `Local` (default) | \\(z_k - z_k^{p}\\), \\(k = \arg\max\|t_i\|\\) | \\(e_k^{T}\\) |
| `PseudoArcLength` | \\(t^{T}(z - z_0) - \sigma\\) | \\([t_x^{T} \mid t_\lambda]\\) — **dense** |

## Why the bordered matrix is not singular at the nose

This is the point of the whole construction, and it is the thing to understand before anything else.

At a simple fold, \\(J\\) has a one-dimensional null space \\(\mathrm{span}(v)\\), and transversality
gives \\(-\Delta s \notin \mathrm{range}(J)\\). The bordered matrix is then non-singular **exactly
when \\(b^{T}v \neq 0\\)**.

- For local parametrization, \\(b = e_k\\) with \\(k = \arg\max|t_i| = \arg\max|v_i|\\) — so
  \\(b^{T}v = v_k \neq 0\\), and it is the *largest* entry of \\(v\\), so not merely nonzero but
  as far from zero as any entry gets.
- For pseudo-arclength, \\(b = t_x\\), and at the nose \\(t = (v, 0)\\) — so
  \\(b^{T}v = \lVert v\rVert^{2} \neq 0\\).

Either way the augmented system stays perfectly well posed at the very point where the power-flow
Jacobian does not.

This also rules out the cheaper-looking alternative. One could keep the plain \\(n \times n\\)
Jacobian, reuse its factorization and recover \\(\Delta\lambda\\) by block elimination — two solves
plus a scalar. That fails precisely where continuation earns its keep: the Schur complement
\\(d - b^{T}J^{-1}c\\) is a quotient of two quantities that both vanish at the nose, and with no
condition estimate available there is no way to detect the transition either.

### What the border costs — measured, and it decides the default

The \\(\lambda\\) **column** is nearly free. COLAMD's `dense_col` threshold trips on it, the column
is ordered last, and the factors are then exactly the bordered factorization. Measured on
`case1354pegase` (\\(n = 2449\\)), a bordered solve with a *sparse* border row costs **1.4×** a plain
Newton solve.

The border **row** is not free, and the reasoning that says it should be is wrong. Emitting it
densely — which pseudo-arclength genuinely requires, since its row *is* the previous tangent — costs
**92×** the plain solve at the same size, while the symbolic analysis stays at 1.0×. So this is
numeric fill during factorization, not a bad ordering, and no argument about COLAMD's dense-row
handling makes it go away. At 119 buses the same comparison is 1.7×, which is exactly why the
problem is invisible on small fixtures and why it was found by measuring rather than by reading.

Hence **`Local` is the default**: its row is the single entry \\(e_k\\), and it carries the same
transversality guarantee shown above. The continuation index \\(k\\) is part of the sparsity
pattern, so a change to it costs one re-analysis — but \\(k\\) is stable, because far from the nose
the tangent is dominated by \\(\lambda\\) (so \\(k = n\\)) and near it a single voltage component
takes over. One index is pinned for a whole step: the tangent solve, the corrector, and every
event-locator trial all share it, so one symbolic factorization serves them all.

End to end, `case1354pegase` traces to its nose in **1.4 s** and `case9241pegase` in **10 s**
(156 points, 424 bordered solves) on one core.

Two consequences are stated rather than hidden:

- **`JacobianBackend::Block` is refused.** Its 2×2-per-bus structure has no home for a scalar
  \\(\lambda\\) unknown, so `run_continuation` returns `BackendUnsupported` instead of quietly
  substituting another backend.
- **BTF is defeated** by the border, so the KLU backends lose any block-triangular split.
  Negligible on connected transmission networks; real on multi-island ones. `Scalar` is the default.

## Two kinds of limit, and why the difference matters

```text
|V|                                          |V|
 |    ___                                     |    ___
 |   /   \___                                 |   /   \___
 |  /        \__                              |  /        \__
 | /            \  <- saddle-node:            | /          X   <- limit-induced:
 |/               J is singular               |/               a machine saturated,
 +------------------ λ                        +------------------ λ    J is not singular
```

A **saddle-node bifurcation** is the smooth fold: \\(d\lambda/d\sigma\\) passes through zero and the
Jacobian is singular there.

A **limit-induced bifurcation** is not. A generator hits its reactive limit, stops holding its
voltage, and the maximum is *that point* — the Jacobian is perfectly well conditioned there. A
continuation that only watches for \\(d\lambda/d\sigma = 0\\) walks straight past it, down a branch
no operating point can reach, and reports a limit that is not one. gridoxide checks the re-seeded
tangent after every reactive-limit event and reports `CriticalPointKind::LimitInduced` when it points
downward in \\(\lambda\\). `pglib_opf_case118_ieee` is such a case.

## Reactive limits, and why the outer-loop driver is not reused

[Outer loops](./outer_loops.md) exist to decide a solver *input* from a solved state, and
[reactive limits](./q_limits.md) are one of them. Continuation cannot use that driver, for two
reasons. `solve_with_loops` drives an ordinary \\(n \times n\\) Newton and cannot solve the bordered
system at all; and its fixed point flips a bus the moment a step overshoots a limit, which is exactly
the imprecision that has to be removed.

So the corrector runs with bus types frozen, and the crossing is **located**: bracket the step in
arclength, re-predict and re-correct at each trial, and narrow with Illinois-damped regula falsi
until the machine sits on its limit. Every trial is a genuine solution of the power flow, so the
located point is a real point on the curve rather than an interpolation between two.

What is *not* duplicated is the switching rule.
[`ReactiveLimits::check`](./q_limits.md) remains the only code in the crate that flips a bus type
and pins `q_spec` to a limit; continuation calls it once, at the located crossing. Measured against a
brute-force bisection that uses nothing but `run_power_flow`, located events land within
\\(10^{-6}\\) in \\(\lambda\\); switching at step granularity instead is off by \\(10^{-2}\\) to
\\(10^{-1}\\).

Two loops are handled differently again. [Distributed slack](./distributed_slack.md) and
[area interchange](./area_interchange.md) move `p_spec` as a function of the solved state — a
dependence invisible to \\(\partial g/\partial\lambda\\), so running them inside the corrector would
silently corrupt the tangent, the one vector continuation needs exact. Their effect is exactly linear
in \\(\lambda\\), so they are folded into \\(\Delta s\\) analytically instead, via
`LoadingDirection::scale_loads_with_pickup`. [Tap control](./tap_control.md) is refused outright: a
tap move is a discrete change to Y-bus *values* and would need its own event function.

## The loading direction

**\\(\lambda_{max}\\) is a property of the direction, not of the network.** Two studies that stress
different buses get different noses and neither is wrong, so the direction is an explicit input:

| Constructor | Scenario |
|---|---|
| `scale_loads` | every net consumer grows at constant power factor; the slack picks it up |
| `scale_buses` | only the named buses grow — a zonal stress |
| `scale_loads_with_pickup` | the pickup is shared over named generators instead of the slack |
| `to_target` / `explicit` | a direction from a scenario or market tool |

It is captured **once**, after the base solve, and never regenerated. Both halves matter. A
sourceless island has already been zeroed by `mark_unreferenced_islands`, so it gets a zero direction
for free. And a machine that `ReactiveLimits` clamped has had its `q_spec` overwritten with the limit
— if the direction is not frozen there too, every later step ramps it straight back off the limit.
The curve still converges at every step, so nothing looks wrong; the reactive limit is simply not
enforced any more. That failure is invisible to any check except an independent one of where each
machine actually saturates.

### ZIP loads are refused

`network::effective_injection` adds voltage-dependent terms at the current \\(|V|\\), but neither
`JacobianPattern::fill` nor its reference oracle carries a \\(\partial s_{eff}/\partial |V|\\) term
for them. An ordinary solve survives that — the mismatch is exact, so only the step *direction* is
approximate. Continuation does not: the Jacobian's singularity **is** the answer, so a missing term
puts the nose in the wrong place while the curve still looks entirely plausible. `run_continuation`
therefore returns `ZipTermsUnsupported` unless `allow_zip` is set, which attaches a warning instead.

## A two-bus nose, in closed form

On two buses the solvability boundary can be written down exactly, which is what the primary test
gate checks against. Slack at \\(E\angle 0\\), a line \\((r, x)\\), one constant-power load.
Eliminating the angle from the two injection equations, with \\(a = rP + xQ\\), \\(b = xP - rQ\\)
and \\(u = |V_2|^2\\):

\\[ u^{2} - (2a + E^{2})\,u + (a^{2} + b^{2}) = 0, \qquad D = 4aE^{2} + E^{4} - 4b^{2} \\]

The nose is \\(D = 0\\). Substituting \\(a = \mu a_0\\), \\(b = \mu b_0\\) with \\(\mu = 1+\lambda\\)
leaves a quadratic in \\(\mu\\):

\\[ \lambda_{max} = \frac{E^{2}\left(a_0 + \sqrt{a_0^{2} + b_0^{2}}\right)}{2\,b_0^{2}} - 1,
   \qquad |V_2|_{nose} = \sqrt{\tfrac{1}{2}\left(2\mu_{max}a_0 + E^{2}\right)} \\]

For a lossless line at unity power factor this collapses to the textbook \\(P_{max} = E^2/2x\\) with
\\(|V| = E/\sqrt 2\\). The formula holds **including resistance** and at any power factor, leading or
lagging, so it gates the solver against arithmetic rather than against itself.

## Against other tools

| | gridoxide | MATPOWER `runcpf` | PSAT | power-grid-model / pypowsybl / pandapower |
|---|---|---|---|---|
| Predictor | tangent | tangent or secant | tangent | — none implements continuation |
| Parametrization | natural / **local** / pseudo-arclength | same three, arclength by default | same three | — |
| Reactive limits | located exactly, by bisection on arclength | event functions, located | yes | — |
| Limit-induced bifurcation | detected and reported as a distinct kind | detected | detected | — |
| Lower branch | traced | traced | traced | — |
| Weakest bus | right tangent at the nose, ranked | available | participation factors | — |

None of the five reference implementations vendored under `references/` has any continuation code,
which is why the gates in `tests/continuation_test.rs` lean on a closed form and on independent
re-checks rather than on a reference to diff against.

## Using it

```bash
gridoxide continuation network.json --enforce-q-limits
gridoxide continuation network.json --enforce-q-limits --curve      # the whole walk
gridoxide continuation network.json --target-lambda 0.5             # stop short
gridoxide continuation network.json --lower-branch                  # past the nose and back
gridoxide continuation network.json --buses 12,13 --pickup 1,2      # a zonal stress
```

```python
import gridoxide

curve = gridoxide.continuation("network.json", enforce_q_limits=True)
print(curve.lambda_max, curve.critical_kind, curve.margin_mw)
for event in curve.events:
    print(f"bus {event.bus} hit q_{event.limit} at lambda {event.lambda_:.6f}")
```

```rust
use gridoxide::continuation::{run_continuation, ContinuationOptions, LoadingDirection};

let direction = LoadingDirection::scale_loads(&buses);
let curve = run_continuation(
    buses, &lines, &transformers, &shunts,
    ContinuationOptions { direction, ..Default::default() },
);
```

## Limitations

- **PV → PQ is one-directional**, inherited from `ReactiveLimits`. A machine that comes back into
  range on the lower branch stays clamped, so lower-branch results are conservative.
- **Multi-island:** one \\(\lambda\\) and one parametrization row couple every island, so
  \\(\lambda_{max}\\) is *the first island to collapse* and the weakest-bus ranking describes only
  that island.
- **Tap changers do not move** during a trace.
- **A degenerate bifurcation** — transversality failure, a pitchfork — makes even the bordered matrix
  singular. The backend returns nothing and the status is `Singular`: honest, but uninformative.
- **The margin in MW is reporting only.** The solver model carries no system base and `Bus::u_rated`
  is a *voltage* base, so `base_mva` is the caller's to state.

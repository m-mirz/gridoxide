# The Iterative-Linear Method

power-grid-model's default calculation method, and gridoxide's optional one. It trades exactness for
a much cheaper iteration.

## The trick

In terms of the complex voltage vector \\(\underline{U}\\), the awkward part of the measurement model
is that power is *bilinear* in voltage: \\(S = U \overline{I}\\). Current is not — \\(I = YU\\) is
perfectly linear. So if the measurements were currents rather than powers, the whole estimation
problem would be a single linear least-squares solve.

So they are converted. A power measurement \\(S\\) at a terminal whose voltage is \\(U\\) becomes a
current measurement

\\[
\underline{I} = \overline{\left( S / U \right)}
\\]

using the voltage from the previous iterate. A magnitude-only voltage measurement becomes a phasor by
borrowing the previous iterate's angle. Each pass redoes the conversion with better voltages, and the
linearization error shrinks as the angles settle.

Zero-injection buses are markedly simpler here than in the nonlinear formulation: a bus with no
appliance carries no injected current, so \\((YU)_i = 0\\) *exactly*, with no linearization involved.
They enter the same augmented KKT system in complex arithmetic.

## Where the speed comes from

The system matrix is built from admittances and measurement weights. Admittances are fixed. The
weights are *nearly* fixed — converting \\(S\\) to \\(I\\) divides by \\(U\\), so a current's variance
scales as \\(1/\vert U \vert^2\\) — and the method assumes \\(\vert U \vert\\) constant for that
purpose.

Granting that assumption, the matrix never changes. It is factorized **once**, and each iteration
moves only the right-hand side. Newton-Raphson, by contrast, rebuilds and refactorizes its Jacobian
every pass.

## What it costs

The constant-\\(\vert U \vert\\) assumption, plus the interpretation of power measurements as current
measurements, means this method's optimum is not exactly the weighted-least-squares optimum.
power-grid-model documents the same caveat and recommends its Newton-Raphson method when precision
matters. gridoxide keeps both, and defaults to Newton-Raphson on the grounds that a library should be
exact unless asked otherwise.

There is also a capability difference worth knowing: a lone \\(P\\) or \\(Q\\) measurement, without
its partner on the same target, is **dropped**. Half a complex power cannot be converted into a
current. The Newton-Raphson path handles the two independently and keeps it.

## Under-relaxation, and why it is safe here

The linearization inverts the voltage, and that can oscillate: an estimate that is too low produces a
current that is too high, which pushes the next estimate too high, and back. The iteration then sits
in a two-cycle indefinitely rather than converging.

This is not hypothetical. power-grid-model's `transmission-case` fixture does exactly that — the step
parks at \\(5.5 \times 10^{-2}\\) and stays there for hundreds of iterations, neither converging nor
diverging:

```text
iteration:  1     2     3     5     10    20    50    100
step:       6.9e-2 5.2e-2 5.6e-2 5.6e-2 5.6e-2 5.5e-2 5.4e-2 5.3e-2
```

gridoxide backs the step off — halving a relaxation factor whenever an iteration fails to improve on
the last, and letting it grow again by 1.2 whenever progress resumes — which breaks the cycle:

```text
iteration:  1     2     3     5     10    20
step:       6.9e-2 5.2e-2 2.8e-2 6.7e-5 2.8e-6 5.9e-9
```

Damping is a suspicious remedy in general, because it can make a solver *look* convergent while
walking toward the wrong answer. It is safe here for a specific reason: it changes the path and never
the fixed point. `iterative::tests::the_true_state_is_a_fixed_point` pins that down directly — seeded
at a state satisfying the measurements, one iteration leaves the residuals at zero and moves the
state only by the global rotation the estimator normalizes away. So a damped run converges to the
same state an undamped one would reach if it got there at all.

Relaxation engages whenever an iteration fails to improve on the last by 10%, which is not the same
thing as "only on a pathological problem" — a flat start alone can trip it, and on case1354pegase it
does, at iteration 3. That is exactly why the factor has to be able to *recover*: see
[Letting the relaxation recover](#letting-the-relaxation-recover) below, where a version that could
only descend turned out to be spending whole runs at half a step because of one transient.

## Measured cost

`examples/bench_se.rs` runs both methods over the MATPOWER benchmark grids, with measurements
synthesized from a converged power flow — a voltage magnitude at every bus and flows at both ends of
every branch, giving a redundancy around 4x.

| case | buses | measurements | Newton-Raphson | iters | iterative-linear | iters | speedup |
|---|---|---|---|---|---|---|---|
| case14 | 15 | 125 | 0.7 ms | 6 | 0.1 ms | 16 | 7x |
| case118 | 119 | 1,083 | 5.5 ms | 6 | 1.0 ms | 37 | 5.5x |
| case300 | 301 | 2,419 | 13.8 ms | 6 | 2.3 ms | 41 | 6x |
| case1354pegase | 1,355 | 11,189 | 137 ms | 6 | 12.8 ms | 45 | 10.7x |
| case2869pegase | 2,870 | 25,204 | 349 ms | 6 | 30.6 ms | 33 | 11.4x |

Both the benefit and the cost show up plainly. Newton-Raphson takes **exactly six iterations at every
scale** — quadratic convergence — while the linearized method takes 16 to 45. It still wins on wall
clock because each of its iterations is so much cheaper, and the margin *widens* with size, from 5.5x
at 119 buses to 11.4x at 2,870: prefactorization amortizes better the larger the matrix.

The accuracy trade is equally visible. Against the state the measurements were read from,
Newton-Raphson lands at ~1e-14 throughout — the data is perfectly consistent here, so it is limited
only by arithmetic. The linearized method runs 8.7e-10, 1.6e-8, 8.9e-8, 5.6e-7, 1.1e-6 down that same
column: still far below any real measurement noise, but degrading with size rather than holding
constant. That is the linearization and the constant-|U| weighting showing through, and it is why
Newton-Raphson remains the default.

### Where the remaining time goes

Profiling was worth doing before optimizing, because it redirected the obvious plan. On
case1354pegase the two methods spend their time quite differently:

| | Newton-Raphson | iterative-linear |
|---|---|---|
| assembly (`h(x)`, `H`, gain) | 22% | — |
| factorization | **78%** | ~30% *including* all setup |
| per-iteration solves | — | **~70%**, over 45 iterations |

The tempting optimization is a symmetric factorization: `G` is symmetric and a general sparse LU is
being used on it, so in principle half of Newton-Raphson's 78% is recoverable. Two measurements argue
against it.

First, zero-injection buses are 9-31% of buses on these grids, so the constrained KKT system is the
normal case, not the exception, and it is symmetric *indefinite* — needing Bunch-Kaufman or a
quasi-definite regularization rather than a Cholesky, which is two code paths and a perturbation of
the constraints.

Second and more decisively, it would speed up the wrong method. Newton-Raphson is where gridoxide is
already ahead — power-grid-model's own Newton-Raphson state estimator raises `SparseMatrixError` on
every case from 300 buses up, where gridoxide's converges. The remaining gap is on *this* method —
1.5 ms against 0.7 ms on case300, 7.2 ms against 3.4 ms on case1354pegase, 18.3 ms against 8.4 ms on
case2869pegase — and there the factorization is not the bottleneck at all. §7 of
`scripts/bench/README.md` has the full table.

So the open lead for this method is its **convergence rate**, not its linear algebra: it needs 45
iterations here where Newton-Raphson needs 6.

### Measured against power-grid-model's own iteration count

That last paragraph was a conclusion about gridoxide on its own. Comparing the *same* method in the
two tools sharpens it into something more useful, and reverses the reading §7 of
`scripts/bench/README.md` used to give.

Neither tool reports an iteration count through its public API, so `scripts/bench/se_iterations.py`
obtains it the same way for both: the smallest `max_iterations` budget that does not fail, found by
bisection. Before comparing anything it is worth knowing the two criteria are the same quantity, and
they are — power-grid-model's `iterate_unknown` returns `max over buses of |u_new − u_old|`
phase-normalized and loops `while (max_dev > err_tol)`; gridoxide's `raw_step` is
`.map(|(a, b)| (a - b).norm()).fold(0.0, f64::max)` over the same normalized voltages, and both
default to `1e-8`. The one asymmetry is that gridoxide tests `raw_step × relaxation`, where
power-grid-model has no relaxation at all and always takes the full step.

On the documents `examples/bench_se.rs --emit` writes:

| case | buses | PGM its | gridoxide its | iterations | ms per iteration |
|---|---|---|---|---|---|
| case14 | 14 | 15 | 31 | 2.07x | **0.20x** |
| case118 | 118 | 18 | 35 | 1.94x | **0.57x** |
| case300 | 300 | 9 | 29 | 3.22x | **0.62x** |
| case1354pegase | 1,354 | 10 | 28 | 2.80x | **0.71x** |
| case2869pegase | 2,869 | 10 | 33 | 3.30x | **0.71x** |

(gridoxide's counts here are the ones that stood before
[Letting the relaxation recover](#letting-the-relaxation-recover), which cuts them by about 40%.)

**gridoxide's iterations are individually cheaper than power-grid-model's — by 30-40% on every case
above 100 buses — and it takes about three times as many of them.** The flat ~2x total is the product
of those two effects pulling against each other, not evidence of a constant-factor gap in the
per-iteration work. Reading a stable ratio as a per-iteration constant was the error; a stable ratio
is equally consistent with two ratios that happen to be stable.

The control that makes this comparison mean anything is that both tools reach the *same answer*: max
|Δu| between their solutions is 7.6e-9 to 5.2e-7 across these cases, i.e. agreement at their shared
tolerance. Same problem, same optimum, different paths to it.

So the lever is convergence rate, confirmed rather than inferred — and specifically **not** the
linear algebra, which is already ahead.

### Letting the relaxation recover

The section above establishes two things that pull in opposite directions: the damping cannot be
removed, because the undamped map does not converge at all; and the damping is what makes gridoxide
take three times power-grid-model's iterations. Both are true, and the way between them is that the
factor could only ever go *down*.

The rule was: halve whenever an iteration fails to improve on the last by 10%. Nothing ever restored
it. And the flat-start transient alone trips that test — at iteration 3 on case1354pegase, before the
iteration has settled into anything — so a run would spend its whole length at half a step because of
one early stumble that had nothing to do with the instability the damping exists for.

Adding one clause fixes it: when the step *is* shrinking, by more than 25%, grow the factor by 1.2
again, capped at 1. The result on the benchmark documents:

| case | before | after | PGM |
|---|---|---|---|
| case14 | 31 | **19** | 15 |
| case118 | 35 | **21** | 18 |
| case300 | 29 | **18** | 9 |
| case1354pegase | 28 | **17** | 10 |
| case2869pegase | 33 | **20** | 10 |

About 40% fewer iterations, and 16-22% less wall clock — measured interleaved, old build and new,
two rounds. The two do not match because a fixed setup-and-factorization cost does not shrink with
the iteration count; that it is now a *larger* share of the total is the point of having cut the
rest. The iteration-count gap to power-grid-model falls from 2.0-3.3x to 1.2-2.0x, and on case118
gridoxide is now the faster of the two outright.

The growth rate is measured rather than derived, and the measurement is the interesting part. Forcing
a *constant* relaxation shows this map's optimum sits near 0.7 — 23 iterations on case300 against 35
at 0.5 — with the map going unstable just above it: 0.8 costs 36 iterations and 0.9 costs 80. So
there is a narrow good range whose location depends on the network, which is an argument for hunting
for it adaptively rather than for hardcoding 0.7. Among growth factors that all converge, 1.2 has the
best worst case: 1.3 costs case2869pegase 30 iterations and 1.4 costs case14 seventy, where 1.2 needs
20 and 19.

The answer is untouched, as it must be — `the_true_state_is_a_fixed_point` already pins that damping
changes the path and never the destination, and the estimates agree to 2.5e-9 across the change,
which is below the 1e-8 the iteration is asked for.

**This does not explain why power-grid-model needs no damping at all.** That remains open, and the
next section narrows it.

### The relaxation is load-bearing, not overhead

The obvious next move from that table is to take the damping out, since power-grid-model manages
without it. That does not work, and finding out why moves the problem somewhere more interesting.

Forced undamped on these same documents, the iteration does not converge slowly — it does not
converge at all. The step falls for three iterations and then locks onto a constant:

```text
iter        1        2        3        4       ...      58       59       60
step   6.09e-1  2.27e-1  1.77e-1  1.76e-1     ...  1.748610e-1  1.748610e-1  1.748610e-1
```

Constant to seven digits for fifty-odd iterations is a limit cycle, not a floor — the step is not
shrinking at all, so the iterate is circulating rather than settling. That under-relaxation at 0.5
restores convergence places the dominant eigenvalue near −1: damping maps `λ` to `0.5 + 0.5λ`, which
sends `λ ≈ −1` to `≈ 0`. power-grid-model's map, on the identical document with the identical
measurements and no damping at all, is stable there.

The rows responsible are the *branch-terminal power* ones. Dropping them from the undamped run drops
the step from 1.7e-1 to 7.1e-4 — three orders of magnitude — while dropping the voltage rows instead
changes nothing at all. That fits their shape: a branch row converts its reading with
`I = conj(S/U_at)`, so its right-hand side depends on the inverse of a voltage the same row solves
for, and on these documents the branch rows outweigh the voltage rows by four orders of magnitude.

Two further candidate mechanisms were tested and both ruled out:

* **The `|U|²` weight scaling.** gridoxide scales a power row's weight by the reference bus's
  starting `|U|²`, where power-grid-model deliberately does not — its
  `iterative_linear_se_solver.hpp` says so directly ("the variance is not scaled as an
  approximation"). Removing the scaling moves the limit cycle's amplitude (1.75e-1 → 1.81e-1 on
  case300) and does not remove it.
* **The zero-injection KKT constraints.** These are 66 to 869 buses on the cases above, and
  power-grid-model has no equivalent augmentation. Dropping them entirely shrinks the amplitude to
  5.7e-2 and still leaves a limit cycle.

Whatever destabilizes the map is therefore in its core — the power-to-current conversion, the phase
normalization, or the interaction between them — rather than in either of the two places gridoxide
visibly departs from power-grid-model. That is the open question, and it is worth answering before
anyone reaches for an accelerator: a scheme layered on an unstable map inherits the instability.

Note this is a different measurement set from the one the next section traces. These documents carry
voltage magnitudes and branch flows only; `examples/se_converge.rs` synthesizes bus injections too,
and its undamped run decays rather than cycling. Both are real, and the difference between them is
itself a clue about which rows drive the instability.

### What actually ends the iteration

Tracing that convergence rate is worth doing before trying to improve it, because it is not one
phenomenon but two, and the second one is not a rate at all.

`examples/se_converge.rs` re-runs the estimate at increasing `max_iter` and prints the step, its ratio
to the previous step, and the error against the state the measurements were read from:

```bash
cargo run --release --example se_converge case1354pegase
```

Forced undamped, case1354pegase decays geometrically at a strikingly stable ratio — 0.757, holding to
three digits from iteration 30 through iteration 53. A single dominant mode, textbook material for
Aitken or Anderson acceleration. Then it stops:

```text
iter        50       51       52       53      ...      197      198      199      200
step    6.3e-6   4.7e-6   3.6e-6   2.7e-6      ...   2.5e-7   3.4e-7   2.9e-7   7.3e-8
```

Out to 200 iterations the raw step never reaches the 1e-8 tolerance; it settles on a floor around
1e-7 and bounces. Forcing any constant relaxation between 0.4 and 1.0 gives the same picture.

What ends the default run is the relaxation ratchet:

```text
iter      1        2        3     ...      39       40       41       42       43       44       45
raw    1.19e0   1.66e0   1.72e0   ...   3.2e-7   3.7e-7   2.1e-7   2.5e-7   2.6e-7   4.7e-7   1.9e-7
relax    1.0      1.0      0.5    ...      0.5     0.25     0.25    0.125   0.0625  0.03125  0.03125
```

Relaxation engages at **iteration 3**, in the flat-start transient, and holds at 0.5 through
iteration 39. Then the raw step stops falling, the 10%-improvement test fails four times over the
next five iterations, and the factor halves to 1/32. Convergence is declared at iteration 45 on a
reported step of
\\(1.86 \times 10^{-7} \times 0.03125 = 5.8 \times 10^{-9}\\).

That reported step is honest about what it measures — the state genuinely moved that little, because
that is how far damping let it move. But it is not the iteration reaching 1e-8. The tolerance is met
by the damping factor.

The answer is unaffected, and this is the part worth being clear about. The error against the true
state bottoms out at 5.5e-7 by iteration 36 — the same 5.6e-7 this method scores on case1354pegase in
the table above, i.e. its own linearization bias, the floor no amount of iterating can go below.
Iterations 37 to 45 buy no accuracy in any case.

Two consequences for whoever picks this up next:

* Accelerating the 0.757 mode wins iterations 3 through 39 and nothing after. The floor is a separate
  phenomenon and would still be there.
* A step floor near 1e-7 on a gain matrix whose weights span 1e4 to 1e6, with exact KKT constraints
  augmented in, is the shape of linear-solve accuracy rather than of the fixed-point map — worth
  testing with iterative refinement or a scaling pass before assuming the iteration is at fault.

An earlier version of this section reported roughly eighty iterations, with relaxation holding at 1.0
until iteration 78. Neither figure reproduces at HEAD; the traces above are what the current code
does.

## Agreement with Newton-Raphson

On every fixture gridoxide checks, the two methods agree to \\(10^{-6}\\) per-unit, bus by bus — a
stricter statement than each agreeing with power-grid-model, since it holds at every bus rather than
only where an expected value was published.

| Fixture | Iterations | Max \\(\Delta \vert V \vert\\) vs power-grid-model |
|---|---|---|
| `1os2msr` | 11 | 3.4e-10 |
| `1os2msr-no-angle` | 11 | 5.7e-10 |
| `inf-measurement-with-injection` | 2 | 4.0e-9 |
| `transmission-case` | 20 | 4.3e-9 |
| `node-injection-sensor-and-zero-injection` | 1 | 4.4e-16 |

Selecting it:

```bash
gridoxide estimate grid.json --iterative-linear
```

```python
model = gridoxide.StateEstimationModel.from_pgm_json(
    "grid.json", method="iterative_linear", max_iter=100
)
```

The larger iteration budget is deliberate: this method converges linearly where Newton-Raphson
converges quadratically. Spending more iterations to make each one far cheaper is the entire point.

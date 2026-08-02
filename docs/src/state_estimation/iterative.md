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
the last — which breaks the cycle:

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

Relaxation engages only when progress stalls; a well-behaved problem runs undamped and is unaffected.

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
| per-iteration solves | — | **~70%**, over roughly 80 iterations |

The tempting optimization is a symmetric factorization: `G` is symmetric and a general sparse LU is
being used on it, so in principle half of Newton-Raphson's 78% is recoverable. Two measurements argue
against it.

First, zero-injection buses are 9-31% of buses on these grids, so the constrained KKT system is the
normal case, not the exception, and it is symmetric *indefinite* — needing Bunch-Kaufman or a
quasi-definite regularization rather than a Cholesky, which is two code paths and a perturbation of
the constraints.

Second and more decisively, it would speed up the wrong method. Newton-Raphson is where gridoxide is
already ahead — power-grid-model's own Newton-Raphson state estimator fails with a sparse-matrix
error on case300 and case1354pegase where gridoxide's converges. The remaining gap is on *this*
method — 1.3 ms against 0.8 ms on case300, 6.7 ms against 3.3 ms on case1354pegase — and there the
factorization is not the bottleneck at all.

So the open lead for this method is its **convergence rate**, not its linear algebra: it needs about
eighty iterations to reach 1e-8 here. Under-relaxation is not the cause — traced on this case, it
stays at 1.0 through iteration 77 and engages once at 78, right at convergence.

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

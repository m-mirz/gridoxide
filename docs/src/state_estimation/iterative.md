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

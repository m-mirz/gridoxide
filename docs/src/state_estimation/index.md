# The State Estimation Problem

## Motivation

[Power flow](../powerflow/index.md) answers a planning question: *given* the injections at every bus,
what are the voltages? Every input is known exactly, and there is one right answer.

Operations does not work that way. What a control room has is a few hundred telemetered readings,
each with its own error, arriving from equipment of varying quality — some quantities measured twice,
many not measured at all, and a handful simply wrong because a transducer has drifted or a sign
convention was mis-wired years ago. No state reproduces all of them, because they contradict each
other.

State estimation asks the question that fits that data: **which grid state best explains these
measurements?** It is the entry point to essentially every operational application, because those
applications need a complete, consistent snapshot of the network and telemetry never provides one.

## Weighted least squares

Let \\(x\\) be the state — every bus's voltage magnitude and angle — and \\(z\\) the vector of
measurements. Each measurement has a *measurement function* \\(h_i(x)\\): what that sensor would read
if the grid were in state \\(x\\). A voltage sensor's is trivial (\\(h = \vert V_k \vert\\)); a branch
power sensor's is the terminal flow; a bus injection's is the familiar power-flow expression.

The estimate is the state minimizing the weighted sum of squared disagreements:

\\[
J(x) = \frac{1}{2} \left( z - h(x) \right)^{T} W \left( z - h(x) \right)
\\]

with \\(W = \Sigma^{-1}\\) diagonal, \\(W_{ii} = 1/\sigma_i^2\\). Weighting by inverse variance is what
makes the estimate *maximum likelihood* under independent Gaussian errors, and it is why a sensor's
declared \\(\sigma\\) matters as much as its value: a reading trusted to 0.1% pulls a hundred times
harder than one trusted to 1%.

Setting \\(\partial J / \partial x = 0\\) and linearizing gives the **normal equations**, solved
repeatedly until the step vanishes:

\\[
G \, \Delta x = H^{T} W r, \qquad G = H^{T} W H, \qquad r = z - h(x)
\\]

where \\(H = \partial h / \partial x\\) is the measurement Jacobian. This is Gauss-Newton.

## How this differs from the power-flow Newton loop

The two loops look alike and are not the same.

**The residual does not go to zero.** `solver::newton_raphson` drives a *mismatch* to zero: an exact
solution exists and Newton converges onto it. Gauss-Newton has no such target — the measurements are
inconsistent by construction, so \\(r\\) stays nonzero at the optimum. Convergence is therefore tested
on the size of the *step*, never on the residual. A converged estimate with a large \\(J(x)\\) has not
failed to converge; it means the data disagrees with itself, which is what
[bad-data analysis](./diagnostics.md) is for.

**There are no PV buses.** A generator's voltage is a quantity to be *estimated*, not asserted. So
every bus contributes a magnitude unknown, and every bus but the angle reference contributes an
angle:

\\[
x = \left[ \theta_0 \ldots \theta_{N-1} \text{ except the reference}, \; V_0 \ldots V_{N-1} \right]
\\]

giving \\(n = 2N - 1\\), against power flow's \\(n_{angle} + n_{pq}\\). Confusing the two layouts is
the easiest way to produce a Jacobian that is subtly and consistently wrong, so `se::jacobian::StateLayout`
owns the mapping and nothing else indexes by hand.

**The angle reference is conditional.** A network measured only in magnitudes and powers is invariant
under a global phase shift, so a reference must be pinned or \\(G\\) is singular however many
measurements there are. But that invariance disappears the moment a *phasor* measurement supplies an
absolute angle — and pinning a reference on top of one is a false constraint that rotates the whole
estimate away from the data. gridoxide pins a reference exactly when no angle is measured.

## Why the normal equations, and what they cost

\\(H\\) is rectangular (\\(m \times n\\)), and every sparse backend gridoxide has is square — they are
power flow's Jacobian solvers. Forming \\(G = H^{T} W H\\) makes the system square *and* symmetric, so
`Scalar`, `Block`, `Klu`, `KluNative` and `Pardiso` all carry state estimation with no new solver
abstraction. That is the single most important reuse in the design.

The price is conditioning: \\(G\\) squares \\(H\\)'s condition number. Two consequences follow, and
both shape the implementation:

- Zero injections are enforced as [hard constraints](./measurements.md#zero-injection-buses) rather
  than as very-high-weight pseudo-measurements, so the weights never span more orders of magnitude
  than the physics requires.
- An orthogonal (QR) formulation, which never forms \\(G\\), remains the escape hatch if a real
  network ever proves too ill-conditioned for this path.

## Validation

gridoxide is checked against power-grid-model's own state-estimation fixtures, committed under
`tests/data/pgm/state_estimation/` with their MPL-2.0 license files. On `transmission-case` — 11
buses, 4 transformers, 59 measurements — the per-unit magnitudes agree to **1.5 × 10⁻⁹**.

Angles are checked in two regimes, because only one of them has an absolute answer. Where an angle is
measured, absolute angles must match. Where none is, the estimate is invariant under a global
rotation and power-grid-model's own fixtures do not agree with each other on the convention
(`transmission-case` reports its source node at exactly 0, `1os2msr-no-angle` reports its source node
at −0.0130). The test then requires gridoxide's angles to match *up to one constant shared by every
bus* — a stronger check than it sounds, since a wrong estimate produces per-bus errors rather than a
uniform offset.

## Running it

From the shell:

```bash
gridoxide estimate tests/data/pgm/state_estimation/1os2msr/input.json
```

From Python:

```python
import gridoxide

model = gridoxide.StateEstimationModel.from_pgm_json("grid_with_sensors.json")
model.solve()
print(model.voltage_mag())          # per-unit, one entry per bus
print(model.observability())        # what the measurements fail to determine
print(model.bad_data())             # chi-squared, p-value, ranked suspects
```

Note the input needs sensors, but does *not* need `p_specified` on its loads or `u_ref` on its
sources — those are quantities state estimation solves for rather than inputs it consumes.

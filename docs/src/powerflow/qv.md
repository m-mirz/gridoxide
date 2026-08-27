# Q-V Curves and the Reactive Margin

## Motivation

[Continuation power flow](./continuation.md) asks how much further the *system*
can be loaded before it collapses. This asks a narrower and more operational
question about one bus: **how much reactive support would it take to hold this
bus at a given voltage, and how much margin is there before that stops being
possible?**

It is the question a reactive-planning study actually asks. A bus with 200 MVAr
of margin can absorb a contingency; one with 20 cannot, and no amount of
system-wide loadability tells you which is which.

## The method

Put a fictitious synchronous condenser at the bus — an ideal machine that holds
a voltage and produces whatever reactive power that takes — by retyping the bus
to `PV`. Then sweep its setpoint downward and read the reactive power it has to
supply at each step:

```text
  Q
  |    \
  |     \___                  Q > 0: the bus must be supported to hold it here
  |         \___
  +-------------\----------------------------- |V|
  |              \___                 the bus's own solved voltage: Q = 0,
  |                  \___             a condenser here has nothing to do
  |                      \__.--
  |                          ^
  |                      the nose: dQ/d|V| = 0.
  |                      No setpoint needs less reactive power than this,
  |                      and −Q there is the margin.
```

## Why this needs no continuation

Every point is an **ordinary power flow**, and fixing \\(|V|\\) is what makes it
so. The bus's reactive equation is dropped and its magnitude stops being an
unknown — which is exactly a `PV` bus — so the Jacobian stays non-singular all
the way through the nose.

A P-V curve needs a predictor–corrector precisely because it does *not* fix a
magnitude: λ is a free parameter and the Jacobian goes singular at the fold.
Here the fold is in \\(Q\\), which is an **output** rather than an unknown, so
nothing goes singular and a sweep is enough.

That is a difference in kind, not an implementation shortcut, and it is why this
reuses [`run_power_flow`](./index.md) rather than continuation's bordered
corrector.

### Interpolating the minimum

The nose falls between samples, and pinning it to the nearest one wastes
accuracy the sweep already paid for. A parabola through the three bracketing
samples is exact where the curve is locally quadratic, which it is at a smooth
minimum, and costs no extra solves. Measured on the two-bus case with samples
0.01 apart, the interpolated nose lands within \\(4 \times 10^{-6}\\) of the
analytic voltage where the nearest sample can be out by 0.005.

The vertex is refused if it falls outside the bracketing samples — that means the
three points do not describe a minimum and the parabola is extrapolating.

## A curve in closed form

On two buses the whole curve can be written down, which is what the solver is
gated against. Slack at \\(E\angle 0\\), a lossless line \\(jx\\), one bus with
net injection \\(P + jQ\\). Eliminating the angle gives

\\[ u^{2} - (2a + E^{2})u + (a^{2} + b^{2}) = 0, \qquad a = xQ,\ b = xP,\ u = |V|^{2} \\]

Read as a quadratic in \\(Q\\) instead of in \\(u\\), the operating branch is

\\[ Q(u) = \frac{u - \sqrt{E^{2}u - x^{2}P^{2}}}{x} \\]

and \\(dQ/du = 0\\) gives the nose directly:

\\[ u_{nose} = \frac{E^{4} + 4x^{2}P^{2}}{4E^{2}}, \qquad
   Q_{nose} = \frac{4x^{2}P^{2} - E^{4}}{4E^{2}x} \\]

`tests/qv_test.rs` matches both to \\(10^{-5}\\) in \\(Q\\) across a range of
reactances and loadings — the solver checked against arithmetic rather than
against itself.

## Q-V and P-V do not agree on the weakest bus

This is worth stating plainly, because the opposite is the natural assumption and
it is false.

Both are voltage-stability measures, so it is tempting to expect the bus a P-V
trace names critical — the largest component of the tangent at the nose — to be
the bus with the smallest Q-V margin. Measured on `case14`:

| | ranking, weakest first |
|---|---|
| P-V (continuation tangent at the nose) | 4, 3, 8, 9, 6 |
| Q-V (reactive margin at base loading) | 7, 13, 5, 9, 11 |

One bus in common out of five. The obvious explanation — that they are evaluated
at different operating points, one at the nose and one at base loading — does not
rescue it either: re-running the Q-V sweep at 98% of \\(\lambda_{max}\\) makes
the agreement *worse*, not better.

They measure related but distinct things. A P-V critical bus is a property of the
**system-wide collapse mode** along one particular loading direction. A Q-V
margin is one bus's **local reactive headroom** at the current operating point.
A bus can be locally weak without participating much in the system's collapse
mode, and the reverse.

This is why utilities run both rather than either, and why `tests/qv_test.rs`
deliberately does *not* assert that they agree — a test to that effect would have
been asserting something untrue about the physics.

## Reading the result

- **The margin is a positive number of MVAr**, and larger is stronger.
- **`nose_not_reached`** means the sweep stopped while the curve was still
  falling. The number is then a *lower bound* on the margin, not the margin, and
  the CLI says so — understating a bus's weakness is the direction that misleads.
- **`already_controlled`** means the bus was already `PV`, so the curve describes
  moving an existing machine's setpoint rather than adding a condenser, and
  \\(Q\\) includes what that machine was already producing.
- The condenser is **deliberately unlimited**. The curve measures how much
  reactive power holding a voltage *would* take, which a limit would truncate
  rather than answer. Other buses keep their own limits.

## Using it

```bash
gridoxide qv network.json --weakest 10          # rank buses by margin
gridoxide qv network.json --bus 13 --curve      # one bus, with the curve
gridoxide qv network.json --bus 13 --v-min 0.2  # sweep further down
```

```python
curve = model.qv_curve(13, v_min=0.20)
print(curve["status"], curve["nose"]["margin_pu"])
for p in curve["points"]:
    print(p["voltage"], p["q"])
```

```rust
use gridoxide::qv::{qv_curve, QvOptions};

let curve = qv_curve(&buses, &lines, &transformers, &shunts, 13, QvOptions::default());
if let Some(nose) = &curve.nose {
    println!("{:.1} MVAr at |V| = {:.4}", nose.margin_mvar(100.0), nose.voltage);
}
```

## Limitations

- **One bus at a time.** A ranking is the caller looping, and each point is a
  full solve, so a sweep over every bus of a large network is
  `buses × steps` power flows. Embarrassingly parallel, and not parallelized.
- **The margin is measured at one operating point.** It says nothing about the
  margin after a contingency, which is the question a security study asks — that
  would be this sweep inside a contingency loop.
- **A slack bus is refused**: it already fixes its own magnitude and already has a
  free reactive injection, so there is no condenser to add.

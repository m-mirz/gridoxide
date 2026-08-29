# The RMS Simulation Problem

## Motivation

Everything else gridoxide computes describes the network at one *instant*. A
[power flow](../powerflow/index.md) says where the power goes; a
[short circuit](../short_circuit/index.md) says how much current a fault draws;
[continuation](../powerflow/continuation.md) traces a curve of instants indexed by loading. None of
them can answer the question a planner actually asks about a fault:

> The breaker will clear in 120 milliseconds. Do the machines stay in step?

That is a question about a **trajectory**, and answering it means integrating.

## Why the machines can fall out of step

A synchronous generator holds position against the grid the way a mass on a spring holds position
against gravity. Electrical power out balances mechanical power in, and the rotor sits at whatever
angle makes those equal.

A fault removes the electrical power almost entirely — a short circuit near the terminals means
the machine is pushing current into something that cannot accept work. The mechanical power does
not change; a steam turbine does not notice a fault for several seconds. So the rotor accelerates.

When the fault clears, the electrical power returns and the rotor decelerates. Whether it
decelerates *enough* depends on how far it got: past a certain angle the restoring torque falls
rather than rises, and the machine runs away. That threshold — the last instant at which clearing
still saves it — is the **critical clearing time**, and it is the number transient stability
exists to produce.

## The differential-algebraic form

```text
ẋ = f(x, V)                machines, exciters, governors — the state
Y V − I_inj(x, V) = 0      the network — a constraint, not a state
```

The second line carries **no derivative**. That is what makes this a differential-algebraic system
rather than an ordinary differential one, and it is not a modelling shortcut: it is the statement
that electromagnetic transients in the network settle far faster than the electromechanical ones
being simulated, so they are taken as instantaneous.

That time-scale separation is exactly what "RMS" means. A voltage is represented by a phasor whose
magnitude and angle vary slowly, rather than by the sinusoid itself. It is what an
[EMT](https://en.wikipedia.org/wiki/Electromagnetic_transient) simulation refuses to assume, and it
is why an RMS run can use a five-millisecond step where an EMT run needs microseconds.

## What that costs, stated plainly

The phasor representation is an approximation, and gridoxide makes one more on top of it: the
stator equations here omit the rotor speed, and the swing equation is written in power rather than
torque. The two forms differ by exactly a factor of \\(\omega\\), so they agree at synchronous speed
and part in proportion to the speed deviation.

That is the classical assumption — Kundur §13.3 states it explicitly — and its size is **measured**
rather than assumed: against Dynawo on Kundur's own Example 13.2 it costs 0.6% of terminal power at
a 0.9% speed deviation. See [Against Dynawo](./validation.md).

## The shape of a run

1. **Solve a power flow.** Every device's initial state is derived from the operating point, so
   there has to be one.
2. **Initialize backwards.** Each machine's states are chosen so its derivatives are *zero*, and
   each control's reference is chosen so that its output is exactly what the machine turned out to
   need. Nothing is read from a file that can be derived from the solve.
3. **Integrate**, stopping exactly on each scheduled event, applying it, re-solving the network
   constraint with the states held fixed, and resuming.

Step 2 is the one that goes wrong, and it has a decisive gate: a run with no disturbance must be a
flat line to machine precision. See [Initialization](./initialization.md).

## Scope

| | |
|---|---|
| Machines | classical (2nd order), transient (4th), subtransient (6th) |
| Excitation | `SEXS`, and a purely proportional regulator |
| Governors | `TGOV1`, and a purely proportional governor |
| Stabilizers | washout plus two lead-lag stages |
| Loads | constant admittance by default; ZIP with a low-voltage cutoff on request |
| Events | bus faults, clearings, branch trip and close, unit trip and close, load steps |
| Input | gridoxide JSON, PSS/E `.dyr`, Dynawo `.dyd`/`.par` |

Deliberately **not** implemented, each for a stated reason:

- **Limits** on exciter, governor and stabilizer outputs. A hard clamp makes the right-hand side
  non-smooth, so doing it properly needs non-windup logic and limiter state; half-implemented
  limits would be worse than none, because they would look present.
- **Saturation.** PSS/E states it as two points on a curve, Dynawo as an exponential characteristic,
  and the two are not convertible without committing to a shape. Both readers parse it, neither
  uses it, and both report a nonzero value rather than dropping it silently.
- **State-triggered events** — a relay opening on an under-voltage threshold. Every event here is
  scheduled at a time, which is what makes the step truncation exact and the root-finding
  unnecessary.

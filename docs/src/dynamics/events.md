# Events and Discontinuities

## Why an event is not just another step

At an event the **algebraic** variables jump and the **differential** ones do not. A rotor angle
cannot change discontinuously — it is the integral of a speed — but a bus voltage can and does,
because the network equations carry no derivative and simply state what `V` must be *right now*,
given `x` and the present topology.

So the integration rule must not be applied across the discontinuity. Doing so would average a
pre-fault derivative with a post-fault one over a step in which neither held, which approximates
nothing. The handling instead:

1. Step exactly onto the event time.
2. Apply the change and reassemble the Y-bus.
3. Hold `x` fixed and re-solve `Y V − I_inj(x, V) = 0` for `V` alone.
4. Resume, with a couple of backward-Euler steps to damp the ringing the trapezoidal rule would
   otherwise show.

Step 3 is done by taking an ordinary step with `h·a = 0`: every differential row then reads
`x₁ − x₀ = 0`, pinning `x` exactly, while the algebraic rows are untouched. One assembly, one code
path, no separate pattern for the sub-block.

Both the pre-event and post-event points are recorded, at the same time value, so the discontinuity
is visible in the trajectory rather than smoothed away by whoever plots it.

### The states are pinned, not merely left alone

With `h·a = 0` the Jacobian is block lower triangular, so *in exact arithmetic* the state update is
zero. It is not zero in floating point: no backend does forward substitution in that order — each
applies its own fill-reducing permutation and partial pivoting — and roundoff leaks across the block
boundary, measured at ~`1e-10` rad on a faulted network whose conditioning the fault admittance
dominates.

Small, but wrong in kind and cumulative over a run with many events. A rotor angle is *defined* to
be continuous across a discontinuity, so the algebraic re-solve declines to apply the update at all,
and the gate asserts bit-identity rather than a tolerance.

## The events

| Kind | Effect |
|---|---|
| `BusFault` | Add a shunt admittance to ground. Replaces any fault already standing at that bus rather than adding to it |
| `ClearFault` | Remove it |
| `BranchTrip` / `BranchClose` | Take a branch out of service, or put it back |
| `UnitTrip` / `UnitClose` | Disconnect a generating unit, or reconnect it |
| `LoadStep` | Change a bus's load |

**All of them are value-only.** None changes the set of unknowns, so none forces a re-analysis, and
one symbolic factorization serves a whole run. See [the DAE chapter](./dae.md) for how the
topology-superset pattern buys that.

A "bolted" fault is a large admittance rather than an infinite one, because an infinite admittance
is not a number. `1e6` per unit against a network whose entries are order 1 leaves `|V|` around
`1e-6`, four orders below any tolerance that matters. Going larger buys nothing and costs
conditioning.

A `LoadStep` is applied as a change of **admittance**, not of power — `Δy = −conj(Δs)/|V₀|²` at the
voltage the case was initialized at — because that is what the loads in this system are. It
therefore delivers exactly `Δs` if the voltage happens to be back at `V₀`, and less, in proportion
to `|V|²`, if it is not.

A tripped unit is **frozen**, not removed. Nothing in the network can observe a disconnected
machine's rotor, so integrating it would track a quantity no result depends on, and reconnecting it
properly would need synchronization, which is not modelled.

## Timing

Steps are truncated to land exactly on each event time, and the time is then *snapped* to the
event's own value rather than accumulated — so an event time need not be a multiple of the step, and
does not drift with the number of steps that preceded it.

No root-finding is needed because every event here is scheduled at a time rather than triggered by a
state. State-triggered events — a relay opening on an under-voltage threshold — are not implemented;
the Illinois locator in [continuation](../powerflow/continuation.md) is the piece to reuse when they
arrive.

## Backward-Euler damping, and what it costs

The trapezoidal rule is A-stable but not L-stable, so a discontinuity excites a numerical
oscillation at the fastest mode that decays only as `(−1)^k`. Two backward-Euler steps after each
event annihilate it.

The cost is exact and known: backward Euler evaluates `δ̇` at the *end* of the step, so on a linearly
growing speed each step overshoots by `Ω_b·a·h²/2` with `a = P_m/2H`. Two steps leave a permanent
offset of `Ω_b·a·h²` — `2.5e-5` rad at a 1 ms step on a typical machine — and it falls as `h²`
despite backward Euler being first-order, because only a fixed number of steps ever use it.

Both of those numbers are gated against their closed forms, on a case where the exact trajectory is
known: with `P_e` removed by a terminal fault the acceleration is constant, the speed is linear, and
the angle is exactly quadratic — which the trapezoidal rule integrates with **no error at all**.

## Islands

A switching event can leave a group of buses with no machine and no fixed-voltage bus in it. Such an
island still solves — every bus has a path to ground through its own load — but the answer is a
de-energized island, not a dynamic one, and reading it as a trajectory would be a mistake. The run
reports it rather than presenting it.

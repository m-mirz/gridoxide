# Outer Loops

## Motivation

A Newton-Raphson solve answers *what is the state, given these injections, these bus types and
these tap positions*. Several real controls are not expressible that way, because each decides
one of those **inputs** from the solved state:

- A generator that runs out of reactive capability stops holding its voltage. Its bus is no
  longer a `PV` bus; it is a `PQ` bus injecting exactly its limit.
- System imbalance is picked up by whichever machines are on governor control, not dumped on
  one slack.
- An on-load tap changer moves until the bus or branch it regulates is inside a deadband. Its
  position is an *output*.

Each is a loop **around** the solve rather than a term inside it.

## Why this is a chapter rather than three functions

gridoxide grew the first two as standalone entry points — `newton_raphson_enforcing_q_limits`
and `newton_raphson_distributing_slack` — each constructing its own
[`PersistentSolver`](../solvers/backends.md) and running its own pass counter. The consequence
was not a missing feature but an impossible combination: **a caller picked one**. Real
transmission networks want both, and tap control would have been a third.

So they are [`OuterLoop`](https://docs.rs/gridoxide) implementations, and `outerloop::solve_with_loops`
drives an ordered list of them.

```rust
let mut slack = DistributedSlack::new(SlackDistribution::uniform(&buses));
let mut qlim = ReactiveLimits::new();
let mut phase = PhaseControl::new();
let mut voltage = TransformerVoltageControl::new().max_tap_shift(3);

let mut ctx = SolveContext::new(&mut buses, &mut ybus)
    .with_branches(&lines, &mut transformers, &shunts)
    .with_taps(&mut tap_changers, &regulation);

// Innermost first.
let mut list: Vec<&mut dyn OuterLoop> = vec![&mut slack, &mut qlim, &mut phase, &mut voltage];
let (islands, report) = solve_with_loops(&mut ctx, 1e-8, 30, backend, &mut list, 40);
```

An empty list is exactly one ordinary solve. That is not a special case in the code — it falls
out of the schedule below — and it is asserted, because it is what makes the layer safe to put
underneath every existing caller.

## The schedule

Transcribed from powsybl-open-loadflow's `AcloadFlowEngine`, because it is not the obvious one:

1. An initial solve runs first. If it does not settle, **no loop is consulted at all** — there
   is nothing useful a control can decide from a state that is not a power flow.
2. The loops are **nested, innermost first**. Each is run to *its own* stability — an inner
   check/re-solve loop — before the next is consulted.
3. Any loop reporting `Unstable` re-solves and becomes the "last unstable" loop. The driver then
   continues down the list, and wraps back to the start.
4. Termination is reaching the last-unstable loop again having changed nothing on the way: the
   whole list has been walked and every criterion holds *simultaneously*, which is what a fixed
   point of the combined controls means.

With an inner loop that moves twice and an outer one that moves once, the consultation order is

```text
inner inner inner   outer outer   inner
└── to its own stability ──┘ └─┘  └─ re-walk, nothing moves, done
```

and not `inner inner inner outer inner outer`. Both halves matter: the first is why an inner
control is never left half-converged, the second is why an outer control's decision is
re-examined by the inner ones.

`budget` caps total re-solves across every loop, as one shared figure rather than a per-loop
count.

### Ordering is a physical claim

The default order — distributed slack, then reactive limits, then phase control, then
transformer voltage control — is powsybl's, and it encodes that **generators respond faster than
tap changers**. A tap should be chosen against a reactive dispatch that has already settled, and
re-examined when the tap move disturbs it.

## Invalidation is the driver's job

Three things must happen together after a tap move — write `Transformer::tap`, restamp the Y-bus
entries derived from it, drop the cached Jacobian — and a loop that does one or two of them is a
bug that surfaces as a wrong answer on the *next* pass rather than as a failure on this one.

A loop therefore declares what it invalidated and the driver acts on it:

| Loop | What it changes | `Invalidates` | What the driver does |
|---|---|---|---|
| Distributed slack | `p_spec` only | `Nothing` | nothing — the Jacobian is unchanged |
| Reactive limits | `bus_type`, hence `n_unknowns` | `Pattern` | `PersistentSolver::reset` |
| Tap control | `Transformer::tap`, hence Y-bus *values* | `Admittances` | restamp, then `invalidate_admittances` |

`invalidate_admittances` drops the cached Jacobian pattern and keeps the per-backend
factorization, so the fill-reducing ordering and elimination tree survive — measured at ~45% of
solve time on a 9,241-bus case. This is the third caller of the same contract, alongside
[switching](../cgmes/node_breaker.md) and [batch](../solvers/backends.md) solving.

## Remote voltage control, and why it is a loop

A generator regulating the far side of its own step-up transformer is ordinary, and gridoxide has
always imported it: `RegulatingControl.Terminal` resolves to the controlled bus, which is pinned to
`PV` at the target. That gets the target right and puts the reactive power **in the wrong place** —
it appears at the bus being held rather than at the machine, so the reactive flow on the path
between them is missing, and the machine's own bus sits at whatever the network makes it rather
than at whatever holding the far bus requires.

Stated exactly, this is a small generalization of the `PV` bus. A `PV` bus means two things at once
— *this bus's magnitude is fixed* and *this bus's reactive injection is free* — and remote control
separates them: fix \(|V|\) at the controlled bus, free \(Q\) at the controller. The unknown
count still balances, one magnitude removed against one reactive equation removed:

| | unknown removed | equation removed |
|---|---|---|
| local control (the ordinary `PV` bus) | \(|V|\) at the bus | \(Q\) at the bus |
| remote control, one machine | \(|V|\) at the **controlled** bus | \(Q\) at the **controller** |
| remote control, \(m\) machines | \(|V|\) at the controlled bus | \(Q\) at each of \(m\) controllers |

The third row is short by \(m-1\) equations, which is what powsybl's `DISTR_Q` supplies. No
vendored fixture has that configuration, so gridoxide does not implement it — see
`plans/REACTIVE_DISPATCH_PLAN.md`.

Implementing the second row *exactly* means the unknown layout stops being derivable from
`BusType`, which is the assumption the Jacobian assembly, the Newton loop, the sensitivities, the
batched solve and the continuation all share. That is a large change to the most load-bearing code
in the crate, for a configuration two vendored fixtures have and one can exercise.

`RemoteVoltageControl` reaches the **same fixed point** without touching any of it. Make the
*controller* bus `PV` — which is what frees its reactive power, and frees it at the right bus — and
drive its own setpoint until the controlled bus reaches the target. At convergence \(|V|\) at the
controlled bus is the target and \(Q\) at the controller is whatever holding it takes, which is
exactly what the exact formulation asserts. The difference is iteration, not answer, and the test
suite checks precisely that: pinning the controller by hand at the setpoint the loop landed on, with
no loop running, reproduces the same state.

It also inherits the reactive limits properly. The controller bus is a `PV` bus carrying the
machine's own capability, so [reactive limits](./q_limits.md) clamp the machine rather than a bus it
is not connected to — the more correct bound as well as the easier one.

The step is `Δ|V|_controller = gain · (target − |V|_controlled)`, with `gain` starting at 1.0 and
refined by secant from the response observed. The 1.0 is not arbitrary: a machine holding the far
side of its own transformer moves that bus nearly one-for-one in per-unit, so the first step lands
close. A derived sensitivity would be the better answer once a second fixture asks for one;
deriving it now would be building the general thing on a sample of one.

**It is opt-in** (`--control-remote-voltage`,
`solve(control_remote_voltage=True)`) because turning it on changes answers. The default is wrong
rather than merely conservative, and that is stated here rather than left to be discovered.

## What this is not

Deliberately **not** an extensibility mechanism. powsybl's `OuterLoop` is ServiceLoader-discovered
so third parties can register their own; that is a Java-ecosystem affordance, and copying it would
buy an abstraction nobody outside this crate can reach. The list is built by the crate, and the
trait exists to make the loops *compose* — which was the actual defect.

## See also

- [Reactive Power Limits (PV → PQ Switching)](./q_limits.md) — including per-machine attribution
- [Distributed Slack](./distributed_slack.md)
- [Transformer Tap Control](./tap_control.md)

# Reactive Power Limits (PV → PQ Switching)

## Motivation

The plain Newton-Raphson formulation on the [Powerflow](./index.md) page treats every PV bus as if
its generator could supply or absorb *any* amount of reactive power needed to hold \\(\vert V_k \vert\\)
at its setpoint. Real generators can't: each has a nameplate reactive capability
\\(Q_k^{min} \le Q_k \le Q_k^{max}\\), usually tightest for round-rotor machines operating near their
active-power limit. If the voltage setpoint on a PV bus would require \\(Q_k\\) outside that range, the
unconstrained solution is not physically achievable — the generator saturates at its limit and the bus can
no longer hold its voltage.

The standard fix is **PV → PQ switching**: solve normally, check every PV bus's computed \\(Q_k\\) against
its limits, and for any bus that violates one, convert it to a PQ bus with \\(Q_k\\) pinned at the violated
limit — trading the voltage-magnitude equation for a reactive-power equation at that bus — then re-solve.
Repeat until no bus violates its limits.

## What changes in the equation system

Recall from the [Powerflow](./index.md) page that a PV bus contributes only a \\(P\\)-mismatch row to
\\(f(x)\\) — its voltage magnitude \\(\vert V_k \vert\\) is *known* (the setpoint), so it isn't part of the
unknown vector \\(x\\) and there's no \\(Q\\)-mismatch row for it. A PQ bus contributes both a \\(P\\)- and
a \\(Q\\)-mismatch row, and its \\(\vert V_k \vert\\) *is* part of \\(x\\).

Switching bus \\(k\\) from PV to PQ therefore:

- adds \\(\vert V_k \vert\\) to the unknown vector \\(x\\) (it's no longer fixed at the setpoint — the
  voltage is now free to float),
- adds a \\(Q\\)-mismatch row \\(Q_k^{calc}(x) - Q_k^{spec}\\) to \\(f(x)\\), with \\(Q_k^{spec}\\) pinned
  at whichever limit was violated (\\(Q_k^{min}\\) or \\(Q_k^{max}\\)) instead of the bus's original
  \\(q\\_spec\\),
- leaves that bus's \\(P\\)-mismatch row untouched — the same equation, unaffected by bus type.

This changes \\(n_{unknowns}\\) itself, so the Jacobian's dimensions grow by one row and one column per
switched bus. Any cached symbolic factorization (fill-reducing ordering) computed for the old sparsity
pattern is invalid once a switch happens and must be redone from scratch.

## Switching strategies

That equation-system change is common to every implementation. What differs is *when* the check runs and
whether a switch can ever be undone — three genuinely different designs, in increasing order of
bookkeeping.

### 1. One-directional outer-loop switching

Solve to full convergence, check limits, switch every violating bus to `PQ`, re-solve; repeat until a pass
switches nothing. A bus that has been switched to `PQ` stays `PQ` for the rest of the call, even if a later
pass's solution would put its \\(Q\\) back within limits.

The appeal is that the anti-oscillation problem simply doesn't arise: with switching one-way, the set of
`PQ` buses grows monotonically, so the outer loop terminates on its own. The cost is that a bus which was
only *transiently* out of range — say during an early pass, still far from the real solution — is committed
to `PQ` permanently, giving a slightly different, slightly more conservative answer than a scheme that could
release it.

### 2. Bidirectional switching with a switch-count cap

The same outer loop, except a bus moved to `PQ` may move back to `PV` in a later pass if its computed
\\(Q\\) has returned within limits — recovering exactly the transiently-violating case strategy 1 gives up
on. This reintroduces the risk strategy 1 avoids: a bus can flip between `PV` and `PQ` indefinitely across
passes, since each type change moves the solution enough to justify the opposite change next time. The
standard mitigation is a hard cap on how many times any one bus may switch, after which it is forced to stay
wherever it last was.

That cap is real bookkeeping — a per-bus counter carried across outer passes — and it makes the result
mildly path-dependent (which bus hits its cap first depends on iteration order). In exchange, no bus is
written off on the strength of one early, poorly-conditioned iterate.

### 3. Mid-iteration switching

Rather than a separate outer loop that fully re-solves after every switch, the check-and-switch step is
folded directly into the Newton iteration: once the mismatch drops below some threshold, bus types are
switched and the *same* iteration sequence continues instead of a fresh solve starting. This saves the
redundant re-convergence strategies 1 and 2 pay for — the iterate is already near the solution when the
switch happens, so it doesn't have to be rediscovered — at the cost of mutating the equation system
underneath a running Newton iteration, so the Jacobian's dimensions (and any cached factorization) change
mid-flight.

Published versions of this technique also widen the switch-*back* criterion: "On PV-PQ Bus Type Switching
Logic in Power Flow Computation" (Jinquan Zhao) bases the decision to release a bus on comparing its voltage
against its setpoint in addition to comparing its \\(Q\\) against its limits, not on \\(Q\\) alone.

## Where this fits in gridoxide today

`outerloop::ReactiveLimits` implements strategy 1, as an outer loop around a
`solver::PersistentSolver`:

1. Solve to convergence with every `PV` bus free (`PersistentSolver::solve`). If this doesn't converge,
   stop and return that status — there's no point checking limits on a non-converged solution.
2. Compute every bus's actual \\(Q_k\\) (`network::power_injections`).
3. For each `PV` bus whose \\(Q_k\\) violates `q_min`/`q_max`: switch `bus_type` to `PQ` and pin `q_spec`
   to the violated limit.
4. If no bus was switched this pass, the solution is self-consistent — stop and return `Converged`.
5. Otherwise, reset the cached factorization (`PersistentSolver::reset`) and go back to step 1.
6. If the outer-loop budget runs out without stabilizing, the run reports `budget_exhausted` rather
   than dressing the state up as converged.

Steps 4 and 5 are the [driver's](./outer_loops.md) job rather than this loop's: it returns
`Unstable` after switching anything, declares `Invalidates::Pattern`, and the driver resets the
factorization and re-solves. That separation is what lets reactive limits run *alongside*
[distributed slack](./distributed_slack.md) and [tap control](./tap_control.md) rather than
instead of them — before the outer-loop layer existed, each was its own entry point and a caller
picked one.

Strategies 2 and 3 both add real complexity (anti-oscillation bookkeeping, or reworking the iteration loop
itself) to recover a case gridoxide's simpler design just accepts as a one-way commitment — a deliberate
scope trade-off, not an oversight.

One further simplification on top of the one-directional rule: **bus voltages carry over between outer
passes** rather than resetting to a flat start. Since only one bus's type changes per pass, the previous
pass's converged state is normally very close to the next equilibrium, so re-solving after a switch
typically takes only a few extra Newton iterations, not a full fresh solve.

### Scope: net injection, not per-device limits

`Bus::q_min`/`q_max` bound the bus's *net* reactive injection — the same aggregate quantity `q_spec`
already represents when multiple loads/generators share a node (see `pgm::PgmVoltageRegulator`'s own
`q_min`/`q_max` parsing, which sums each active generator's limits at a bus exactly the way `p_spec` is
already summed). Pinning `q_spec` to a violated limit only achieves that limit exactly if the bus carries
no voltage-dependent ZIP-model load terms (`Bus::zip_terms`) — true for the common case of a bus that's
purely a PV/generator connection, not true in general if a voltage-dependent load is co-located there.

### Validated against real data

`tests/q_limits_test.rs` checks the mechanics directly: a 3-bus fixture with one PV bus whose `q_min` is
set tight enough to force a violation, confirmed to switch to `PQ`, pin `q_spec` at exactly `q_min`, and
converge to the same voltages across every Jacobian backend (`Scalar`, `Block`, `Klu`) — as well as a
control case with a *loose* `q_min` that shouldn't trigger any switch at all, converging to the same answer
as plain unconstrained Newton-Raphson.

Beyond the unit fixture, this has been exercised against all 12 real MATPOWER benchmark cases
(`gridoxide.matpower` — `python/gridoxide/matpower.py`, with `scripts/bench/matpower_to_pgm.py` as a thin
CLI wrapper around it — populates `voltage_regulator.q_min`/`q_max` from each case's own `gen` matrix): 11 of the 12 have at least one PV bus whose unconstrained \\(Q\\) genuinely exceeds its nameplate
limit — from 4 violations on the smallest case up to 166 simultaneous violations on `case3120sp`.

## Tool reference

| Tool | Strategy | Where |
|---|---|---|
| **gridoxide** | 1 — one-directional outer loop | `outerloop::ReactiveLimits` (opt-in; plain `newton_raphson` ignores `q_min`/`q_max`) |
| MATPOWER | 1 — one-directional outer loop | `runpf`'s `enforce_q_lims` option |
| pandapower | 1 — one-directional outer loop, pypower/MATPOWER-derived | `enforce_q_lims` (NR algorithm only, per its own docstring) |
| powsybl-open-loadflow | 2 — bidirectional, capped switch count per bus | `ReactiveLimitsOuterLoop` (handles capability curves too, not just fixed limits) |
| VeraGrid | 3 — mid-iteration, per Zhao's switching logic (cited in its own source) | `PowerFlowOptions.control_q` |

## Which machine, not just which bus

A `PV` bus has one reactive injection and the solver decides it. The limits above are the *summed*
capability of everything holding that bus — which is the right bound to enforce, and is what the
importer accumulates. But it leaves a question unanswered: when six machines hold one bus, which of
them is carrying the reactive power?

`gridoxide solve --dispatch` (and `PowerFlowModel.machine_dispatch()`) answers it, by attributing
the solved injection after the fact:

- by an explicit per-machine key where the source document states one;
- else in proportion to each machine's own reactive range;
- else evenly.

The fallback is **wholesale**: one implausible or missing value discards the whole basis rather than
leaving a split half-derived from capability and half from nothing. Shares are then clamped to each
machine's own range and the excess re-split, so no machine is handed more than it can produce while
another sits idle.

**This changes no answer.** It is arithmetic on a solved state, not a constraint inside the solve.
Every shared-control case in the vendored CGMES corpus is machines sitting at the bus they regulate,
so that bus is an ordinary `PV` bus with a single injection. The genuinely different problem —
several machines at *different* buses holding one remote bus — needs extra equations inside the
Newton system, and is not built, because no fixture exercises it. See
`plans/REACTIVE_DISPATCH_PLAN.md`.

### What the attribution exposed

It is worth reading `unattributed` on the result. A non-zero value means the machines at that bus
are all at their limits and *still* cannot account for the reactive power the solve put there.

That is not an arithmetic failure. It happens because the clamp above compares `q_min`/`q_max`
against the bus's **net** injection, while the CGMES importer fills those fields from the machines'
**own** capability. Where a reactive load shares the bus, the machine must cover it too — so the
machine saturates before the net injection reaches the bound, and the clamp fires late. On RealGrid,
50 of 416 regulated buses share their bus with another reactive injection and 6 come out
unattributable, 4.4 MVAr in total.

The simplification was already documented (`pgm::PgmVoltageRegulator`'s doc comment names it);
nothing had measured it before. Correcting it means bounding the machine part rather than the net,
which changes power-flow answers and so is deliberately not folded into an attribution pass.

# Reactive Power Limits (PV → PQ Switching)

## Motivation

The plain Newton-Raphson formulation on the [Powerflow](./powerflow.md) page treats every PV bus as if
its generator could supply or absorb *any* amount of reactive power needed to hold \\(\vert V_k \vert\\)
at its setpoint. Real generators can't: each has a nameplate reactive capability
\\(Q_k^{min} \le Q_k \le Q_k^{max}\\), usually tightest for round-rotor machines operating near their
active-power limit. If the voltage setpoint on a PV bus would require \\(Q_k\\) outside that range, the
unconstrained solution is not physically achievable — the generator saturates at its limit and the bus can
no longer hold its voltage.

The standard fix, used by MATPOWER's `runpf` (`enforce_q_lims`) and implemented the same way in
gridoxide, is **PV → PQ switching**: solve normally, check every PV bus's computed \\(Q_k\\) against its
limits, and for any bus that violates one, convert it to a PQ bus with \\(Q_k\\) pinned at the violated
limit — trading the voltage-magnitude equation for a reactive-power equation at that bus — then re-solve.
Repeat until no bus violates its limits.

## What changes in the equation system

Recall from the [Powerflow](./powerflow.md) page that a PV bus contributes only a \\(P\\)-mismatch row to
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

## The outer-loop algorithm

`solver::newton_raphson_enforcing_q_limits(buses, ybus, tol, max_iter, backend, max_outer_iter)` implements
this as an outer loop around a `solver::PersistentSolver`:

1. Solve to convergence with every `PV` bus free (`PersistentSolver::solve`). If this doesn't converge,
   stop and return that status — there's no point checking limits on a non-converged solution.
2. Compute every bus's actual \\(Q_k\\) (`network::power_injections`).
3. For each `PV` bus whose \\(Q_k\\) violates `q_min`/`q_max`: switch `bus_type` to `PQ` and pin `q_spec`
   to the violated limit.
4. If no bus was switched this pass, the solution is self-consistent — stop and return `Converged`.
5. Otherwise, reset the cached factorization (`PersistentSolver::reset`) and go back to step 1.
6. If `max_outer_iter` outer passes pass without stabilizing, stop and return `MaxIterationsReached`.

Two simplifications versus a fully general implementation (deliberate, and both match MATPOWER's own
default `runpf` behavior rather than being a gridoxide-specific shortcut):

- **Switching is one-directional.** A bus switched to `PQ` never switches back to `PV` within the same
  call, even if a later pass's solution would put it back within limits. A bidirectional scheme needs real
  anti-oscillation logic (a bus can otherwise flip back and forth indefinitely across outer passes); this
  avoids that whole problem by not attempting it.
- **Bus voltages carry over between outer passes** rather than resetting to a flat start. Since only one
  bus's type changes per pass, the previous pass's converged state is normally very close to the next
  equilibrium, so re-solving after a switch typically takes only a few extra Newton iterations, not a full
  fresh solve.

## Alternative switching strategies

One-directional switching is a deliberate simplification, not the only viable design. Two genuinely
different strategies exist for the same underlying problem:

- **Bidirectional switching with a hard switch-count cap.** A bus moved to `PQ` is allowed to move back to
  `PV` in a later pass if its computed \\(Q\\) returns within limits — recovering a bus that was only
  transiently out of range (e.g. during an early, poorly-conditioned iteration far from the solution)
  instead of committing it to `PQ` for the rest of the solve. The risk this reintroduces is a bus
  oscillating between `PV` and `PQ` indefinitely across passes; the standard mitigation (used this way by
  powsybl-open-loadflow's `ReactiveLimitsOuterLoop`) is a hard cap on how many times any one bus may switch,
  after which it's forced to stay wherever it last was — real bookkeeping gridoxide's one-directional rule
  avoids needing at all.
- **Checking mid-iteration instead of only after full convergence.** Rather than a separate outer loop that
  fully re-solves after every switch, the check-and-switch step can instead be folded directly into the
  Newton iteration itself once the mismatch drops below some threshold, switching bus types and continuing
  the same iteration sequence rather than starting a fresh solve. One published version of this technique
  ("On PV-PQ Bus Type Switching Logic in Power Flow Computation", Jinquan Zhao — cited directly in VeraGrid's
  own source as the basis for its implementation) also bases the switch-*back* decision on comparing the
  bus's voltage against its setpoint in addition to its \\(Q\\) against its limits, not \\(Q\\) alone.

Both alternatives add real complexity (anti-oscillation bookkeeping, or reworking the iteration loop itself)
to recover a case gridoxide's simpler design just accepts as a one-way commitment — a deliberate scope
trade-off matching MATPOWER's own default behavior, not an oversight.

## Scope: net injection, not per-device limits

`Bus::q_min`/`q_max` bound the bus's *net* reactive injection — the same aggregate quantity `q_spec`
already represents when multiple loads/generators share a node (see `pgm::PgmVoltageRegulator`'s own
`q_min`/`q_max` parsing, which sums each active generator's limits at a bus exactly the way `p_spec` is
already summed). Pinning `q_spec` to a violated limit only achieves that limit exactly if the bus carries
no voltage-dependent ZIP-model load terms (`Bus::zip_terms`) — true for the common case of a bus that's
purely a PV/generator connection, not true in general if a voltage-dependent load is co-located there.

## Validated against real data

`tests/q_limits_test.rs` checks the mechanics directly: a 3-bus fixture with one PV bus whose `q_min` is
set tight enough to force a violation, confirmed to switch to `PQ`, pin `q_spec` at exactly `q_min`, and
converge to the same voltages across every Jacobian backend (`Scalar`, `Block`, `Klu`) — as well as a
control case with a *loose* `q_min` that shouldn't trigger any switch at all, converging to the same answer
as plain unconstrained `newton_raphson`.

Beyond the unit fixture, this has been exercised against all 12 real MATPOWER benchmark cases
(`scripts/bench/matpower_to_pgm.py` populates `voltage_regulator.q_min`/`q_max` from each case's own `gen`
matrix): 11 of the 12 have at least one PV bus whose unconstrained \\(Q\\) genuinely exceeds its nameplate
limit — from 4 violations on the smallest case up to 166 simultaneous violations on `case3120sp`.

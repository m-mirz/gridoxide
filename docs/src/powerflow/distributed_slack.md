# Distributed Slack

## Motivation

An ordinary power flow needs one bus whose active power is free. Load is specified, generation
is specified, but transmission losses are not known until the solve finishes — so *something*
has to absorb whatever the schedule does not cover. That something is the slack bus.

The trouble is that "whatever the schedule does not cover" is not small. It is the sum of every
scheduling mismatch **plus all system losses**, and an ordinary solve puts all of it on one
machine. On `case118_ieee` that is 16.5 per-unit — 1,650 MW — appearing at a single bus that was
never scheduled for it.

Real systems do not behave that way. Generators on governor control pick up imbalance together,
in proportion to their droop settings, and the plant at the reference bus is not special. The
single-slack answer is therefore wrong in a specific and predictable place: the branches around
the slack, which carry an injection no real dispatch would put there. Those are frequently the
branches you were looking at.

## The formulation

Give each bus a **participation weight** \\(\alpha_i\\), and let the slack's own `p_spec` be its
schedule. Then:

1. Solve the power flow normally.
2. Measure the slack's excess over its schedule, \\(\Delta = P^{calc}_{slack} - P^{sched}_{slack}\\).
3. Add \\(\alpha_i \Delta / \sum_j \alpha_j\\) to every participating bus's schedule.
4. Solve again, and repeat until \\(|\Delta|\\) is below tolerance.

```rust
use gridoxide::solver::{newton_raphson_distributing_slack, JacobianBackend, SlackDistribution};

// Every generator bus — Slack and PV — takes an equal share.
let distribution = SlackDistribution::uniform(&buses);

let (islands, report) = newton_raphson_distributing_slack(
    &mut buses, &ybus, 1e-10, 30, JacobianBackend::Scalar, &distribution,
);

println!("moved {:.3} pu off the slack", report.shift.iter().sum::<f64>());
```

`SlackDistribution::from_weights` takes explicit weights instead. They are **normalized**, so
raw megawatt headroom or droop constants can be passed without pre-dividing.

### The slack's schedule is its own `p_spec`

Ordinary power flow ignores that field for a slack bus — the slack's output is an *answer*, not
an input — which is exactly what leaves it free to carry the schedule here.

The consequence is worth stating plainly: **a document that never needed the field may leave it
at zero**, in which case the slack is treated as scheduled for nothing and its entire output is
redistributed. That is a coherent answer to a coherent question, but it is probably not the
question you meant. Set it if the reference plant has a real schedule.

### Weights normalize per island, not globally

A disconnected network has one slack and one imbalance *per island*. Normalizing globally would
size one island's correction by another island's generators — an island with large machines
would quietly dominate the split in an island it is not even connected to.

So the sum in step 3 runs over that island's buses only, which is also why the report names
islands it could not distribute over rather than skipping them.

## Convergence

The first-order term cancels in a single pass, and that is what makes an outer loop affordable
here at all. Distributing \\(\Delta\\) adds \\((1 - \alpha_s)\Delta\\) to the *other*
participants' schedules, so the next solve asks the slack for exactly that much less — while its
own schedule has risen by \\(\alpha_s\Delta\\). The two cancel.

**What is left over does not vanish.** Moving generation changes the flows, which changes the
losses, which changes the imbalance. So convergence is linear at roughly the fractional loss
sensitivity — a few per cent per pass — rather than the one-and-done the cancellation alone
suggests:

| Case | Passes to 1e-8 | Moved off the slack |
|---|---|---|
| `case14_ieee` | 7 | 2.3 pu |
| `case30_ieee` | 7 | 2.4 pu |
| `case118_ieee` | 7 | 16.5 pu |

Seven passes is not seven full solves' worth of work, because of the next point.

### The factorization survives the whole loop

Only `p_spec` changes between passes. Bus types do not, so `n_unknowns` does not, so the
Jacobian's sparsity pattern does not — and
[`PersistentSolver`](../solvers/backends.md) keeps its symbolic factorization across every pass.

That is the opposite of the [Q-limit loop](./q_limits.md), which switches buses `PV → PQ` and
must therefore `reset()` each time it does. The two outer loops look alike and cost very
differently.

## Reading the result

`buses[i].p_spec` ends up holding the **dispatched** value rather than the schedule it went in
with. That is the answer — who ended up producing what — and `report.shift` records how far each
bus moved, so the original is recoverable.

`report.undistributed` names any island left on a single slack, with the reason: no reference
bus, an ambiguous one, or no participating generator in that island. This matters more than it
looks. A caller who mis-specifies weights gets a perfectly ordinary single-slack answer, which
is a correct answer to a *different* question — the report is the only thing distinguishing it
from success.

## How it is checked

`tests/distributed_slack_test.rs`. The defining property is that the slack ends up producing its
schedule and nothing more, but the more interesting check is that the shifts **sum to** the
imbalance a single-slack solve put on that bus: a formulation that quietly created power would
still converge, and would still leave the slack at its schedule, so only the accounting catches
it.

Every check re-derives injections from `network::power_injections` at the returned state rather
than reading the loop's own bookkeeping, so the outer loop cannot corroborate itself.

## Is area-interchange control the same thing?

**Closely related — it is this, generalized from one balance target to several.** Not a
different mechanism.

Distributed slack has one balance condition per island: *the slack produces its schedule*. Area
interchange control partitions the network into control areas, each with a scheduled net
exchange with its neighbours, and asks that *every area's tie-line flows sum to its schedule*.
Both are "adjust generation among participating units until a power-balance target is met"; the
difference is how many targets there are and what defines them.

powsybl-open-loadflow makes the relationship explicit rather than incidental. Its
`AcAreaInterchangeControlOuterLoop` constructor reads:

```java
super(activePowerDistribution,
      new DistributedSlackOuterLoop(activePowerDistribution, slackBusPMaxMismatch),
      slackBusPMaxMismatch, areaInterchangePMaxMismatch, LOGGER);
```

It *constructs a distributed-slack outer loop* and holds it as the fallback used when the
network defines no areas, shares the same `ActivePowerDistribution` engine for the redistribution
itself, and reuses `DistributedSlackContextData` for its bookkeeping. Distributed slack is
literally the degenerate case.

gridoxide does not implement area interchange control. What it would need is not a new algorithm
but three additions: an area assignment per bus, a scheduled interchange per area, and a
per-area mismatch computed from tie-line flows in place of the slack deviation used here. The
per-island normalization already in `newton_raphson_distributing_slack` is the same shape that
generalizes to per-area — islands and areas are both just partitions of the bus set with their
own balance target.

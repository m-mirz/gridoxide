# Node-Breaker Topology

A CGMES model does not describe a network as buses joined by lines. It describes it the way the
substation is actually built: equipment terminates on **connectivity nodes**, and connectivity nodes
are joined by **switching devices** — breakers, disconnectors, load-break switches, fuses. The buses
a power flow needs are a *derived* view, obtained by fusing every group of nodes reachable through
closed switches.

gridoxide has always performed that fusion — see [Reading CGMES Input](./index.md) and
`cgmes::merge_closed_switches`. What it did not do until recently was *keep* the switches. They were
consumed at import and discarded, which meant switching state was frozen at the moment the file was
read: no switch could be operated between solves, no flow through a breaker could be reported, and no
measurement could attach to one.

The node-breaker path keeps them.

## Two views of the same file

```rust
use gridoxide::cgmes::{cgmes_node_breaker_to_buses_and_branches, cgmes_to_buses_and_branches};
use gridoxide::switches::SwitchTreatment;
use gridoxide::topology::RetentionPolicy;

// The bus-branch view: switches are fused away. This is the original path.
let (buses, lines, transformers, shunts) = cgmes_to_buses_and_branches(&ds, 100e6)?;

// The node-breaker view: switches survive as network elements.
let net = cgmes_node_breaker_to_buses_and_branches(
    &ds,
    100e6,
    &RetentionPolicy::RetainAdjacentToBusbar,
    SwitchTreatment::Regularize,
)?;
```

`net` is a `switches::NodeBreakerNetwork`: the same `buses`/`lines`/`transformers`/`shunts` any other
path produces, plus the `BusView` they were derived from and the switch↔branch mapping.

Both read the same profiles. The difference is what happens to the switching devices, and that is
governed entirely by the retention policy.

## Retention policies

A `RetentionPolicy` answers one question per switch: *does this device survive into the solved
network as an element of its own, or is it fused away?*

| Policy | Keeps | Typical use |
|---|---|---|
| `MergeAll` | nothing | Reproduces the classical bus-branch view exactly. The baseline. |
| `RetainAdjacentToBusbar` | switches with a `BusbarSection` on at least one side | Busbar coupling and bay isolation — the switching that actually reconfigures a station. The default. |
| `RetainKinds(set)` | switches of the listed `SwitchKind`s | "Model the breakers, merge the disconnectors." |
| `Explicit(set)` | the switches you name | Studying a specific bay. |
| `RetainAll` | everything | The full node-breaker network, one bus per connectivity node. |

The cost is bus count, and it is not small. On the conformance configurations:

| Model | Connectivity nodes | `MergeAll` | `RetainAdjacentToBusbar` | `RetainAll` |
|---|---|---|---|---|
| MiniGrid | 103 | 15 buses, 0 switches | 45 buses, 30 switches | 105 buses, 90 switches |
| SmallGrid | 1,369 | 167 buses, 0 switches | 540 buses, 373 switches | 1,369 buses, 1,266 switches |
| Svedala | 1,179 | 228 buses, 0 switches | 638 buses, 857 switches | 1,179 buses, 1,464 switches |

(Bus counts exceed the merged-node count slightly — MiniGrid's 105 against 103 connectivity nodes —
because equipment conversion adds buses of its own, notably the star point of a three-winding
transformer.)

All of these converge, and in the *same iteration count* as their own bus-branch solve: 2 for
MiniGrid, 5 for SmallGrid, 6 for Svedala, whether every switch is merged or every switch is retained.
That is worth stating plainly, because it was not the expected outcome — the working assumption was
that retaining switches at this scale would wreck the conditioning of the Jacobian. It does not. See
[Ideal Switches and Zero-Impedance Branches](../powerflow/zero_impedance_branches.md) for why, and for
what the alternative formulations would have cost.

## How a retained switch is represented

A retained switch is stamped as an ordinary branch with a small series reactance — the
**regularization** approach. Two consequences follow, and both are the reason this approach was
chosen over merging:

- **A switch is a branch.** Nothing downstream needs to know it is a switch. It has a flow, so
  `switch_flow` is just a branch flow. It has a PTDF/LODF column, so a switch's `lodf_column` *is*
  its bus-split distribution factor. A switching campaign is an ordinary
  `BatchSolver::solve_contingencies` sweep. None of that needed new solver code.
- **Operating a switch does not move the sparsity pattern.** A switch's position is carried as its
  terminal *status*, not as a change to the node set. Opening one changes Y-bus *values* while the
  structural entries stay in place, which is what lets `PersistentSolver` keep its symbolic
  factorization across states. Merging cannot do this — closing a switch would change `n`.

The series reactance is `switches::switch_reactance()`. It is small enough that the voltage drop
across a closed switch is negligible and large enough not to dominate the matrix; the same value is
used by the ideal-connection handling described in the zero-impedance chapter.

### Degenerate switches

A retained switch whose two nodes land in the same bus anyway — because some *other* closed path
already joins them — is **degenerate**: stamping it would create a self-loop. `bus_view` detects
these and `switches::degenerate_switches` reports them. They are retained in the topology (so they
keep their identity) but not stamped as branches. None of the three models above has any — retained
and stamped counts match exactly — but a model with a closed ring inside one bay would.

## Operating a switch

```rust
for (switch, branch) in net.switch_branches() {
    println!("{} -> branch {branch}", net.switch_label(switch));
}

net.set_switch_open(switch, true);            // false to close it again
assert_eq!(net.is_switch_open(switch), Some(true));

// Re-solve, then read the flow through the switch itself.
let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
stamp_shunts(&mut ybus, &net.shunts);
let report = run_power_flow_analysis_from_ybus(net.buses.clone(), ybus);
let v = gridoxide::branch_flow::bus_voltages(&report.buses);
let (p, q) = net.switch_flow(switch, &v).unwrap();
```

`set_switch_open` mutates the stamped branch's status, which is the single source of truth —
`is_switch_open` reads it back from there rather than from the imported model, so a listing never
shows a stale position.

## From Python

See [Python Bindings](../getting_started/python.md#node-breaker-topology-and-switches):

```python
model = gridoxide.PowerFlowModel.from_cgmes(
    paths, topology="node_breaker", retain="busbar_adjacent"
)
for switch_id, mrid, kind, bus_from, bus_to, is_open, branch in model.switches():
    ...
model.set_switch(switch_id, True)
```

## State estimation

```rust
let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
stamp_shunts(&mut ybus, &net.shunts);
let se = net.se_network(ybus.finish());

let mut buses = net.buses.clone();
se::nr::linear_start(&mut buses, &se, &measurements);
let report = se::nr::estimate(&measurements, &mut buses, &se, &SeOptions::default());
```

The measurement model needs no changes at all: a sensor on a breaker is a `Target::BranchTerminal`
at `branch_of_switch(switch)`, and the estimator never learns that any of its branches is a switch —
the same reason phases 4 and 6 of the plan came free.

What *is* different is the constraint set, and it inverts the usual proportions. A bus-branch model
has a handful of zero-injection buses; a node-breaker model is mostly zero-injection buses, since
every internal node of a bay carries switches and nothing else. MiniGrid goes from 7 constrained
buses to 37. That is exact, free information — no sensor, no noise, no weight — and it is why the
finer topology is observable without any more sensors than the coarse one needs. gridoxide has
carried zero-injection buses as hard equality constraints (the KKT augmented system) all along; see
[The State Estimation Problem](../state_estimation/index.md).

The flags come from the model's structure, not from the snapshot's numbers. A load whose SSH `p` and
`q` happen to be zero this hour is not a bus that injects nothing, and a hard constraint saying
otherwise would bias every estimate around it.

Two things are worth knowing before running one:

- **Start with `linear_start`, not `flat_start`.** CGMES marks a de-energized bus by leaving it out
  of every `TopologicalIsland`, and such a bus must *start* at zero. Nothing measures it, so it is
  pinned wherever the start left it; pinned at 1 p.u. when the truth is 0, it poisons every
  measurement that touches it. On MiniGrid a flat start diverges to an objective of 2.2e4 where
  `linear_start` converges in three iterations to 4e-10.
- **De-energized buses are not an observability gap.** `observability::analyze` reports their
  unknowns separately, in `de_energized`, and excludes them from the rank test — they are determined
  by the network at exactly zero rather than by any sensor. Counting them would report every model
  with one switched-out node as unobservable.

## From the command line

```
gridoxide switches <profile.xml>... [--retain none|busbar_adjacent|all]
                   [--open <mrid>] [--solve]
```

```console
$ gridoxide switches MiniGrid_{EQBD,EQ,SSH,TP,SV}.xml --solve
45 bus(es) from 103 connectivity node(s); retain = busbar_adjacent
90 switch(es) in the model, 30 retained, 30 stamped as branches
power flow: Converged in 2 iteration(s)

mrid                                   kind          bus  bus  state branch    P (MW)
_3a783d1d-…                            Disconnector    8   22 closed     18    -9.000
…
```

Switches are named by **mRID**, which is what an operator and the source file have in common;
`--open` takes one. The requested profiles must include TP and SV even without `--solve`, because
converting the equipment needs the angle reference from
`TopologicalIsland.AngleRefTopologicalNode` — EQ and SSH alone give the topology but not a slack, and
gridoxide will not invent one.

## What is not there yet

- **Measurements** are not read from CGMES. The estimator accepts a node-breaker network (below),
  but the profiles carry nothing to feed it: only the OP profile holds `Analog`/`AnalogValue`, only
  FullGrid ships one in the conformance set, and it holds four of them. A caller supplies its own
  measurements.
- **The equality-constrained formulation** (`SwitchTreatment::Constrain`) is unimplemented.
  Regularization was expected to fail at scale and did not, so the case for it is weaker than it
  looked; it remains the right answer if a model ever appears where regularization *does* break down.
- **FullGrid** does not solve on either path. This is pre-existing and unrelated to switches.

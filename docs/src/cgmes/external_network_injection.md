# ExternalNetworkInjection

## Motivation

Any CGMES sub-model that covers less than the full interconnected synchronous area — which is to say,
almost every real CGMES file, since even a large national TSO's own model is a *part* of Continental
Europe's actual network — needs some way to represent what lies beyond its own boundary. CGMES has (at
least) two distinct classes for this: `EquivalentInjection` and `ExternalNetworkInjection`. Both carry
P/Q; the difference is that `ExternalNetworkInjection` additionally derives from `RegulatingCondEq` (the
same base `StaticVarCompensator` and `SynchronousMachine` share), so it can also carry a
`RegulatingControl` and behave like a voltage-regulated source, not just a fixed injection. ENTSO-E's
MiniGrid conformance fixture uses it this way for its two external-grid connection points ("Q1"/"Q2").

## The concepts

### 1. An external injection is a generator, not a new element type

Structurally, "the rest of the interconnected system" behaves exactly like a generator at the boundary
bus: it injects P and Q, it may hold a voltage setpoint, and it has (possibly unbounded) reactive limits.
Nothing about it needs a distinct element in the internal network model — the same
`RegulatingControl`-driven PV upgrade a `SynchronousMachine` gets applies unchanged, with the originating
CGMES class kept only as a tag for round-trip export.

This applies to `EquivalentInjection` and `ExternalNetworkInjection` equally: the two CGMES classes need
not map to two internal types.

### 2. Sign convention: both P and Q are negated

CGMES's SSH power-flow values for these classes are given in the *load* sign convention — power drawn from
the network at the terminal — so converting to an injection means negating both:

\\[ P_{inj} = -p_{SSH}, \qquad Q_{inj} = -q_{SSH} \\]

This is worth stating explicitly because it is *not* universal across CGMES's regulating-equipment
classes: `SynchronousMachine`'s own `q` follows a different rule in practice (no negation), so "it derives
from `RegulatingCondEq`, therefore it signs like a machine" is exactly the wrong inference. The right
grouping is by *conceptual role*: an external injection stands in for a network, not for a physical
rotating machine.

### 3. Boundary-point folding applies to only one of the two classes

An `EquivalentInjection` sitting exactly at a network boundary point can be folded into the virtual
generation/load of the boundary line that crosses it, rather than becoming a standalone element.
`ExternalNetworkInjection` has no equivalent path, because it isn't itself a boundary-defining class — it
is always a standalone element at an ordinary bus. This is the one real structural difference between the
two, and it only matters to a tool that models boundary lines as first-class objects.

### 4. Under-specified limits interact with slack selection

CGMES often leaves an external injection's active-power limits unspecified, and the natural default is
"unbounded" (\\(\pm\infty\\), or `±Double.MAX_VALUE`). Any heuristic that picks a slack bus by generator
size then has to cope with a generator whose stated `maxP` is absurd — the usual treatment is a
plausibility threshold above which a candidate is *excluded*. So an under-specified external injection
tends to be filtered out of slack consideration rather than favored, purely as a side effect of the
defaulting.

## Where this fits in gridoxide today

The conversion (`src/cgmes.rs`) applies concept 1 by mirroring `StaticVarCompensator`'s block almost
exactly — same `RegulatingCondEq`/`RegulatingControl`-driven PV-bus upgrade, same fallback to a fixed
injection when not actively regulating — with concept 2's sign convention:

```rust
for mrid in by_type(ds, "ExternalNetworkInjection") {
    let eni: &cimstructs::ExternalNetworkInjection = require(ds, mrid, "ExternalNetworkInjection", mrid, "(self)")?;
    ...
    buses[bus].p_spec += -eni.p.unwrap_or(0.0) * 1e6 / s_base_va;
    buses[bus].q_spec += -eni.q.unwrap_or(0.0) * 1e6 / s_base_va;
    ...
```

Both P *and* Q are negated here — `EquivalentInjection`'s convention, not `SynchronousMachine`'s Q
exception (see the [StaticVarCompensator](./static_var_compensator.md) page for why that exception
exists and why it doesn't extend here).

Concepts 3 and 4 don't arise: gridoxide has no boundary-line object for an injection to fold into, and its
slack bus comes from CGMES's own `TopologicalIsland.AngleRefTopologicalNode` (see
[Multi-Island Power Flow](../powerflow/multi_island.md)) rather than from any generator-size heuristic, so an
unbounded `maxP` has nothing to distort.

## Tool reference

| Tool | Internal type (§1) | Sign (§2) | Boundary folding (§3) |
|---|---|---|---|
| **gridoxide** | bus-level injection; PV upgrade if voltage-regulating, else fixed P/Q (`src/cgmes.rs`) | both negated | ❌ no boundary-line model |
| powsybl-core | IIDM `Generator` (`EnergySource.OTHER`) for *both* CGMES classes, distinguished only by a `PROPERTY_CGMES_ORIGINAL_CLASS` string for round-trip export | both negated: `targetP = -updatedPowerFlow.p()`, `targetQ = -updatedPowerFlow.q()` — same in `EquivalentInjectionConversion.update()` | ✅ for `EquivalentInjection` at a boundary point (folded into a `BoundaryLine`); n/a for `ExternalNetworkInjection` |
| powsybl-open-loadflow | none of its own — nothing in the tool references `ExternalNetworkInjection` or its origin-class property; it is an ordinary generator by the time the solver sees it | inherited | inherited |

powsybl-open-loadflow's only indirect sensitivity is concept 4: `ExternalNetworkInjectionConversion`
defaults `maxP` to `±Double.MAX_VALUE`, and `LargestGeneratorSlackBusSelector` filters out generators
whose `maxP` exceeds a plausibility threshold when choosing a slack candidate.

power-grid-model and lightsim2grid have no CGMES import, so neither has an equivalent concept.

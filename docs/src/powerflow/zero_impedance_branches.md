# Ideal Switches and Zero-Impedance Branches

## Motivation

Real networks contain connections that are, by design, not really "branches" in the sense the
[Powerflow](./index.md) page assumes: breakers, disconnectors, bus-bar couplers, jumpers — elements meant
to either tie two points electrically together with (idealized) zero impedance, or fully separate them, with
no impedance value in between and no partial state.

Modeling one as an ordinary branch runs straight into the admittance formulation the rest of that page
relies on. A branch's contribution to the bus admittance matrix is built from \\(Y = 1/Z\\); with \\(Z = 0\\)
that's an infinite entry, not a large-but-finite one. And this isn't just a numerical inconvenience to work
around — a *closed* zero-impedance connection isn't naturally a "branch" at all. Its physical meaning is an
**equality constraint**, \\(V_i = V_j\\), not a current flowing in proportion to a voltage difference. That's
a fundamentally different kind of equation from every other term in \\(f(x)\\), which is exactly why real
tools solve this with dedicated mechanisms rather than a variant of the ordinary branch stamp.

## Three approaches

### 1. Topological reduction (merge before you formulate)

The most direct fix is to never let the zero-impedance connection reach the admittance formulation in the
first place. Build a graph of nodes connected by *closed* zero-impedance edges, find its connected
components, and treat each component as a single node in the actual power-flow model — an open switch is
simply not an edge, so it doesn't merge anything.

This adds no new equations, no new unknowns, and no numerical stiffness. The costs are that it happens
*before* the solver ever runs, as a separate graph pass over the input model, and that the two original
terminals lose their distinct identity — nothing downstream can distinguish "flow through the left side of
the coupler" from "flow through the right side" once they've become one node. Switching state also can't
change between solves without redoing the reduction and rebuilding the model.

### 2. Large-admittance regularization

A second approach keeps the zero-impedance connection as an ordinary branch, but assigns it a series
admittance many orders of magnitude larger than anything else in the network (say 10<sup>6</sup> S in a
10 kV network) and zero shunt admittance. The same nodal equations that already exist for every other branch
then force near-equality of the two bus voltages as a natural numerical consequence — no new equation type,
no graph pass, just an extreme parameter value. Open/closed status needs no special handling either: it's
the same connection-status flag every other branch already has.

The trade-off is conditioning: introducing one admittance value five or six orders of magnitude larger than
the rest of the matrix widens its dynamic range substantially, which is exactly the kind of thing a direct
sparse solver's pivoting has to work harder to stay accurate through as a network grows. That's a real cost,
paid deliberately in exchange for needing no separate topological pre-processing stage at all — the ideal
connection is just a branch, as far as the rest of the solver is concerned.

### 3. Equality-constrained augmented system

The third approach takes the constraint interpretation from the Motivation section literally: instead of
merging the two nodes away or approximating the constraint with an extreme admittance, add the equation
\\(V_i = V_j\\) (and, in polar form, \\(\theta_i = \theta_j\\)) directly into the Newton-Raphson system as
its own row. Since adding an equation without adding an unknown would leave the system over-determined, a
pair of new "dummy" variables is introduced alongside it — one at each of the two buses, equal and opposite,
representing the (otherwise unmodeled) power flowing through the ideal connection to make the constraint
hold. The system stays exactly square: one new equation, one new pair of unknowns.

A complication this approach has to handle and the other two don't: a *loop* of zero-impedance branches
produces a linearly dependent constraint set (the last edge's constraint is implied by the others), which
would make the augmented system singular. The standard fix is to compute a minimum spanning tree over each
connected group of zero-impedance branches and constrain only the tree edges, leaving the redundant non-tree
edges as inactive constraints that can be reactivated if the tree has to be re-routed after a later
switching operation.

Compared to the other two, this is real bookkeeping — the spanning-tree maintenance in particular has no
analogue in either simpler approach — but it's also the only one of the three that keeps both original
terminals numerically distinct after the fact, which matters if anything downstream needs to report or
reason about flow through each side of the connection separately.

## Where this fits in gridoxide today

gridoxide's solver core has no node-breaker layer at all: every `Bus` reaching `network::build_ybus` is
already assumed to be a single, fully-resolved electrical node. Topology resolution is a precondition of the
`Bus`/`Line`/`Transformer` model, not something the solver participates in — so where gridoxide needs one of
the three mechanisms above, it uses approach 1, in the importer.

- **CGMES (`src/cgmes.rs`)** — `merge_closed_switches` is a direct implementation of approach 1: a
  path-compressing union-find over every closed, in-service `Breaker`/`Disconnector`/`LoadBreakSwitch`/
  `Fuse`/`Jumper`/`Cut`/`GroundDisconnector`/`DisconnectingCircuitBreaker`/`Switch` (plus `Junction`, which
  CIM defines as a permanent zero-impedance tie with no `open` state at all), collapsing each group into one
  bus and remapping every terminal index onto it.

  This exists because the converter's original assumption — that CGMES's TP profile always pre-merges a
  closed switch's two ends into a single `TopologicalNode`, making switches topologically invisible — turned
  out to be false in practice. It holds for MiniGrid/MicroGrid-BE/RealGrid, but FullGrid has a plain `Switch`
  that is `open=false` in SSH and still resolves to two distinct `TopologicalNode`s in TP. Real exporters
  don't universally do the reduction, so gridoxide does it itself.
- **PGM-JSON (`src/pgm.rs`)** — maps each PGM `node` directly to a `Bus` and doesn't implement PGM's own
  `link` component at all yet. That's the one remaining place gridoxide would need a mechanism from this
  page (most naturally approach 1 again, at import) to correctly ingest a PGM dataset that uses it.

Since the merge happens once at import, gridoxide inherits approach 1's limitations directly: switching state
can't be changed between solves without re-importing, and no per-side flow reporting is possible across a
merged switch. Neither matters for gridoxide's current scope — nothing downstream reports per-terminal flows.

## Tool reference

| Tool | Approach | Where |
|---|---|---|
| **gridoxide** | 1 — topological reduction, at CGMES import | `cgmes::merge_closed_switches` (union-find over closed switches; PGM `link` not implemented) |
| powsybl-core | 1 — topological reduction, in the bus/branch view | graph traversal of a `VoltageLevel`'s node-breaker topology terminates at open switches and fuses everything reachable through closed ones into one `CalculatedBusImpl` |
| power-grid-model | 2 — large-admittance regularization | the `Link` component: an ordinary two-terminal branch with a large fixed series admittance ("1e6 Siemens in a 10kV network", scaled to the network's base) and zero shunt; no special status in `Topology::build_topology` |
| powsybl-open-loadflow | 3 — equality-constrained augmented system | `LfZeroImpedanceNetwork` groups zero-impedance branches per component and runs Kruskal's algorithm; `AcEquationSystemCreator.createNonImpedantBranch` emits `ZERO_V`/`ZERO_PHI` equations with a `DUMMY_P`/`DUMMY_Q` variable pair per spanning-tree edge, non-tree edges inactive |

powsybl uses both approaches 1 and 3, at different layers and for different reasons: closed switches that its
bus/branch view merges away never reach the solver at all, while retained switches that survive into a
node/breaker solve get the augmented-system treatment.

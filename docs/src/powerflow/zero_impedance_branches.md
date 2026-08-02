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

gridoxide's solver core has no node-breaker layer: every `Bus` reaching `network::build_ybus` is
already a fully-resolved electrical node. Topology resolution is a precondition of the
`Bus`/`Line`/`Transformer` model, so the choice above is made once, at import.

### The rule: identity, not impedance

> **Merge only when the element has no identity in the output model. Otherwise it stays a branch.**

| Element | Identity? | Treatment |
|---|---|---|
| CGMES closed switch | No — the bus/branch view *is* the merged view | Merge (`cgmes::merge_closed_switches`) |
| PGM `link` | Yes — power-grid-model's output schema carries a `link` record with its own flows | Branch, at `pgm::LINK_Y` |
| Any branch with `\|Z\|` below a threshold | Yes — it is a line | Branch, clamped to a link's stiffness |

The deciding evidence is what the fixtures assert. All four upstream power-flow cases containing a
`link` publish that link's own current, and `vision-validation-network` publishes its full `p`/`q`/`s`
at both ends. Two state-estimation fixtures publish per-node injections either side of a link. Merging
deletes the branch those numbers describe, so it is unavailable wherever they are asserted — branch
satisfies all eight, merge satisfies two.

Note this is decided by *what the element is*, not by what is attached to it. An earlier version of
this rule asked whether the endpoints carried appliances, and `vision-validation-network`'s link 74
refutes it: the link joins two nodes with no appliances at all — apparently the safest possible merge
— and power-grid-model still reports 1.964 MW flowing through it.

The threshold row exists because a connection can be electrically zero-impedance without being
declared as one. `ill-conditioned-by-line-meshed` carries a line at 7.07e-9 p.u.; only a value-based
test catches that, which is powsybl's framing rather than power-grid-model's.

### The numbers, and why they were measured

Both constants were chosen by sweeping, not derived, because the two calculation types pull in
opposite directions:

- **Power flow** wants a link stiff. The drop across it is \\(\Delta V = I/y\\), and the fixtures
  check node voltages and the link current at 1e-5 relative.
- **State estimation** wants it soft. \\(G = H^{T}WH\\) squares the admittance, so
  power-grid-model's `1e8` becomes `1e16` in the gain matrix.

| `y` | power flow | state estimation |
|---|---|---|
| `1e8` (power-grid-model's) | pass | **singular** |
| `1e6` | pass | **singular** |
| **`2e5`** (`pgm::LINK_Y`) | **pass** | **converge** |
| `1e5` | **fail**, exactly at tolerance | converge |

The window is about one order of magnitude wide. That narrowness is the argument for treating these
as regularization parameters with measured values rather than physical constants: a network far
outside these fixtures' power scale may need them re-measured, and if no value serves both, approach
3 is the exit — it imposes \\(V_i = V_j\\) exactly, with no large number anywhere.

`topology::ZERO_IMPEDANCE_THRESHOLD` is `1e-7` p.u. on the same basis: measured across all 86
branches in the committed PGM fixtures and every CGMES one, it sits above the single pathological
line at 7.07e-9 and below the smallest legitimate branch (1.0e-6 in PGM, 2.92e-6 in CGMES), so it
disturbs nothing currently modelled.

**Detection and treatment are separate numbers.** The threshold only decides *whether* a branch is an
ideal connection; a branch it catches is clamped to `IDEAL_CONNECTION_Y`, the same admittance a
declared link gets. One number cannot do both jobs: it has to sit below every legitimate branch, yet
clamping merely to that level leaves \\(|Y|\\) as high as `1e7` — inside the range measured as
singular for state estimation, and 35x stiffer than a link. Separating them satisfies both, and says
the right thing besides: a line that short *is* an undeclared link, so it should be treated as one.

powsybl's own threshold is `1e-8`, but its default treatment is the equality-constrained formulation
rather than a clamp, so it never has to reconcile the two roles in a single value.

### Consequences of the merge, where it is used

CGMES inherits approach 1's limitations directly: switching state cannot change between solves
without re-importing, and no per-side flow is reportable across a merged switch. There is also an
empirical reason that side merges rather than stamping branches — it was tried, and the AC
Newton-Raphson solve *diverged* on FullGrid with 20-odd such branches active at once, which is
exactly the conditioning cost approach 2 carries.

## Tool reference

| Tool | Approach | Where |
|---|---|---|
| **gridoxide** | 1 and 2, by element identity | `cgmes::merge_closed_switches` merges closed CGMES switches (union-find, shared via `topology`); PGM `link` is stamped as a branch at `pgm::LINK_Y`; any branch below `topology::ZERO_IMPEDANCE_THRESHOLD` is clamped to the same admittance |
| powsybl-core | 1 — topological reduction, in the bus/branch view | graph traversal of a `VoltageLevel`'s node-breaker topology terminates at open switches and fuses everything reachable through closed ones into one `CalculatedBusImpl` |
| power-grid-model | 2 — large-admittance regularization | the `Link` component: an ordinary two-terminal branch with a large fixed series admittance ("1e6 Siemens in a 10kV network", scaled to the network's base) and zero shunt; no special status in `Topology::build_topology` |
| powsybl-open-loadflow | 3 — equality-constrained augmented system | `LfZeroImpedanceNetwork` groups zero-impedance branches per component and runs Kruskal's algorithm; `AcEquationSystemCreator.createNonImpedantBranch` emits `ZERO_V`/`ZERO_PHI` equations with a `DUMMY_P`/`DUMMY_Q` variable pair per spanning-tree edge, non-tree edges inactive |

powsybl uses both approaches 1 and 3, at different layers and for different reasons: closed switches that its
bus/branch view merges away never reach the solver at all, while retained switches that survive into a
node/breaker solve get the augmented-system treatment.

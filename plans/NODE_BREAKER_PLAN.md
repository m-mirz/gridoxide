# Node-breaker topology in gridoxide

Status: **proposal**, not implemented. Written 2026-08-09 against `818211f`; revised the same day
against `f96a660`.

> **What the revision changed.** Four commits landed between the two revisions — DC power flow's
> cross-validation and batching, multi-branch DC outages, and AC branch contingencies — and they
> touch this document in three places. Two are good news and one is a correction.
>
> - **§4.2's central argument is no longer theoretical.** The claim that a constant sparsity pattern
>   across states buys factorization reuse has now been *measured*, by a different mechanism, on the
>   AC path: `network::build_ybus_with_outages` takes a branch out while keeping its structural
>   entries at zero, and the symbolic factorization survives a whole N-1 sweep for 2.0–2.7×.
> - **§5.1 was missing a hazard, and it is the one that will bite Phase 3.** A constant sparsity
>   pattern is *not* sufficient for cache reuse, because `JacobianPattern` caches the admittance each
>   entry was analyzed against. See §5.1's new bullet — this cost real debugging in the AC
>   contingency work and was invisible to inspection.
> - **§5.4 and §5.6 got cheaper and partly stale.** `feature_comparison.md` gap 2 is now closed, and
>   the Woodbury machinery §5.4 calls for already exists.
>
> Everything else in the document was re-checked against `f96a660` and still holds: §1's table,
> §1.1's counts, `se::constraints::augment`'s generic signature, and `measurement::Target`'s
> variants are all unchanged.
>
> **Implementation status.** **Phases 0 and 1 are done; phase 2 is partly done** (§9).
>
> *Phase 0* turned `src/topology.rs` into a directory — `model`
> (`NodeIdx`/`BusIdx`/`SwitchIdx`, `Switch`, `SwitchKind`, `NodeBreakerTopology`), `bus_view`
> (`BusView`, `RetentionPolicy`, `bus_view`) and `reduction` (everything that was there, at its
> original paths) — and rerouted `cgmes::merge_closed_switches` through it. All 14 CGMES fixture
> tests and the full PGM suite are unchanged.
>
> *Phase 1* added `cgmes::cgmes_node_breaker_topology` — `ConnectivityNode`s as nodes, the nine
> switch classes plus `Junction` as edges, `BusbarSection`s as busbars — validated against the
> exporter's own `TopologicalNode` partition on all four node-breaker configurations
> (`tests/cgmes_node_breaker_test.rs`). **MiniGrid reproduces the exporter exactly**: 103
> connectivity nodes and 90 switches reduce to precisely its 13 topological nodes. Every
> disagreement elsewhere is explained — see §6.1, which the results corrected in a direction this
> document did not anticipate.
>
> *Phase 2* — **gate met, and it went further than the gate asked.** Every node-breaker
> configuration in the tree except FullGrid now solves under *every* retention policy, including the
> full node-breaker view: MiniGrid (105 buses, 90 switches), SmallGrid (1,369 buses, 1,266 switches)
> and Svedala (1,179 buses, 1,464 switches), each in the same iteration count as its own bus-branch
> solve. §1.1(d) records the consequence: §4.1's "dead on arrival at real scale" is refuted.
> FullGrid fails on both paths and is pre-existing.
>
> It added `src/switches.rs` (`SwitchTreatment::Regularize`, with a switch's position carried
> as terminal *status* so a state change moves Y-bus values without moving the sparsity pattern) and
> `cgmes::cgmes_node_breaker_to_buses_and_branches`, which derives buses from a `BusView` and reuses
> every existing equipment loop. **Its gate is met**: MiniGrid converges under
> `RetainAdjacentToBusbar` with 30 retained switches, each reporting a flow, and SmallGrid converges
> at real scale — 540 buses with 373 retained switches. Two caveats are recorded in
> `tests/cgmes_node_breaker_solve_test.rs` and §9. §1.1(d) also records a negative result worth
> reading before phase 3, and two latent bugs in shared code that node-breaker import surfaced:
> `connected_components` counting an out-of-service branch as a connection, and `classify` counting
> a de-energized placeholder as a reference bus. Both are fixed.
>
> Phases 3–7 are not started.

This document answers: *what would it take for gridoxide to model node-breaker topology as a
first-class thing, across every calculation it already supports — AC Newton-Raphson, DC Bθ, the
constant-admittance linearization, PTDF/LODF sensitivities, and weighted-least-squares state
estimation?*

The short version:

> gridoxide already *reads* node-breaker data. `cgmes::merge_closed_switches` union-finds across
> nine CGMES switch classes plus `Junction`, honoring `open` and `inService`. What it does not do is
> *keep* it: the switches are consumed at import and discarded. There is no switch element in the
> network model, so switching state is frozen at import time, no switch flow is reportable, and no
> measurement can attach to a breaker.
>
> The fix is not a bigger importer. It is a **topology layer** sitting between the importers and the
> solver: a node/switch graph, three views over it, and a per-switch **retention policy** deciding
> which view each switch survives into. Retained switches then need a formulation, and the right
> one is the **equality-constrained augmented system** — approach 3 of
> [`zero_impedance_branches.md`](../docs/src/powerflow/zero_impedance_branches.md) — because it is
> the only one of the three that simultaneously (a) keeps switch identity, (b) adds no
> large number to the matrix, which *already measurably diverged* on FullGrid at ~30 switches, and
> (c) **holds the node set and the sparsity pattern constant across switching states**.
>
> Property (c) is the load-bearing one, and it is easy to miss. `PersistentSolver`'s symbolic
> factorization reuse and `batch::BatchSolver`'s block-diagonal embedding are this project's two
> best performance assets, and both are keyed on a fixed sparsity pattern. Merging cannot preserve
> it — closing a switch changes `n`. The constrained formulation can, which is what turns "we
> support switches" into "we can screen a thousand switching states on one symbolic factorization".

---

## 1. Where things actually stand

Node-breaker support is not absent so much as **consumed and thrown away**. The table is per
subsystem, with the evidence:

| Subsystem | Node-breaker today | Evidence |
|---|---|---|
| Core network model | **None.** `Bus`/`Line`/`Transformer`/`ShuntAdm`, all `usize`-indexed. No switch, no node, no substation, no voltage level. | `src/types.rs` is 99 lines and has no switch type |
| Topology utilities | Union-find + group compaction, importer-agnostic | `src/topology.rs`: `UnionFind`, `merge_groups`, `union_all` |
| CGMES import | Reads switch state, merges it away at import | `cgmes::merge_closed_switches` — 9 switch classes + `Junction`, returns `(Vec<Bus>, remap)` |
| CGMES topology source | **Requires the TP profile.** `TopologicalNode` *is* the `Bus`. | `src/cgmes.rs:6-9`, `CgmesError::NoTopologicalNodes` |
| PGM import | `link` kept as a branch at `LINK_Y = 2e5+j2e5`, no open/close | `src/pgm.rs:1008`, `topology::IDEAL_CONNECTION_Y` |
| AC Newton-Raphson | Bus-branch only | `network::build_ybus` takes fully-resolved buses |
| DC Bθ + PTDF/LODF | Bus-branch only; `DcSensitivity` is built from `(buses, branches)` | `src/linear/btheta.rs`, `src/linear/sensitivity.rs:150` |
| Branch contingencies | **Done, bus-branch only.** AC keeps the symbolic factorization across an outage sweep; DC does N-1 and N-k with no re-solve. None of it knows what a switch is. | `batch::BatchSolver::solve_contingencies`, `DcSensitivity::{outage_flows, multi_outage_flows}` |
| State estimation | Bus-branch only. **But** it already has the augmented KKT machinery. | `se::constraints::augment` builds `[[G Cᵀ],[C 0]]` |
| Measurements | Can target a bus, a branch terminal, a source, a shunt. **Not a switch.** | `measurement::Target`, 6 variants |

Three observations from that table shape everything below.

**The SE side is much closer than the power-flow side.** `se::constraints::augment` already
assembles the exact augmented system a retained switch needs, and its signature —
`augment(triplets, rhs, n, constraint_values, constraint_rows)` — is *already generic over what the
constraint is*. It takes values and Jacobian rows, not buses. Adding a switch-equality constraint
alongside the zero-injection ones requires no change to that function at all. The power-flow side
has no equivalent and will have to grow one.

**The CGMES importer's index bookkeeping is already a known hazard.** `src/cgmes.rs` carries
several long comments about pre-merge versus post-merge indices, including one describing a real bug
where `AngleRefTopologicalNode` resolution picked "an unrelated `TopologicalNode`" once merging was
introduced. That is the classic symptom of two different index spaces sharing one `usize` type. A
proper topology layer with distinct `NodeIdx`/`BusIdx` newtypes eliminates the whole bug class, and
that is a correctness argument for this work independent of any new feature.

**Contingency analysis arrived first, and it arrived bus-branch.** `BatchSolver::solve_contingencies`
and `DcSensitivity::multi_outage_flows` now screen N-1 and N-k over *branches*. That is genuinely
useful and it is also the wrong shape for a substation: what an operator switches is a breaker, not
a line, and a bus-split contingency is not expressible as a branch outage at all. So the demand this
document serves is now concrete rather than speculative — the contingency machinery exists and is
asking for a switch model to point at. It also means §5.4's remaining work shrank considerably; see
there.

### 1.1 How much node-breaker data is actually in the tree

Counted directly from the committed conformance configurations
(`tests/data/CGMES-Test-Configurations/v3.0/`) — `cim:ConnectivityNode` in EQ, `cim:TopologicalNode`
in TP, and all nine switch classes `merge_closed_switches` already reads:

| Configuration | CN (EQ) | TN (TP) | CN/TN | Switches | Character |
|---|---:|---:|---:|---:|---|
| PowerFlow | 2 | 2 | 1.0 | 0 | bus-branch |
| PST Type1 / Type2 / Type3 | 2 | 2 | 1.0 | 0 | bus-branch |
| **FullGrid** | 43 | 25 | 1.7 | **29** | node-breaker (the only `Cut` in the tree) |
| **MiniGrid** | 101 | 13 | **7.8** | **90** | node-breaker |
| **Svedala** | 1,179 | 191 | **6.2** | **1,464** | node-breaker |
| **SmallGrid** | 1,366 | 167 | **8.2** | **1,266** | node-breaker |
| RealGrid | 6,252 | 6,252 | 1.0 | 0 | bus-branch |
| MicroGrid (BE IGM, Type1/Type2) | — | — | — | 22 / 26 | node-breaker |

Four things fall out of this table, and they set the shape of the whole plan.

**(a) Four of the configurations already in the tree are genuine node-breaker models.** SmallGrid,
Svedala, MiniGrid and FullGrid all carry substantially more connectivity nodes than topological
nodes, and hundreds to thousands of switches. There is no fixture authoring to do — the test data
for this feature is already committed and already downloaded.

**(b) The blowup factor is 6–8×, not 2×.** SmallGrid goes 167 buses → 1,366 nodes, Svedala 191 →
1,179. A full node-breaker AC system is roughly an order of magnitude larger than today's, which
makes the retention policy (§3.3) load-bearing rather than a convenience.

**(c) The largest case is *not* affected.** RealGrid — the 6,252-bus case the benchmarks lean on —
is 1:1 CN:TN with zero switches. It is already a bus-branch export. So the performance risk of this
work does not land on the headline benchmark, which is a genuinely favourable accident.

**(d) The `Regularize` approach looks dead on arrival at real scale.** §4.1 recounts that the AC
solve diverged on FullGrid with "20-odd" large-admittance switch branches active. FullGrid has 29
switches — the numbers agree. SmallGrid has 1,266 and Svedala 1,464, i.e. **45–52× past the count
already measured to diverge**. That is not a margin to be optimistic about.

> **Refuted (phase 2).** Both models converge with *every* switch retained — SmallGrid at 1,266 and
> Svedala at 1,464 — in the same iteration count as the bus-branch solve of the same model
> (`tests/cgmes_node_breaker_solve_test.rs`). The synthetic probe found no divergence either, at any
> count, arrangement or admittance sign (`switches::conditioning_probe`).
>
> Both models *did* fail at first, which is what made this look confirmed. The cause was a bug with
> nothing to do with switches: a de-energized bus is `Slack` at `V = 0`, a placeholder rather than a
> reference, and `network::classify` counted it as one. A live `PQ` bus paired with such a
> placeholder then gets an identically zero angle row, since `H_ii = −Q_i − V_i²B_ii` cancels exactly
> at zero volts. The bus-branch importer never produces that pair — it merges the dead node into a
> live one — so the bug was latent until node-breaker import stopped merging.
>
> So the scaling argument against `Regularize` does not survive contact with the data, and the
> historical FullGrid divergence remains unexplained by count, shape, sign or scale. `Constrain`
> keeps its other advantages — no large number in the matrix, and a principled answer for a switch
> flow inside a loop of closed switches (§4.3) — but §4.1 should no longer be cited as the reason
> to build it.

---

## 2. What "node-breaker support" has to mean

Borrow powsybl's taxonomy, because it is the one the CGMES data is shaped for and the one the
existing `zero_impedance_branches.md` already references. Three views over one model:

| View | Definition | Who wants it |
|---|---|---|
| **Node-breaker** | Every connectivity node and every switch, as imported. Nothing merged. | Substation-level analysis, topology-error detection, breaker-flow reporting |
| **Bus-breaker** | Retained switches survive as elements; everything else merged. | Switching studies, bus-split contingencies, SCADA-shaped state estimation |
| **Bus-branch** | Every closed switch merged away. | Everything gridoxide does today |

gridoxide today has exactly one of the three, permanently. The goal is all three, selected per
solve, with **bus-branch remaining the default so that nothing existing shifts**.

The per-feature requirement matrix — what each calculation needs beyond a bus view:

| Feature | Needs from node-breaker | Difficulty |
|---|---|---|
| AC Newton-Raphson | Augmented Jacobian, dummy `p`/`q` per retained edge, spanning forest | **High** — the main lift |
| DC Bθ | Same, but real-valued and one dummy per edge instead of two | Low, once AC is done |
| Linear impedance | Same as AC but a single linear solve, no iteration | Low |
| PTDF/LODF | Sensitivities through the augmented system; switch opening as a low-rank update | Medium |
| State estimation | Switch constraint rows (`augment` unchanged), `Target::SwitchFlow`, observability accounting | Medium |
| Batch | Constant sparsity across switching states | Falls out of the formulation |

---

## 3. The topology layer

A new module, `src/topology/` (promoting today's single `topology.rs` into a directory, keeping
`UnionFind`/`merge_groups`/`clamp_branch_impedance` where they are as `topology::reduction`).

### 3.1 Types

```rust
// topology::model
pub struct NodeIdx(usize);      // a connectivity node
pub struct BusIdx(usize);       // a resolved electrical bus
pub struct SwitchIdx(usize);

pub enum SwitchKind {
    Breaker, Disconnector, LoadBreakSwitch, GroundDisconnector,
    Jumper, Cut, Fuse, DisconnectingCircuitBreaker, Generic,
    Junction,   // no `open` state at all — CIM defines it as a permanent tie
}

pub struct Switch {
    pub kind: SwitchKind,
    pub nodes: [NodeIdx; 2],
    pub open: bool,
    pub in_service: bool,
    pub retained: bool,         // survives into the bus-breaker view
}

pub struct NodeBreakerTopology {
    pub nodes: Vec<Node>,               // + optional VoltageLevel / Substation / Bay containment
    pub switches: Vec<Switch>,
    pub busbars: Vec<NodeIdx>,          // CGMES BusbarSection
    pub equipment_terminals: Vec<(EquipmentRef, NodeIdx)>,
}
```

The newtypes are not ceremony. `src/cgmes.rs` already documents an index-space bug that cost real
debugging; making the compiler reject `bus_of[node_idx]` is the cheapest possible prevention.

### 3.2 The processor

```rust
pub struct BusView {
    pub bus_of: Vec<BusIdx>,                        // indexed by NodeIdx
    pub n_buses: usize,
    pub retained: Vec<RetainedSwitch>,              // (SwitchIdx, BusIdx, BusIdx, open)
    pub nodes_of: Vec<Vec<NodeIdx>>,                // inverse, for reporting
}

pub fn bus_view(topo: &NodeBreakerTopology, policy: RetentionPolicy) -> BusView;
```

Merge rule: union two nodes iff the switch between them is closed, in service, **and not retained**.
That is today's `merge_closed_switches` with one extra clause, and it reuses `UnionFind` +
`merge_groups` verbatim.

### 3.3 Retention policies

```rust
pub enum RetentionPolicy {
    MergeAll,                          // bus-branch — today's behaviour, the default
    RetainAll,                         // full node-breaker
    RetainKinds(EnumSet<SwitchKind>),  // e.g. breakers only, not disconnectors
    RetainAdjacentToBusbar,            // the classic bus-breaker view
    Explicit(HashSet<SwitchIdx>),      // caller-chosen, for a contingency campaign
}
```

`RetainAdjacentToBusbar` is the one that matters in practice, and §1.1's numbers say why. On
SmallGrid, `MergeAll` gives 167 buses and `RetainAll` gives 1,366 nodes plus 1,266 retained edges —
an AC unknown count of roughly `2×1366 + 2×1266 ≈ 5,264` against today's `≈334`, a **16× larger
system to answer the same question**. Nobody wants that by default. `RetainAdjacentToBusbar` buys
the bus-breaker view for a retained count in the tens; `Explicit` is what a contingency screen uses —
retain the twenty switches you intend to operate, merge the other 1,246.

The policy is therefore not a configuration nicety. It is the mechanism that keeps node-breaker
support from being a 16× performance regression, and it should exist from phase 0.

**Measured (phase 1), confirming one half and correcting the other.** `RetainAll` on SmallGrid comes
out at ~5,270 AC unknowns against `MergeAll`'s ~334 — a 15.8× blowup, against the ~5,264 estimated
just above. That estimate was sound and `MergeAll` staying the default is not negotiable.

But `RetainAdjacentToBusbar` is **not** "a retained count in the tens". That holds only on the small
models (30 on MiniGrid, 26 on FullGrid); on the real ones it is **373 on SmallGrid and 857 on
Svedala** — the latter retaining more switches than SmallGrid despite having fewer nodes, because it
is the most switch-dense model in the tree. It still buys a 3× reduction against `RetainAll`, so it
remains the right shape for a bus-breaker view, but a contingency campaign wanting a handful of
switches must reach for `Explicit` rather than assume this policy is already small. Full numbers in
`scripts/bench/README.md` §10.

**`MergeAll` must be the default everywhere.** The regression gate for phase 0 is that all 14 CGMES
fixture tests produce bit-identical results after the importer is rerouted through this layer.

---

## 4. Formulating a retained switch

`zero_impedance_branches.md` already surveys the three approaches and states gridoxide's rule
("merge only when the element has no identity in the output model"). Node-breaker support is
precisely the case where the element *does* have identity, so the rule sends us to a branch — and
the question becomes which of approaches 2 and 3.

### 4.1 Why not approach 2 (large-admittance regularization)

It is tempting because it needs no new machinery: stamp the switch as a branch at
`IDEAL_CONNECTION_Y` and everything downstream — flows, Jacobian, SE rows — works unchanged. It
should be built anyway, as `SwitchTreatment::Regularize`, because it is nearly free and delivers
switch flows immediately.

But it does not scale, and this repo already has the measurement proving it. From `src/cgmes.rs:274`
and the doc's §"Consequences": stamping CGMES switches as large-admittance branches was tried, and
**the AC Newton-Raphson solve diverged on FullGrid with 20-odd such branches active at once**.
FullGrid carries 29 switches (§1.1), so that is the whole model, not a stress case.

Against that, SmallGrid has 1,266 switches and Svedala 1,464 — 45–52× the count already measured to
diverge. Compounding it, `IDEAL_CONNECTION_Y = 2e5` was chosen from a window "about one order of
magnitude wide" that had to satisfy power flow and state estimation pulling in opposite directions,
so there is no headroom left to re-tune it upward for stiffness or downward for conditioning.

So: `Regularize` is supported and documented as suitable for a handful of retained switches — which,
paired with `RetentionPolicy::Explicit`, is a real and useful operating point. It is not the answer
for the general case, and the plan should not pretend otherwise.

### 4.2 Approach 3, and the property that makes it the answer

Per retained edge `k`, introduce a dummy variable pair `(p_k, q_k)` — the power flowing through the
ideal connection — appearing with `∓1` in the two endpoint buses' mismatch rows. Then add two
equations whose *content* depends on the switch state but whose *variables* do not:

| State | Equations |
|---|---|
| Closed | `θ_i − θ_j = 0`, `V_i − V_j = 0` |
| Open | `p_k = 0`, `q_k = 0` |

One equation pair, one variable pair, per edge, in both states. The system stays square either way.

This is powsybl-open-loadflow's `LfZeroImpedanceNetwork` / `ZERO_V`/`ZERO_PHI` / `DUMMY_P`/`DUMMY_Q`
design, and the doc already cites it.

**The constant-pattern property.** Closed-state rows touch columns `(θ_i, θ_j, V_i, V_j)`;
open-state rows touch `(p_k, q_k)`. Different patterns — but if the symbolic factorization is
computed over the *union* of the two, both states factor with the same symbolic structure, at the
cost of a few explicit zeros. The result:

> Every switching state of a given retained set shares one sparsity pattern, so one symbolic
> factorization serves the entire campaign, and every scenario is the same size — which is exactly
> the precondition `batch::BatchSolver`'s block-diagonal embedding requires.

Merging cannot offer this. Closing a switch reduces `n` by one; every scenario is a different matrix
of a different size, symbolic analysis has to rerun per scenario, and the batch embedding does not
apply at all. This is the strongest architectural argument in the document and it should drive the
phase ordering.

**This is now measured rather than argued.** `BatchSolver::solve_contingencies` applies the same
principle to *branch* outages, and it works: `network::build_ybus_with_outages` takes a branch out of
service while keeping its structural entries at zero, so the pattern is bit-for-bit the intact
network's, the symbolic factorization carries across the whole sweep, and an N-1 sweep comes in at
1.98× (case118), 2.08× (case1354pegase) and 2.72× (case9241pegase) against independent solves —
single-threaded, so that is factorization reuse alone, before any parallelism
(`scripts/bench/README.md` §9). The ~2× floor is what this repo's own "symbolic factorization is
~45% of solve time" figure predicts.

Two things follow for this document. First, the mechanism is validated on real cases at real scale,
so §4.2 no longer rests on reasoning about what *ought* to reuse. Second, the payoff for a switching
campaign should be *larger* than 2×, because a state flip changes only constraint-row values while a
branch outage changes the whole Y-bus block — but that is an expectation, not a measurement, and
Phase 3 should report the real number rather than inherit this one.

There is a precondition the original text missed, and it is not optional. See §5.1.

### 4.3 Loops of closed switches, and an honesty note

A cycle of closed retained switches yields linearly dependent constraints — the last edge's equation
is implied by the others — and the augmented system goes singular. The standard fix, which powsybl
uses (Kruskal): compute a spanning forest per connected group of closed retained edges, constrain
only tree edges, and deactivate non-tree edges by setting their `p_k = q_k = 0` instead.

The consequence deserves to be stated plainly in the user-facing docs, because it will otherwise
generate bug reports:

> **The flow through a closed switch inside a loop of closed switches is mathematically
> indeterminate in the ideal-switch model.** Only the total around the loop is determined. gridoxide
> reports the spanning-tree solution, which assigns zero to non-tree edges — a legitimate answer, not
> the only one. `SwitchTreatment::Regularize` does produce a determinate split, but only because it
> gives every switch the same finite impedance, so the split it reports reflects that assumption
> rather than the physical breakers.

### 4.4 Recommendation

| Treatment | Keeps identity | Scales | Constant pattern | Use for |
|---|---|---|---|---|
| `Merge` (default) | ✗ | ✓✓ | ✗ | Everything today; anything not being switched |
| `Regularize` | ✓ | ✗ (diverged at ~30) | ✓ | A handful of retained switches; a quick path to switch flows |
| `Constrain` | ✓ | ✓ | ✓ | Node-breaker proper, switching studies, contingency campaigns |

---

## 5. Per-feature integration

### 5.1 AC Newton-Raphson — the main lift

Touches `src/jacobian.rs`, `src/solver.rs`, `src/lib.rs`.

- A state layout that admits dummy variables. The AC side has no `StateLayout` (that is an SE type);
  it indexes `θ` then `V` implicitly. This needs to become explicit, which is worthwhile cleanup
  regardless.
- Mismatch: subtract dummy injections at both endpoints.
- Jacobian: `∓1` entries coupling bus rows to dummy columns; constraint rows per §4.2.
- `PersistentSolver`: the symbolic factorization must be computed over the union pattern (§4.2) so
  it survives a state flip. This is a small change to where the pattern is derived, and it is what
  unlocks everything in §5.6.
- **A constant sparsity pattern is not sufficient to reuse the cached Jacobian, and this is the
  trap in the whole phase.** `PersistentSolver` caches two different things with two different
  validity conditions:

  | Cache | Depends on | Survives a state flip? |
  |---|---|---|
  | Backend symbolic factorization (`scalar`/`klu`/…) | the `(row, col)` pairs alone | **Yes**, given the union pattern |
  | `JacobianPattern` (`jacobian`) | those pairs **and the admittance value at each one** | **No** |

  `JacobianPattern::Entry` carries the `y: Complex<f64>` it was analyzed against, and `fill` reads
  `e.y.re`/`e.y.im` on every iteration. So a pattern carried across two states silently evaluates
  the *previous* state's network. The field's doc comment used to claim the recipe "depends only on
  topology and bus types", which is exactly why this is easy to miss; that comment is now corrected
  in `src/solver.rs`.

  The AC contingency work hit this head-on. It was invisible to inspection and to a per-scenario
  correctness test — every contingency was right when solved alone. What caught it was an
  *interleaving* test: the same outage came back `Converged` in one position of the batch and
  `MaxIterationsReached` in another. **Phase 3 should write that test before it writes the feature.**

  The primitive already exists: `PersistentSolver::invalidate_admittances` drops the recipe and
  keeps every backend's symbolic factorization. Re-analysis is O(nonzeros) against the symbolic
  phase's ordering and elimination-tree work, so the reuse §4.2 promises survives intact — that is
  precisely the split that delivers the 2.0–2.7× measured there.

  Whether a *switch* state flip perturbs the bus-block admittances at all is a separate question
  (under `Constrain` it should not — the change lives in the constraint rows), but the union-pattern
  scheme deliberately makes both states share one structure with explicit zeros, so something in the
  cached recipe changes between states by construction. Phase 3 must establish which cache is
  invalidated by what, and test it by interleaving states rather than by solving each alone.
- Interaction with `mark_unreferenced_islands`: opening a retained switch can strand a node in a
  sourceless island. That machinery already exists and already handles it; it needs a test, not a
  change.
- Interaction with Q-limit enforcement: the outer PV→PQ loop changes bus types, not the constraint
  block. Orthogonal, but worth a combined test since both perturb the system between iterations.

### 5.2 DC Bθ

Strictly easier: real-valued, one dummy `p_k` per edge, one equation (`θ_i − θ_j = 0` closed,
`p_k = 0` open). `src/linear/btheta.rs` builds and factors one matrix once, so the augmented system
costs one larger factorization and no iteration. Phase-shifter handling is unaffected — a switch has
no tap.

### 5.3 Constant-admittance linearization

`src/linear/impedance.rs` assembles a Y-bus and does a single linear solve. The augmented block
appends to that system exactly as in §5.1 with the nonlinearity removed — the constraint rows are
already linear, and the bus rows are linear in this mode by construction. Lowest-effort of the
three power-flow modes.

### 5.4 PTDF/LODF and switching contingencies

`DcSensitivity::new(buses, branches, n_branches)` builds and factors the reduced `B` matrix. Two
things change:

1. **With retained switches present**, sensitivities must be taken through the augmented system.
   PTDF columns come out of the same factorization via the Schur complement onto the bus block; no
   new factorization, just a longer solve vector.
2. **Switch opening as a contingency** is the genuinely new capability, and classical LODF does not
   cover it — LODF is derived for a branch with finite reactance, and a closed ideal switch has none.
   The augmented formulation gives the right route: flipping edge `k` between states swaps one
   row (and, by symmetry, one column) of the augmented matrix, a rank-≤2 modification. Woodbury over
   the existing factorization therefore yields every single-switch-opening sensitivity from **one**
   factorization — the exact analogue of what LODF does for line outages, and the reason bus-split
   contingencies become tractable rather than requiring a refactorization each.

**This is materially cheaper than when it was written.** `DcSensitivity::multi_outage_flows` already
implements Woodbury over a set of removed branches — the rank-`k` update, the `(I − Ψ_LL) c = f_L`
system, and `sparse::solve_dense` for the small dense solve are all in place and validated against
outage re-solves. A switch flip is a different rank-≤2 update against the augmented matrix rather
than a branch removal against `B`, so the algebra differs, but the pattern of "assemble a small
dense correction, solve it against the cached factorization, correct the flows" is written and
tested. What remains is the augmented system itself, i.e. §5.1.

`feature_comparison.md` gap 2 no longer lists contingency analysis as open — it was closed for
branches on both the AC and DC paths. That does not diminish this item, it sharpens it: the
contingency machinery now exists and is bus-branch only, so a bus-split contingency remains
inexpressible. This section is what makes the existing capability reach the thing operators actually
switch.

### 5.5 State estimation

The least new machinery, because the augmented system is already there.

- **Constraints.** Generalize `se::constraints::Constraints` from `buses: Vec<usize>` to a list of
  constraint kinds — `ZeroInjection(bus)` and `SwitchClosed(edge)` — and extend `evaluate` to emit
  the `θ_i − θ_j` / `V_i − V_j` rows alongside the injection rows. **`augment` needs no change**; it
  already consumes `(values, rows)` generically. `se::nr` and `se::bad_data` pick this up for free.
- **`Target::SwitchFlow { switch, terminal }`.** Under `Constrain` this is the prettiest row in the
  system: the switch flow *is* a state variable, so `H` gets a single `±1` and nothing else. Under
  `Regularize` it is an ordinary branch terminal flow. Under `Merge` it is unmeasurable, and should
  return a typed error rather than being silently dropped.
- **Zero injections multiply.** A node-breaker model is full of junction nodes with nothing attached
  — every one is an exact zero injection, which is *information*, and `Constraints` already exploits
  it. Node-breaker import therefore tends to *improve* observability while enlarging the system. Both
  effects need measuring.
- **Observability has a gap to close first.** `se::observability::analyze(measurements, buses, net,
  layout)` takes no constraints, so it currently judges observability from measurements alone and
  ignores the zero-injection constraints that in fact determine part of the state. That is already
  slightly wrong today; with node-breaker it becomes badly wrong, since a merged group's internal
  structure is entirely constraint-determined. Fix: give `analyze` the constraint rows and count them
  in the structural rank.
- **Bad-data detection** is structurally unaffected — `bad_data.rs` already builds on `augment`.

### 5.6 Batch

Once §4.2's constant pattern holds, a switching campaign is a batch: same topology, same size, same
pattern, different constraint-row *values*. That is exactly the shape `batch::BatchSolver` and
`bde.rs`'s block-diagonal embedding were built for, and it needs no new solver work — only a
`SwitchingScenario` input type alongside the existing scaled-injection scenarios.

`BatchSolver::solve_contingencies` has since demonstrated the same shape for branch outages,
including the two things a switching campaign will also need: a per-scenario decision about which
cache survives (§5.1), and a fallback for scenarios that disconnect the network. That fallback is
worth copying rather than reinventing — an outage that severs the network cannot use the
structural-zero trick, because `connected_components` cannot tell a zeroed structural entry from a
live one, so those scenarios are detected up front with `network::structural_component_count` and
rebuilt properly. **Opening a retained switch has exactly this failure mode**, and §5.1's note that
`mark_unreferenced_islands` "already handles it" is true only once the *classification* sees the
open switch. Under `Constrain` the node set is constant by design, so the analogous question is
whether the classifier is reading switch state at all — Phase 3 should answer that explicitly rather
than assume it.

---

## 6. Importers

### 6.1 CGMES — and a free ground-truth test

Today `src/cgmes.rs` requires TP and uses `TopologicalNode` as `Bus`. Add a mode selector:

```rust
pub enum CgmesTopologyMode {
    Auto,            // node-breaker if EQ has ConnectivityNodes, else TP  (proposed default)
    BusBranchFromTp, // today's path, unchanged
    NodeBreakerFromEq,
}
```

`NodeBreakerFromEq` builds nodes from `ConnectivityNode`, switches from the nine classes already
enumerated in `merge_closed_switches`, busbars from `BusbarSection`, and containment from
`VoltageLevel`/`Substation`/`Bay`. Most of the class-by-class reading code already exists and moves
rather than being written.

The validation opportunity is unusually good and should be exploited early:

> For any dataset carrying **both** EQ connectivity and TP, the bus view computed from EQ+SSH must
> reproduce the TP-derived `TopologicalNode` partition exactly. Every configuration in §1.1's table
> supplies that ground truth for free, with no fixture authoring at all.

Four of them are genuine node-breaker exports and are the ones that will actually exercise the
processor:

| Config | The test | Why it is the interesting one |
|---|---|---|
| **MiniGrid** | 101 CN + 90 switches must collapse to 13 TN | Small enough to debug by hand; build it first |
| **FullGrid** | 43 CN + 29 switches → 25 TN | The known-awkward case: its plain `Switch` is `open=false` in SSH yet spans two distinct `TopologicalNode`s, which is what forced `merge_closed_switches` into existence |
| **SmallGrid** | 1,366 CN + 1,266 switches → 167 TN | First case at real scale |
| **Svedala** | 1,179 CN + 1,464 switches → 191 TN | Most switch-dense in the tree |

This test is worth building in phase 1 *before any solver work*, because it validates the topology
processor against real exporter output rather than against gridoxide's own assumptions, and because
a discrepancy here is cheap to diagnose and would be miserable to chase through an augmented
Jacobian later.

One caveat to expect rather than be surprised by: exact agreement is the goal, but exporters do not
universally reduce closed switches into one `TopologicalNode` — FullGrid demonstrably does not. So
the assertion is "the EQ+SSH partition refines to the TP partition under the same open/closed
state", and where the two genuinely disagree the test should record the discrepancy count per
configuration rather than assert zero blindly.

**Measured (phase 1).** The caveat was right to expect disagreement, and wrong about its direction —
disagreement runs *both* ways, and the unanticipated one is the more interesting.

| Config | CN | switches (conducting) | TN | buses | splits | multi-TN buses |
|---|---:|---:|---:|---:|---:|---:|
| MiniGrid | 103 | 90 (90) | 13 | 13 | 0 | 0 |
| FullGrid | 48 | 29 (28) | 25 | 22 | 1 | 2 |
| SmallGrid | 1,369 | 1,266 (1,203) | 167 | 167 | 4 | 1 |
| Svedala | 1,179 | 1,464 (988) | 191 | 228 | 29 | 0 |

*MiniGrid agrees exactly*, which is the result this section was hoping for.

A *multi-TN bus* is the anticipated direction: gridoxide merging further than the exporter, because
a closed switch was left unreduced. FullGrid has 2 and SmallGrid 1.

A *split* is the direction the plan did not anticipate: the exporter put connectivity nodes in one
topological node that gridoxide keeps **apart**. All 34 of them, across three configurations, have
one cause — the nodes are joined only by switches that are open or out of service. On FullGrid it is
a single open `GroundDisconnector`; on Svedala it is 29, unsurprising for a model with 1,464 switches
of which only 988 conduct. gridoxide is right here and the exporter is loose: an open switch does not
tie two points together.

So "refines" is not the right relation in either direction, and the test asserts the *cause* instead:
every split must be bridged solely by non-conducting switches. That would fail if the reader ever
missed an edge, which a bare discrepancy count would not.

### 6.2 PGM

power-grid-model has no node-breaker concept, so nothing is required. One optional improvement:
represent `link` as a retained switch rather than a hardcoded `LINK_Y` branch. It would gain an
open/close state and let the `Constrain` treatment remove the large admittance entirely — which, per
§4.1, is the number with no tuning headroom left. Behaviour must stay bit-identical by default, so
this is opt-in and low priority.

**A second argument for it surfaced in the DC work.** `LINK_Y` is `topology::IDEAL_CONNECTION_Y =
2e5 + j2e5`, whose *positive* imaginary part was inherited from power-grid-model's own `1e8 + j1e8`
and is a regularization choice rather than a claim that the element is capacitive. Inverting it
gives `x = −2.5e-6` — a **negative** reactance. AC never noticed, because only `|y|` and the link's
own reported Q depend on that sign, which is what `tests/link_test.rs` pins. DC noticed immediately:
a negative susceptance flips that branch's coupling and makes `B` indefinite, so
`linear::btheta::dc_branches` carries a guard matching on the exact constant. It cannot use a
blanket `x < 0` test, because case9241pegase contains 16 genuine series capacitors that such a clamp
would silently corrupt.

So the `LINK_Y` constant now has a sign problem as well as a magnitude problem, and one special-case
guard already exists to work around it. Modelling `link` as a switch under `Constrain` deletes both
the constant and the guard. That moves this from "optional improvement" to a real, if still
low-priority, cleanup with a named beneficiary.

### 6.3 Native JSON

`src/json.rs` is 8 lines. Node-breaker models need to be expressible natively for tests, for the
Python API, and so that a switching study does not require a CGMES round trip. Add optional `nodes`
and `switches` sections; absent them, the format is unchanged.

---

## 7. API surface

```rust
// Rust
let topo = cgmes::to_topology(&ds, CgmesTopologyMode::Auto)?;
let view = topo.bus_view(RetentionPolicy::RetainAdjacentToBusbar);
let mut model = PowerFlowModel::from_topology(&topo, &view, SwitchTreatment::Constrain);
model.solve()?;
model.switch_flow(switch_idx);          // (p, q) — the dummy pair, directly
model.set_switch_open(switch_idx, true);
model.solve()?;                         // same factorization, no re-import
```

```python
# Python — mirroring the existing PowerFlowModel surface in src/python.rs
model = gridoxide.PowerFlowModel.from_cgmes(paths, topology="node_breaker",
                                            retain="busbar_adjacent")
model.switches()                  # ids, kinds, endpoints, state
model.set_switch(id, open=True)
model.solve()
model.switch_flow_p()
```

Plus a `--topology` / `--retain` pair on the CLI in `src/main.rs`, and a book page under
`docs/src/powerflow/` alongside `zero_impedance_branches.md` (which should gain a forward reference
— its "Where this fits in gridoxide today" section becomes historical once this lands).

---

## 8. Risks

| Risk | Assessment | Mitigation |
|---|---|---|
| **Node count blowup** — 6–8× more nodes on real node-breaker exports (§1.1), and a 16× larger AC system on SmallGrid under `RetainAll` | Real, and this repo's benchmark culture will notice. Mitigated by the accident that RealGrid, the headline benchmark, is already bus-branch with zero switches | `MergeAll` stays the default; node-breaker is opt-in; retention policy caps the retained count; benchmark row before merging phase 1 |
| **`Regularize` divergence** | Already measured: FullGrid diverged at 29 switches. SmallGrid and Svedala are 45–52× past that | Document the ceiling explicitly; pair it with `RetentionPolicy::Explicit`; make `Constrain` the recommended mode for anything larger |
| **Augmented system is symmetric indefinite** | Not a problem — `se::constraints` already relies on this, and all five backends are general sparse LU | None needed; note it so no one reaches for Cholesky |
| **Silent behaviour change in existing paths** | The real regression risk | Phase 0's gate is bit-identical results on all 14 CGMES fixture tests plus the full PGM suite |
| **Loop indeterminacy surprises users** | Certain, if undocumented | §4.3's note goes in the book, not just here |
| **Index-space confusion** | Already bitten this codebase once | `NodeIdx`/`BusIdx` newtypes from phase 0, before any new code depends on the old convention |
| **An out-of-service branch counted as a connection** | **Was real, now fixed.** `connected_components` walked the Y-bus's *structure*, so a transformer with an open terminal — whose entries `build_ybus` stamps at zero regardless — made the bus behind it a member of somebody else's island, with an all-zero Jacobian row. Node-breaker import surfaced it because it stops merging such nodes away | `connected_components` now ignores numerically-zero off-diagonals. This also removes the phase-0 caveat that the Y-bus and branch-list partitions disagree about half-open transformers — they now agree |
| **Stale cached Jacobian across switching states** | **Has already bitten the AC contingency work.** A constant sparsity pattern makes the *symbolic factorization* reusable but not `JacobianPattern`, which caches the admittance per entry (§5.1). Silent: every state is correct when solved alone | Use `PersistentSolver::invalidate_admittances`, not `reset`, so the symbolic half survives. Test by **interleaving** states in one batch and asserting each state matches its own isolated solve — a per-scenario test cannot see this |

---

## 9. Phased plan

Each phase is independently shippable and leaves the tree green. Docs are a deliverable of the phase
that introduces the feature, not a phase of their own — this project documents as it goes.

**Phase 0 — the topology layer, no behaviour change. ✅ Done.**
`topology::model` + `topology::bus_view` + `NodeIdx`/`BusIdx` newtypes. Reroute
`cgmes::merge_closed_switches` through it. *Gate: all 14 CGMES fixture tests and the full PGM suite
bit-identical.* — **met**, and the reroute is genuinely exercised rather than merely compiled:
FullGrid (29 switches), MiniGrid (90), SmallGrid (1,266) and Svedala (1,464) all resolve their buses
through `bus_view` now.

Two notes for later phases, from doing it:

- **It did not delete code on net** (+80/−45 in `cgmes.rs`), contrary to this document's original
  claim. The class-by-class reading of the nine CIM switch classes is irreducible — it is ten
  near-identical blocks differing only in how deep `base.base.base` goes — and extracting it moved
  rather than removed it. The deletion this phase *does* buy is conceptual: `merge_closed_switches`
  no longer contains a union-find at all.
- **`BusView::representative` exists because of a subtlety worth flagging.** The merged bus clones
  the union-find *root*'s record, and `cgmes.rs`'s own comment records that this was chosen over the
  lowest member to be "provably behaviour-preserving rather than merely equivalent-looking". A bus
  view that only exposed `nodes_of` (ascending) could not reproduce it, so the root is exposed
  explicitly — and documented as arbitrary-but-deterministic, since it depends on switch order.
  That makes `NodeBreakerTopology::switches` order load-bearing, which every future importer must
  respect.

**Phase 1 — CGMES node-breaker import without TP. ✅ Done.**
`CgmesTopologyMode`, `ConnectivityNode`/`BusbarSection` reading, as
`cgmes::cgmes_node_breaker_topology`. *Gate: the EQ+SSH-derived bus view reproduces the TP partition
on MiniGrid, FullGrid, SmallGrid and Svedala (§6.1).* — met, in the sharper form §6.1 now records:
MiniGrid exactly, and every disagreement elsewhere explained by an open switch or an unreduced
export. Benchmark row added as `scripts/bench/README.md` §10 via
`examples/bench_node_breaker.rs`; reading the switch graph costs 1.5–5.4% of the XML decode that has
to happen anyway. No solver changes — it ends with a validated topology processor and nothing
consuming it yet, exactly as scoped.

Two notes for later phases:

- **Substation/voltage-level/bay containment is not read.** Nothing needs it yet: the only policy
  that consults structure is `RetainAdjacentToBusbar`, which needs busbars alone. A policy that
  retains per substation or per bay would need it, and that is when to add it.
- **Node identity is now real**, so `NodeBreakerTopology`'s `n_nodes: usize` is ready to become
  `nodes: Vec<Node>` — `CgmesNodeBreaker::node_mrids` is exactly the data such a record would hold,
  currently carried alongside rather than inside.

**Phase 2 — switch identity via `Regularize`. ✅ Done.**
Retained switches as branches; switch flows; `set_switch_open`. *Gate: MiniGrid (90 switches) under
`RetainAdjacentToBusbar` converges and reports switch flows.* — **not met.**

Done: `src/switches.rs` with `SwitchTreatment`, `regularized_branches`, `degenerate_switches`, and
`conditioning_probe`. Two design points worth carrying forward:

- **Position is carried as terminal status, not by adding and removing branches.** An open switch
  keeps its structural Y-bus entries at zero, exactly as `build_ybus_with_outages` does for an
  outaged branch, so *flipping a switch preserves the sparsity pattern*. This document attributes
  that property to `Constrain` alone (§4.2); `Regularize` has it too. What `Regularize` still lacks
  is the scaling, not the pattern stability. The corollary is that
  `PersistentSolver::invalidate_admittances` — not `reset` — is the right call after a position
  change, for the reason §5.1 now spells out.
- **A regularized switch is purely inductive**, unlike `IDEAL_CONNECTION_Y`, so DC needs no guard
  for it. See §6.2.

The importer wiring landed too: `build_node_breaker_skeleton` derives buses from a `BusView`, with
`u_rated` read through `ConnectivityNode.ConnectivityNodeContainer` → `VoltageLevel.BaseVoltage`
(via `Bay` where the container is one). The equipment conversion was factored into
`convert_equipment` and is shared verbatim with the bus-branch path — nothing below the skeleton
cares how a bus came to exist, since it all resolves through `terms.bus(...)`.

**Gate met:** MiniGrid converges under `RetainAdjacentToBusbar` (45 buses, 30 retained switches, all
reporting flows) **and under `RetainAll`** — the full node-breaker view, 105 buses and all 90
switches retained, in two iterations. SmallGrid converges at 540 buses with 373 retained.

**On §4.1's prediction.** SmallGrid under `RetainAll` (1,266 retained) *does* fail, which is the
shape §4.1 predicts. But it cannot yet be attributed to conditioning, because Svedala fails at
**zero** retained switches — so a bus-derivation problem of the same family is a live alternative
explanation, and the two cannot be separated until Svedala is fixed. Combined with §1.1(d)'s
synthetic result (no divergence at any count, shape or sign), the honest position is that §4.1
remains unproven in either direction.

Two caveats, both in the open rather than in a footnote:

- **FullGrid does not converge on *either* path.** The ordinary `TopologicalNode` importer returns
  `MaxIterationsReached` on it too. Pre-existing and unrelated — there is no `cgmes_fullgrid_test`
  in the tree for the same reason.
- **Svedala — resolved.** It failed on the node-breaker path at *zero* retained switches. Three
  hypotheses were tested and eliminated before the real cause was found:
  1. *Multi-slack components.* The TN path has **more** of them (7 vs 5) and converges, and all of
     them on both paths are all-placeholder (`V = 0`, zero injection) components with nothing to
     solve. Not the cause.
  2. *Numerically-dead rows.* 34 split-off connectivity nodes have no equipment and no conducting
     connection, but each forms its own component and is correctly pinned `NoReferenceBus`. Zero
     dead rows survive inside any singular island.
  3. *Out-of-service branches counting as connections.* Real, and fixed — see below — but not the
     cause of this.

  The 108-bus island that remained under suspicion turned out to be full rank (182 of 182) in
  isolation — collateral, exactly as `IslandStatus::Singular`'s own doc warns ("*every*
  still-unconverged component is marked singular whether or not it was the actual cause"). Rank
  analysis of every component, with sourceless ones pinned first, found the real culprit: a two-bus
  component holding one live `PQ` bus and one de-energized `V = 0` placeholder, whose angle column
  was identically zero. `classify` now disregards a zero-voltage slack as a reference, which fixes
  Svedala and, with it, SmallGrid under `RetainAll` — see §1.1(d).

**Phase 3 — `Constrain` for AC Newton-Raphson.**
Explicit AC state layout, dummy variables, constraint rows, Kruskal spanning forest, union-pattern
symbolic factorization. The heavy lift, and the phase everything after depends on. *Gate: SmallGrid
and Svedala converge under `RetainAll` — the two cases phase 2 is expected to fail — and every
existing power-flow test is unchanged.* **Write the interleaving test first** (§5.1): flip a
retained switch back and forth within one batch and assert each state matches its own isolated
solve. That is the test that catches the cache hazard, and it costs nothing to write before the
feature exists.

**Phase 4 — DC, linear impedance, and switching contingencies.**
§5.2, §5.3, then §5.4's Woodbury bus-split sensitivities. Smaller than originally scoped: the
Woodbury machinery landed with `DcSensitivity::multi_outage_flows`, so what remains is applying it
to a switch-state flip against the augmented matrix rather than to a branch removal against `B`.
`feature_comparison.md`'s contingency item is already closed for branches; this is what extends it
to the bus splits operators actually perform.

**Phase 5 — state estimation.**
Generalized `Constraints`, `Target::SwitchFlow`, and the observability constraint-accounting fix
(§5.5) — which is worth doing on its own merits even if node-breaker were dropped.

**Phase 6 — batch switching campaigns.**
`SwitchingScenario` over the existing `BatchSolver`. Small, given phase 3.

**Phase 7 — API, CLI, Python, benchmarks.**
Ongoing through 2–6 rather than deferred; called out separately only for the cross-cutting bits
(`--topology` flag, Python surface, a benchmark section in `scripts/bench/README.md`).

---

## 10. Deliberately out of scope

- **Topology error detection in state estimation** — inferring that a reported breaker status is
  *wrong* from measurement residuals. This is the classic node-breaker SE payoff, and generalized
  state estimation solves it by treating switch status as estimated rather than given. It is out of
  scope here, but note that §4.2's formulation is exactly its precondition: the residual on a
  `ZERO_V` constraint is the signal a status error produces. Phase 5 should not close off the door.
- **Automatic switching / remedial action optimization.** A solver capability, not a modeling one.
- **Bay-level protection modeling** — CGMES `ProtectedSwitch` semantics beyond the `open` flag.
- **Three-phase node-breaker.** The 3-phase path (`build_ybus_3ph`, `SeNetwork::from_3ph`) indexes
  `3k + p`; the topology layer is phase-agnostic and a per-phase switch is a coherent extension, but
  no fixture in the tree needs it.

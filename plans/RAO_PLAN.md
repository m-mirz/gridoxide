# Remedial action optimization in gridoxide

Status: **in progress.** Written 2026-08-17 against `0549f7e`; phases 1 and 2 landed 2026-08-18.

> **Implementation status.** Phase 1 (UCTE importer) is done and its pypowsybl gate is met — see
> §9. Phase 2 is half done: `ratings::BranchLimits` and `types::TapChanger` exist and the CGMES
> `OperationalLimit` importer converts every declared limit in every conformity fixture, but CGMES
> tap *tables* are still discarded at import (`cgmes.rs` evaluates the current step and drops the
> rest). Phases 3-12 are unstarted.

## Context

gridoxide can now answer *what is the state* (`solver::newton_raphson`), *what changes if something
moves* (`ac_sensitivity`, `DcSensitivity`), *what breaks if something trips*
(`BatchSolver::solve_contingencies`, `multi_outage_flows`), and *what should the injections be*
(`opf::dc`, `opf::ac`). It cannot answer the question an operator actually asks at 09:00:

> Tomorrow's forecast overloads a line after outage C. I am allowed to open these three couplers,
> move those two phase shifters, and redispatch within this range. **Which of them should I do, and
> when — before the outage or after it?**

That is remedial action optimization. It is the question the CORE, Nordic and SWE capacity
calculation processes run daily across Europe, and it is the last of the classical operational-
planning problems gridoxide does not touch.

Two plans in this repo currently disclaim it. `plans/OPF_PLAN.md` §10 puts security-constrained OPF
out of scope and §9 risk 6 names "scope creep toward SCOPF" as a hazard; `plans/NODE_BREAKER_PLAN.md`
§10 lists "automatic switching / remedial action optimization — a solver capability, not a modeling
one". **This document supersedes both disclaimers.** They were right when written: the modeling was
not there. It is now. Node-breaker landed with retained switches as first-class branches, AC and DC
contingency analysis both reuse their factorizations, and the OPF layer shipped a solver-independent
`LinearProgram` boundary with two backends. What is left is genuinely the solver capability, plus
three data gaps (§5).

The intended outcome is a `gridoxide rao` that reads a network and a CRAC, and returns the remedial
actions to take per state, the resulting margins, and the cost — cross-validated against a mature
implementation rather than against itself.

## 1. What this is

Given a base-case network, a set of contingencies \\(\mathcal{C}\\), a set of monitored elements
with limits, and a **catalogue** of actions the operator is permitted to take:

\\[ \max_{a \in \mathcal{A},\ u \in \mathcal{U}(a)} \quad \min_{c \in \mathcal{C},\ e \in \mathcal{E}(c)} \ \text{margin}(e, c, a, u) \\]

with \\(a\\) the discrete choices (open this switch, close that coupler) and \\(u\\) the continuous
ones (this PST to 4.2°, this redispatch to 300 MW). Three things separate it from everything already
in the tree:

1. **The action set is given, not derived.** OPF optimizes over whatever the physics admits. A RAO
   optimizes over what a TSO has agreed is permissible — a curated list with owners, costs, and
   rules about when each may be used.
2. **The decisions are mostly discrete.** Switch positions and tap positions are integers. A
   continuous relaxation gives a bound, not an answer.
3. **There is a time axis.** Preventive actions are taken before the outage and must be paid for
   whether or not it happens; curative actions are taken after, and only for the contingency that
   occurred. The same network element can be a decision variable twice, at different instants, with
   different limits applying.

The third point is what makes this more than "SCOPF with integers", and it is the part a naive
formulation gets wrong.

## 2. Decisions taken

Confirmed before writing this:

- **CRAC data model plus search tree, DC evaluation first.** The OpenRAO shape, for the reason §3
  gives: it is the only open-source structure that expresses the time axis, and the industry data
  formats are built around it. DC first for the same reason DC-OPF came first — the whole pipeline
  gets exercised at a fraction of the risk.
- **Build a UCTE reader and an IIDM reader.** Not a conversion script. §6.2 argues these pay for
  themselves independently of RAO, and they are the only route to the external validation gate.
  IIDM is version-tolerant across the `1_x` namespaces, bus-branch **and** node-breaker.
- **MILP through HiGHS first, an in-house branch-and-bound second.** `LinearProgram` gains an
  integrality vector; `opf-highs` gains `Highs_changeColsIntegrality`. Then a bespoke B&B over the
  existing `IpmSolver` brings the discrete path back under CI, with HiGHS as its cross-check. This
  is the third instance of a pattern the crate has already run twice (§8.4).
- **Three input paths: a native companion document, an OpenRAO JSON CRAC reader, and CGMES
  `OperationalLimit` import.** The CGMES CSA/NC profiles are out of scope — cimstructs has none of
  those classes and adding them is a cimoxide change, not a gridoxide one (§11).
- **Angle and voltage CNECs are out of scope for the optimizer.** OpenRAO does not optimize them
  either; its `monitoring/` module checks them afterwards. Copying that boundary is free.

## 3. The two references, and what each actually is

Both are checked out under `references/`. They are far less alike than the shared phrase "topology
optimization" suggests, and the difference decides the architecture.

| | **powsybl-open-rao** | **ToOp** |
|---|---|---|
| Origin | RTE / Linux Foundation Energy, `7.4.0-SNAPSHOT` at `46d286f4` | Elia Group + 50Hertz, open-sourced May 2026 |
| Licence | MPL-2.0 | MPL-2.0 |
| Stack | Java, OR-Tools (CBC/SCIP/XPRESS) via JNI | Python 3.11, JAX/Equinox, GPU-native, `qdax` |
| Size | 86k Java LOC (main); algorithm proper ≈ 26k | 93k Python LOC (src); no LP/MILP solver anywhere |
| Action set | **Curated** by the operator in a CRAC file | **Generated**: all \\(2^{d-1}\\) busbar splits per substation, filtered |
| Actions | Topological + PST + HVDC + redispatch, **with cost** | Splits, reassignments, line switching, PST taps. No cost, no redispatch |
| Method | Greedy search tree over discrete actions; a full **MILP re-solved at every leaf** for the continuous ones | **MAP-Elites** quality-diversity GA over a weighted-sum fitness, plus a brute-force mode |
| Time axis | Preventive / outage / auto / curative instants, PATL vs TATL | None. One snapshot |
| Output | One optimized result per state | A Pareto archive of diverse topologies |
| Validation | 109 Cucumber `.feature` files with expected margins, costs, activated actions | Two-stage DC screen → AC re-validation with hard rejection thresholds |

**OpenRAO supplies the structure; ToOp supplies the kernel ideas.** Take the CRAC model, the
perimeter decomposition, the search tree and the MILP fillers from the first. Take from the second:
the two-stage DC-screen-then-AC-validate discipline, the idea that *diversity* of answers is worth
more to an operator than a single optimum, and — most concretely — the observation that a busbar
split is a rank-one update to the PTDF rather than a re-solve.

That last one is where gridoxide arrives unusually well-equipped, and it is worth stating plainly.
ToOp calls it BSDF, cites a PowerTech paper for it, and implements it as 750 lines of JAX. In
gridoxide a retained switch is already an ordinary branch with its own flat index, so:

```rust
let branch = net.branch_of_switch(sw).unwrap();
let split_factors = sens.lodf_column(branch).unwrap();  // this *is* the BSDF
```

`docs/src/cgmes/node_breaker.md:86` says so in as many words. Neither reference has this for free;
gridoxide does, because the node-breaker work chose to keep switches as branches rather than merging
them away.

## 4. What already exists, and what it buys

More than for any previous plan in this repo, because a RAO is mostly an orchestration layer over
things that already work.

| Need | Already there |
|---|---|
| Post-contingency DC flows, N-1 and N-k | `DcSensitivity::outage_flows`, `multi_outage_flows` (Woodbury, one factorization) |
| Post-contingency AC flows | `batch::BatchSolver::solve_contingencies` — symbolic factorization held across a whole sweep |
| Contingencies that sever the network | `is_breaking_set`, `structural_component_count`; an islanded contingency is a *result*, not an error |
| Sensitivity of a flow to a PST angle (the PSDF the MILP needs) | `ac_sensitivity::FunctionRow::d_phase`, `d_ratio` — adjoint, one solve per monitored quantity |
| Sensitivity of a flow to an injection (redispatch) | `DcSensitivity::ptdf_row` / `ptdf_column`; `FunctionRow::d_active` |
| **Bus-split distribution factors** | `lodf_column` at `branch_of_switch(sw)` — see §3 |
| Applying a topological action | `NodeBreakerNetwork::set_switch_open` — flips values, not the sparsity pattern |
| Switch opening as a contingency | an ordinary `Scenario::branch_outages` entry |
| An LP/QP with duals, two backends | `opf::LinearProgram` / `Solution` / `Solver`, `opf::ipm`, `opf::highs` |
| Parallel evaluation of many scenarios | `BatchSolver` — rayon `scope` + atomic work counter, deterministic output order |
| A companion-document convention for data the network format cannot hold | `src/opf/model.rs` and `<network>.opf.json` |
| Branch ratings, PATL/TATL | **nothing** (§5.1) |
| Tap positions as discrete state | **nothing** (§5.2) |
| Integer variables | **nothing** (§5.3) |
| A network format any of this can be validated against | **nothing** (§6.2) |

The honest summary: the *evaluation kernel* of a RAO is finished and benchmarked. The data model,
the optimizer's discrete half, and the ability to read the files the reference implementations test
against are not.

## 5. Three gaps that must close first

These are not phases of the RAO. They are preconditions, and each is useful on its own.

### 5.1 Nothing in the network model carries a rating

`Line` has `r`, `x`, `b_shunt`, `g_shunt`. `Transformer` has admittances and a tap. Neither has a
limit. The only rating anywhere is `opf::model::BranchLimit { id, rate_a, unlimited }` — a single
MVA figure per branch, arriving through the companion OPF document.

A CNEC needs strictly more than that: a **permanent** limit (PATL) and one or more **temporary**
limits (TATL) with acceptable durations, potentially different per side and per direction. The
whole preventive/curative split turns on the difference — preventive optimization is judged against
the TATL, curative against the PATL, and the gap between them is precisely the room a curative
action is sized to cover.

Worse, `src/cgmes.rs` imports **no** limit classes at all. cimstructs generates
`OperationalLimit`, `OperationalLimitSet`, `OperationalLimitType` (with `acceptable_duration`,
`is_infinite_duration`, `direction`, `kind`), `CurrentLimit`, `ActivePowerLimit`,
`ApparentPowerLimit` and `VoltageLimit` — 527 generated classes, and the importer reads none of
these seven. So a CGMES network reaches gridoxide today with no ratings whatsoever, which means
"is this network secure?" is currently unanswerable from CGMES input alone. That is a gap worth
closing regardless of this plan.

**Resolution.** A `network::BranchLimits { patl: Option<f64>, tatl: Vec<(Duration, f64)>, side }`
carried alongside the branch arrays, populated by three importers: CGMES `OperationalLimit*`, UCTE
`##L`/`##T` current ratings, and IIDM `currentLimits` / `operationalLimitsGroup`. MATPOWER's
`rate = 0 means unlimited` convention, already honoured by `BranchLimit::rate_pu`, carries over.

### 5.2 Tap positions are discarded at import

`Transformer::tap` is one `Complex<f64>` — `k·e^{jα}`. `network::transformer_tap(...)` builds it
from `(tap_pos, tap_min, tap_max, tap_nom, tap_size, clock)` and then **throws the arguments away**.
There is no `set_tap`, no step count, no tap→angle map.

A `PstRangeAction` cannot be expressed without those. Its setpoint is an angle, its decision variable
is a tap, the map between them is nonlinear and not necessarily monotone, and OpenRAO's
`DiscretePstTapFiller` linearizes it piecewise around the current position — recalibrating
\\(k^{+} = f(t{+}1) - f(t)\\) and \\(k^{-} = f(t) - f(t{-}1)\\) at each MIP iteration. All of that
needs the map retained.

**Resolution.** A `TapChanger { position: i32, low: i32, high: i32, neutral: i32, steps: Vec<Complex<f64>> }`
retained on the transformer, with `set_tap_position` mutating `Transformer::tap` from it. The CGMES
importer already computes every step for all four `PhaseTapChanger` flavours plus
`RatioTapChangerTable` — it currently evaluates one and discards the rest. UCTE `##R` gives the
same thing in five fields (`δu`, `θ`, `n`, `n'`, `SYMM|ASYM`); IIDM gives it as explicit `<step>`
elements, of which the fixtures contain 2,181.

### 5.3 The LP boundary cannot express an integer

`LinearProgram` has `col_lower`, `col_upper`, `col_cost`, `hessian`, `rows`, `row_lower`,
`row_upper` — and no integrality. `HighsSolver::solve` calls `Highs_passLp` + `Highs_passHessian`
and nothing else, so the backend is capable in principle (HiGHS has a full branch-and-cut MIP
solver) while the boundary type cannot say so. `docs/src/opf/index.md` already notes this:
*"HiGHS solves MIPs, so the backend would carry it, but nothing above the solver boundary models
it."*

**Resolution.** `pub col_integral: Vec<bool>` on `LinearProgram`, empty meaning all-continuous so
every existing caller is untouched; `validate()` rejects integrality on a problem with a Hessian
(HiGHS does not take MIQPs and neither will the in-house B&B). `HighsSolver` calls
`Highs_changeColsIntegrality` when the vector is non-empty. `IpmSolver` returns
`OpfError::IntegralityUnsupported` rather than silently relaxing — a silent relaxation would be
the worst possible failure mode here.

## 6. The data layer

### 6.1 The CRAC

A new `src/rao/crac.rs`, modeled on OpenRAO's domain but flattened for Rust. The type list, with the
one piece of hard-won advice from reading the reference: **`Instant` must be data, not an enum, from
day one.** OpenRAO supports several `CURATIVE` instants (`curative1`, `curative2`, …) and the
"latest state at or before this instant at which this device is controllable" chain is load-bearing
throughout the MILP. Retrofitting it is described in the reference's own porting notes as painful.

```rust
pub enum InstantKind { Preventive, Outage, Auto, Curative }
pub struct Instant { pub id: String, pub kind: InstantKind, pub order: usize }
pub struct Contingency { pub id: String, pub elements: Vec<usize> }   // flat branch indices
pub struct State { pub instant: usize, pub contingency: Option<usize> }

pub struct FlowCnec {
    pub id: String, pub branch: usize, pub state: State,
    pub thresholds: Vec<Threshold>,        // per side, MW or A, upper and/or lower
    pub reliability_margin: f64,
    pub optimized: bool,                   // enters the min-margin objective
    pub monitored: bool,                   // must not get worse — an MNEC
    pub operator: Option<String>,
}

pub enum ElementaryAction {
    Switch { switch: SwitchIdx, open: bool },
    TerminalsConnection { branch: usize, connected: bool },
    PstTapPosition { transformer: usize, tap: i32 },
    InjectionSetpoint { bus: usize, p: f64 },
    SwitchPair { open: SwitchIdx, close: SwitchIdx },
}
pub struct NetworkAction { pub id: String, pub elementary: Vec<ElementaryAction>, /* … */ }

pub enum RangeAction {
    Pst { transformer: usize, ranges: Vec<TapRange>, initial_tap: i32 },
    Injection { keys: Vec<(usize, f64)>, ranges: Vec<StandardRange> },
    Hvdc { line: usize, ranges: Vec<StandardRange> },
}
pub enum RangeType { Absolute, RelativeToPreviousInstant, RelativeToInitialNetwork }

pub enum UsageRule {
    OnInstant { instant: usize },
    OnContingencyState { state: State },
    OnConstraint { instant: usize, cnec: usize },
    OnFlowConstraintInCountry { instant: usize, country: String, contingency: Option<usize> },
}
```

Note what is *not* here: `UsageMethod`. The current OpenRAO has removed that enum — availability is
now a predicate evaluated at RAO time (`RaoUtil::canRemedialActionBeUsed`), and the
available-versus-forced distinction is implicit in the instant kind: everything at `Auto` is forced
and simulated, everything else is offered to the optimizer. Copy the current design, not the one in
the older literature.

Three ways in, all agreed:

- **`<network>.rao.json`**, a companion document beside the network file keyed by component ids —
  the pattern `src/opf/model.rs` established, for the reason its module docs give: inventing fields
  inside someone else's format is how a converter becomes a liability.
- **`src/rao/crac_json.rs`**, reading OpenRAO's own documented JSON CRAC schema. This is what turns
  724 CRAC files in the reference's test tree into usable fixtures.
- **CGMES `OperationalLimit*`** for the limits half (§5.1).

### 6.2 Two importers, and why they earn their place

The validation gate for this work is OpenRAO's 109 Cucumber `.feature` files, which state expected
margins, costs, and activated remedial actions per scenario in plain text:

```gherkin
Then the total cost for timestamp "2025-11-04 04:30" is 625010.0
Then the remedial action "redispatchingAction" is used at timestamp "2025-11-04 04:30" in preventive
```

That is the RAO equivalent of pglib's `BASELINE.md`, and it is the only external gate available.
Reaching it means reading the networks those scenarios run against: 176 `.uct` and 51 `.xiidm`.

**UCTE-DEF is small.** Across all 176 fixtures only five record types appear — `##C` (date),
`##N`/`##Z<country>` (nodes by zone), `##L` (lines), `##T` (transformers), `##R` (tap changer
regulation). No `##TT` tap tables, no `##E`, no `##DD`. The largest fixture is 74 lines; all 176
together are 6,402. Fixed-column parsing, a few hundred lines of Rust. And it arrives carrying
exactly the two things §5.1 and §5.2 say are missing:

```
##L
BBE1AA1  BBE2AA1  1 0 0.0000 10.000 0.000000   5000          <- r, x, b, and the current rating
##R
BBE2AA1  BBE3AA1  1        -0.68 90.00 16  0        SYMM      <- δu, θ, n steps, current step, kind
```

`docs/src/reference/resources.md` already lists UCTE-DEF as "potential fallback path if CGMES export
isn't available" — this promotes it from a note to an importer.

**IIDM is the larger lift, and the version spread is the reason.** The 51 fixtures span eleven
schema versions, `1_0` through `1_16`, and the model moved underneath them: limits are spelled
`currentLimits` in 1,446 places and `operationalLimitsGroup` in 273. The parser therefore accepts
any `.../schema/iidm/1_*` namespace, handles both spellings, and **skips unknown elements rather
than failing** — the alternative is a parser that breaks on the next powsybl release.

Element coverage, from what the fixtures actually contain: `substation`, `voltageLevel`,
`busBreakerTopology`/`bus`, `nodeBreakerTopology`/`busbarSection`/`switch`, `generator`, `load`,
`shunt`, `line`, `twoWindingsTransformer` with `ratioTapChanger`/`phaseTapChanger`/`step`,
`danglingLine`, `tieLine`, `hvdcLine`/`vscConverterStation`, and both limit spellings.

The reassuring part: gridoxide already *models* every one of those. Node-breaker topology with nine
switch classes, all four phase-tap-changer flavours, a real DC-side network with VSC converters,
linear and nonlinear shunts — the CGMES importer built all of it. IIDM is a parsing-and-mapping job
onto existing types, not new physics. `quick-xml` 0.37.5 is already in `Cargo.lock` via cimdecoder,
so it is promoted from a transitive to a direct dependency, exactly as rayon was for `src/batch.rs`.

Both importers go behind their own features (`ucte`, `iidm`) and both are **pure Rust with no
system library**, so unlike `klu`/`pardiso`/`opf-highs` they are built and tested in CI.

## 7. The algorithm

### 7.1 Perimeters

The CRAC is partitioned once, into a preventive perimeter and one scenario per contingency. This is
OpenRAO's `StateTree` and it is not an optimization — it is the decomposition that makes the problem
tractable at all, by refusing to put every contingency's curative actions into one MILP.

- **Preventive perimeter**: the base case judged against PATL, plus every outage state judged
  against TATL. Solved first; its actions are then fixed.
- **Auto perimeter** (per contingency): forced automatons, *simulated* not optimized — a speed-
  ordered loop that shifts each range action toward relieving the worst-violated CNEC until it is
  clear or the range is exhausted. Phase 5, not phase 1.
- **Curative perimeters** (per contingency, per curative instant): judged against PATL, solved
  greedily in chronological order with the previous instant's result fixed.

One subtlety worth building in from the start because retrofitting it is ugly: a curative CNEC for
which *no* curative action exists must be pulled forward into the preventive perimeter and judged
there against its PATL rather than its TATL. Otherwise the preventive optimizer happily parks the
flow between PATL and TATL, and nothing downstream can rescue it.

### 7.2 The evaluation kernel

For each candidate, a perimeter must be evaluated: every CNEC's flow under every contingency, and
the sensitivity of each to each range action. This is the hot loop — OpenRAO's own porting notes
say the search tree is not where the time goes, the per-leaf re-solve is.

It is also where gridoxide should be fastest, and the plan should say so with numbers rather than
adjectives:

| Quantity | Route | Measured cost |
|---|---|---|
| Post-outage flows, N-1 | `DcSensitivity::outage_flows` | one triangular solve, no refactorization |
| Post-outage flows, N-k | `multi_outage_flows` | 0.40 ms/pair on `case9241pegase` vs 13.4 ms re-solve |
| Bus-split flows | `lodf_column(branch_of_switch(sw))` | same — a switch is a branch |
| AC contingency sweep | `BatchSolver::solve_contingencies` | 2.0x (`case118`) to 2.7x (`case9241pegase`) single-threaded |
| PST sensitivity | `AcSensitivity::function_row` (adjoint) | one solve per monitored quantity, not per PST |

The DC path is exact, because DC is linear — so a DC-evaluated search tree has no linearization
error in its *evaluation*, only in its LP. Phase 1 is therefore DC throughout, and the AC path
enters as ToOp's second stage: re-validate the survivors, with explicit rejection thresholds, rather
than optimizing in AC.

### 7.3 The linear problem

Built with `opf::LinearProgram`, solved through the existing `Solver` boundary. The formulation
follows OpenRAO's fillers, whose LaTeX specifications in
`references/powsybl-open-rao/docs/algorithms/castor/linear-problem/` are a better spec than the Java.

**Core (`rao::filler::core`)** — variables \\(F(c)\\) per CNEC-side, \\(A(r,s)\\) setpoint per range
action and state, \\(\Delta^{+}(r,s), \Delta^{-}(r,s) \ge 0\\). The keystone row is the
linearization:

\\[ F(c) \;=\; f_n(c) \;+\; \sum_{r \in \mathcal{RA}(s)} \sigma_n(r,c,s)\,\bigl[A(r,s) - \alpha_n(r,s)\bigr] \\]

with \\(\sigma\\) coming from `ptdf_row` (injection) or `d_phase` (PST). Terms below a sensitivity
threshold are **dropped entirely**, which is what keeps the MILP sparse. Plus the variation link
\\(A(r,s) = A(r,s') + \Delta^{+} - \Delta^{-}\\) chaining each state to its predecessor, and — when
injection range actions are present — a global balance row
\\(\sum_r (\Delta^{+} - \Delta^{-})\sum_d d = 0\\).

**Objective (`rao::filler::margin`)** — one scalar \\(MM\\), with \\(MM \le f^{+}(c) - F(c)\\) and
\\(MM \le F(c) - f^{-}(c)\\) per optimized CNEC, minimizing \\(-MM\\). Plus a small penalty
\\(\sum c^{pen}_r(\Delta^{+} + \Delta^{-})\\) so that among equally good answers the one that moves
least wins.

**Discrete PSTs (`rao::filler::pst`)** — the first place integrality bites. Integer tap variations
\\(\Delta t^{\pm}\\), binaries \\(\delta^{\pm}\\) with \\(\delta^{+} + \delta^{-} \le 1\\), and the
two-piece tap→angle linearization recalibrated each iteration as described in §5.2.

**Usage limits (`rao::filler::limits`)** — a binary \\(\delta(r,s)\\) per range action with the
big-M activation row, then \\(\sum_r \delta(r,s) \le \text{maxRa}\\) and its per-TSO variants.

The outer loop is iterate-and-relinearize: solve, round taps to the better neighbour, re-run
sensitivities at the new point, **rebuild the MILP from scratch**, keep the result only if the true
cost improved. Do not over-engineer incremental model updates — the reference rebuilds every
iteration and its own notes say so. When the linearization oscillates, shrink each range action's box
by \\((2/3)^n\\) around the previous iterate.

### 7.4 The search tree

Greedy, best-per-depth, over combinations of network actions:

```
evaluate(root); optimize_range_actions(root)
for depth in 0..max_depth:
    candidates = bloom(best_leaf)                 // filtered, see below
    for each candidate (in parallel over rayon):
        leaf = apply(candidate)                   // set_switch_open, set_tap_position
        evaluate(leaf)                            // §7.2
        optimize_range_actions(leaf)              // §7.3 — the full MILP, every leaf
    if no leaf improved by more than the impact thresholds: stop
```

Two details that are easy to miss and expensive to omit. **Every leaf re-runs the full range-action
MILP** — that is the whole point, because a topological action changes the sensitivities the PSTs
are optimized against, and optimizing the two separately gives a worse answer than optimizing them
together. And **candidate ordering must be deterministic** despite parallel evaluation; OpenRAO
sorts by (detected-during-RAO, user-predefined, size, CRC32 of the concatenated ids) precisely so
that a parallel run reproduces a serial one. `BatchSolver` already establishes the pattern of
sorting results back into input order for exactly this reason.

Candidate filtering, in rough order of value: drop already-applied actions; drop incompatible
combinations (two actions on the same element); enforce the usage limits; drop actions with no
network impact; optionally drop actions electrically far from the most limiting CNEC.

### 7.5 Deliberately not ported

Named here so the phase table stays honest: the MARMOT multi-timestamp RAO with generator ramping
(3.2k Java LOC), the CNE XML exporters (6.7k), loop-flow decomposition, relative margins with zonal
PTDF sums, counter-trade range actions (not modeled in OpenRAO's LP either), and the FastRao CNEC-
subsetting heuristic. Each is a well-defined later addition; none is needed to answer the question in
the Context section.

## 8. Validation

Four gates, in increasing order of strength.

### 8.1 Analytic cases

Small networks where the right action is derivable by hand: a two-branch parallel path where opening
one is obviously wrong; a three-bus case with one PST where the optimal tap is computable from the
PSDF directly; a case where the preventive answer and the curative answer differ *because* TATL
exceeds PATL, which is the one behaviour no simpler tool exhibits.

### 8.2 The evaluation kernel against itself

Every DC screening result must agree with a full re-solve. This is already the shape of
`tests/ac_contingency_test.rs`, which oracles every single-branch contingency against a from-scratch
rebuild. Extend it to bus-split contingencies: `lodf_column` at a switch branch against an explicit
`set_switch_open` + re-solve. **This gate is independent of the RAO entirely** and can land with
phase 1.

### 8.3 OpenRAO's Cucumber expectations

The external gate. 109 `.feature` files stating expected margins, costs and activated actions, run
against the `.uct`/`.xiidm` networks the new importers read and the CRACs the JSON reader reads.

Two honest caveats, stated now rather than discovered later. First, **the search tree is a heuristic**
— agreement on the objective value is meaningful, but a different *set* of actions achieving the same
margin is not a bug, and the test assertions must be written to that standard. Second, a mismatch may
be an importer bug rather than an optimizer bug, which is why §8.2 exists as the independent check on
the evaluation half.

### 8.4 Two independent MILP solvers

The pattern the crate has now run twice: `ipm` versus `highs` on 300 randomized convex QPs caught a
split primal/dual step length that is correct for LPs and wrong for QPs; `nlp` versus `ipopt` agrees
to 5e-9 on every fixture. The third instance is the in-house branch-and-bound versus HiGHS MIP.

For a MILP the comparison is *stronger* than for the nonconvex NLP case and weaker than for the
convex QP one: the optimal objective value is unique, so a disagreement in objective is a bug in
one — but the optimal *solution* need not be unique, so disagreement in which taps moved is not.
Assert on the objective and on feasibility, never on the argmin.

## 9. Phases

| # | Deliverable | Gate |
|---|---|---|
| 1 | ✅ **Done.** **UCTE importer** (`src/ucte.rs`, feature `ucte`) | 175 of 176 vendored `.uct` files parse (the 176th is a deliberately malformed fixture pypowsybl rejects too). Against pypowsybl: exact to solver tolerance — <1e-9 pu voltage, <1e-6 deg, <1e-3 MW — on every fixture whose transformers declare no magnetizing admittance, **including a phase shifter at tap 16 of 16 and 400/225 transformers**. Where a magnetizing admittance is declared the gap is ~2e-4 deg / 0.3 MVar and is entirely the crate's π-split-vs-Γ shunt model: zeroing just those fields restores 2e-9 deg / 1.1e-7 MW |
| 2 | ⏳ **Half done.** `ratings::BranchLimits`, `types::TapChanger`, `set_tap_position`, and the CGMES `OperationalLimit*` importer. **Still open:** retaining CGMES tap *tables* — `cgmes.rs` evaluates the current step and discards the rest, so a CGMES-sourced PST has no `TapChanger` | Limits: every declared `CurrentLimit` is accounted for — 13 → 8 PATL + 5 TATL on the PST fixture, 768 → 398 + 370 on SmallGrid, zero unattached, zero valueless. UCTE taps round-trip through import → `set_position` → re-read |
| 3 | **IIDM importer** (`src/iidm.rs`, feature `iidm`) — version-tolerant `1_x`, bus-branch and node-breaker | All 51 `.xiidm` fixtures parse; `nordic32` and `ieee14` solve |
| 4 | **CRAC data model and readers** (`src/rao/crac.rs`, `crac_json.rs`, `<network>.rao.json`) | All 724 OpenRAO JSON CRACs import; round-trip through the native format without loss |
| 5 | **Evaluation kernel** (`src/rao/evaluate.rs`) — perimeters, margins per CNEC per state, DC | §8.2: every screened flow matches a re-solve, bus splits included |
| 6 | **Integrality on the LP boundary** (§5.3) + HiGHS MIP backend | Solves a hand-built MILP; `IpmSolver` refuses integrality rather than relaxing it |
| 7 | **Linear optimizer** (`src/rao/linear.rs`) — core + margin + discrete-PST + usage-limit fillers, iterate-and-relinearize | §8.1 analytic cases; the Cucumber scenarios that use range actions only |
| 8 | **Search tree** (`src/rao/search.rs`) — bloom, filters, deterministic ordering, parallel leaves | §8.3 on the preventive-only scenarios |
| 9 | **Curative perimeters and `RaoResult`** — multi-instant, PATL/TATL, the pull-forward rule | §8.3 on the multi-step scenarios; CLI `gridoxide rao` and Python `rao()` |
| 10 | **In-house branch-and-bound** over `IpmSolver`, MILP back in CI | §8.4 against HiGHS on every fixture and on randomized MILPs |
| 11 | **AC re-validation stage** — ToOp's second stage: re-check survivors in AC with explicit rejection thresholds | Rejection rate and reasons reported per case, not just a pass/fail |
| 12 | **Automaton simulation** (`auto` instant) — speed-ordered forced actions | The Cucumber scenarios involving ARAs |

Phases 1–3 are independently useful and ship without any RAO. Phase 5 is a genuine deliverable on its
own: a security analysis that says which CNECs are violated under which contingency, which gridoxide
cannot report today. Phases 1–9 are the minimum that answers the Context question.

Size estimate, against the reference's 26k Java LOC of algorithm plus 11k of data model: a functional
subset in Rust is **8–12k lines**, with serde collapsing the ~4.5k of hand-written JSON handling and
the result-adapter layer collapsing hard. The two importers are perhaps 1.5k more.

## 10. Risks

1. **The search tree is a heuristic, and the plan's main gate compares against another heuristic.**
   Two greedy searches with different tie-breaking can reach different local optima of equal quality.
   Mitigation: assert on objective values and feasibility, never on which actions were chosen; keep
   §8.1 and §8.2 as the gates that *can* be exact.
2. **Per-leaf MILP cost.** The reference's own notes say this dominates, and that `FastRao` exists
   solely to shrink the CNEC set. gridoxide's DC evaluation is fast, but the MILP is not gridoxide's
   code in phase 7. Watch the leaf count × MIP time product early; it is the number that decides
   whether this is usable.
3. **IIDM version drift.** Eleven schema versions in 51 fixtures is not an accident; powsybl evolves
   the format. A strict parser will rot. The "skip unknown elements" rule is the mitigation and must
   be a deliberate design property with a test, not an accident of implementation.
4. **The discrete half is not in CI until phase 10.** Phases 6–9 are verified only where
   `libhighs-dev` is installed — the position `pardiso` and `opf-ipopt` already occupy and document.
   This is an accepted cost, but it means phase 10 is not optional polish; it is what makes the
   feature testable on a bare checkout.
5. **`Instant` as data, not an enum.** Called out in §6.1 because the reference's own porting notes
   flag retrofitting it as painful, and because the obvious Rust instinct — a four-variant enum — is
   exactly the wrong one.
6. **Reading the CRAC does not mean agreeing on its meaning.** The OpenRAO JSON importer will parse
   files whose semantics depend on network-element resolution, aligned-PST groups and relative range
   types. A silently mis-resolved network element produces a plausible wrong answer. Every importer
   should carry a creation-report equivalent naming what it could not resolve, as the reference's
   `CracCreationContext` does.
7. **Licence hygiene.** Both references are MPL-2.0 and gridoxide is Apache-2.0. This is a
   *from-the-documentation* port — the LaTeX filler specifications and the format documentation —
   not a transliteration of Java. `docs/src/reference/provenance.md` should record that distinction
   explicitly, as it already does for the KLU translation.

## 11. Deliberately out of scope

- **CGMES CSA/NC profiles.** The standards-native CRAC path. cimstructs has zero of the required
  classes across 527 generated files — no `RemedialAction`, `AssessedElement`, `ContingencyEquipment`
  or `GridStateAlteration`. Adding them is a cimoxide schema-generation change, and should be planned
  there rather than assumed here.
- **Angle and voltage CNECs in the optimizer.** OpenRAO checks these post-RAO in a separate
  monitoring module. Copying that boundary costs nothing and keeps the LP flow-only.
- **Multi-timestamp / time-coupled RAO** (MARMOT) — generator ramping, min-up/min-down, ranges
  relative to the previous time step. Needs a time-series input path that does not exist, the same
  reason `plans/OPF_PLAN.md` §10 defers multi-period OPF.
- **Loop flows and relative margins.** Both need GLSK and a reference program — a whole data layer
  for zonal PTDF decomposition, relevant to capacity calculation rather than to security analysis.
- **CNE XML export.** 6.7k Java LOC of ENTSO-E report formats. Output as JSON; the report formats
  are a business-process concern.
- **ToOp's action-set generation.** Automatically enumerating all \\(2^{d-1}\\) busbar splits per
  substation and filtering them by islanding tests is a genuinely valuable capability, and gridoxide
  has the kernel for it (§3). But it answers a different question — *what could I do?* rather than
  *which of these am I allowed to do?* — and mixing them dilutes both. Worth its own plan once the
  CRAC path works.

### 11.1 Other sources considered

`docs/src/reference/resources.md` has no RAO section today; this plan should add one. Assessed
against this work:

| Source | Bearing |
|---|---|
| **powsybl-open-rao** | **Adopted** as the primary reference — structure, CRAC model, MILP formulation, and the validation gate (§8.3) |
| **ToOp** | **Adopted** for two ideas: DC-screen-then-AC-validate with explicit rejection thresholds, and diversity of answers as an output. Its BSDF kernel is already present in gridoxide (§3) |
| **powsybl-core `action-api`** | Already checked out. The elementary-action vocabulary OpenRAO's `NetworkAction` wraps — 44 action classes. The right taxonomy to copy for `ElementaryAction`, and it also defines `OperatorStrategy`/`Condition`, a lighter-weight "simulate this named strategy" layer worth knowing about |
| **PyPSA `optimize_security_constrained`** | The clean textbook SCLOPF: N-1 branch constraints added to a linear OPF. A good sanity reference for the preventive-only, continuous-only special case, and nothing more — no discrete actions, no time axis |
| **MATPOWER MOST** | Multiperiod, stochastic, contingency-constrained scheduling with *corrective* post-contingency redispatch. The closest thing to a preventive/curative split outside the CRAC world, and the reference to read if the redispatch half ever grows a time axis |
| **PowerModelsSecurityConstrained.jl** | The ARPA-E GO Challenge 1 benchmark SCOPF. Continuous, AC, no discrete actions — relevant to a future AC-SCOPF, not to this |
| **ExaGO `SCOPFLOW`** | Preventive and corrective SCOPF modes at HPC scale. Same position as the above: the continuous half done well |
| **Grid2Op / L2RPN** | An RL environment whose action space *is* substation reconfiguration. Not a solver to compare against, but the largest public body of work on topological action spaces, and the source of the "most actions do nothing" intuition behind the candidate filters (§7.4) |
| **pypowsybl `rao`** | Python bindings to OpenRAO. Not needed given the checkout, but the fastest route to generating additional reference results on networks the fixtures do not cover |
| **pglib-opf ARPA-E GO cases** | Already vendored. Security-constrained by design, but with no remedial-action catalogue — usable for the evaluation kernel's scale testing, not for the RAO itself |

## 12. API surface

```rust
// Rust
let net  = gridoxide::ucte::read("TestCase12Nodes.uct")?;      // or iidm::read, or cgmes::load_profiles
let crac = gridoxide::rao::Crac::from_openrao_json(&text, &net)?;
let result = gridoxide::rao::Rao::new(&net, &crac)
    .with_parameters(RaoParameters { max_search_depth: 3, ..Default::default() })
    .run()?;

result.margin(instant, cnec, Unit::Megawatt);
result.activated_network_actions(state);
result.optimized_tap(state, pst);
result.cost(instant);                                          // functional + virtual
```

```
# CLI
gridoxide rao <network> --crac <crac.json> [--depth 3] [--dc|--ac] [--json out.json]
gridoxide security <network> --contingencies <file>            # phase 5, standalone
```

```python
# Python — alongside dc_opf / ac_opf in src/python.rs
result = gridoxide.rao(network_path, crac_path, max_depth=3)
result.margin("curative", "cnec-id")
result.activated_actions("co1", "curative")
```

Touch-points, per the repo's established list: `Cargo.toml` (three features — `ucte`, `iidm`, `rao`
— each with the doc comment explaining its CI position; `[[test]]` blocks per gated test file),
`src/lib.rs`, `src/main.rs` (new match arm plus `USAGE` prose plus the `#[cfg(not(feature))]` stub),
`src/python.rs` (registered in the module init behind `#[cfg(feature = "rao")]`),
`docs/src/SUMMARY.md` plus a new `docs/src/rao/index.md` chapter,
`docs/src/reference/feature_comparison.md` (a new row; note that of the six comparison tools only
VeraGrid claims anything adjacent — SRAP support inside its contingency analysis — so that cell needs
checking rather than assuming a clean sweep), `docs/src/reference/resources.md` (the new section from
§11.1),
`docs/src/reference/provenance.md` (risk 7), `tests/rao_*_test.rs` + `tests/cli_rao_test.rs`,
`tests/data/rao/`, and `.github/workflows/build.yml`. No C API — neither OPF, batch, sensitivity nor
short-circuit has one, and this follows that precedent.

## Verification

End to end, after phase 9:

```bash
cargo test --features ucte,iidm                 # importers: all 227 network fixtures parse
cargo test --features rao                       # data model, evaluation kernel, relaxed LP
cargo test --features rao,opf-highs             # the MILP half, where HiGHS is installed
cargo run --features ucte,rao -- rao tests/data/rao/TestCase12Nodes.uct \
    --crac tests/data/rao/crac_pst.json --depth 2
```

and, as the external check, a harness that walks the `.feature` files, runs each scenario, and
compares against the stated costs and activated actions — reporting objective-value agreement and
feasibility separately from action-set agreement, for the reason risk 1 gives.

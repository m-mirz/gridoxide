# Multi-Island Power Flow

## Motivation

Real networks aren't always one connected system. Switched-out feeder sections,
decommissioned equipment, or genuinely separate synchronous areas can leave a Y-bus with
more than one disconnected component. Before this feature, `newton_raphson` solved the
whole bus list as a single system regardless: if *any* disconnected component had no
slack/reference bus in it, that component's Jacobian rows were structurally singular (no
reference angle to anchor them against), and the **entire solve failed** — even when
every other component was perfectly solvable on its own.

## The concepts

Supporting disconnected networks means answering four separate questions. They're
largely independent — a tool's answer to one doesn't determine its answer to the next.

### 1. What counts as "one island"?

The obvious definition is a connected component of the network graph. But there are two
defensible graph definitions once DC links exist:

- **Connected component** — traverse AC branches *and* DC/HVDC links.
- **Synchronous component** — traverse AC branches only.

Two AC areas linked *only* by an HVDC tie are one connected component but two separate
synchronous components. That's physically the right distinction, because a DC link
carries power without providing angle/frequency synchronization between its two sides:
each side needs its own angle reference, so each side is its own power-flow problem.

A tool with no DC modeling at all doesn't need the distinction — the two notions
coincide for it — but the seam is worth keeping conceptually, since retrofitting it
after the fact is harder than carrying a second, currently-redundant notion of
"component".

### 2. Which components actually get solved?

Two answers in practice:

- **Main component only** — find the largest component, solve it, drop the rest. Cheap
  and matches the common case where everything but the main grid is modeling debris, but
  the dropped buses silently produce no result at all.
- **Every component** — solve each one independently. Every bus in the input gets an
  answer, at the cost of doing work on components a user might not care about.

Note that this choice is separable from the classification itself: the classification
(what the components *are*) can run unconditionally while only the *selection* of which
to solve is configurable.

### 3. What happens to a component with no reference bus?

A component containing no slack/source bus has no anchor for voltage angle — it's not a
solvable power-flow problem, no matter how well-formed the rest of the network is. Three
distinct responses:

- **Fail the whole solve** — structurally correct in that the system really is singular,
  but it destroys the results for every other component too. This is the behavior the
  motivation above describes as the actual bug.
- **Emit a null/zero placeholder** — mark the component's buses with fixed
  \\(V = 0,\ P = Q = 0\\) so they contribute nothing to the shared system, and report
  them as unsolved. There's no principled way to fabricate a reference phasor for a
  genuinely sourceless island, so none is invented.
- **Drop the component entirely** — the natural consequence of "main component only";
  the buses just aren't in any output.

The mirror-image case is a component with *more than one* reference bus. That's
physically over-determined — two independently fixed slack phasors in one electrically
connected component — but it is not necessarily numerically singular, so Newton-Raphson
will happily converge to a result satisfying neither slack's true power balance. A
mismatch-based convergence check can't detect this after the fact; it has to be decided
up front from the topology.

### 4. How is the result reported?

A single overall status (`Converged`/`Diverged`) can't describe a network where one
island converged and another didn't. Per-island reporting means one status per
component, so a caller can tell "the part I care about converged" from "everything
failed".

There's a real limitation to be honest about here: if every component is solved in one
*shared* sparse factorization (as opposed to genuinely separate solves), a singularity
detected on the result vector can't be attributed to a specific component — the
factorization is one object. Per-island status after a shared solve is therefore
best-effort for that particular failure mode, and exact only for the ones decided from
topology up front.

## Where this fits in gridoxide today

gridoxide solves **every** solvable component in one call — no mode to choose, no opt-in;
this is simply what `run_power_flow_analysis`/`run_power_flow_analysis_from_ybus` do.
Sourceless components get the zero-voltage placeholder, and each component gets its own
status.

The key architectural fact that makes this cheap rather than a solver rewrite:
`network::YBusSparse` already exposes everything needed (`n()`, `row(i)`) to build a
connectivity graph with zero new plumbing, and *every* Jacobian backend already excludes
`BusType::Slack` buses from its unknowns with no assumption that there's exactly one
`Slack` bus anywhere in the whole bus list. So marking a sourceless component's buses as
fixed `Slack` placeholders makes the *existing* `newton_raphson`/`PersistentSolver`
correctly solve every other component in the same shared call — mathematically
equivalent, iteration for iteration, to solving each disconnected component
independently, since there's no Jacobian coupling between them. **Zero changes to the
solver algorithm itself** were needed; this is a data-preparation pass before the
existing solve and a status-reporting pass after it.

gridoxide has no DC-side traversal in this pass today, so its notion of "component" is
both notions at once (§1) — `src/dc.rs`'s HVDC converters are resolved into AC-side
injections before `build_ybus` runs, so a DC link is not an edge in the graph this pass
walks, making its components synchronous components by construction.

### The pipeline

The partitioning pipeline lives inside `solver::PersistentSolver::solve` /
`solver::newton_raphson_with_backend` / `solver::newton_raphson` themselves — not in
`run_power_flow_analysis_from_ybus`. That means every *production* entry point shares
**one** mechanism: `run_power_flow_analysis_from_ybus` is a thin convenience wrapper
(build a Y-bus, run `linear_initial_guess`, construct a one-shot `PersistentSolver`, call
`.solve()`), and a caller who reaches for `PersistentSolver`, `newton_raphson_with_backend`,
or `newton_raphson` directly — to reuse a cached factorization, pick a specific backend, or
set a custom `tol`/`max_iter` — gets the exact same island handling for free, with no
separate opt-in and no risk of silently bypassing it. All three return
`Vec<solver::IslandReport>` (previously `SolveStatus`/`()`).

`batch::BatchSolver` inherits it too, since each of its workers calls
`PersistentSolver::solve` per scenario — batching changes how many solves run and on which
thread, not what a solve does.

**One deliberate exception:** `bde::solve_batch_block_diagonal` does *not* partition
islands. It stacks every scenario's Jacobian into one block-diagonal sparse matrix to
validate the GPU path's architecture (see `scripts/bench/README.md` §4d), and is an architecture
validator rather than a production entry point — its own module
documentation says so and points CPU callers at `BatchSolver`. Worth stating here rather than
only there, because "every entry point partitions islands" is exactly the kind of invariant
that gets relied on later. If BDE ever becomes a production path, island partitioning has to
be added to it: a sourceless component would otherwise make its whole block singular, and
because the blocks are structurally disjoint (§4 below), that block alone — not the batch —
would be the part that fails.

Each of those functions does, internally:

1. `network::connected_components` — an iterative DFS over `YBusSparse`'s own adjacency,
   generic over the finished Y-bus (so it works uniformly for native JSON, PGM-JSON, and
   CGMES input with no format-specific code).
2. `network::classify` — counts each component's existing `Slack` buses:
   - **exactly one** → normal, solvable.
   - **zero** → `network::mark_unreferenced_islands` pins every member bus to a fixed
     `V = 0`, `P = Q = 0` placeholder. There is no principled way to fabricate a
     reference voltage/angle for a genuinely sourceless island, so none is attempted —
     every unit/sign-convention bug this project has actually fixed got fixed by
     matching verified physical or reference-implementation behavior, never by
     guessing, and inventing a slack here would repeat that mistake class.
   - **more than one** → left untouched in the shared solve, and reported as
     `AmbiguousReferenceBus`. This verdict is decided once, up front, and is never later
     overwritten by a numerically-convergent-looking mismatch check — for the reason §3
     gives.
3. One shared solve (via whichever `JacobianBackend` the caller picked) across the whole
   (now correctly-classified) bus set.
4. `solver::finish_island_reports` recovers each component's own status *after* the
   shared solve, restricting the same `power_injections`/`effective_injection` mismatch
   computation `newton_raphson_cached`'s own convergence check already uses to just that
   component's bus indices.

`newton_raphson_enforcing_q_limits` composes on top the same way it always did: each of
its outer PV→PQ-switching passes calls `PersistentSolver::solve` (now returning
`Vec<IslandReport>`) and returns that same `Vec<IslandReport>` — from its last pass once
Q-limits have stabilized, or immediately if some island's own status is `Singular`/
`MaxIterationsReached`.

### `PowerFlowReport` and `IslandStatus`

```rust
pub struct PowerFlowReport {
    pub buses: Vec<Bus>,
    pub islands: Vec<solver::IslandReport>,
    /// Iteration count and per-iteration convergence trace. The solve loop
    /// itself prints nothing — see `solver::SolveStats`.
    pub stats: solver::SolveStats,
}

pub enum IslandStatus {
    Converged,
    MaxIterationsReached,
    Singular,
    NoReferenceBus,        // no Slack bus at all — placeholder values only, never solved
    AmbiguousReferenceBus, // more than one Slack bus — solved, but not necessarily meaningful
}
```

### A real limitation, stated plainly rather than papered over

`Singular` **cannot, in general, be attributed to a specific island** — the shared-solve
caveat from §4, concretely. Every backend solves one combined sparse factorization, and
singularity is detected via a finiteness check on the *result vector* (`sparse.rs`), not a
specific failing pivot or row. When the overall solve reports `Singular`, every component
whose own post-hoc mismatch is still above tolerance gets marked `Singular` too —
best-effort, not precise. A component whose own mismatch is already below tolerance is
marked `Converged` regardless of the overall status, since every backend returns
`Singular` *before* applying that iteration's update (confirmed directly in both the
`Scalar` and `Block` backends), so `buses` always holds the last fully-applied,
fully-finite state — never a partially-updated or NaN-poisoned one — making this post-hoc
check meaningful even after an overall `Singular` result.

### One existing "sourceless placeholder" precedent

**`cgmes.rs`'s de-energized-bus block** — uses CGMES's own `TopologicalIsland`
membership (a semantic guarantee from the standard: "only energised
TopologicalNode-s shall be part of the topological island"), which encodes real domain
knowledge the raw admittance graph doesn't have — not just a bypass-caller
accommodation, unlike the PGM case above. Kept unconditionally: per the FullGrid
diagnosis in Motivation, if the generic pass *replaced* this block, FullGrid's real
converter gap (`EquivalentBranch` unimplemented) would get silently repackaged as 4
spurious `NoReferenceBus` islands instead of surfacing as the connectivity bug it
actually is.

The generic pass layers on top of CGMES's own block, unconditionally, for every entry
point — a harmless, idempotent no-op for input that's already de-energization-resolved
by CGMES's own `TopologicalIsland` logic, and the mechanism that gives CGMES richer
per-island reporting once a file genuinely declares more than one `TopologicalIsland`.

### Validated

`tests/multi_island_test.rs` covers, entirely through the public API (no internal-only
test module needed, since `PersistentSolver::solve` is itself public and does the full
partitioning/reporting internally): two well-formed islands matching independent
single-island solves exactly; a no-slack island correctly placeholder-reported without
poisoning a well-formed neighbor (the literal repro of the original bug); an ambiguous
(two-slack) island likewise not poisoning a neighbor; all three non-trivial statuses
coexisting in one call, in `connected_components`'s bus-index-ascending order; and an
easy-converging island correctly reported `Converged` even while a deliberately hard one
drags the overall solve to `MaxIterationsReached` — exercised by calling
`PersistentSolver::solve` directly with a custom low `max_iter`, since
`run_power_flow_analysis_from_ybus`'s own `tol`/`max_iter` are fixed.

## Tool reference

| Tool | Island definition (§1) | Which are solved (§2) | Sourceless component (§3) |
|---|---|---|---|
| **gridoxide** | connected == synchronous (no DC edges in the graph) | all, unconditionally, in one shared solve; per-island `IslandReport`/`IslandStatus`. `batch::BatchSolver` inherits this per scenario; `bde::solve_batch_block_diagonal` deliberately does not partition at all (see "The pipeline") | zero-voltage placeholder, reported `NoReferenceBus` |
| power-grid-model | connected == synchronous (no DC/HVDC modeling at all) | all, unconditionally — DFS from every `Source` builds a separate internal "math model" per reachable sub-graph (`topology.hpp`: "divide grid into several math models... start search from a source") | null/zero output (`get_null_output()`), rest of the calculation unaffected |
| powsybl-core / powsybl-open-loadflow | both, distinctly: `AbstractConnectedComponentsManager` traverses AC + DC/HVDC, `AbstractSynchronousComponentsManager` AC only | opt-in via `LoadFlowParameters.ComponentMode`: `MAIN_CONNECTED` (default), `ALL_CONNECTED`, `MAIN_SYNCHRONOUS`. Under `ALL_CONNECTED`, `LfNetworkLoaderImpl.load()` groups buses by `(connectedComponentNum, synchronousComponentNum)` and builds one independent `LfNetwork` per group, each with its own slack selection and its own `LoadFlowComponentResult` | dropped under the default `MAIN_CONNECTED` — this is why pypowsybl's bus counts run below gridoxide's on every CGMES fixture |

Neither tool auto-detects a mode from the network's actual shape: powsybl's
`DEFAULT_COMPONENT_MODE` is a fixed constant, and PGM has no mode at all. The
classification itself always runs unconditionally; only *selection* is ever configurable.

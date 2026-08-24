# Reactive dispatch between machines sharing a voltage target

Status: **Phase 1 implemented**, 2026-08-24, against `bc16494`. Phases 2 and 3 remain
proposals; §11 records what Phase 1 found.

> **The scope changed once the fixtures were measured.** This was going to be powsybl's `DISTR_Q`
> — n−1 reactive-distribution equations inside the Newton system, one per controller bus. Measuring
> what the vendored CGMES corpus actually contains says that is the wrong first move: **every**
> shared-control case in it is several machines at the *controlled bus itself*, which needs no new
> equations at all. §2 has the numbers. The plan below follows them.

## 1. The question

`docs/src/reference/feature_comparison.md` marks shared voltage control ⚠️ — "capability yes,
dispatch no". `cgmes::VoltageControl` sums reactive limits across every machine holding one bus,
which is right and was a real fix, but nothing says **which machine produces what**. Its own doc
comment names the gap and defers it:

> Doing it properly means reactive dispatch inside the Newton system, which is a different job.

That sentence assumes the powsybl shape. It is worth checking before adopting it.

## 2. What the corpus actually contains

Instrumented `VoltageControl::regulate` to record, per controlled bus, the *own* bus of every
machine regulating it — the pairing the importer computes and discards. Across every vendored
configuration:

| fixture | shared, all controllers **at the controlled bus** | shared, controllers at **distinct** buses | single remote |
|---|---|---|---|
| RealGrid | **62** | **0** | 0 |
| FullGrid | **2** | **0** | 1 |
| MicroGrid | 0 | 0 | 1 |
| SmallGrid, Svedala | 0 | 0 | 0 |

Two findings, and they point opposite ways.

**The expensive half has no coverage.** Not one case anywhere in the corpus has several machines at
*different* buses jointly regulating one bus. That is precisely the configuration `DISTR_Q` exists
for. Building it now would mean writing the crate's most delicate code — new equations inside the
Newton system — against zero real data, and the only way to know it worked would be a case
constructed to prove it. `VoltageControl::regulate` already refuses to guess at target conflicts
for exactly this reason ("no vendored fixture has two controllers disagreeing about a target… any
resolution rule would be untested"). The same standard applies here.

**The cheap half covers 64 real buses.** When every machine sits at the bus it regulates, the bus is
an ordinary `PV` bus with one reactive injection. Splitting that injection between the machines is
**allocation after the solve, not an equation inside it**. The power-flow answer is already right;
what is missing is only the per-machine attribution — and with it, which specific machine is at its
limit rather than merely that the bus is.

## 3. Three shapes, and which are real here

Write the controlled bus \\(B\\) and the controllers \\(A_1 \dots A_m\\). The unknown/equation
accounting is the whole story:

| shape | unknowns removed | equations removed | balance |
|---|---|---|---|
| local, one machine (\\(A_1 = B\\)) | \\(\|V\|_B\\) | \\(Q\\) at \\(B\\) | balanced — **this is the ordinary `PV` bus** |
| local, \\(m\\) machines (all \\(A_i = B\\)) | \\(\|V\|_B\\) | \\(Q\\) at \\(B\\) | balanced — still one `PV` bus. **Allocation, not equations** |
| remote, one machine (\\(A_1 \neq B\\)) | \\(\|V\|_B\\) | \\(Q\\) at \\(A_1\\) | balanced — but the two are *different buses* |
| remote, \\(m\\) machines | \\(\|V\|_B\\) | \\(Q\\) at each \\(A_i\\) | short by \\(m-1\\) → **`DISTR_Q`** |

So the existing `PV` bus is the degenerate case of one general rule: *fix a magnitude somewhere, free
a reactive injection somewhere else.* Today gridoxide can only do it when "somewhere" and "somewhere
else" are the same bus, and it forces the remote case into that mould by pinning \\(B\\) to `PV`
and leaving the machine's own bus alone — which puts the reactive power out at the wrong bus.

## 4. Phase 1 — per-machine allocation at a shared bus

The 64-case job, and no Newton change.

### Retain the discarded half

The importer computes each machine's own bus and its reactive range, then throws both away.
`TapChanger`'s doc comment already states this pattern and why:

> every importer until now computed the position it was told to use and threw the rest away…
> Anything that *moves* a tap needs the discarded half back.

Same here. A new `types::RegulatingMachine { id, at_bus, controls_bus, q_min, q_max, key: Option<f64> }`,
returned alongside the buses, plus the per-bus **non-regulating** reactive injection — because the
bus's solved \\(Q\\) is the machines *plus* the load, and only the machines are being allocated.

### Allocate

At the converged state, for each controlled bus:

\\[ Q_{machines} = Q_{calc}(B) - Q_{non\text{-}regulating}(B) \\]

split by normalized keys, then a clamp-and-redistribute loop: any machine whose share falls outside
its own \\([q_{min}, q_{max}]\\) is pinned there and its excess re-split across the rest, repeating
until nothing moves or every machine is pinned. This is what turns "the bus is at its limit" into
"machine 4 of 6 is at its limit and the others have headroom".

### Keys

Transcribed from `references/powsybl-open-loadflow`'s `Control.createReactiveKeys`, which falls back
in a fixed order and falls back **wholesale**, not per machine — one implausible value discards the
whole basis:

1. explicit per-machine reactive key, where the document states one;
2. else proportional to each machine's max reactive range, unless any range is outside plausible
   bounds;
3. else uniform — one unit per regulating machine at the bus.

Whether CGMES supplies (1) needs checking against the corpus; if nothing does, say so and let (2)
carry, rather than writing a reader for a field no file has.

### Validation — there is a published oracle, and it is not currently read

`RealGrid_SV.xml` carries 122,895 `SvPowerFlow` elements: the published per-terminal P and Q of the
reference solution, including at each machine's own terminal. **gridoxide reads no `SvPowerFlow`
today** — `cgmes_common::assert_matches_sv` compares voltages via `SvVoltage` only.

So Phase 1 gets a real gate rather than a self-consistency check: allocate, then compare each
machine's share against its own published terminal flow. Reading `SvPowerFlow` is independently
worth having — it is a per-branch oracle for flows the crate currently validates only at buses.

Expect the comparison to be loose, and say why in the test rather than tightening the tolerance
until it passes: the published dispatch comes from another tool's allocation rule, which need not be
the key chain above. The gate worth asserting is that the shares **sum** to the bus total exactly,
that every share respects its own machine's limits, and that the ranking of machines by output
matches the published one. An exact per-machine match is a bonus, not the criterion.

## 5. Phase 2 — remote control where the reactive power actually is

Two fixture cases (MicroGrid, FullGrid), and a wrong answer today rather than a missing feature: a
machine at \\(A\\) regulating \\(B\\) has its reactive output produced at \\(B\\), so the reactive
flow on the path between them — typically a step-up transformer — is missing, and \\(A\\)'s voltage
is whatever the network makes it rather than whatever the machine holds.

The fix is the general rule from §3: keep \\(\|V\|_B\\) out of the unknowns and drop the \\(Q\\)
equation at \\(A\\). Concretely, the unknown layout stops being derivable from `BusType` alone —
`Layout::analyze` currently reads `bus_type` and nothing else — and needs a control map beside it.
That is the same generalization `src/continuation/augmented.rs::Layout` already had to make for the
λ unknown, so the shape is familiar.

Worth measuring first, and cheap to measure: how far apart are \\(A\\) and \\(B\\) in the two cases?
If they are the two ends of one transformer, the size of the error is that transformer's reactive
loss, and knowing the number decides whether Phase 2 is worth doing before Phase 3.

## 6. Phase 3 — `DISTR_Q`, deferred and gated on a fixture

Only reachable once Phase 2 exists, and **not to be built until something exercises it.** Two ways
that could change, in preference order:

1. a fixture that has it — worth a look at the wider ENTSO-E conformity set and at
   powsybl-open-loadflow's own test resources, which certainly construct such cases;
2. failing that, a hand-derived three-bus case whose answer can be written down, in the style of the
   continuation plan's closed-form two-bus nose.

The formulation is settled and can be transcribed when the time comes —
`AcEquationSystemCreator.createGeneratorReactivePowerDistributionEquations`, with the constant term
from `AcTargetVector.getGeneratorReactivePowerDistributionTarget`:

\\[ 0 = (k_i - 1)\,q_i + k_i \sum_{j \neq i} q_j, \qquad \text{target } (k_i - 1)\,Q_{spec,i} \\]

## 7. Reactive limits, and what changes

`ReactiveLimits` clamps the bus at the summed limit and pins `q_spec`. That stays correct for
Phase 1 and needs no change — allocation happens after it. It is worth reporting, though, that when
the bus clamps, the allocation says which machines are individually saturated; a bus can be inside
its summed limit while a machine inside it is not, and today nothing can see that.

Phase 2 changes this materially: once a controller is a distinct bus, a controller hitting its limit
must drop out of the control and hand its bus back to `PQ` — and, in Phase 3, the keys must be
recomputed over the survivors. powsybl does exactly that
(`GeneratorVoltageControl.updateReactiveKeys`, re-run when a controller is disabled). That is an
outer-loop interaction, and the plan for it belongs with Phase 2 rather than here.

## 8. API surface

```rust
// New, alongside the existing report
pub struct MachineDispatch {
    pub id: String,
    pub at_bus: usize,
    pub controls_bus: usize,
    pub q: f64,          // per-unit, allocated
    pub q_min: f64,
    pub q_max: f64,
    pub at_limit: bool,  // the thing the bus-level clamp cannot tell you
    pub key: f64,        // normalized share, and which fallback produced it
}
```

surfaced on `VoltageControlReport`, through the CLI's `solve` output where shared buses exist, and
as `PowerFlowModel.machine_dispatch()` in Python.

## 9. Out of scope

- Target conflicts between controllers of one bus stay reported rather than resolved, for the
  reason already recorded in `VoltageControl::regulate`: no fixture has one.
- Reactive **power** control (a machine holding a branch's Q rather than a bus's V) — powsybl's
  `GeneratorReactivePowerControl` shares the `DISTR_Q` machinery, so it follows Phase 3, not this.
- Continuous SVC/shunt susceptance control.

## 10. Risks

1. **Phase 1's oracle may disagree for a legitimate reason.** The published dispatch is another
   tool's allocation. If per-machine numbers do not match, the useful outcome is a documented
   comparison, not a tuned key rule — see §4.
2. **Phase 2 touches the unknown layout**, which every solver in the crate shares. The bit-for-bit
   Jacobian oracle (`solver::jacobian_triplets_reference`) is the guard, and it only covers what it
   is given: it will need cases with a remote control in them or it guards nothing.
3. **The temptation to do Phase 3 first**, because it is the interesting one. The measurement in §2
   is the argument against, and it should be re-run rather than trusted if the corpus changes.

---

## 11. What Phase 1 found

Built as §4 describes: `types::RegulatingMachine` retained at import, `src/dispatch.rs` doing the
allocation, `gridoxide solve --dispatch`, `PowerFlowModel.machine_dispatch()`, and gates in
`tests/cgmes_dispatch_test.rs` plus unit tests for the key chain and the redistribution.

### The oracle worked, once the two errors were separated

`SvPowerFlow` gave a per-machine reference for all **496** of RealGrid's regulating machines. Taken
naively the comparison reads 8.0% mean relative error, which says almost nothing — it conflates two
unrelated things. Split apart:

| | mean relative error |
|---|---|
| bus totals — *gridoxide's solve* against the published one | 7.5% |
| the split rule, measured against the *published* bus total | 0.5% overall, **7.4% on shared buses** |

The solve error is pre-existing and has nothing to do with attribution. What the split rule is
responsible for is the second row, and only the 62 shared buses are a real test of it, since a lone
machine's "split" is not a choice. Capability-proportional reproduces the published attribution to
7.4% there. Both numbers are asserted loosely and recorded, per §4 — tightening either would be
fitting the rule to one document's convention.

### It measured a documented simplification for the first time

The invariant "shares sum to the bus total" does not hold everywhere, and chasing that turned up
something worth having. `ReactiveLimits` compares `Bus::q_min`/`q_max` against the bus's **net**
injection, while the CGMES importer fills those fields from the machines' **own** capability. Where
a reactive load shares the bus the machine has to cover it too, so it saturates before the net
injection reaches the bound and the clamp fires late.

`pgm::PgmVoltageRegulator`'s doc comment already names this as a deliberate simplification. Nothing
had ever measured it. On RealGrid: of 416 regulated buses, 50 share their bus with another reactive
injection and **6 end up unattributable, 0.0436 p.u. (4.4 MVAr) in total**. Reported through
`BusDispatch::unattributed` rather than absorbed, and surfaced by the CLI, because the honest
reading is "the solve put this bus outside what its machines can produce", not "the arithmetic did
not close".

Fixing it means bounding the machine part rather than the net, which **changes power-flow answers**
and therefore belongs with Phase 2, not here. Phase 1 changes no answer, as promised.

### Smaller things

- **A real bug, in code this touched rather than in the new code.** `VoltageControl` is sized from
  `buses.len()` before branch conversion, and a three-winding transformer pushes a star bus
  afterwards — so the final bus list is longer than the accumulator. Indexing it in `finish` panicked
  on MiniGrid. The §10 risk about touching shared import code was the right one to write down.
- **Redistribution needs asymmetric capability to fire at all.** With `q_max = −q_min` a
  capability-proportional share exceeds a machine's own maximum only once the *bus* is past its joint
  capability, at which point every machine saturates together. Every RealGrid machine is symmetric,
  so the clamp-and-redistribute loop is exercised by unit tests rather than by the fixture — worth
  knowing before someone reads the fixture passing as coverage.
- **CGMES states no per-machine key.** `RegulatingControl` gives a target and `SynchronousMachine` a
  capability; nothing apportions between machines sharing a target. `key` is therefore always `None`
  from this importer and the capability basis carries, which is why `KeyBasis::Explicit` appears in
  no fixture. Left in the chain rather than removed, because it is the branch a format that *does*
  state one would take.
- **One bus falls back to uniform**: 6232, whose two machines each declare a ±0.001 p.u. (0.2 MVAr)
  range — an order of magnitude below anything plausible, so the capability basis is rejected. Both
  declare the *same* implausible range, so uniform and capability agree there anyway. The
  plausibility guard costs nothing and stops one mis-stated nameplate from taking a bus.

### Still not done

Phase 2 (remote control where the reactive power actually is) and Phase 3 (`DISTR_Q`), both
unchanged from above — and the clamp fix that Phase 1's `unattributed` measurement now justifies.

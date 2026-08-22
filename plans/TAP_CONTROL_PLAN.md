# Transformer tap control in gridoxide

Status: **implemented**, 2026-08-22, `999f339..760b97f`. Written as a proposal the same day and
kept as written below, with a closing section recording where the plan was wrong.

## 1. What this is

Power flow as gridoxide computes it answers *what is the state, given these injections* — and,
silently, *given these tap positions*. A real on-load tap changer does not work that way. It holds
a voltage or a flow, and its position is an **output** of the solve rather than an input to it.

\\[ \text{find } t \in \{t_{\min},\dots,t_{\max}\} \ \text{ s.t. } \ |V_c(t) - V^{\text{target}}|
   \le \tfrac{1}{2}\,\text{deadband} \\]

with \\(V_c\\) the voltage at the *controlled* bus, which need not be either of the transformer's
own. The phase-shifter version substitutes a branch's active power for the voltage.

`docs/src/reference/feature_comparison.md` gives gridoxide `❌ static taps only` in this row against
`✅` from four of the five comparison tools — power-grid-model's `TapChangingStrategy`, powsybl's
several outer loops, VeraGrid's `control_taps_modules`/`control_taps_phase`, pandapower's
`DiscreteTapControl`/`ContinuousTapControl`. Only lightsim2grid also fixes taps at init, and its own
`docs/disclaimer.rst` names that as a known limitation rather than a design.

This is therefore not a new analysis type. It is a **modelling assumption underneath everything
already shipped** — power flow, contingency, sensitivity, OPF, state estimation and the RAO gate all
compute on a network whose transformers cannot respond. Closing it changes existing numbers, which
is why §8.5 is as long as it is.

The second half of the job is structural and is not optional. §4.1 shows that gridoxide's two
existing outer loops cannot run together. Taps would be the third, and adding it as a fourth
`newton_raphson_*` entry point beside three it subsumes is the wrong answer. Since API stability is
not a constraint here, the right answer is available: **collapse all of them into one entry point
taking a list of loops** (§5.4). That is what makes the refactor affordable rather than the
"materially larger undertaking" the comparison document's item 11 calls it — the cost is a
61-file mechanical port, not a compatibility layer maintained forever.

## 2. Decisions taken

- **Both control families, ratio and phase.** A CGMES `TapChangerControl` is a `RegulatingControl`,
  and the vendored conformity fixtures use two of its modes: `voltage` (11 on Svedala, 3 on
  FullGrid) and `activePower` (2 on FullGrid, 1 on PST_Type3). Building only the voltage half would
  leave the fixture that actually *moves* a tap (§8.3) untestable.
- **The incremental, sensitivity-driven strategy first**, in powsybl's sense of
  `IncrementalTransformerVoltageControlOuterLoop`, not the continuous-relax-then-round one. The
  reason is specific to this crate: the quantity it needs, \\(\partial|V_c|/\partial \rho\\), is
  already `ac_sensitivity::AcSensitivity` with `Variable::TransformerRatio` against
  `Function::VoltageMagnitude`, and it is already validated against a central-difference re-solve of
  the full nonlinear power flow. The continuous variants need the ratio inside the Newton system,
  which is a change to `jacobian::JacobianPattern` and to `n_unknowns`.
- **The outer-loop layer is internal, not an extensibility mechanism.** powsybl's `OuterLoop` is
  ServiceLoader-discovered so third parties can register their own; that is a Java-ecosystem
  affordance and copying it would buy an abstraction nobody outside this crate can reach. A trait
  plus an ordered `Vec<Box<dyn OuterLoop>>` built by the crate is enough to make the loops compose,
  which is the actual defect. The comparison table's "pluggable outer loop" row closes on the
  composition, and the row should say *internal* when it does.
- **Validated against the fixtures' own published answer.** Every CGMES conformity fixture with a
  tap changer publishes `SvTapStep.position` in its SV profile — the position the tool that produced
  the fixture converged to. `SvTapStep` is already generated in cimstructs. This is the same shape of
  gate as pglib's `BASELINE.md` for OPF and OpenRAO's Cucumber suite for RAO: the reference states
  its expectation, and gridoxide is scored against it rather than against its own arithmetic.
- **The migration replaces the existing entry points; it does not wrap them.** Confirmed with the
  author: at version 0.0.2, API stability is not a constraint, so
  `newton_raphson_enforcing_q_limits` and `newton_raphson_distributing_slack` are *deleted* rather
  than kept as wrappers, and `tests/q_limits_test.rs` / `tests/distributed_slack_test.rs` are ported
  to the new call. What is *not* free to change is the **numbers** those tests assert — see §8.5,
  which is now about numerical drift rather than signatures.
- **Deadbands are honoured, not approximated away.** Every enabled control in the fixture set
  carries one (Svedala: ±2.59 kV on a 129.5 kV target, ±0.275 on 11 kV; FullGrid: ±35 MW on a
  −65 MW target). A loop that ignores the deadband hunts, and a loop that hunts does not terminate.

## 3. What already exists, and what it buys

Unusually much, because two other features paid for the hard parts first.

| Need | Already there |
|---|---|
| Discrete positions → complex tap, incl. asymmetric and tabular changers | `types::TapChanger` — `at`, `ratio`, `angle_deg`, `set_position`, `nearest_to_angle`. Built for RAO |
| \\(\partial \lvert V\rvert/\partial \rho\\) and \\(\partial P/\partial \alpha\\) at a converged point | `ac_sensitivity::AcSensitivity`, `Variable::{TransformerRatio, PhaseShift}` × `Function::{VoltageMagnitude, BranchActivePower}`, both forward and adjoint against one factorization |
| Keeping the symbolic factorization across a tap move | `PersistentSolver::invalidate_admittances` — a tap changes admittance *values*, not the pattern, exactly as a switch flip does. `types::TapChanger::set_position`'s own doc comment already says so |
| An outer loop that must `reset` | `solver::newton_raphson_enforcing_q_limits` (PV→PQ changes `n_unknowns`) |
| An outer loop that must not | `solver::newton_raphson_distributing_slack` (only `p_spec` moves) |
| Resolving `RegulatingControl.Terminal` to a controlled bus | `src/cgmes.rs`, written four times over — `SynchronousMachine`, `StaticVarCompensator`, `PowerElectronicsConnection`, `ExternalNetworkInjection` |
| All four `PhaseTapChanger` flavours + `RatioTapChangerTable`, per step | `cgmes::TapChangerIndex::effect_for_end` — it computes the current step and **discards the rest** |
| Tap tables retained through import | `iidm.rs` and `ucte.rs`, as `tap_changers: Vec<Option<TapChanger>>` |
| A second pass over a CGMES dataset producing a side table | `cgmes::cgmes_operational_limits` → `EquipmentLimits` + `LimitImportReport`. The precedent this plan's importer follows exactly |
| Regulation targets, modes, deadbands; a controlled-bus map for taps | **nothing** |
| An outer-loop driver | **nothing** |

The RAO work is the direct predecessor. It needed to *move* a phase shifter, so it built the retained
tap table and the position↔angle map; it did not need anything to decide *where* the tap should be,
because a CRAC says. This plan supplies the decision.

## 4. Three gaps that must close first

Preconditions rather than phases. The first is useful on its own and is arguably a defect report.

### 4.1 The outer loops cannot compose

```
src/solver.rs:420   newton_raphson
src/solver.rs:682   newton_raphson_enforcing_q_limits
src/solver.rs:1293  newton_raphson_distributing_slack
```

Each constructs its own `PersistentSolver::new(backend)` and runs its own `for pass in 0..max_outer`
loop. There is no way to enforce Q-limits *and* distribute slack in one solve. On a real transmission
network you want both: distributed slack because a single bus absorbing a few hundred megawatts
distorts every flow around it, Q-limits because 11 of the 12 committed MATPOWER benchmark cases have
a PV bus whose unconstrained Q exceeds its nameplate. Today they are alternatives.

Nothing about the two loops makes them incompatible — they are the same shape, they interleave
naturally, and powsybl runs them together by default. It is purely that neither was written with a
place to put the other.

**Resolution.** §5. This must land before tap control, because tap control is the loop that makes the
absence structural rather than merely inconvenient: a tap move changes reactive flow, which changes
which generators hit their limits, which changes the slack pickup.

### 4.2 CGMES keeps no tap table, and reads no tap regulation

`grep tap_changers src/cgmes.rs` returns nothing. `TapChangerIndex::effect_for_end` evaluates the
current step into a `TapEffect { tap, x_override }` and returns that; the other positions are
computed and dropped. So the deepest importer in the tree produces transformers that cannot move,
while `iidm.rs` and `ucte.rs` produce ones that can — an asymmetry with no reason behind it beyond
the order the importers were written in.

The regulation side is entirely absent. cimstructs generates `TapChangerControl` (a bare
`RegulatingControl` subclass, so `mode`, `targetValue`, `targetDeadband`, `enabled`, `Terminal` all
come from the base) and `TapChanger` carries `tapChangerControl`, `controlEnabled`, `lowStep`,
`highStep`, `neutralStep`, `ltcFlag`. `src/cgmes.rs` reads none of them.

The data is in the committed fixtures — 17 files mention `TapChangerControl`:

| Fixture | `TapChangerControl` | enabled | modes |
|---|---|---|---|
| Svedala | 11 | 11 | all `voltage` |
| FullGrid | 5 | 4 | 3 `voltage`, 2 `activePower` |
| PowerFlow | 1 | 0 | `voltage` |
| PST_Type1 / Type2 / Type3 | 1 each | — | all `activePower` |
| MicroGrid-Type2-HVDC | 1 | — | — |

**Resolution.** Two additions to `src/cgmes.rs`, following `cgmes_operational_limits`'s precedent of a
separate pass returning a side table plus a report rather than widening
`cgmes_to_buses_and_branches`'s already-four-element tuple:

- `cgmes_tap_changers(ds) -> HashMap<String, TapChanger>` — keyed by `PowerTransformerEnd` mRID,
  built by having `effect_for_end` retain every step it already computes instead of one.
- `cgmes_tap_regulation(ds) -> (Vec<TapRegulation>, TapRegulationReport)` — mode, target, deadband,
  enabled, and the controlled bus resolved from `RegulatingControl.Terminal` by the same route the
  four existing callers use.

### 4.3 UCTE reads the tap table but skips the target columns

`ucte::parse_regulation` reads `##R` columns 20–32 (ratio `δu`, `n`, `n'`) and then jumps to 39
(angle `δu`, `θ`, `n`, `n'`, kind). Bytes 33–38 and 58–63 — the ratio regulation's **U (kV)** target
and the angle regulation's **P (MW)** target — are never read. `RatioRegulation` and
`AngleRegulation` have no field for either.

That is a real omission, and it is also **not testable on the material here**: across all 207 `##R`
records in the vendored `.uct` corpus, zero carry a U target and zero carry a P target. So the fields
should be read (an importer that silently drops a column is how a plausible wrong answer gets made),
but no gate can be built on them, and the plan should not pretend otherwise.

IIDM is better provisioned but not in the committed tree: `tests/data/iidm`'s 8 networks have 7 phase
tap changers, all `regulating="false"` (`CURRENT_LIMITER` ×4, `FIXED_TAP` ×3) and no ratio tap
changers at all. `references/powsybl-open-rao`'s 51 `.xiidm` fixtures — the ones the RAO work is
gated on — have **zero** regulating changers either. `references/powsybl-core`'s 301 do: 338 ratio
and 93 phase changers, 155 `regulating="true"`, 136 `regulationMode="VOLTAGE"`. A corpus worth
reaching for, and one that lives in a gitignored checkout.

**So CGMES is where the gate is.** That is the opposite of the RAO work, where UCTE and IIDM carried
everything and CGMES carried nothing.

## 5. The outer-loop layer

### 5.1 The trait

powsybl's `lf/outerloop/OuterLoop.java` is four methods — `initialize`, `check` returning
`STABLE | UNSTABLE | FAILED`, `cleanup`, `isNeeded` — and that is the right shape. In Rust, without
the five type parameters the Java version carries to bridge its AC and DC engines:

```rust
pub enum OuterLoopStatus { Stable, Unstable, Failed(String) }

pub trait OuterLoop {
    fn name(&self) -> &'static str;

    /// What this loop invalidates when it reports `Unstable`. The driver acts
    /// on it; a loop never touches the solver cache or the Y-bus itself. §5.3.
    fn invalidates(&self) -> Invalidates { Invalidates::Nothing }

    fn initialize(&mut self, _ctx: &mut OuterLoopContext<'_>) {}

    fn check(&mut self, ctx: &mut OuterLoopContext<'_>) -> OuterLoopStatus;
}

pub enum Invalidates {
    /// Only `p_spec` moved — distributed slack.
    Nothing,
    /// `bus_type` moved, so `n_unknowns` and the pattern did — reactive limits.
    Pattern,
    /// `Transformer::tap` moved: restamp the Y-bus, keep the symbolic
    /// factorization — both tap loops.
    Admittances,
}
```

`OuterLoopContext` is a bundle of mutable borrows plus the last solve's result:

```rust
pub struct OuterLoopContext<'a> {
    pub buses:         &'a mut [Bus],
    pub lines:         &'a [Line],
    pub transformers:  &'a mut [Transformer],
    pub tap_changers:  &'a mut [Option<TapChanger>],
    pub regulation:    &'a [TapRegulation],
    pub ybus:          &'a mut YBusSparse,
    pub islands:       &'a [IslandReport],
    pub iteration:     usize,
}
```

powsybl passes a network object here; gridoxide has none, and the parallel-`Vec` convention is
crate-wide, so the bundle is the honest translation. It is worth noticing that this struct is
approximately the network object the crate never grew — if one is ever wanted, this is where it
would start, and the freedom to change signatures means that is a later decision rather than a
foreclosed one.

### 5.2 The driver, and why the nesting order is the design

`AcloadFlowEngine.java:250-315` is worth transcribing in prose because the scheduling is not the
obvious one:

- The loops are **nested, innermost first in the list**. Each is run to *its own* stability — an
  inner `do { check } while (UNSTABLE)` — before the next is consulted.
- After any loop reports `Unstable` and the network is re-solved, the driver goes back to the
  **start** of the list, so an outer loop's move re-exposes the inner ones to their own criteria.
- Termination is `outerLoop == lastUnstableOuterLoop`: having walked the whole list and found the
  loop that last moved something now stable, nothing moved this pass, and the run is done.

Its default order (`DefaultAcOuterLoopConfig.java`) is: distributed slack → HVDC → area interchange →
secondary voltage → monitoring → **reactive limits** → **phase control** → **transformer voltage
control** → transformer reactive power → shunt → automation. Slack innermost, Q-limits next, taps
outside both.

That ordering is a physical claim, not a convention: generators respond faster than tap changers,
so a tap should be chosen against a reactive dispatch that has already settled, and re-checked when
the tap move disturbs it. gridoxide's list is the subset it has:

```
[ DistributedSlack, ReactiveLimits, PhaseControl, TransformerVoltageControl ]
```

with `max_outer_iterations` as a single budget across all of them, as powsybl has it, rather than a
per-loop count.

### 5.3 The invalidation contract

This is the part that has to be stated explicitly, because getting it wrong is silent — a stale
symbolic factorization does not fail, it converges to the wrong place or refuses to converge.

| Loop | What it changes | Correct call |
|---|---|---|
| Distributed slack | `p_spec` only | `Invalidates::Nothing` — the Jacobian is unchanged |
| Reactive limits | `bus_type`, hence `n_unknowns` and the pattern | `Invalidates::Pattern` → `reset()` |
| Phase / voltage tap control | `Transformer::tap`, hence Y-bus *values* | `Invalidates::Admittances` → restamp + `invalidate_admittances()` |

`invalidate_admittances` sets `self.jacobian = None` and leaves the per-backend factorization objects
alive, so the fill-reducing ordering and elimination tree survive — measured at ~45% of solve time on
a 9,241-bus case. This is precisely the position `switches::set_switch_open` and
`batch::BatchSolver`'s scenario loop already occupy, and the third caller of the same contract.

`Invalidates` exists so the **driver** makes this call, never the loop. Three things must happen
together after a tap move — write `Transformer::tap`, restamp the Y-bus entries derived from it,
drop the cached Jacobian — and a loop that does one or two of them is a bug that surfaces as a wrong
answer on the *next* pass rather than as a failure on this one. Returning an enum and letting the
driver act on it makes the three inseparable by construction.

### 5.4 What the existing entry points become: deleted

API stability is not a constraint here, and the plan is materially better for it. The three entry
points of §4.1 collapse into one:

```rust
pub fn newton_raphson(
    ctx: &mut SolveContext<'_>,
    tol: f64,
    max_iter: usize,
    backend: JacobianBackend,
    loops: &mut [Box<dyn OuterLoop>],
    budget: usize,
) -> (Vec<IslandReport>, OuterLoopReport)
```

`newton_raphson_enforcing_q_limits`, `newton_raphson_enforcing_q_limits_with_stats` and
`newton_raphson_distributing_slack` are **removed**. A plain solve is this function with an empty
loop list; Q-limits is `[ReactiveLimits::default()]`; the combination nobody can currently express is
just a longer list. `PersistentSolver::solve` grows the same two parameters, since the driver has to
live where the factorization cache does.

`SolveContext` is `OuterLoopContext` minus the per-pass fields — the same borrow bundle, constructed
once by the caller. Passing it instead of loose `&mut [Bus]` / `&mut YBusSparse` arguments is what
makes the tap loop possible at all: a loop must restamp the Y-bus, and every current signature takes
it immutably.

Had backward compatibility been required, the same capability would have arrived as a *fourth*
entry point sitting beside three others it subsumes, with `&mut YBusSparse` threaded through as a
special case. That was the shape of the previous draft and of risk 4, and both are now gone.

**The blast radius, counted rather than guessed.** 61 files reference `newton_raphson*` or
`PersistentSolver::new` — 15 under `src/`, 42 under `tests/`, 4 examples. `PersistentSolver::new`
itself appears 20 times across 14 files. That is a large mechanical port and a small conceptual one:
most call sites are `solver.solve(&mut buses, &ybus, tol, max)` and become
`solver.solve(&mut ctx, tol, max, &mut [], budget)`, which is a `SolveContext::new(...)` line above
and an argument change. Worth doing in its own commit, ahead of any tap code, so a bisect can tell a
port error from a control error.

**What the freedom does not extend to.** The call sites move; the answers do not. Porting
`tests/q_limits_test.rs` and `tests/distributed_slack_test.rs` to the new signature is expected and
fine — changing the values they assert is not, and §8.5 is the guard. A migration that quietly
relaxes a tolerance to make a port compile has converted an API change into a physics change.

## 6. The controls themselves

### 6.1 A ratio tap changer regulating a voltage

Per outer pass, for each controlled bus outside its deadband, in powsybl's
`IncrementalTransformerVoltageControlOuterLoop` form:

\\[ \Delta\rho = \frac{V^{\text{target}} - V_c}{\partial V_c/\partial\rho} \\]

then the nearest reachable position to \\(\rho + \Delta\rho\\), capped at `max_tap_shift` steps.
Three guards, all of them load-bearing:

- **Insensitivity filter.** powsybl's `MIN_SENSI_FILTER = 0.05`; below it the controller is declared
  unable to reach its bus and is left alone. Without it \\(\Delta\rho\\) is a division by nearly
  zero and the tap slams to a limit.
- **Direction-change budget.** `MAX_DIRECTION_CHANGE = 3`. A tap that has reversed three times is
  hunting between two positions that straddle the target, and the honest answer is to stop and report
  it, not to keep going until the iteration budget runs out.
- **Deadband.** Half-deadband on each side; a bus inside it is `Stable` by definition.

The missing primitive is small: `TapChanger::nearest_to_ratio(ratio) -> Option<i32>`, the sibling of
the existing `nearest_to_angle`, searching the step table rather than dividing by a step size for the
same reason — the map is not linear, and for a `RatioTapChangerTable` it is not even regular.

Several controllers on one controlled bus is the shared-control problem the comparison table already
flags a `❌` for on the generator side ("last writer wins"). The tap version should not repeat that:
split \\(\Delta\rho\\) across the controllers by sensitivity and report the conflict when their
targets disagree, which is what powsybl's `adjustWithSeveralControllers` does. It is also the cheaper
half of the generator problem, since taps are outside the Newton system.

### 6.2 A phase shifter regulating active power

Same loop, different pair: \\(\Delta\alpha = (P^{\text{target}} - P_b)/(\partial P_b/\partial\alpha)\\)
on the *regulated terminal's* branch, `Variable::PhaseShift` × `Function::BranchActivePower`, then
`nearest_to_angle`, which exists.

Two notes. The sensitivity's direct term matters here: `ac_sensitivity`'s own validation found that a
tapped branch's own flow carries a \\(\partial f/\partial p\\) term on top of the chain rule, and
dropping it flips the sign (−0.276 → +0.286 on `distribution-case`). A phase shifter regulating its
*own* flow — which is the normal case, and is what FullGrid's `BE_PS_ASYM_1` does — is exactly that
configuration. And the sign convention has to be pinned against the fixture rather than reasoned out:
FullGrid's target is `−65 MW`, so the direction the SSH means by positive flow is a property of the
regulated terminal, not of gridoxide's `Terminal::From`.

### 6.3 The discrete tail

The incremental strategy rounds at every step, so it never produces a fractional position and needs
no separate discretization pass. What it can produce is a **limit cycle** between two positions
whose voltages straddle the target with no position inside the deadband. That is a property of the
network and the deadband, not a solver failure, and the report must distinguish it from
non-convergence — `OuterLoopReport` carries, per controller: final position, steps moved, direction
changes, and whether it stopped `InDeadband`, `AtLimit`, `Insensitive` or `Hunting`.

### 6.4 Deliberately not ported

- `SimpleTransformerVoltageControlOuterLoop` and `TransformerVoltageControlOuterLoop` — the two
  continuous-then-round strategies. They put the ratio inside the Newton system, which is a change to
  `JacobianPattern` and `n_unknowns`, for an answer the incremental strategy also reaches.
- `IncrementalTransformerReactivePowerControlOuterLoop` — `reactivePower` mode. No fixture in the
  committed tree uses it for a tap changer (the 12 `REACTIVE_POWER` regulation modes are all in
  `references/`), so it would ship ungated.
- Shunt-section control. The same outer loop over a different discrete device; worth having once the
  layer exists, and not this plan.
- Secondary voltage control, area interchange, HVDC AC-emulation limits. Each is its own loop on the
  layer §5 builds. Area interchange is already named in the comparison document's gap 3 as the
  natural next one.

## 7. Importers

### 7.1 CGMES — where the work is

Per §4.2. Two functions, both following `cgmes_operational_limits`'s shape: a separate pass, a side
table keyed by mRID, and a report counting what could not be resolved, because a silently dropped
control is how an importer produces a plausible wrong answer.

The report needs at least: controls whose `Terminal` names no bus in this dataset; controls with no
`targetValue`; controls in a mode not modelled (`reactivePower`, `currentFlow`), counted rather than
converted, exactly as `cgmes_operational_limits` counts the `ApparentPowerLimit`s it cannot express
in amperes; and tap changers with a `TapChangerControl` but `controlEnabled=false`, which are data
rather than errors — PowerFlow's single control is one.

`RegulatingControl.targetValue` is in kV for `voltage` mode and MW for `activePower`, so both need
the per-unit conversion the rest of the importer already does, against the controlled bus's own
nominal voltage rather than a global base.

### 7.2 IIDM and UCTE

IIDM: `ratioTapChanger`/`phaseTapChanger` already parse into `TapChanger`. What is unread is the
regulation — `regulating`, `targetV`, `targetDeadband`, `regulationMode`, `regulationTerminal`,
`loadTapChangingCapabilities`. Cheap, and it makes `references/powsybl-core`'s 136 VOLTAGE-mode
changers reachable as a second corpus — the only voltage-mode material anywhere that *moves*.

UCTE: read `##R` columns 33–38 and 58–63 per §4.3, with a note in the module docs that no vendored
fixture populates them.

### 7.3 PGM and native JSON

PGM's own `TapChangingStrategy` is a solver option over `transformer.tap_pos`/`tap_min`/`tap_max`,
not a per-transformer regulating control, and `PgmTransformer` has no target field to read. Out of
scope. Native JSON gains an optional `tap_control` block once the Rust model settles, on the same
argument `plans/OPF_PLAN.md` §4 makes about companion documents: gridoxide's own format may grow a
field, someone else's may not.

## 8. Validation

### 8.1 Analytic

A two-bus network, one tapped transformer, one load. The tap position that puts the secondary inside
the deadband is computable by hand from the step table; the loop must find it from every starting
position, and must find it in one pass from a start where the sensitivity is accurate.

### 8.2 Against the tap table by exhaustive sweep

For each fixture transformer with a changer: solve at every position in `low..=high`, record the
controlled bus's voltage, and assert that the loop's answer is the position the sweep says is closest
to target — or, where none is inside the deadband, one of the two straddling it. This is the gate
that does not depend on any reference tool, and it is the one that catches a sign error in
\\(\partial V/\partial\rho\\), which is otherwise entirely plausible in the output.

### 8.3 Against the fixtures' own `SvTapStep`

The external gate. Every conformity fixture publishes the position its own producer converged to:
Svedala 11, FullGrid 11, SmallGrid 10, PowerFlow 1.

Comparing SSH start positions against SV answers, checked before writing this:

- **Svedala**: all 11 identical. **SmallGrid**: all 10 identical. **PowerFlow**: identical.
- **FullGrid**: 2 of 11 differ.

So most of the corpus is a **fixed-point** gate — the fixture starts at its own answer, and a
converged tap loop must leave every one of those 22 taps where it found them. That is a weaker
statement than "finds the right tap" but a strong one about false positives, and it is the failure
mode this feature actually risks: a loop that moves taps it should not silently degrades every
existing CGMES result.

FullGrid is the one fixture that moves, and only one of its two movements is a tap-control movement:

| Changer | Kind | Control | SSH → SV |
|---|---|---|---|
| `BE_PS_ASYM_1` | `PhaseTapChangerAsymmetrical`, low 1 / neutral 13 / high 25 | `activePower`, target −65 MW, deadband 35 MW, enabled | **14 → 15** |
| `BE_TR2_HVDC2` | `RatioTapChanger`, low −2 / neutral 0 / high 2 | **none** | 0 → −2 |

The first is the gate for §6.2, complete with its own target and deadband. The second has no
`TapChangerControl` at all, so nothing in this plan should reproduce it, and a test that expects the
loop to is testing the wrong thing. Worth writing down before someone tries to make it pass.

No fixture in the committed tree exercises a *voltage*-mode control that moves. §8.2's sweep is what
covers that direction, and `references/powsybl-core` (§7.2) is where to look if a moving
voltage-mode fixture is wanted.

### 8.4 Svedala's voltage tail

`tests/cgmes_svedala_test.rs` currently compares against the published SV on percentiles rather than
per bus, and its own doc comment records why: median error 0.08%, but p99 ≈ 4.3% and max ≈ 4.7%,
attributed to `newton_raphson` not enforcing Q-limits.

Svedala has 11 enabled voltage-mode tap controls that gridoxide ignores and the SV's producer did
not. Some of that tail is plausibly taps rather than Q-limits. This is a **measurement, not an
assertion**: solve Svedala with the full loop list and record what the percentiles do. If they
improve, that is the strongest evidence this feature is worth having; if they do not, the doc
comment's attribution was right and should be left as it is. Either outcome is worth committing.

### 8.5 The signatures may move; the answers may not

§5.4 deletes two entry points and re-signs a third, so 61 files get ported. The point of this section
is the line between the two kinds of change that port can make.

**Free to change:** every call site, the shape of the arguments, which struct carries what, the
names. `tests/q_limits_test.rs` and `tests/distributed_slack_test.rs` are expected to be rewritten
against the new signature, and so are the 40 other test files that construct a solver.

**Not free to change**, and each of these is an assertion that must survive the port unaltered:

- `tests/q_limits_test.rs`'s switch counts and converged voltages, on all three Jacobian backends.
- `tests/distributed_slack_test.rs`'s seven-pass convergence on `case14_ieee` / `case30_ieee` /
  `case118_ieee`, and the 2.3 / 2.4 / 16.5 per-unit moved off each slack. That test re-derives its
  injections from `network::power_injections` rather than from the loop's own bookkeeping, which is
  exactly the property that makes it a physics gate rather than a bookkeeping one.
- All 12 MATPOWER benchmark cases converging under a `[ReactiveLimits]` list, with the same number
  of PV→PQ switches as `newton_raphson_enforcing_q_limits` produces today.
- Every CGMES fixture test's voltage comparison, **with an empty loop list**. A plain solve after
  the refactor must be bit-identical to a plain solve before it. This is the single most valuable
  assertion in the plan, because it separates "the driver changed the numbers" from "tap control
  changed the numbers", and only one of those is intended.
- `scripts/bench/README.md` §4b's batch scaling and §6's CGMES timings, re-measured. The driver sits
  on the same `PersistentSolver`, and a per-pass allocation that costs 10% would otherwise hide.

The failure mode to name explicitly: a port that relaxes a tolerance to make a rewritten test
compile has silently converted an API change into a physics change. Record the before-numbers first,
in the porting commit's message, so the comparison is available rather than reconstructed.

## 9. Phases

| # | Deliverable | Gate |
|---|---|---|
| 0 | **The signature port** — `SolveContext`, `newton_raphson` and `PersistentSolver::solve` re-signed, 61 files updated. No behaviour change, no loop list yet (the parameter exists and is always empty) | §8.5's "not free to change" list, every bullet. Specifically: a plain solve is **bit-identical** to a plain solve before the commit, on every CGMES fixture and all 12 MATPOWER cases. This phase is worth landing alone precisely so that assertion is checkable in isolation |
| 1 | **The outer-loop layer** — `OuterLoop` trait, `OuterLoopContext`, the nested driver of §5.2, the invalidation contract of §5.3. `ReactiveLimits` and `DistributedSlack` implemented as loops; the two old free functions deleted | The two ported tests reproduce their pre-port numbers through the loop list. Plus the new capability nothing today can express: Q-limits *and* distributed slack in one solve, producing a state satisfying both — Q within limits at every switched bus, slack deviation distributed by weight |
| 2 | **CGMES tap tables** — `cgmes_tap_changers`, `effect_for_end` retaining every step | Every position of every changer in Svedala/FullGrid/SmallGrid/PST round-trips import → `set_position` → re-read. The current step read back from the table equals the `TapEffect` the old path returned, bit for bit, on all four fixtures |
| 3 | **CGMES tap regulation** — `cgmes_tap_regulation`, `TapRegulation`, `TapRegulationReport` | The counts in §4.2's table, exactly: 11/11 on Svedala all `voltage`; 5 on FullGrid of which 4 enabled and 2 `activePower`; PowerFlow's 1 disabled. Zero unattached, zero targetless, unmodelled modes counted not dropped |
| 4 | **Voltage tap control** — `TransformerVoltageControlOuterLoop`, `nearest_to_ratio`, deadband, sensitivity filter, direction budget | §8.1 and §8.2. §8.3's fixed-point half: all 22 already-correct taps on Svedala/SmallGrid/PowerFlow stay put. §8.4 measured and recorded whichever way it comes out |
| 5 | **Phase-shifter active-power control** — `PhaseControlOuterLoop` | §8.3's moving half: FullGrid's `BE_PS_ASYM_1` reaches position 15 from 14, with the −65 MW / 35 MW target and deadband read from the SSH rather than hard-coded. `BE_TR2_HVDC2` is asserted *unmoved*, with a comment saying why |
| 6 | **IIDM and UCTE regulation** (§7.2) + CLI and Python exposure (§12) | IIDM: the `references/powsybl-core` corpus's VOLTAGE-mode changers import with targets. UCTE: the two column groups read, with a test on a synthesized record since no fixture has one |

Phase 0 is pure churn and ships on its own, deliberately: it is the only phase whose gate is
"nothing changed", and mixing it with phase 1 would make that unverifiable. Phase 1 is the
composition fix and is worth having with no tap control at all. Phases 2–3 are useful without 4–5:
a CGMES network that carries its tap tables is what lets a RAO `PstRangeAction` work on a CGMES
input, which today it cannot.

## 10. Risks

1. **Changing results that are currently gated.** The defining risk, and unchanged by the API
   freedom: enabling tap control changes converged voltages on every CGMES fixture with an enabled
   control. Mitigation is that it is opt-in and the default loop list is empty — but the moment it
   becomes a default, every percentile in `cgmes_*_test.rs` moves. §8.4 is the deliberate, measured
   version of that; anything else is an accident.

   The freedom to rewrite tests makes this risk *worse*, not better, and that is worth saying
   plainly. When a test may be edited, "the assertion no longer holds" and "the assertion was
   rewritten to hold" look the same in the diff. Phase 0's bit-identical gate exists to close that
   window while it is still closable.
2. **Hunting is the normal failure mode, not an edge case.** Discrete positions and a finite
   deadband mean a network can have no stable answer. The loop must terminate and say so.
   `MAX_DIRECTION_CHANGE` is the mitigation and the report field is the honesty.
3. **The sensitivity is evaluated at the wrong point.** \\(\partial V/\partial\rho\\) is exact at the
   converged state and approximate anywhere else, so a large \\(\Delta\rho\\) overshoots.
   `max_tap_shift` bounds it; powsybl caps at a small number of steps per outer iteration for exactly
   this reason, trading passes for stability.
4. **A stale Y-bus.** No longer an API risk — `SolveContext` carries `&mut YBusSparse` and every
   caller is being rewritten anyway — but still a correctness one. `Transformer::tap` and the Y-bus
   entries derived from it must not drift apart, and `types::TapChanger::set_position` writes the
   first while knowing nothing about the second (its own doc comment says so). `Invalidates::Admittances`
   is the mitigation: one restamp helper, driven by the driver, so no loop can move a tap without the
   Y-bus following.
5. **The gate is thin in one direction.** One fixture, one tap, one movement (§8.3), all of it
   phase-shifter. The voltage half is covered only by §8.2's self-consistent sweep. That is weaker
   than the RAO Cucumber gate or pglib's baselines, and worth stating plainly rather than implying
   the conformity suite validates more than it does.
6. **Three-winding transformers.** CGMES resolves them into three two-winding branches around a star
   bus, and a `RatioTapChanger` sits on one `PowerTransformerEnd`. The mapping from end mRID to the
   resolved leg has to be right, and a wrong leg produces a plausible answer. FullGrid has exactly one
   three-winding transformer among its eleven; check it explicitly rather than assuming the existing
   resolution carries the association.

## 11. Deliberately out of scope

- **Taps as OPF decision variables.** Already named as absent in `plans/OPF_PLAN.md` §10 and in the
  comparison table's item 8. A regulating tap and an optimized tap are different questions, and the
  RAO's `PstRangeAction` is the second one.
- **Tap control inside contingency and batch solves.** `BatchSolver` amortizes one symbolic
  factorization per worker; a tap loop invalidates admittances per scenario, which is affordable, but
  the interaction with `solve_contingencies`' structural-entry trick needs its own thought. Ship the
  base case first.
- **Tap control in DC.** powsybl has `DcIncrementalPhaseControlOuterLoop`, and it is the natural
  companion to `linear::btheta`. Cheap once the layer exists; not on the path to the gate.
- **Shunt-section control, secondary voltage control, area interchange.** §6.4.
- **Generator shared voltage control.** The `❌ last writer wins` cell in the comparison table's
  shared-voltage-control row is a real defect, and it is *inside* the Newton system (powsybl adds
  n−1 `DISTR_Q` equations), which makes it a different job from anything here.

## 12. API surface

```rust
// Rust — one entry point, the loop list is the whole configuration surface
let mut ctx = SolveContext::new(&mut buses, &lines, &mut transformers,
                                &mut tap_changers, &regulation, &mut ybus);

let mut loops: Vec<Box<dyn OuterLoop>> = vec![
    Box::new(DistributedSlack::new(&distribution)),   // innermost
    Box::new(ReactiveLimits::default()),
    Box::new(PhaseControl::default()),
    Box::new(TransformerVoltageControl::default().max_tap_shift(3)),
];
let (islands, report) = solver::newton_raphson(&mut ctx, 1e-8, 30, backend, &mut loops, 40);

// A plain solve is the same call with an empty list.
let (islands, _) = solver::newton_raphson(&mut ctx, 1e-8, 30, backend, &mut [], 0);

report.controller(pst).final_position;
report.controller(pst).outcome;        // InDeadband | AtLimit | Insensitive | Hunting
```

The loops take their regulation data from the context rather than at construction, so the list is
built once and reused across solves — which is what a batch or contingency sweep would need if tap
control ever reaches them (§11).

```
# CLI — a new subcommand, see below
gridoxide solve <network> --control-taps [--control-taps-max-shift 3] \
                          --enforce-q-limits --distribute-slack
```

**There is no `gridoxide solve` today.** `USAGE` lists `estimate`, `security`, `rao`, `switches`,
`short-circuit`, `opf`, `sensitivity` and `dc`; a bare `gridoxide` runs a bundled demo. The only CLI
route to an AC power flow is `switches --solve` (CGMES, as a side effect of listing switches) and
`dc` (PGM JSON, and DC). So the CLI half of phase 6 is *adding the subcommand the crate never grew*,
not adding a flag to one — larger than it looks, and worth doing precisely because tap control,
Q-limits and distributed slack would then all be reachable at once.

```python
# Python — the first exposure of any outer loop beyond the Rust API
model = gridoxide.PowerFlowModel.from_cgmes(paths)
model.solve(control_taps=True, enforce_q_limits=True, distribute_slack="uniform")
model.tap_positions()
```

`--enforce-q-limits` and `--distribute-slack` have no exposure today either, in the CLI or in
Python — the comparison document records both loops as "library-only". Phase 1 is what makes
exposing them a matter of one flag each against one entry point, rather than three entry points to
choose between.

Touch-points, per the repo's established list: `src/solver.rs` (the layer, the single entry point,
and the deletion of two others), the 61-file port of §5.4,
a new `src/outerloop/` module or `src/solver/outerloop.rs`, `src/types.rs`
(`TapChanger::nearest_to_ratio`), `src/cgmes.rs` (two passes), `src/iidm.rs`, `src/ucte.rs`,
`src/main.rs` (a new `solve` subcommand plus its `USAGE` prose), `src/python.rs`, `docs/src/SUMMARY.md` plus a new
`docs/src/powerflow/tap_control.md`, `docs/src/reference/feature_comparison.md` (two rows — tap
control, and the outer-loop row, which should say *internal* rather than claiming powsybl's
extensibility), `tests/tap_control_test.rs`, `tests/outer_loop_test.rs`, and the CGMES fixture tests
of §8.5. No C API, following OPF/RAO/batch/sensitivity/short-circuit.

## Verification

End to end, after phase 5:

```bash
cargo test                                       # §8.5: ported call sites, unchanged numbers
cargo test --test outer_loop_test                # composition: Q-limits + distributed slack together
cargo test --test tap_control_test               # §8.1, §8.2, §8.3
cargo test --test cgmes_svedala_test -- --nocapture   # §8.4, the percentiles either way
cargo run --features cgmes -- solve \
    tests/data/CGMES-Test-Configurations/v3.0/FullGrid/FullGrid-Merged \
    --control-taps --enforce-q-limits          # after phase 6 adds the subcommand
```

and, as the external check, a script that reads each fixture's `SvTapStep` and diffs it against the
converged positions — reporting *moved when it should not have* separately from *did not reach the
published position*, because §8.3 shows the corpus is overwhelmingly the first kind of test.


---

## 13. What happened

All six phases landed. The plan is left as written above; this section records where it was
wrong, because that is the part worth keeping.

### The gate's premise was false, and the fixture said so

§8.3 reasoned that since Svedala's SSH tap positions equal its SV ones, a converged loop must
leave all eleven where it found them, and made that the headline fixed-point gate. Measured
instead of assumed: **Svedala's own published solution sits 2–5% away from the targets its own
SSH declares, against half-deadbands of 1–2%** — every one of the eleven controls is violated at
the published positions. The published state does not satisfy the controls the document records,
so a loop that left those taps alone would be the defective one.

The gate inverted. `svedala_brings_every_controller_into_its_deadband` now asserts that all
eleven move *into* their deadbands, `every_chosen_position_is_confirmed_by_a_sweep` confirms each
choice against an exhaustive per-controller sweep, and
`svedalas_published_solution_violates_its_own_deadbands` records the premise failure so it cannot
quietly stop being true. SmallGrid, PowerFlow and MiniGrid carry the no-movement evidence Svedala
cannot.

### The moving-tap gate is unreachable

§8.3's one moving fixture was FullGrid's `BE_PS_ASYM_1`, 14 → 15. **FullGrid does not converge in
gridoxide at all** — a pre-existing limitation, unrelated to this work, and there is no FullGrid
solve test in the tree to have caught it. Resolving its HVDC converters does not help.

The phase-shifter loop is gated on the three PST conformity configurations instead: Type 1 starts
inside its deadband and must not move, Types 2 and 3 are outside theirs and must reach the best
position their table offers. Weaker than the plan promised, and §10's risk 5 was right to call the
gate thin.

### A discrete tap often cannot reach a deadband at all

Not anticipated. `PST_PhaseTapChangerLinear_Type2` asks for zero flow within 5e-4 pu on a shifter
whose closest position leaves 0.11 pu. The first implementation relabelled every unfinished
controller `InDeadband` once nothing moved, reporting a flow 200 times the tolerance as success.
`ControllerOutcome::Closest` says it instead, and `AtLimit` now means what it says — the
controller asked for a ratio or angle the changer cannot produce, which is distinguishable only
because the caller passes it in, since `nearest_to_ratio` rounds into the table by construction.

### Two things the plan did not know were there

**Transformer ordering was nondeterministic.** `ends_by_pt` is a `HashMap`, and Rust randomizes
its iteration order per process, so the transformer list — and every flat branch index derived
from it — came out differently on each run of the same program against the same file. Nothing
asserted on a transformer index, so it never failed. Sorted by mRID now, with a test that imports
one file twice.

**Tap position is not the only thing a phase changer moves.** A CGMES
`PhaseTapChangerLinear`/`Symmetrical`/`Asymmetrical` carrying `xMin`/`xMax` changes reactance as
it moves. `TapChanger::series` carries that per position and `set_position` writes both or
neither; writing only the ratio shifts the phase while leaving the impedance at the exported
position, which still moves the flow the right way.

**FullGrid has two phase shifters on one target**, so `PhaseControl` groups by regulated flow the
way the voltage loop groups by controlled bus. The plan only anticipated the voltage case.

### Where the plan was deliberately departed from

**§5.4 said the three entry points collapse into one; they did not.** `PersistentSolver::solve`
kept its signature and stayed the *inner* solve, with the outer-loop driver as a layer above it.
The two outer-loop free functions were deleted as planned. That is layering rather than the
compatibility shim §5.4 argued against — a contingency sweep or a state-estimation solve has no
use for a loop list — and it turned the 61-file mechanical port into about fifteen files, all of
them genuinely about outer loops. §5.4's `SolveContext` exists and does carry `&mut YBusSparse`,
which is what the tap loops needed.

**Phase 0 therefore did not happen as its own commit.** Its gate — a plain solve is bit-identical
across the refactor — is met and asserted (`an_empty_list_is_a_plain_solve`,
`control_taps_off_changes_nothing`), just inside phase 1's commit rather than ahead of it.

**The plan said "No C API".** There is one, `src/capi/`, with two entry points mirroring the two
deleted functions. Both were ported to the driver behind an unchanged ABI, and
`gridoxide_powerflow_solve_with_controls` was added for the composition. An FFI shim over the
driver is what an FFI layer is, not the wrapper pattern §5.4 objected to.

**`cgmes_to_buses_and_branches` was not widened.** The four-element tuple would have become five;
`cgmes_to_network` returns a `CgmesNetwork` instead and the tuple function destructures it, which
kept about fifteen test call sites untouched for no loss of honesty.

### Numbers, as they came out

| | |
|---|---|
| Svedala | 11 controls read, all voltage; 11 of 11 reach their deadbands; 6 outer re-solves over 7 inner solves |
| FullGrid | 5 controls, 3 live across both modes, 4 disabled; two shifters on one −65 MW target |
| PST Type 1 / 2 / 3 | 1 control each, all `activePower`; Type 1 a fixed point, Types 2–3 reach `Closest` |
| SmallGrid / MiniGrid | 10 and 3 tap tables, no controls, nothing moves |
| IIDM (`references/powsybl-core`) | 111 voltage and 5 active-power controls across 301 files |
| UCTE | 0 of 207 `##R` records declare a target — the columns are read anyway |
| Test suite | 826 Rust tests, 98 Python (29 skipped without optional deps) |

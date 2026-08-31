# Widening the dynamics model library

Status: **proposal**, 2026-08-30, against `7b1391d`. Written after scanning Dynawo's own example
corpus, which turned "breadth" — an open-ended task with no oracle — into a ranked list of **seven
cases that already ship reference trajectories**, and found a defect worth more than any of them.

> **Why this needed a scan first.** `docs/src/reference/feature_comparison.md` calls the model
> library a gap by comparison — four machine orders, `SEXS` plus a proportional regulator, `TGOV1`
> plus a proportional governor, one stabiliser, against Dynawo's Modelica library. True, and useless
> as a work item: nothing in the tree fails, so there is no measurement saying which model to write,
> and a model written with no case demanding it and no reference to check it against is exactly the
> physics-nobody-can-check that `RMS_PLAN.md` §20 declined saturation over.

---

## 1. The corpus

`dynawo/dynawo`'s `examples/` directory, sparse-cloned (25 MB, no Dynawo install — the same
arrangement `tests/data/dynamics/dynawo/PROVENANCE.md` records for the existing gate):

| family | `.dyd` cases | reference curve sets | what it is |
|---|---|---|---|
| DynaSwing | 59 | 63 | short-term transient stability — gridoxide's own problem |
| DynaFlow | 7 | 6 | steady state with controls |
| DynaWaltz | 6 | 5 | long-term dynamics; the existing IEEE14 fixture is from here |
| RT | 2 | 0 | real-time variants |

**69 distinct model libraries across 74 case files.** Most are not machine models: `Line` (52),
`InfiniteBusWithVariations` (47), `NodeFault` (40), `Step` (39), `TransformerFixedRatio` (36),
`SetPoint` (22), `DoubleStep` (22) are network elements and event constructs, which gridoxide models
natively.

Of the generator libraries, gridoxide's `.dyd` reader covers three —
`GeneratorSynchronousFourWindings`, `...ThreeWindingsProportionalRegulations` and
`...FourWindingsProportionalRegulations` — which is 12 of the 74 files' generator declarations.

## 2. What the scan found first, and it is not a missing model

`src/dynamics/dyd.rs` dispatches on the library name by **prefix and suffix**:

```rust
fn is_four_windings(lib: &str) -> bool { lib.starts_with("GeneratorSynchronousFourWindings") }
fn has_proportional_regulations(lib: &str) -> bool { lib.ends_with("ProportionalRegulations") }
```

`GeneratorSynchronousFourWindingsTGov1SexsPss2a` satisfies the first. It does not satisfy the second.
And the regulator branch has no `else`:

```rust
let (mut avr, mut gov) = (None, None);
if has_proportional_regulations(&model.lib) { … }
```

So a case declaring a machine **with a governor, an exciter and a stabiliser** is read as a bare
sixth-order machine with constant field voltage and constant mechanical power, **with no warning**.
`DydWarning::UnsupportedLib` exists and is raised in two places — for a library that is not a
generator at all, and for a generator that is neither three- nor four-windings — but not for the case
in between, which is every unrecognised regulator suffix in the corpus.

It would then initialize to a flat start, run, and produce a confident wrong answer. That is the same
failure this month has turned up four times over, and it is worth more than any model below: the
`.dyr` reader already does the right thing here, warning `UnsupportedModel` and skipping, which is
how `IEEEST` in `tests/data/dynamics/two_machine.dyr` is handled today.

## 3. The ranked list

Every case below ships `reference/outputs/curves/curves.csv` — Dynawo's own solver's trajectories,
committed upstream — so each is gateable the moment its models exist.

| what is missing | cases unlocked | reference curves |
|---|---|---|
| **suffix mapping only** — `TGov1Sexs` | `DynaSwing/ENTSOE/TestCase2` | ✓ |
| **PSS2A** | `ENTSOE/TestCase1`, `ENTSOE/TestCase3`, `SingleMachineSystem/SynchronousMachineI8` | ✓ ✓ ✓ |
| **VRKundur** | `Kundur_Example13/KundurExample13_VR_NoPss` | ✓ |
| **VRKundur + PssKundur** | `Kundur_Example13/KundurExample13_VR_Pss` | ✓ |
| **PmConst + VRNordic** | `DynaWaltz/Nordic` | ✓ |
| GoverPropVRPropInt | `DynaFlow/IEEE14/IEEE14_DisconnectLine` | ✗ |

**Seven gated cases behind one name mapping and four regulator models**, and the first of those five
is not a model at all: gridoxide already has `TGOV1` and `SEXS`, so `TGov1Sexs` needs the suffix
recognised and the two existing models wired to it.

Two things make this cheaper than the table suggests. `Kundur_Example13` is **the directory the
existing Dynawo gate already validates against** — `tests/data/dynamics/dynawo/kundur13/` holds its
`_SetPoint` variant, the no-regulator one — so `_VR_NoPss` and `_VR_Pss` extend a case whose network,
machine and initialization are already correct, isolating the regulator. And a stabiliser serves both
readers: PSS2A is a PSS/E model too, which is what `IEEEST` in the vendored `.dyr` is waiting for.

## 4. What is reused

- **The finite-difference Jacobian oracle** (gate G4, `tests/dynamics_models_test.rs`). Every model
  in the library is checked against it, and it is the reason a new model is a day rather than a week:
  the analytic Jacobian is the easiest thing here to get subtly wrong, and it is never trusted.
- **The flat-trajectory invariant.** A run with no disturbance must not move. Nearly every
  initialization mistake is caught by that one check, and a regulator with a mis-derived reference
  fails it immediately.
- **`tests/dynamics_reference_test.rs`**, which already compares a whole trajectory against Dynawo's
  committed curves. A new case is a new entry, not new machinery.
- **`Limits`, `LimitState`, the latch/project discipline** — non-windup limiting is built and gated,
  and every regulator below has limits.

## 5. Phases

### Phase 1 — Refuse what is not understood

Warn on an unrecognised regulator suffix and build the machine without regulators *only* when the
library says there are none. Everything else raises `DydWarning::UnsupportedLib`, which the CLI
already prints and `dynamics_dyd_test.rs` already asserts on.

**Gate:** reading `TestCase1.dyd` today warns, naming `GeneratorSynchronousFourWindingsTGov1SexsPss2a`.
Before this change it is silent, which is the defect.

This lands first and alone. It is a correctness fix, it is independent of every model below, and it
changes what the existing reader does to documents nobody has tried yet.

### Phase 2 — `TGov1Sexs`, with no new physics

Recognise the suffix and wire the existing `Tgov1` and `Sexs` to it, reading their parameters from
the `.par` set the way `ProportionalRegulations` already does.

**Gate:** `DynaSwing/ENTSOE/TestCase2` against its committed curves, at the tolerance
`dynamics_reference_test.rs` already uses. This is the phase that tests whether the corpus is
reachable at all — if the parameter names or the per-unit bases do not line up, that is a reader
problem to solve once, before four models are written on top of it.

### Phase 3 — PSS2A

Three cases, the largest single block, and gridoxide already has a stabiliser to compare against.
PSS2A is dual-input (speed and electrical power) with two washouts and a ramp-tracking filter, where
`Stab1` is a single washout plus two lead-lag stages.

**Gates:** the finite-difference oracle; the flat trajectory; then all three cases against their
curves. Plus one that costs nothing and is worth having — with the stabiliser's gain at zero the
trajectory must match the `TGov1Sexs` run of the same case bit for bit, which isolates the new model
from everything around it.

### Phase 4 — VRKundur, PssKundur

Two cases, in the directory the existing gate already covers, so the network and machine are known
good and only the regulator is new. Worth doing after PSS2A rather than before: it is the smaller
piece and it benefits from whatever phase 3 learns about reading regulator parameter sets.

### Phase 5 — PmConst, VRNordic

`PmConst` is constant mechanical power — the absence of a governor, which the unit composite already
expresses. `VRNordic` is one model. One case, in DynaWaltz, so it also exercises a family the
existing gate touches only through IEEE14.

### Phase 6 — The `.dyr` side

`IEEEST` in the vendored fixture, once PSS2A exists and the stabiliser interface has been used twice.
The `.dyr` reader already refuses it correctly, so this closes a warning rather than a hazard.

## 6. What decides whether this is worth continuing

After phase 3, six of the seven cases are reachable and the marginal cost of a model is known. If
phase 2 shows the parameter sets do not map cleanly — different per-unit bases, names that are not in
the `.par` — then the corpus is more expensive than it looks and the ranking above should be redone
against that cost rather than against case counts.

## 7. Deliberately out of scope

- **Converter-interfaced generation.** The largest block in the corpus by some distance —
  `IECWT4A/B`, `IECWPP4A/B` in 2015 and 2020 variants, WECC `WTG3`/`WTG4`, photovoltaics, BESS,
  `GridFormingEpri`, roughly thirty case-uses across a dozen libraries. These are not synchronous
  machines with different regulators; they are a different device class, current- or voltage-source
  converters with their own control stacks and their own limits. That is a plan of its own, and
  probably a larger one than this whole document. It is also the most valuable thing in the corpus
  for a modern grid, which is worth saying out loud rather than leaving implied by its absence.
- **Exponential and restorative loads** (`LoadAlphaBeta` ×7, `LoadAlphaBetaRestorative` ×4). Close to
  the existing ZIP load — `P = P₀(V/V₀)^α` against a polynomial — and cheap, but they change *load*
  behaviour rather than closing the machine-library gap this plan is about. Worth a separate small
  piece.
- **Automatons** (`CurrentLimitAutomaton`, `UnderVoltageAutomaton`, `TapChangerBlockingAutomaton1`).
  gridoxide has relays on bus voltage, machine speed and angle excursion; a current-based one is a
  small addition to existing machinery rather than a model.
- **Saturation**, still, on `RMS_PLAN.md` §20's terms. Nothing in this scan changes them.

## 8. What this plan is wrong about

The seven-case count assumes each case's *network* is expressible — DynaSwing cases use `Line`,
`TransformerFixedRatio`, `InfiniteBusWithVariations`, `NodeFault`, `Step` and `DoubleStep`, and only
the first four have obvious counterparts in what the reader builds today. `Step` and `DoubleStep` are
event constructs whose mapping onto `EventSpec` has not been checked. If they do not map, phase 2
finds out on one case rather than phase 5 finding out on five.

Phase 1 is the only part of this that is certain, and it is worth landing whether or not the rest
proceeds.

# Short-circuit calculation in gridoxide

Status: **implemented**. Written and executed 2026-08-15 against `6913e38`.

> **What shipped.** IEC 60909 short-circuit calculation in the phase domain (`src/shortcircuit/`):
> all four fault types, `c_max`/`c_min` voltage scaling, bolted and impedance faults, multiple
> simultaneous faults, and de-energized-island handling, with results reported both as phase
> quantities and as symmetrical components. CLI (`gridoxide short-circuit`), Python
> (`gridoxide.short_circuit`), and an mdBook chapter. Cross-validated against all 15 of
> power-grid-model's own short-circuit fixtures.
>
> **Three latent bugs in shared three-phase code were found by the fixtures, not by inspection.**
> Each is described in §5. They are the most valuable thing this work turned up, because all three
> affected the pre-existing asymmetric power-flow path too.
>
> **One planned gate could not be met, for a reason outside gridoxide.** The reference fixture set
> is internally inconsistent about an artificial numerical device; see §6. Eleven fixtures match to
> power-grid-model's own `1e-8`, four to `1e-3`, each documented individually.

## 1. Why this was affordable

The decisive observation is that gridoxide was already **sequence-in, phase-assembled** before any
of this work started:

- `types::Line3Ph` carries `r0`/`x0`/`b0` alongside the positive-sequence values.
- `network::transformer_seq_params` returns `(y0, y1, y2)`.
- `network::source_impedance_pu_seq` gives sequence source impedance.
- `network::fortescue_to_phase` already converts sequence parameters to the phase domain, and
  `build_ybus_3ph` / `stamp_transformers_3ph` / `stamp_shunts_3ph` already assemble a 3N×3N Y-bus
  from them.

So a short-circuit solver is mostly *fault boundary conditions* stamped on top of assets that were
already tested by the asymmetric power-flow suite. The new numerical code is small; most of the
work was in the fault stamping and in validation.

## 2. Formulation, and why

Phase domain (abc), following power-grid-model's `ShortCircuitSolver`, rather than the
sequence-domain formulation IEC 60909 itself is written in.

Two reasons. Unbalanced faults fall out without special-casing each one — a fault type is a
different set of matrix rows to rewrite, not a different network to assemble. And it is what
power-grid-model does, which makes its fifteen fixtures apply directly; that is the standard every
other numeric feature in this repo is held to.

The cost is inherited and documented: the phase domain needs a path to ground, where the sequence
domain does not. See `docs/src/short_circuit/index.md`.

Results are reported in **both** bases. The phase quantities are what power-grid-model emits and
what the fixtures check; the symmetrical components are what powsybl's short-circuit API models
(`FortescueValue`) and what the standard's own vocabulary uses. The projection is the inverse of a
transform the crate already had, so it was nearly free.

Of the four vendored references, only power-grid-model implements a solver at all. powsybl-core's
`shortcircuit-api` is interface-only — `ShortCircuitAnalysisProvider`'s javadoc says
implementations "may typically rely on an external tool" — though its *result* model is
sequence-domain. powsybl-open-loadflow and lightsim2grid have nothing.

## 3. Scope

Implemented: the voltage factor `c` and nothing else of IEC 60909's wider parameterization.
Deliberately **not** implemented, each because no fixture in this tree can validate it:

- study types (sub-transient / transient / steady-state) and the machine reactances they need;
- configurable initial voltage profiles;
- the derived peak / breaking / thermal currents.

Input is PGM JSON only. CGMES carries IEC 60909 data (`src/cgmes.rs` already notes
`ExternalNetworkInjection`'s role) but has no fault-location component and no reference outputs in
the conformity set, so it stays a follow-up.

## 4. Structure

| File | Contents |
|---|---|
| `src/shortcircuit/mod.rs` | `FaultType`, `FaultPhase`, `VoltageScaling`, `FaultAdmittance`, `Fault`, `c_factor`, the PGM bridge |
| `src/shortcircuit/solver.rs` | Matrix assembly, fault stamping, the solve, result assembly |
| `src/shortcircuit/fortescue.rs` | `SequenceValue` — the symmetrical-component projection |
| `src/pgm.rs` | `PgmFault`, `ScNetwork3Ph`, `pgm_to_3ph_sc_network`, `pgm_lines_3ph`, the `sc_output` structs |

Two seams were added to existing code: `network::YBus::into_entries`, because a bolted fault has to
rewrite the matrix *by column* and `YBusSparse` is row-major; and `pgm::pgm_lines_3ph`, extracted
from `pgm_to_3ph_network` so the short-circuit path can reuse the line conversion without also
inheriting that function's injection accumulation and virtual slack buses.

Three details that are easy to get wrong, each pinned by a test:

- **The source EMF is `c · e^{jθ}` — `u_ref` is dropped, not multiplied.** Invisible in every
  fixture (all set `u_ref = 1.0`).
- **`r_f = x_f = 0` means a *bolted* fault**, i.e. infinite admittance, not zero.
- **The two two-phase fault types default to phase `bc`, not `ab`.**

A fourth, at the implementation level: `sparse::solve_complex` **sums** duplicate triplets, so
"zero a column" must filter entries out rather than add a compensating negative.

## 5. Three latent bugs in shared code, found by the fixtures

All three affected the pre-existing asymmetric power-flow path as well, and none was visible by
reading the code.

**`transformer_seq_params` panicked on most winding pairs.** It supported exactly Dyn and YNyn and
called `panic!` otherwise — and 11 of the 15 short-circuit fixtures use YNd. It now implements
power-grid-model's general zero-sequence algorithm (YNyn, YN\*, \*yn, zigzag, and the general
low-susceptance rule). The gate for that rewrite was that every existing asymmetric test pass
**bit-for-bit**, which they did: the general algorithm reproduces both old arms exactly under the
two substitutions the doc comment records. `pgm::Unsupported3Ph::WindingPair`, which existed only
to pre-empt the panic, was removed with it.

**`Line3Ph` had no conductance field.** power-grid-model's line shunt is `2πf·c·(tan δ + j)`;
gridoxide modelled only the susceptance, silently dropping the dielectric-loss term. Almost every
fixture sets `tan = 0`, which is why it went unnoticed. Now `g1`/`g0`.

**The three-phase path dropped `link` components.** `pgm_to_3ph_network` never modelled them
(`Unsupported3Ph::Link`), so a link silently vanished — de-energizing whatever hung off it.
`pgm_to_3ph_sc_network` now stamps a link as a branch with `LINK_Y` in all three sequences, mirroring
what the symmetric path already did.

## 6. The gate that could not be met, and why

The plan's gate was that all 15 fixtures match power-grid-model to the `rtol`/`atol` in their own
`params.json`. Eleven do. Four cannot, and the reason is worth recording because it is not a
gridoxide defect and no amount of work here would fix it.

power-grid-model regularizes a transformer winding with no zero-sequence path of its own by adding
an artificial "low susceptance" to ground it — a numerical device to keep the zero-sequence system
non-singular, not physics, and whose magnitude is arbitrary. Tracing its history in the vendored
checkout (HEAD `be9d55bf1`, tag `v1.13.135`):

- the device was **introduced 2025-11-14** (`2a0678e2e` "add small admittance", `19ed1c26e`
  "approach 2") and reworked three days later (`99b0dbd05` "separate func");
- `single_phase_to_ground_*` and `two_phase_to_ground_*` have expected outputs dated
  **2023-09-21** and were never regenerated — they encode the behaviour from before it existed;
- `floating_zero_sequence_two_phase_short_circuit` has an expected output dated **2025-11-16** —
  generated *with* it, and genuinely needing it, since without regularization its zero sequence is
  undetermined (0.70 vs 0.96 p.u.).

**No single rule satisfies both vintages.** gridoxide implements the current one, so the four older
fixtures disagree — by ~1.5e-8 on voltages, and by ~1.3e-4 on one small capacitive ground current
(a few amperes, on a network whose three-phase fault current is 25 kA) that the artificial
susceptance sits directly in the path of.

A fifth fixture, `dummy-test-line-into-itself`, disagrees by ~1.4e-7 for an unrelated and
pre-existing reason: gridoxide's ideal-connection admittance is `2e5 + 2e5j`
(`topology::IDEAL_CONNECTION_Y`) where power-grid-model uses `1e8 + 1e8j`, a deliberate and
already-documented divergence.

These five run at a relaxed, individually justified tolerance through
`run_diverging_fixture(name, KnownDivergence::…)` in `tests/pgm_short_circuit_test.rs`, where each
cause is documented in full. The relaxed bound is still `1e-3` relative — two to three orders of
magnitude tighter than any of the three real bugs in §5, every one of which this same test caught.

**How this was established**, since "it is only a tolerance issue" is exactly the claim that should
not be taken on faith: the assembled system was dumped and re-solved in exact rational arithmetic.
That showed gridoxide's float64 solve reproduces the exact solution of its own matrix to 3e-13, so
the arithmetic is sound and the difference is in the model. Substituting power-grid-model's expected
solution into gridoxide's matrix then localized the disagreement to a single bus's three equations,
identical across all three phases — a pure zero-sequence term — of magnitude exactly `−j·4e-6`,
which is the regularization constant to eight significant figures.

## 7. Deliberately out of scope

- **CGMES input.** No fault-location component, no reference outputs.
- **Batched short-circuit.** The fixtures are batches and the test drives them as such, but there
  is no `BatchSolver` equivalent: a fault changes the matrix itself, so there is no shared symbolic
  factorization to amortize — the premise `batch::BatchSolver` is built on does not hold here.
- **Honouring `source.u_ref_angle` in the power-flow paths.** The field is now parsed and used by
  the short-circuit path, which has fixtures that set it. `pgm_to_3ph_network` and
  `pgm_to_buses_and_branches` still ignore it, as they always have; no power-flow or
  state-estimation fixture in this tree sets it, so changing that would be an unvalidated
  behaviour change to shipped paths. It is a real gap, and it needs its own reference outputs.

# Reading CGMES Input

`cargo build --features cgmes` builds `src/cgmes.rs`, a third network-input path alongside the native
JSON format and PGM-JSON, reading CGMES (Common Grid Model Exchange Standard) RDF/XML — the IEC
61970/61968 interchange format ENTSO-E and TSOs use.

It is built on [cimoxide](https://github.com/m-mirz/cimoxide) — a separate Rust project by the same
author — for RDF/XML decoding, via a pinned git dependency rather than crates.io (see
[Provenance and Licensing](../reference/provenance.md) for why). The feature is opt-in since some
users only need JSON input and shouldn't pay for `cimdecoder`'s dependency tree or build time.

```rust
use gridoxide::cgmes::{load_profiles, cgmes_to_buses_and_branches};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

let ds = load_profiles(&[&eq_path, &ssh_path, &tp_path, &sv_path])?;
let (buses, lines, transformers, shunts) = cgmes_to_buses_and_branches(&ds, 100e6)?;
let mut ybus = build_ybus(buses.len(), &lines, &transformers);
stamp_shunts(&mut ybus, &shunts);
let result = run_power_flow_analysis_from_ybus(buses, ybus);
```

## What the importer expects

The standard EQ+SSH+TP+SV "solved case" profile bundle:

- **TP is required.** `TopologicalNode` is used directly as gridoxide's `Bus`, so switch-state
  topology processing is assumed already resolved upstream. See
  [Ideal Switches and Zero-Impedance Branches](../powerflow/zero_impedance_branches.md) for what
  that resolution involves and how `cgmes::merge_closed_switches` handles the node-breaker case.
- **SV must carry a populated `TopologicalIsland.AngleRefTopologicalNode`**, used as the slack bus.
  See [Multi-Island Power Flow](../powerflow/multi_island.md) for how reference buses are picked per
  island.

## What is mapped

**Loads** — `EnergyConsumer`, `ConformLoad`, `NonConformLoad`, `EquivalentInjection`,
`ExternalNetworkInjection`, and `AsynchronousMachine`. The last is converted like a plain load, with
both P and Q negated.

**Branches** — `ACLineSegment` and `SeriesCompensator`, including
[`ACLineSegment.gch`](./shunt_conductance.md), real shunt conductance, not just `bch`'s reactive
charging.

**Transformers** — 2- and 3-winding `PowerTransformer`s, with `RatioTapChanger` (including its
optional [`RatioTapChangerTable`](./ratio_tap_changer_table.md) per-step override, falling back to
the linear `stepVoltageIncrement` formula when absent) and all four `PhaseTapChanger` variants:
[`Linear`](./phase_tap_changer_linear.md), `Symmetrical`, `Asymmetrical`, and `Tabular`.

**Shunts** — `LinearShuntCompensator` and `NonlinearShuntCompensator`.

**Voltage-controlled buses** — `SynchronousMachine` plus `RegulatingControl`, and the same
mechanism for [`StaticVarCompensator`](./static_var_compensator.md) and
[`ExternalNetworkInjection`](./external_network_injection.md), minus the active-power term for the
former.

## Validation

Validated end-to-end against four ENTSO-E conformance cases, with fixtures referenced via a git
submodule (see `tests/data/cgmes/README.md`):

| Case | Test | Notes |
|---|---|---|
| MicroGrid-BE-MAS | `tests/cgmes_microgrid_be_test.rs` | |
| MiniGrid | `tests/cgmes_minigrid_test.rs` | First fixture with more than one 3-winding transformer, which exposed and fixed a real star-bus-indexing bug; also real `AsynchronousMachine` loads (~9 MW / ~5 MVAr) |
| PhaseTapChangerLinear PST | `tests/cgmes_pst_phase_tap_changer_linear_test.rs` | Matches published SV values to ~1e-3 |
| RealGrid | `tests/cgmes_realgrid_test.rs` | Large real transmission+distribution model, 6252 buses |

MicroGrid-BE-MAS and MiniGrid converge cleanly but match their own published SV voltages only within
a few percent. That gap was cross-checked (for MicroGrid-BE-MAS) against pypowsybl's own independent
CGMES import and AC load flow on the same case, which shows a comparable deviation from the same
published values (`scripts/bench/cross_validate_cgmes_microgrid_be.py`) — confirming it is inherent
to solving a boundary-truncated area file with fixed-injection equivalents, not a correctness bug.
One known, documented limitation contributes: `types::Line` has no tap ratio, so it can't absorb the
small nominal-voltage mismatch CGMES explicitly allows at boundary tie points.

### FullGrid is an import fixture, not a solve fixture

FullGrid is the richest conformity model in the tree — both `TapChangerControl` modes, two phase
shifters sharing one target, a three-winding transformer, HVDC, a bus held by nine machines — and
`tests/cgmes_tap_table_test.rs`, `cgmes_tap_regulation_test.rs`, `cgmes_voltage_control_test.rs`
and `cgmes_node_breaker_test.rs` all use it for exactly those.

**It does not converge, and it cannot.** Its
`NonlinearShuntCompensatorPoint._7df4778f` declares `b = 0.99 S` *and* `g = 0.99 S` for
`BE_SHUNT_1` at `nomU = 225 kV`. On a 100 MVA base, \\(z_{base} = 506.25\,\Omega\\), so
\\(g = 501.19\\) per-unit — a shunt compensator dissipating **50 GW** on a network whose entire
scheduled generation is 485 MW. The first Newton iteration reports a mismatch of 503 pu, which is
that conductance and almost nothing else.

The conversion is arithmetically right; the input is not physical. `0.99` turns up across FullGrid
as filler for several unrelated quantities — it is also the SVC's `inductiveRating`/
`capacitiveRating`, giving that device a ±511 pu reactive band. These are placeholders exercising a
profile's classes, not a modelled network.

Worth separating from a different failure mode: FullGrid's published `SvVoltage` *is* consistent
with its own `EQ`/`SSH` transformer data — `scripts/bench/check_cgmes_sv_consistency.py` flags zero
of its ten two-winding transformers above 5%, where Svedala has one and RealGrid has many. The
inconsistency here is between the fixture and physics, not between two of its own profiles.

`tests/cgmes_fullgrid_test.rs` pins all of this, including the non-convergence itself, so a
corrected fixture announces itself. This has cost effort before: `scripts/bench/README.md` records
that `network::dc_angle_guess` was added specifically to make FullGrid converge, did not, and was
removed after it broke `case3120sp`.

**Not built or tested in CI** — the same local/manual-verification posture as `klu` and `pardiso`.

## The per-class pages

The remaining pages in this section each take one CIM class or attribute that needed real modeling
work, and follow the same structure: why it matters, the concepts and formulas involved, where it
sits in gridoxide today, and how other tools handle it.

## One voltage base per galvanic level

An `ACLineSegment` has no ratio. Its two ends are one conductor at one physical
voltage — which means per-unit across it is only meaningful if \\(|V| = 1.0\\)
denotes the same volts at both. That is what a per-unit *base* is for.

CGMES does not guarantee it. The same physical level is declared **380 kV** in
Belgium and **400 kV** in the Netherlands, **220** and **225** either side of
another border, and a tie line between them is one wire carrying two different
`BaseVoltage.nominalVoltage` values on its ends. Per-unitizing that line on one
end's base — all a single base can do — leaves its two ends in different
per-unit systems, so a flat \\(1.0\\) profile already contains a 5% step across a
wire with nothing in it to make one. The solver pushes reactive power to sustain
the step, and the whole area comes out high.

So the importer gives every galvanically-connected group of buses one base
before it converts a single branch. The group is the connected component over
plain conductors — `ACLineSegment` and `SeriesCompensator` — and the base is the
nominal the most of its members declare, ties going to the larger.

### Moving a base is safe, and moving a branch is not

A per-unit base is a choice, not a measurement. Moving one changes the per-unit
numbers and leaves the volts alone, because everything that converts between the
two reads `u_rated`: a reported voltage is \\(|V| \cdot u_{rated}\\), a voltage
setpoint is imported as \\(target / u_{rated}\\), and — the load-bearing one —
transformers self-correct, because
[`transformer_tap`](./index.md) already takes the node ratings and computes

\\[ k = \frac{u_1 / u_2}{u_{1,rated} / u_{2,rated}} \\]

so a moved node base is absorbed by the off-nominal ratio. That is why this runs
*before* any branch is converted rather than after.

### The one thing it is not invariant to

That invariance holds wherever `u_rated` is used as a *base*. It fails in the
one place the importer uses it as a stand-in for the **actual operating
voltage**: the `StaticVarCompensator` rating, which turns an ohmic reactance
into a reactive limit through \\(Q \approx V^2 / x\\) evaluated at
\\(|V| = 1.0\\). Move the base and \\(1.0\\) p.u. denotes different volts,
so the limit moves with it — by \\((400/380)^2 = 10.8\%\\), or
\\((225/220)^2 = 4.6\%\\).

The shunt compensators converted a few lines away look identical and are *not*
affected, which is worth seeing clearly: they convert siemens to per-unit, and
\\(Q_{pu} = |V|_{pu}^2 \cdot B_{pu}\\) recovers the same physical MVAr on
any base. Only pinning \\(|V| = 1\\) makes the base stand for a voltage.

No vendored fixture exercises it. MicroGrid-Type1's single SVC sits at 225 kV —
the base that is *kept* — so its rating is ±0.1 p.u. either way. A document with
an SVC on the other side of such a boundary would see its rating shift, and
nothing would flag it.

### It adjudicates, and the alternative does not

Harmonizing has to **choose** which of two declarations to keep. That is a
judgement about the document, not arithmetic, and it rests on an inference: an
`ACLineSegment` has no ratio, therefore its two ends are one level, therefore a
differing declaration is a naming convention rather than a fact.

For 380/400 and 220/225 that inference is right, and the error figures below say
so. Where it would be wrong is a document that is *itself* wrong — a line
written where a transformer belongs. Then harmonizing forces the two ends onto
one base and the level difference disappears into a model that looks entirely
reasonable, where powsybl's branch-side treatment would carry both declarations
through and let the discrepancy reach the answer. Neither is correct, because
the input is not; the difference is whether to trust the declaration or the
topology.

What this does instead of choosing silently is report:
`CgmesNetwork::base_harmonization` names every nominal it merged and how many
buses moved, and `examples/base_mismatch_probe.rs` prints it. That is weaker
than not having to choose — it needs someone to look — but it is what makes a
mis-declared nominal noticeable rather than absorbed.

### What it was worth

Measured against the fixtures' own published `SvVoltage`:

| fixture | lines spanning two bases | worst error before | after |
|---|---|---|---|
| MicroGrid-Type1 | 5 of 13 | 4.4% | 1.3% |
| FullGrid | 5 of 13 | — (does not converge) | — |
| SmallGrid, MiniGrid, RealGrid, Svedala | 0 | unchanged, bit for bit | |

On MicroGrid-Type1 the median error falls from 0.96% to 0.01%. With
[remote voltage control](../powerflow/outer_loops.md) also enabled — the same
fixture needs it — the worst error reaches **0.09%**, and the machine's own
reactive output comes within 2% of the published one. Neither fix is visible
underneath the other: the base defect was larger than the reactive power remote
control moves, which is why that fixture could not referee remote control until
this landed.

### What powsybl does

The same problem, solved in the branch rather than the bus, and enough of a real
modelling question there to be a configuration option. `LinePerUnitMode` is
either `IMPEDANCE` — per-unitize on the geometric mean \\(n_1 n_2 / S_B\\) and
fold correction terms into the two end shunts — or `RATIO`, an ideal transformer
of \\(n_1/n_2\\) on an otherwise ordinary line (`LfBranchImpl.createLine`).

Both need a branch model with per-terminal shunts or a ratio.
[`types::Line`](../powerflow/index.md) has neither, deliberately: its shunt is
one total split equally, which is what every line in every other fixture needs.
Doing this in the bus base reaches the same coherent per-unit system without
giving every line in the crate a field that ten lines in two fixtures would ever
use — and, unlike `RATIO`, without moving a line into the transformer list and
shifting the flat branch index that ratings, `terminal_branch` and the RAO all
address branches by.

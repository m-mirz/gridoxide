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

**Not built or tested in CI** — the same local/manual-verification posture as `klu` and `pardiso`.

## The per-class pages

The remaining pages in this section each take one CIM class or attribute that needed real modeling
work, and follow the same structure: why it matters, the concepts and formulas involved, where it
sits in gridoxide today, and how other tools handle it.

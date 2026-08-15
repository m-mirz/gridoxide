*SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>*  
*SPDX-License-Identifier: MPL-2.0*

*Adopted from Power Grid Model `tests/data/short_circuit`, at `be9d55bf1` (tag
`v1.13.135`). Only the `.json` files are copied; the `.license` sidecars are
dropped, since this file carries the same attribution for all of them.*

---

# Short-circuit validation cases

The reference outputs for `tests/pgm_short_circuit_test.rs`. Between them they
cover all four fault types, both IEC 60909 voltage-scaling choices, bolted and
impedance faults, multiple simultaneous faults, and two degenerate topologies.

| Fixture | What it pins down |
|---|---|
| `three_phase_c_{maximum,minimum}` | The balanced case, both `c` factors |
| `single_node_source_three_phase_c_{maximum,minimum}` | A source alone on a node — fault current set purely by source impedance and `c` |
| `single_phase_to_ground_c_{maximum,minimum}` | The commonest fault in practice; excites the zero sequence |
| `branch_source_single_phase_to_ground_c_maximum` | Single-phase-to-ground with no transformer in the path |
| `two_phase_c_{maximum,minimum}` | Phase-to-phase, clear of ground — no zero-sequence component at all |
| `two_phase_to_ground_c_{maximum,minimum}` | Phase-to-phase *and* to ground |
| `floating_zero_sequence_two_phase_short_circuit` | A node that is delta-wound on both sides, so its zero sequence has no path to ground |
| `multiple_short_circuits_same_subgrid` | Two faults on one bus — the current has to divide between them |
| `multiple_short_circuits_different_subgrids` | Two faults in electrically separate subgrids, one of them de-energized |
| `dummy-test-line-into-itself` | A line with both terminals on one node, plus a `link` and nonzero `tan δ` |

Each directory holds `input.json`, `params.json` (which names the voltage
scaling and the tolerances), and either `sc_output.json` or the batch pair
`update_batch.json` + `sc_output_batch.json`.

## A note on two fixtures that cannot be matched exactly

Eleven of the fifteen are reproduced to power-grid-model's own `1e-8`. Four are
not, for reasons that are documented in full on `KnownDivergence` in
`tests/pgm_short_circuit_test.rs` and summarised here:

- **`single_phase_to_ground_*` and `two_phase_to_ground_*`.** power-grid-model
  regularizes a transformer winding that has no zero-sequence path of its own
  by adding an artificial "low susceptance" to ground it. That device was
  introduced on **2025-11-14**; these four fixtures' expected outputs date from
  **2023-09-21** and were never regenerated, so they encode the behaviour from
  before it existed. `floating_zero_sequence_two_phase_short_circuit`, whose
  expected output is dated **2025-11-16**, encodes the behaviour *with* it — and
  needs it, since without regularization its zero sequence is undetermined.

  No single rule satisfies both vintages. gridoxide implements the current one,
  so the four older fixtures disagree by ~1.5e-8 on voltages, and by ~1.3e-4 on
  one small capacitive ground current (a few amperes, on a network whose
  three-phase fault current is 25 kA) that the artificial susceptance sits
  directly in the path of.

- **`dummy-test-line-into-itself`.** A `link` is an ideal connection, which a
  nodal formulation has to give some large-but-finite admittance.
  power-grid-model uses `1e8 + 1e8j`; gridoxide uses `2e5 + 2e5j`
  (`topology::IDEAL_CONNECTION_Y`), a pre-existing and deliberate choice. The
  two put slightly different voltage drops across the link — ~1.4e-7 relative,
  at the node on its far side only.

Both are differences of convention about quantities that are not physics.
Neither is a disagreement about a short-circuit current anyone would measure.

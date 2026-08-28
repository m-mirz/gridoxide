# Kundur Example 13 — Dynawo's own case, and its own answer

Copied verbatim from [dynawo/dynawo](https://github.com/dynawo/dynawo),
`examples/DynaSwing/Kundur_Example13/`. Copyright (c) 2015-2019, RTE
(http://www.rte-france.com), **MPL-2.0**; SPDX-License-Identifier: MPL-2.0.

| File | What it is |
|---|---|
| `KundurExample13_SetPoint.dyd` | The architecture: infinite bus, two parallel lines, a fixed-ratio transformer, one four-windings machine with **no regulators**, a node fault and a line opening |
| `KundurExample13.par` | Every parameter, including the machine's and the operating point |
| `KundurExample13.crv` | What the case records |
| `reference_setpoint.csv` | **Dynawo's own solver's answer**, from `reference/outputs_SetPoint/curves/curves.csv` |

## Why this case

It is the one published Dynawo case in which **every element is something
gridoxide models exactly**: a sixth-order machine with constant field voltage
and constant mechanical power, an infinite bus (which `SystemSpec::fixed_buses`
represents exactly rather than approximating), pure series reactances, a bolted
node fault, and a branch trip. No tap changers, no limits, no saturation, no
model this library lacks. Nothing has to be excused before the comparison
starts.

It is Kundur's *Power System Stability and Control* Example 13.2, and the
operating point can be checked by hand against the book: `P = 0.9` pu and
`E_t = 1.0∠28.34°` on the machine's 2220 MVA base.

## What the reference file already confirms, before any simulation

- `SM_generator_theta` at `t = 0` is **1.224004** rad. gridoxide's own
  initialization, from an entirely independent derivation, puts `δ₀` there too.
  That is the dq convention checked against a second implementation.
- `SM_generator_PmPu` is **0.903** while `PGenPu` is 0.900 on the machine base.
  The difference is exactly the `0.003` pu stator copper loss — which is the
  statement that Dynawo's mechanical power, like gridoxide's, is the **air-gap**
  power and not the terminal power.

## The one thing the file does not carry

Dynawo's solver is IDA at order 2 with `1e-4` accuracy and a variable step;
gridoxide integrates at a fixed step with the trapezoidal rule. The two are
different methods converging on the same trajectory, so the comparison is
between two approximations, not against truth.

# Reading MATPOWER Input

MATPOWER's case format is the one the standard transmission test cases are
published in: the IEEE cases, PEGASE, RTE, PGLib. A case file is a MATLAB
function that assigns literal matrices to `mpc.baseMVA`, `mpc.bus`, `mpc.gen`
and `mpc.branch`, and those four are what `src/matpower.rs` reads.

```rust
let net = gridoxide::matpower::read("case118.m")?;
let report = gridoxide::run_power_flow(
    net.buses, &net.lines, &net.transformers, &net.shunts,
    gridoxide::TapData::none(),
    PowerFlowOptions { init: PowerFlowInit::Flat, tol: 1e-8, ..Default::default() },
);
```

```python
model = gridoxide.PowerFlowModel.from_matpower("case118.m", tol=1e-8)
model.solve(init="flat")
voltages = dict(zip(model.bus_numbers(), model.voltage_mag()))
```

From the command line, `gridoxide solve case118.m`. Behind the `matpower`
feature, which has no dependency; the Python bindings always include it.

## The problem it solves is MATPOWER's

The import follows `runpf`'s rules, from MATPOWER's `bustypes`, `ext2int` and
`makeSbus`/`makeYbus`, so that a case means here what it means there:

| What the case says | What is solved |
|---|---|
| Bus type 3 or 2 with an online generator | Reference or PV bus |
| Bus type 3 or 2 without one | PQ bus; if no reference remains, the first PV bus becomes it |
| A generator on a PQ bus | A fixed `Pg + jQg` injection |
| The voltage a regulated bus holds | Its generators' `Vg` (the last online one), never the bus's `Vm`, which MATPOWER reads only as a starting value |
| The reference angle | The reference bus's `Va` |
| Bus type 4 (isolated) | Left out with its branches and generators; reported at zero voltage |
| A branch with a ratio or a phase shift | A two-winding transformer: the π-model with the complex ratio `τ·e^{jθ}` on the from side, exactly `network::branch_calc_param`'s model |
| `Gs`, `Bs` | A shunt admittance, per unit |

Everything is per unit on the case's own `baseMVA`, so a mismatch tolerance is
the same quantity MATPOWER measures. A PV bus also carries the summed `Qmin`
and `Qmax` of its generators, used only if reactive limits are asked for.

## Why not the PGM conversion

`gridoxide.matpower` (the Python package's converter) turns a case into PGM
JSON, and that cannot carry a MATPOWER case exactly. PGM has no slack bus, only
a `source` behind a short-circuit impedance, so the reference bus sits behind an
impedance the case does not have and drifts off its setpoint by however much
current it carries. PGM's transformer `clock` also holds a phase shift only in
multiples of 30°, so the PEGASE and RTE phase shifters are rounded. Read
directly, neither loss happens.

## What it refuses

A case file is MATLAB code, and some cases finish with code that edits the
matrices they just wrote: `case33bw` converts its loads from kW and its
impedances from ohms that way. Reading the literals and skipping the code would
solve a network 1000 times more heavily loaded, without a word. So an
assignment to `mpc.bus`, `mpc.gen`, `mpc.branch` or `mpc.baseMVA` that is not a
literal matrix is an error, and so is a version-1 case or a binary `.mat` file.
benchmark-grids keeps plain-data copies of such cases under `matpower-plain/`.

## How it is tested

`tests/matpower_test.rs` reads the transmission cases from case14 to
case9241pegase and six distribution feeders, solves each from a flat start, and
substitutes the solution into the power-flow equations rebuilt from the raw
matrices with MATPOWER's own formulas. Every case lands below 1e-7 p.u., and
flipping the sign of the phase shift in the importer fails case1354pegase by
0.44 p.u.

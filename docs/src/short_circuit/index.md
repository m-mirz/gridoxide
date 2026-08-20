# The Short-Circuit Problem

## Motivation

[Power flow](../powerflow/index.md) and [state estimation](../state_estimation/index.md) both
describe a network that is *working*. Short-circuit analysis describes one that has just failed:
a conductor has touched ground, or two phases have touched each other, and for the next few cycles
an enormous current flows through whatever path the impedances allow.

The number that comes out — the **initial symmetrical short-circuit current** \\(I_k''\\) — is one of
the most consequential in power engineering. A circuit breaker is chosen because it can interrupt
it. A conductor is sized because it can survive it for the fraction of a second before protection
clears. A protective relay is set because it can *distinguish* it from load. Get it wrong in one
direction and equipment explodes; wrong in the other and faults go undetected.

## What makes this not a power flow

Three things, and all three are deliberate simplifications that
[IEC 60909](https://webstore.iec.ch/publication/24100) makes so the answer does not depend on the
operating point:

**Loads and generation are ignored entirely.** Not linearized, not approximated — dropped. A
short-circuit current is a property of the network's *impedances*, and the standard isolates it
from whatever the dispatch happened to be that afternoon. This is the assumption that makes a
short-circuit study a property of the grid rather than of a moment. Shunt admittances are kept:
a shunt is a passive element, not a dispatch decision.

**Sources sit behind a fixed EMF**, scaled by a voltage factor \\(c\\) rather than by their own
setpoint — see below.

**The problem is linear.** With every load gone, what remains is a linear circuit: impedances,
fixed voltage sources, and a fault. One sparse factorization, no initial guess, no iteration
count, no convergence to fail — the same character as
[the constant-admittance linearization](../powerflow/linear_impedance.md), and for the same
reason. A short-circuit calculation either has a solvable network or it does not.

## The voltage factor `c`

Rather than model the pre-fault voltage profile, IEC 60909 replaces every source EMF with
\\(c \cdot U_n / \sqrt{3}\\), where \\(c\\) is a single tabulated factor:

| | \\(c_{max}\\) | \\(c_{min}\\) |
|---|---|---|
| \\(U_n \le 1\\) kV | 1.10 | 0.95 |
| \\(U_n > 1\\) kV | 1.10 | 1.00 |

The two columns are **two different studies**, not two guesses at one answer:

- \\(c_{max}\\) gives the largest credible fault current. This is what breaker ratings and thermal
  withstand are sized against.
- \\(c_{min}\\) gives the smallest. This is the protection-sensitivity study: if the relay can still
  see a fault at the end of the feeder under the least favourable conditions, it will always see it.

gridoxide exposes the choice as `VoltageScaling::{Maximum, Minimum}` and nothing else — see
[Deliberate omissions](#deliberate-omissions). The standard distinguishes a 6% and a 10%
low-voltage tolerance for \\(c_{min}\\); like power-grid-model, gridoxide assumes 10%.

One detail worth stating because it is easy to get wrong: the factor **replaces** the source's
`u_ref` rather than multiplying it. A source declared at 1.02 p.u. contributes \\(c\\), not
\\(1.02c\\). That is precisely what isolates the result from the operating point, and it is
invisible in every reference fixture because they all set `u_ref = 1.0`.

## Formulation: the phase domain

Two formulations are in common use, and they are not approximations of each other — they are the
same problem in different coordinates.

**The sequence domain (0-1-2)** is what IEC 60909 itself is written in, and what most textbooks
teach. Build three decoupled networks — zero, positive, negative — reduce each to a Thévenin
equivalent at the fault point, and connect them in the pattern the fault type dictates: positive
alone for a three-phase fault, all three in series for single-phase-to-ground, and so on.

**The phase domain (abc)** keeps the full \\(3N \times 3N\\) nodal system and writes the fault's
boundary conditions directly into it, rewriting the rows and columns at the faulted bus:

\\[ I_N = Y_{bus} \, U_N \\]

**gridoxide solves in the phase domain.** Two reasons. First, unbalanced faults fall out without
special-casing each one — a fault type is a different set of rows to rewrite, not a different
network to assemble. Second, and decisively for this project, it is what power-grid-model does, so
its fifteen short-circuit fixtures cross-validate gridoxide's numerics directly. That is the same
standard every other numeric feature here is held to.

Results are then *reported* in both bases: phase quantities, and symmetrical components via the
inverse Fortescue transform. The sequence view is where a fault type's signature is legible at a
glance, which is most of why it is worth having — see [Fault Types](./faults.md).

Both formulations are written out in
[Symmetrical Components and the Fault Equations](./sequence.md), and
[A Fault Current, by Hand](./worked_example.md) computes three of them on a one-node system where
every step is a calculator away.

### The cost of that choice

The phase domain needs the network to have a path to ground. An ungrounded network has a singular
zero-sequence system in the phase domain — it does not in the sequence domain, where the
zero-sequence network is simply absent — and comes back as `ShortCircuitError::Singular`.

This is inherited from the formulation, and power-grid-model documents the same limitation for
itself. In practice a real network is grounded somewhere; the case that bites is a small test
model with a delta winding on both sides of a node. gridoxide follows power-grid-model in adding a
tiny artificial susceptance to ground a winding that has no zero-sequence path of its own, purely
so the matrix stays non-singular. It is a numerical device, not physics, and its exact value is
arbitrary — a point that matters when comparing against reference data, as
`tests/data/pgm/short_circuit/README.md` records in detail.

## Islands

A bus with no path to any source cannot carry a fault current, so a fault declared there returns
zero and the bus is reported de-energized and pinned at zero volts. The energized part of the
network is solved on its own.

This is the same contract every other solver in gridoxide offers — see
[Multi-Island Power Flow](../powerflow/multi_island.md) — and it exists for the same reason:
a sourceless component has no reference, and inventing one would be guessing.

## Deliberate omissions

gridoxide implements the voltage factor \\(c\\) and nothing else of IEC 60909's wider
parameterization. In particular it does **not** model:

- **Study type** (sub-transient / transient / steady-state) and the associated machine reactances.
  powsybl's short-circuit API models these; gridoxide does not read the machine data they need.
- **Configurable initial voltage profiles** — user-supplied voltage ranges, or seeding from a
  previously solved power flow.
- **Peak, breaking and thermal-equivalent currents** (\\(i_p\\), \\(I_b\\), \\(I_{th}\\)), which are
  derived from \\(I_k''\\) by further standard factors.

Each is a real capability elsewhere. None is implemented here because no fixture in this tree can
validate it, and this project's rule is that a number it prints is a number something checked.

## Using it

From the CLI:

```bash
gridoxide short-circuit network.json --scaling max
```

From Rust:

```rust
use gridoxide::shortcircuit::{short_circuit_from_pgm, ShortCircuitOptions, VoltageScaling};

let opts = ShortCircuitOptions { scaling: VoltageScaling::Maximum };
let (net, report) = short_circuit_from_pgm(&input, 1e6, 50.0, opts)?;
for fault in &report.faults {
    println!("fault {}: {:.1} A on phase a", fault.id, fault.i_f[0]);
}
```

From Python:

```python
import gridoxide

result = gridoxide.short_circuit("network.json", scaling="max")
for fault in result.faults:
    print(fault.id, fault.i_f)
```

The fault itself is declared in the input document, as power-grid-model's `fault` component: a
node, a fault type, an optional phase, and an optional fault impedance. **A `fault` with no
`r_f`/`x_f` is a bolted fault** — a dead short — not one that draws no current.

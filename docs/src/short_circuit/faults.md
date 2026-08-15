# Fault Types and Their Boundary Conditions

Four fault situations are distinguished, and they differ from each other only in what is
constrained at the faulted bus. Everything else — the network, the sources, the voltage factor —
is identical. This page is about those constraints and how to read the result.

## The four types

| Type | Phases | Rough frequency in practice | Zero sequence? |
|---|---|---|---|
| Three-phase | all | rare | no (balanced) |
| Single-phase-to-ground | one | the large majority | yes |
| Two-phase | two, clear of ground | uncommon | **no** |
| Two-phase-to-ground | two, plus ground | uncommon | yes |

The three-phase fault is the balanced one and usually — though not always — gives the largest
current, which is why it is the default sizing case. The single-phase-to-ground fault is by a wide
margin the most *common* in real systems, and it is the one whose magnitude depends most on
transformer winding configuration, because its current has to return through the zero-sequence
network.

## Default phases

A fault may name its phases explicitly (`a`, `b`, `c`, `ab`, `ac`, `bc`) or leave it to the type:

| Type | Default |
|---|---|
| Three-phase | `abc` |
| Single-phase-to-ground | `a` |
| Two-phase | **`bc`** |
| Two-phase-to-ground | **`bc`** |

The two two-phase types default to `bc`, not `ab`. This follows power-grid-model, and it is worth
stating loudly because getting it wrong produces an entirely plausible magnitude on the wrong pair
of conductors — the kind of error that survives a casual review.

## Bolted versus impedance faults

A **bolted** fault is a dead short: zero impedance, therefore infinite admittance. An **impedance**
fault goes through some finite \\(Z_f\\) — an arc, a tree, a resistive earth path.

The two are not the same formula evaluated at a limit, and gridoxide does not treat them that way.
An impedance fault *adds* an admittance to the faulted bus's diagonal. A bolted fault **replaces**
that bus's equations: its voltage is known to be zero, so the unknown in that position becomes the
injected current instead. Letting an infinity into the matrix would produce `NaN`, not a large
current.

The practical consequence for input data: **`r_f = x_f = 0` means bolted**, and that is the
default when a `fault` omits them. A fault that draws no current is not something the component can
express, because it would not be a fault.

## Reading the symmetrical components

Every result carries the symmetrical-component view alongside the phase quantities, and it is the
fastest way to tell whether an answer is sane:

- **Three-phase** — positive sequence only. Zero and negative should be numerical noise. If they
  are not, either the network is genuinely unbalanced or something is wrong.
- **Single-phase-to-ground** — all three components present, comparable in size.
- **Two-phase** (clear of ground) — **no zero-sequence component at all.** With no ground return
  there is nowhere for zero-sequence current to go. This is the single sharpest check available on
  a short-circuit result.
- **Two-phase-to-ground** — all three present.

A second useful check: a delta winding blocks zero-sequence current. If a node sits behind one and
still shows zero-sequence voltage, the transformer's zero-sequence branch is wrong.

Both properties are asserted directly in `python/tests/test_short_circuit.py`.

## Multiple simultaneous faults

Several faults may be declared at once, on the same bus or on different ones, subject to one
restriction: **they must all share a fault type and phase.** The boundary conditions of different
types rewrite the same matrix entries in incompatible ways, so "one of each, at once" is not a
well-posed linear system. Mixing them returns `ShortCircuitError::MixedFaultTypes` rather than a
plausible-looking answer.

Two faults on one bus divide the current between them. A bolted fault short-circuits its bus
outright, so any other fault sharing that bus — bolted or not — is bypassed and carries nothing.

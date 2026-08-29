# Kundur's Example 13.2, Against Dynawo

Every other check in this module is self-supporting: a closed form, an identity, a reduction, an
oracle. Those catch a great deal, but there is one thing they cannot catch — a formulation that is
internally consistent and physically wrong. For that you need a second implementation.

## The case, and why this one

Kundur's *Power System Stability and Control* Example 13.2, as
[Dynawo](https://github.com/dynawo/dynawo) ships it: a sixth-order machine with **no regulators**, an
infinite bus, two parallel lines, a fixed-ratio transformer, a bolted fault at the line junction and
a line trip on clearing.

It was chosen because every element is something gridoxide models *exactly* — no tap changers, no
limits, no saturation, no fifth-order machine. Nothing has to be excused before the comparison
starts, which is what makes a disagreement mean something.

Dynawo's repository ships its own solver's reference curves alongside the case, so the comparison
needs **no Dynawo install** — the same arrangement `tests/data/ucte/` and `tests/data/iidm/` already
use for pypowsybl.

## What the reference file settles before a single step

Two things, checked first because a disagreement in either would make the rest meaningless:

- **`theta` at `t = 0` is 1.224004 rad.** gridoxide's own initialization, from an entirely
  independent derivation, puts `δ₀` within `6e-5` of it. That is the rotor-frame convention agreeing
  with a second implementation — much stronger evidence than any self-consistency check.
- **`PmPu` is 0.903 while `PGenPu` is 0.900** on the machine base. The difference is exactly the
  0.003 pu stator copper loss, which is the statement that Dynawo's mechanical power, like
  gridoxide's, is the **air-gap** power.

The operating point is checkable by hand against the book, too: `P = 0.9` pu and
`E_t = 1.0∠28.34°`, and gridoxide's power flow reproduces the file's stated terminal angle of
0.49445 rad to `1e-4`.

## The agreement

| | |
|---|---|
| Rotor angle through the fault | 9.1e-4 rad |
| Rotor speed through the fault | 6.6e-5 pu |
| Terminal voltage during the fault | 9.0e-4 pu |
| Rotor angle over the first swing | 1.9e-2 rad |
| The published 70 ms clearing time | survivable in both; 500 ms in neither |

## The disagreement, and what it turned out to be

After clearing the two separate steadily. That was chased rather than absorbed into a tolerance, and
what it is *not* was established first, each by experiment:

- **not step size** — gridoxide's answer is converged to `1e-4` rad between `h = 4 ms` and
  `h = 0.0625 ms`;
- **not the nominal frequency** — at 60 Hz the machine loses synchronism outright;
- **not the tripped branch** — tripping the other line, or neither, gives a wholly different
  trajectory;
- **not the sign of the `q`-axis damper coupling** that the model's derivation left open — flipping
  it moves the answer by under `1e-3` rad, which also means this case does not settle that question
  either way.

Localizing it needed separating the power-angle relation from the accumulated angle. At matched
rotor angle in the first tenth of a second after clearing, gridoxide's terminal power is **0.6%
below** Dynawo's, and the voltage error *steps* at the clearing instant rather than growing
smoothly. So it is algebraic, not numerical.

Reading Dynawo's own Modelica settled it:

```text
udPu = (Ra + RTfo)·idPu − omegaPu·lambdaqPu
uqPu = (Ra + RTfo)·iqPu + omegaPu·lambdadPu
2·H·der(omegaPu) = cmPu·PNomTurb/SNom − cePu − DPu·(omegaPu − omegaRefPu)
PePu = cePu·omegaPu
```

Dynawo keeps `ω` on the speed-voltage terms and writes the swing equation in **torque**. gridoxide
makes the classical RMS approximation in both places: `ω ≈ 1` in the stator, and the swing equation
in power. The two differ by exactly a factor of `ω`, so they agree at synchronous speed and part in
proportion to the speed deviation — which is the observed pattern exactly: nil at `t = 0`, 0.04%
early in the fault where `ω − 1 = 0.0015`, and 0.6% after clearing where `ω − 1 = 0.009`.

**Neither form is wrong.** Kundur §13.3 states the approximation explicitly; Sauer & Pai keep the
terms. What changed is that the cost is now *measured* — 0.6% of terminal power per 0.9% of speed
deviation — recorded where the equations are, and pinned by a gate, so that adopting the full form
would visibly drive it to zero rather than pass unnoticed.

This is what an external gate is for: a difference invisible to every self-consistency check,
because both formulations are internally perfect.

## What is not yet done

A second external reference. ANDES is installed and its power flow reproduces the case — once its
lines are given matching `Vn1`/`Vn2`, since it defaults them to 110 kV and silently rescales every
impedance — but its own initialization then fails on the hand-built case, with residuals of 0.16 in
the bus-3 angle equation, and the machine loses synchronism where both other tools keep it. That is
a setup problem rather than a finding, and it is recorded as outstanding rather than reported as a
result.

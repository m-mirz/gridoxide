# Small-Signal Analysis

## A different question

A [time-domain run](./index.md) answers "what happens after *this* disturbance". Small-signal
analysis answers one no single run can: **what dynamic character does this system have at all?**
Which oscillations exist, how fast each decays, and — the part that makes it actionable — which
machines are taking part in each.

It is also the only reliable way to find a *negatively damped* mode. A run finds one if the
disturbance happens to excite it; an eigenvalue finds it whether anything excited it or not.

## The reduction

Linearize the DAE about an equilibrium:

```text
Δẋ = A_x·Δx + A_v·ΔV
0   = C_x·Δx + C_v·ΔV
```

The algebraic half carries no derivative, so it is a *constraint* on `ΔV` rather than a state of its
own — which means it can be eliminated outright:

\\[ \Delta V = -C_v^{-1} C_x \Delta x, \qquad A = A_x - A_v C_v^{-1} C_x \\]

That reduced `A` is the whole dynamic system, of dimension the differential states alone. The
network contributes to every mode without contributing a single one of its own — which is the same
statement, seen from a different angle, as the one that makes this an RMS simulation rather than an
EMT one.

## The four blocks are not new code

They are the very blocks [`DaePattern::fill`](./dae.md) already assembles for every Newton iteration
of every step, read out at `h·a = 1`:

```text
⎡ I − A_x   −A_v ⎤     so   A_x = I − (top-left),   A_v = −(top-right)
⎣   C_x      C_v ⎦          C_x and C_v as they stand
```

That is worth more than the saved code. Two hand-written derivations of the same Jacobian would be
two things to keep in step, and a small-signal analysis that had quietly drifted from the simulation
it describes would be worse than none. There is one Jacobian, and it is the one already checked
against a finite-difference oracle for every model in the library.

## Participation factors

An eigenvalue says a mode exists. A participation factor says *whose* it is:

\\[ p_{ki} = \frac{|u_{ki}\,w_{ik}|}{\sum_j |u_{ji}\,w_{ij}|} \\]

with `u` the right eigenvectors and `w = u^{-1}` the left ones. The product is dimensionless and
scale-free, which is what lets a rotor angle in radians and a field voltage in per unit be compared
at all.

This is the output that gets used. A poorly damped mode at 0.8 Hz with 90% participation from two
machines' speeds is an inter-area oscillation between them, and it names the machines to tune.

## What it looks like

```text
$ gridoxide dynamics case.json --modes 6

13 mode(s) over 13 differential state(s)

                eigenvalue    damping   freq (Hz)    tau (s)   participation
    -0.73018     +7.99385j     0.0910      1.2723      1.370   G1.omega 41%, G1.delta 36%, G1.eqp 6%
    -0.73018     -7.99385j     0.0910      1.2723      1.370   G1.omega 41%, G1.delta 36%, G1.eqp 6%
    -1.02972     +2.23736j     0.4181      0.3561      0.971   G1.avr_lead 37%, G1.eqp 34%, G1.delta 7%
    -1.02972     -2.23736j     0.4181      0.3561      0.971   G1.avr_lead 37%, G1.eqp 34%, G1.delta 7%
   -40.55915     +5.54508j     0.9908      0.8825      0.025   G1.psi1d 39%, G1.pss_lead2 19%, G1.pss_lead1 19%
   -40.55915     -5.54508j     0.9908      0.8825      0.025   G1.psi1d 39%, G1.pss_lead2 19%, G1.pss_lead1 19%

every mode decays
```

Three distinct physical mechanisms, each named by the states it belongs to: the electromechanical
swing at 1.27 Hz on the rotor, an excitation-and-field-flux mode at 0.36 Hz, and a fast mode on the
damper winding and the stabilizer's lead stages. None of that has to be inferred from a trajectory.

## Gates

The load-bearing one is that the eigenvalue agrees with the **closed form** the time-domain run is
independently checked against. For an undamped classical machine against an infinite bus the swing
mode is exactly

\\[ \lambda = \pm j\sqrt{\Omega_b K_s / 2H}, \qquad K_s = P_{max}\cos\delta_0 \\]

and both halves are checked: the frequency, and that the real part is zero, since nothing dissipates.

Two more compare against the run itself, through almost no shared code — one integrates a trajectory
and counts zero crossings, the other assembles a Jacobian, eliminates the network and takes an
eigenvalue:

- the predicted period matches the observed one to `2e-3` relative;
- the real part predicts the decay: successive peak amplitudes fall as `exp(σ·Δt)`, to 2%.

And a fourth-order machine's modes separate the way they should: the swing mode is over 80% the
rotor's, and the slowest real mode is the field flux with a time constant of the order of `T'_d0`.

Analysing a point the system is not sitting at is **refused**. A linearization about a mid-transient
state describes nothing in particular, and its eigenvalues would look entirely plausible.

## Scope

Dense: the reduced `A` is eigen-decomposed whole, which is `O(n³)` and is the normal approach for
this analysis. A system of a few thousand states wants a sparse Arnoldi method targeting a region of
the complex plane instead, and that is not implemented.

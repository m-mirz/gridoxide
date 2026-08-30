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

## Mode shape

A participation factor says *whose* a mode is. It cannot say **how they move**, and for an
oscillation that is the more useful half. The mode shape is the right eigenvector read at the rotor
angles, rotated and scaled so the largest component is `1∠0`:

```text
G1.delta 1.00∠+0°, G2.delta 1.00∠-180°
```

Two rotors near opposite phase are swinging *against* each other — an inter-machine or inter-area
mode. Two near the same phase are riding together, which is a different phenomenon needing a
different remedy.

Participation cannot make that distinction. On two islanded machines it says both rotor modes belong
to the rotors, which is true of both and useful about neither; the shape separates the oscillation
from the free drift of the island's absolute angle, and that drift is reported with no shape at all,
because a relative phase between things that are not oscillating means nothing.

## What it looks like

```text
$ gridoxide dynamics case.json --modes 6

13 mode(s) over 13 differential state(s)
every mode, by a dense decomposition of the state matrix

                eigenvalue    damping   freq (Hz)    tau (s)   residual   participation
    -0.73018     +7.99385j     0.0910      1.2723      1.370      0.0e0   G1.omega 41%, G1.delta 36%, G1.eqp 6%
    -0.73018     -7.99385j     0.0910      1.2723      1.370      0.0e0   G1.omega 41%, G1.delta 36%, G1.eqp 6%
    -1.02972     +2.23736j     0.4181      0.3561      0.971      0.0e0   G1.avr_lead 37%, G1.eqp 34%, G1.delta 7%
    -1.02972     -2.23736j     0.4181      0.3561      0.971      0.0e0   G1.avr_lead 37%, G1.eqp 34%, G1.delta 7%
   -40.55915     +5.54508j     0.9908      0.8825      0.025      0.0e0   G1.psi1d 39%, G1.pss_lead2 19%, G1.pss_lead1 19%
   -40.55915     -5.54508j     0.9908      0.8825      0.025      0.0e0   G1.psi1d 39%, G1.pss_lead2 19%, G1.pss_lead1 19%

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

## Targeting a region

Everything above computes *every* mode, by forming `A` and decomposing it whole. That is `O(n³)` in
the states and `O(n²·n_net)` in the reduction, and it is the right method up to a couple of
thousand states. A four-thousand-bus case has twenty thousand of them, where it is not slow but
hours — and almost none of what it computes is wanted.

The question a stability study actually asks is narrower: *the modes near this frequency*, or *the
least damped ones*, six or ten of them. Naming where to look is what makes that affordable:

```text
$ gridoxide dynamics ring4096.json --modes 4 --modes-freq 1.0

4 mode(s) over 20480 differential state(s)
the 4 nearest -0.3146+6.2832j (1.000 Hz) — sparse Arnoldi, 3 restart(s). NOT every mode the system has.

                eigenvalue    damping   freq (Hz)    tau (s)   residual   participation
    -1.02387     +5.82374j     0.1732      0.9269      0.977    2.5e-14   G2068.omega 0.14%, G2070.omega 0.14%, G2066.omega 0.14%
                                                    shape: G2062.delta 1.00∠+0°, G2060.delta 1.00∠+21°, G2064.delta 1.00∠-21°, G2058.delta 1.00∠+42°
    -1.03416     +5.83962j     0.1744      0.9294      0.967    2.5e-14   G2052.omega 0.14%, G2054.omega 0.14%, G2050.omega 0.14%
                                                    shape: G2048.delta 1.00∠+0°, G2050.delta 1.00∠-21°, G2046.delta 1.00∠+21°, G2052.delta 1.00∠-42°
```

Twenty thousand differential states, on a case the dense method cannot begin.

Read the second line. A **partial** list of modes supports no statement about the system as a whole:
"nothing unstable here" means *near this shift*, and the output says which kind of list it is rather
than leaving a reader to remember. The Python binding carries the same distinction as
`result.complete`.

Then read the shape. Equal magnitudes with the phase advancing 21° from one rotor to the next is a
**travelling wave** going round the ring — which is what a ring of near-identical machines should
have, and is not something a participation factor could have told you: every machine takes part, at
0.14% each, and the interesting content is entirely in the phases.

## How the sparse method works

Shift-invert. The eigenvalues of \\((A - \sigma I)^{-1}\\) are \\(1/(\lambda_i - \sigma)\\), so the
ones *nearest* `σ` become the *largest* — and largest is what a Krylov method finds first. Aiming
the analysis is choosing `σ`, and because the modes worth aiming at oscillate, `σ` is complex:
`--modes-freq 0.5` is the shift \\(-\zeta\omega_d/\sqrt{1-\zeta^2} + j\omega_d\\).

`A` is never formed. Write the linearized DAE as a *pencil* rather than a reduced matrix:

```text
 ⎡ A_x  A_v ⎤ ⎡x⎤       ⎡ I  0 ⎤ ⎡x⎤
 ⎣ C_x  C_v ⎦ ⎣v⎦ = λ   ⎣ 0  0 ⎦ ⎣v⎦
        J                    E
```

Its finite eigenvalues are exactly `A`'s — the bottom block-row carries no `λ`, so it reads
`C_x x + C_v v = 0` whatever `λ` is, which is the elimination written down rather than performed.
Then for a vector `b` in the state space,

\\[ S(b) = \text{the leading } n_x \text{ entries of } (J - \sigma E)^{-1} \begin{bmatrix} b \\\\ 0 \end{bmatrix} \\]

**is** \\((A - \sigma I)^{-1} b\\), by the Schur-complement identity. Each application costs one
sparse solve on the bordered system, against a factorization computed once per shift.

And the shift is free. `J - σE` is the same assembly [`DaePattern::fill`](./dae.md) produces at
`h·a = 1`, with its top block-rows negated and `(1 − σ)` added on the state diagonal — and that
diagonal is structurally present already, because the pattern carries a dense `∂f/∂x` block per
device. A shift adds no fill-in and changes no pattern, which is the same property that makes every
event in a [run](./events.md) value-only.

Left eigenvectors — which participation factors need, and which the dense method gets by inverting
`U` — come from a second Arnoldi pass on the adjoint operator, against the *same* factorization. The
two passes converge from different subspaces and are matched by eigenvalue; where that match is
ambiguous, the mode is reported **without** participation rather than with a plausible wrong one.

Two things measured on the way are worth carrying:

- **A restart means *growing*.** On a ring of two thousand machines the electromechanical modes are
  packed into a tenth of a hertz, and restarting from a combination of Ritz vectors at a fixed
  Krylov dimension barely helped — five restarts at dimension 32 moved the residual from `6e-2` to
  `7e-3`. Raising the dimension to 128 converged to `2e-19` in one cycle, and faster in wall time.
  So a restart doubles the dimension, up to a cap set by memory.
- **A residual is not an error bar.** It is measured by applying the operator rather than by the
  textbook estimate `|h_{m+1,m}||e_m^T y|`, which reported `8e-52` where the truth was `1.3e-14`.
  But even a true residual only certifies that the pair solves the problem it claims to. How far the
  *eigenvalue* could still move is the eigenvalue's condition, `1/|w^H u|`, and on that same ring the
  forward and adjoint passes disagree at the `1e-4` level with residuals at `3e-14` — not a defect in
  either, but the honest accuracy available for modes that ill-conditioned.

## Eigenvalue sensitivities

An eigenvalue says a mode is poorly damped. A sensitivity says what to change:

\\[ \frac{d\lambda}{dp} = \frac{W^H (\partial J/\partial p)\, U}{w_x^H u} \\]

in the pencil's own left and right eigenvectors — which avoids differentiating `C_v^{-1}`, since
`∂J/∂p` touches only the rows of the device that owns `p`. It is reported as `dλ/dp` and, more
usefully, as `dζ/dp`: how much damping ratio a unit of the parameter buys.

```text
$ gridoxide dynamics case.json --modes 3 --modes-sensitivity

sensitivity of the least-damped mode (-0.73018+7.99385j, ζ = 0.0910):
         parameter       value                  dlambda/dp    dzeta/dp  dzeta per 1%
              G1.h      5.0000      +0.10135     -0.83398j   -3.110e-3     -1.555e-4
              G1.d      1.0000      -0.05231     -0.00156j    6.480e-3      6.480e-5
```

`∂J/∂p` is a **central difference of the assembly** — set the parameter, refill the pattern,
subtract — so nothing is re-derived and nothing can fall out of step with the models as they change.
The pattern does not move, because sparsity depends on structure and not on values.

**Only the inertia and the damping coefficient are offered**, and the restriction is the honest part
rather than an omission. The formula above holds the operating point fixed, so it is the whole
derivative only for parameters the equilibrium does not depend on. `H` appears only as `1/2H` in the
swing equation and `D` only multiplying `(ω − 1)`, which is zero at rest. A reactance is different:
initialization picks `δ`, `e'_q` and the flux states *from* the reactances, so changing one moves the
point being linearized about, and the true derivative carries a `∂J/∂x · dx/dp` term this does not
have. Offering it would produce a number that looks like an answer and is a fraction of one.

## Scope

Both methods are available and both say which one answered. The dense one computes every mode and
is the default below two thousand states; past that, or whenever a shift is named, the sparse one
computes the modes near a point. Sensitivities work from either.

Not here: the **EMT** half of modal analysis, which is a different simulation entirely; block Arnoldi
for genuinely degenerate spectra — a ring of identical machines has near-exact repeated modes, and a
Krylov space holds one vector per invariant direction, so one of each pair is found and the
multiplicity is not; and sensitivities to network parameters, which perturb through the Y-bus rather
than through a device's own fill.

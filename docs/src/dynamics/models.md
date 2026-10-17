# The Model Library

## A machine and its controls are one device

An exciter writes `E_fd` into its machine, a governor writes `P_m`, a stabilizer writes `v_s` into
the exciter, and all three read a machine state or the terminal voltage. Modelled as separate
devices those couplings would be cross-device Jacobian entries — a second sparsity problem on top of
the network's, for what is physically one machine.

A **generating unit** composes them into one device with one contiguous state block, so every
coupling is an ordinary partial derivative inside one diagonal block.

The couplings are not written out per combination, which would be combinatorial. Each part declares
its derivatives with respect to its own states and its own *scalar* input, and the unit composes
them by the chain rule over the signal graph:

```text
   Δω ──► PSS ────► v_s ──┐
                          ├──► AVR ──► E_fd ──┐
   |V| ────────────(−)────┘                   │
                                              ├──► machine ──► I, δ, ω
   Δω ──► governor ─────────────► P_m ────────┤
   V   ───────────────────────────────────────┘
```

The graph is acyclic and no block's output depends on its own input through another block, so one
forward pass evaluates it and one sweep of the chain rule differentiates it. The governor's direct
feedthrough from `Δω` to `P_m` is not a loop: `P_m` enters `ω̇`, and `ω̇` does not enter `P_m` — only
`ω` does.

## Machines

| Model | States | What it adds |
|---|---|---|
| Classical | `δ, ω` | Constant voltage behind transient reactance. The model the equal-area criterion is written for |
| Transient (4th) | `+ e'_q, e'_d` | A field winding — the first model an exciter can act on — and one `q`-axis damper |
| Salient (5th) | `+ ψ_1d` on `d`, one flux on `q` | A salient-pole rotor: a damper on each axis, but no `q`-axis transient — there is no `q`-axis field for the flux to hide in |
| Subtransient (6th) | `+ ψ_1d, ψ_2q` | Two more damper windings. Immediately after a disturbance the machine's effective impedance is `x''`, not `x'` |

Every machine offers **both stator formulations**. By default the speed-voltage terms omit the
rotor speed and the swing equation is written in power — the classical RMS assumption. With
`speed_voltages` on, the speed is carried and the swing equation is written in torque, which is what
Dynawo and Sauer & Pai do. The two agree exactly at synchronous speed and part in proportion to the
speed deviation; on Kundur's Example 13.2 the full form removes 96% of the disagreement with
Dynawo. See [Against Dynawo](./validation.md).

`P_e` throughout is the **air-gap** power, not the terminal power. With armature resistance the two
differ by the stator copper loss, and it is the air-gap power the rotor feels. Using the terminal
power gives a machine damped by its own resistance — wrong in a way that looks entirely plausible,
since the swing still decays. Dynawo agrees: its `PmPu` exceeds its `PGenPu` by exactly the copper
loss.

The fourth-order model's air-gap power keeps the **saliency term**
`(x'_q − x'_d)·i_d·i_q`. Dropping it is a common and invisible mistake: the machine still swings,
just at a slightly wrong frequency and to a slightly wrong equilibrium.

### Why the subtransient conventions are derived rather than cited

Published statements of the sixth-order model disagree with each other about the sign of `ψ_2q` and
of `e'_d`. A formulation copied from one source and checked against another is internally consistent
and physically wrong.

So the conventions here are pinned by two requirements the code can be held to:

1. **The steady state must reduce to the fourth-order model's**, term for term. Setting the flux
   derivatives to zero and substituting into the subtransient EMFs must turn the subtransient stator
   equations into the transient ones. That fixes the interpolation coefficients and the damper
   equations.
2. **The two axes must map onto each other** under `(e'_q, ψ_1d, i_d) ↔ (e'_d, ψ_2q, −i_q)`. That
   fixes the sign of the damper-coupling correction in `ė'_d`, which the first requirement cannot
   see, because that correction vanishes at steady state.

A confirmation falls out: both damper equations reduce to `−Δ/T''` for the same `Δ` the correction
terms use. One expression serving both is itself evidence the signs agree.

The gate is the reduction — take `x''` up to `x'` and the sixth-order machine must trace the
fourth-order machine's trajectory through a fault, which it does to `1e-5` rad.

## Controls

| Kind | Models |
|---|---|
| Excitation | `SEXS` (lead-lag into a lag), and a purely proportional regulator |
| Governor | `TGOV1` (droop into two lags), and a purely proportional governor |
| Stabilizer | washout into two lead-lag stages |

The two **proportional** variants carry no states at all. They are not simplifications of the
lagged ones — they are different devices, and they are what Dynawo's `VRProportional` and
`GoverProportional` implement, so a Dynawo case maps onto them exactly rather than through invented
time constants. They are also what Kundur's worked examples use.

A zero-state control drops into the chain rule with no special case, which is the same property that
lets a ZIP load be a device with no states.

### What droop does

`TGOV1`'s `R` decides how a disturbance is shared. Synchronized machines settle at *one* frequency
deviation, and each one's steady-state contribution is then `−Δω/R`, so a machine with half the
droop picks up twice the power. That is a system-level property no single model can state on its
own, and it is gated as such.

### The washout, and why a stabilizer is safe to add

A stabilizer adds a small signal in phase with rotor speed to the exciter's input, to damp the
electromechanical oscillation a high-gain exciter erodes. Its washout passes **no steady signal at
all**, so `v_s = 0` at any equilibrium whatever the speed is, and it has no reference to latch.

That is the property that makes it safe: it cannot shift the voltage setpoint, however it is tuned,
so it can only affect the transient. Note that the washout's *state* does not go to zero — it goes
to the *input*. `ẋ = (u − x)/T_w`, so `x → u`, and it is the output `u − x` that vanishes. The state
absorbing the steady component is precisely the mechanism.

## Loads

The default is constant admittance from the operating point, which is optimistic: it sheds power as
`|V|²`, so it helps the voltage recover from exactly the disturbance that depressed it.

`ZipLoad` puts the choice in the caller's hands as three fractions, with a low-voltage cutoff. A
constant-power load is singular at zero voltage — `I = conj(S)/conj(V)` diverges, and a nearby fault
drives it there — so all three parts are written in one form

```text
I = conj(S_z)·V/V₀²  +  conj(S_i)·V/(V₀·m)  +  conj(S_p)·V/m²,     m = max(|V|, V_cut)
```

Above the cutoff those are exactly the three exponents. Below it `m` freezes and the whole load
becomes the constant impedance that reproduces its own current at the cutoff voltage. The current is
continuous there; its derivative is not, and that kink is a real property of the model rather than an
implementation artifact.

**The cutoff changes answers.** It is a modelling parameter, not a numerical tolerance, and it
should be reported alongside a result that depended on it.

## Limits, and why they are non-windup

An exciter ceiling, a governor valve limit and a stabilizer output clamp are all implemented, and
both readers carry them through from their files.

They are **non-windup**, which is the distinction that matters. With windup the state keeps
integrating past the boundary while the output is pinned there, so when the error finally reverses
the output stays pinned for however long the state takes to travel back — a delay with no physical
basis. On the gated case an unlimited exciter reaches 18 pu against a 2.6 pu ceiling; a wound-up one
would sit at that ceiling for as long as it took to fall back through fifteen per unit. Held at the
boundary, the state never exceeds it at all.

Three pieces make that work, and each answers a real failure:

**The active set is fixed for the duration of a step.** Recomputing it from each Newton iterate
makes the residual non-smooth *inside* the solve: an iterate landing just above the boundary sees a
zeroed derivative, the next lands just below and sees the full one, and the two alternate. Measured,
that made an ordinary exciter ceiling fail the step outright. Frozen, the step is smooth and the
Jacobian is exact for what is actually being solved. A limit therefore engages one step late, which
at five milliseconds is not worth the machinery to avoid.

**States are projected onto their limits after each accepted step.** The step that *crosses* a
boundary can still carry a state past it, because the trapezoidal rule averages a start-of-step
derivative that was still driving hard with an end-of-step one that has been zeroed — and a frozen
derivative cannot then bring it back. The projection is exact rather than a correction: a non-windup
state has no legitimate value outside its limits.

**Rate limits** are implemented too, on the governor valve, and they constrain a different thing:
not where the valve may be but how fast it may travel. A steam valve that opens in a fifth of a
second and a hydro gate that takes five are the same model with different rates, and the difference
decides whether a machine can arrest a frequency excursion at all.

The two compose in one order and not the other. The rate limit clips the derivative first; the
position limit then decides whether the valve may move at all. Reversing them would let a valve
pinned at its ceiling still "travel" at its rate limit, which is nothing.

What remains genuinely absent is saturation — see the readers.

## Every model is checked against an oracle

The analytic Jacobian is what runs; a central-difference oracle is what proves it. Every model, in
every combination of controls, is checked at points well away from the equilibrium — a Jacobian that
is only right at the operating point is right for nothing.

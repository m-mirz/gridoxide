# Initialization from a Power Flow

## The rule

Every device is initialized **backwards**: from its terminal voltage and power, choose the states
that make its derivatives *zero*, and latch whatever internal reference that implies.

A classical machine latches its internal EMF magnitude and its mechanical power. A transient
machine additionally derives the field voltage its flux state requires. An exciter latches `V_ref`
so that its output is exactly *that* field voltage. A governor latches `P_ref` likewise.

**Nothing reads a setpoint from the input file.** A file's stated `V_ref` and its stated power flow
are two independent claims, and they will not agree to machine precision. Deriving the setpoint from
the operating point makes them agree by construction.

## The invariant, and the gate that checks it

After the build, at `t = 0`:

- every device's derivative vector is zero, and
- the network residual `Y V − I_inj(x, V)` is zero,

both to **machine precision**, not merely to a solver tolerance. That is not an aspiration; it falls
out algebraically. Each machine's Norton current is `E·y` with `E = V + z·I` for the very
`I = conj(S/V)` the power flow produced, so the network residual at its bus reduces to the
power-flow equation that was already satisfied. Each remaining injection becomes an admittance
`y = −conj(R)/|V|²`, which reproduces `R` exactly at the voltage it was derived at.

So the builder **checks both** and refuses a system that fails either. Which gives the single most
useful check in the whole module:

> Run with no disturbance at all. The trajectory must be a flat line.

If initialization is consistent, every derivative is zero at `t = 0` and stays zero. If a sign is
flipped, a per-unit base missed, a Norton stamp inconsistent with the impedance its own model
initialized against, or a residual assembled wrongly, the state drifts immediately and visibly.
Nearly every mistake in this module is caught by that one check, and it is run for every model and
every combination of controls.

## What the check cannot see

A wrong **device/load split**. If a caller declares that a machine makes half of what it really
makes, the result is *self-consistent*: the machine initializes to an equilibrium at the power it
was told, and the remainder is absorbed into the bus's admittance, so both residuals are still
exactly zero.

What comes out is a *different machine* — smaller output, smaller internal EMF, smaller rotor angle
— swinging against a network that makes up the difference from something inert. The trajectory is
plausible and the answer is wrong.

This is why the equilibrium gate is a **code-correctness** check and not an input-validation one,
and why the file format removes the need to state the split at all: a device may give its own `p`
and `q`, and if it does not, it takes what its bus's solved injection has left. One device omitting
it at a bus is unambiguous; two is refused by name rather than split by a guess.

## Deriving the rotor angle

For any machine with a `q` axis, the rotor angle comes from the steady-state reference

\\[ E_q = V + (r_a + j x_q)\,I, \qquad \delta = \arg E_q \\]

That is not a convenience. Choosing `δ` this way forces the `d`-axis component of `E_q` to zero,
which reduces the stator relation for `e'_d` to exactly `(x_q − x'_q)·i_q` — the expression
`ė'_d = 0` demands. The same construction works for the subtransient machine, because at steady
state a machine looks like `x_q` behind its terminal on the `q` axis whatever damper windings it
carries.

Two identities check the result exactly, and both are asserted to `1e-12`:

1. **The machine reproduces its own terminal current.** `injection(x₀, V₀) − y_norton·V₀` must equal
   `conj(S/V)`. A transposed sine and cosine, a `q` axis defined the other way round, or a sign slip
   in the stator solve all survive a plausibility reading; none survives this.
2. **A machine with no subtransient saliency cancels its own Norton stamp**, so `∂I/∂V` is exactly
   zero.

Against Dynawo, gridoxide's `δ₀` on Kundur's Example 13.2 agrees to `6e-5` rad — two implementations
deriving the same rotor angle from the same terminal condition by independent routes.

## What loads become

Constant admittance, derived at the solved voltage, unless a `ZipLoad` device says otherwise. This
is exact at `t = 0` by construction, and it is a real modelling choice with real consequences: a
constant-impedance load sheds power as the square of the voltage, so it is *optimistic* about
voltage recovery compared with a constant-power one. See [The Model Library](./models.md).

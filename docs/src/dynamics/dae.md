# The DAE and Its Jacobian

## The formulation, and the one decision everything follows from

gridoxide solves the differential and algebraic halves **simultaneously**: one Newton per step over
the whole vector, with no inner iteration between a device solver and a network solver. There is no
interface error to converge away and no per-step lag between the two halves.

The trapezoidal rule on the differential half, the constraint enforced at the new point:

\\[ x_{k+1} - x_k - \tfrac{h}{2}\big(f(x_k,V_k) + f(x_{k+1},V_{k+1})\big) = 0 \\]
\\[ \mathbf{Y}V_{k+1} - I_{inj}(x_{k+1},V_{k+1}) = 0 \\]

giving the Newton matrix

```text
      ⎡ I − (h/2)·∂f/∂x       −(h/2)·∂f/∂V ⎤
  J = ⎢                                     ⎥
      ⎣    −∂I_inj/∂x        Y − ∂I_inj/∂V  ⎦
```

## Why rectangular coordinates, and why current balance

Write the network in rectangular coordinates and its block is the real form of the admittance
matrix:

```text
Y_real = ⎡ G  −B ⎤        I = YV  ⇒  i_re = G v_re − B v_im
         ⎣ B   G ⎦                   i_im = B v_re + G v_im
```

which is **constant** for the lifetime of a topology. Nothing about it depends on `x` or on `V`.
Every moving part is then a device stamp, and every device stamp is local: `∂f/∂x` is one dense
block per device, `∂f/∂V` and `∂I/∂x` couple that block to its own bus's two columns and rows, and
`∂I/∂V` is 2×2 on one diagonal.

A polar power-mismatch Jacobian — what [the power flow](../powerflow/index.md) builds — has none of
that structure: every entry moves every iteration. So `dynamics` borrows that module's *shape*
(analyze once, refill values at fixed offsets, hand the values to the backend) and none of its
arithmetic.

## One symbolic factorization for a whole run

The sparsity pattern is fixed for the lifetime of a topology, and **every event preserves it**.
That is deliberate. The pattern is built from a *topology superset* — with every branch in service
and every bus diagonal stamped, whether it needs one yet or not — and an out-of-service branch has
its positions re-stamped at zero rather than dropped.

So a fault, a clearing, a branch trip and a unit trip are all numeric refills against one symbolic
factorization. Nothing re-analyzes for the whole run.

Tripping a generating unit looks like the exception, since its states would leave the system.
Freezing it rather than removing it keeps the layout identical — its rows become `x₁ − x₀ = 0` and
its Jacobian block the identity the implicit rule contributes — and freezing is also the better
model: nothing in the network can observe a disconnected machine's rotor.

## The Norton stamp, which changes no answer

Each machine declares a constant admittance that is stamped into `Y` at build, and its injection
returns

```text
I_inj = I_machine(x, V) + y_norton·V
```

so the `y_norton·V` the network row adds is exactly cancelled. The stamp is *purely* a conditioning
device: it keeps `Y` diagonally dominant at generator buses without moving a single result.

For a machine with no subtransient saliency the cancellation is exact and `∂I/∂V` comes out zero to
the last bit — which is the sharpest available check on the whole rotor-frame algebra, and is
asserted as an identity rather than a tolerance. For a salient machine a residual remains: the
machine is genuinely not an impedance in the network frame, because its response depends on the
rotor's orientation. That residual *is* the saliency.

## Two integration rules, one assembly

The trapezoidal rule is second-order and A-stable, which is what a stiff phasor DAE wants. It is
**not** L-stable: its amplification factor tends to `−1` rather than `0`, so a discontinuity excites
a numerical oscillation that alternates sign and decays only as slowly as the physical mode does.

The remedy is a couple of **backward Euler** steps immediately after each event, then back to
trapezoidal. Both rules are the same residual with different coefficients —

```text
x₁ − x₀ − h(a·f₁ + b·f₀) = 0     (a, b) = (½, ½) trapezoidal
                                         (1, 0)  backward Euler
```

— so nothing branches on which is active beyond choosing the pair. That matters because the rule
switches *within* a run, and two separate assemblies would be two places to get the damping wrong.

The damping is not free, and its cost is known in closed form: backward Euler evaluates `δ̇` at the
end of the step, so on a linearly growing speed each damping step overshoots by exactly
`Ω_b·a·h²/2`. Two steps leave a permanent offset of `Ω_b·a·h²`, which is `2.5e-5` rad at a 1 ms
step on a typical machine.

## The backends

All five of gridoxide's [sparse LU backends](../solvers/backends.md) serve this, unchanged, because
the DAE Jacobian satisfies the same contract the power flow's does: pattern fixed, values change
every call. `Block` is the exception and is refused — it assumes a uniform 2×2 block per bus, and a
DAE has variable-size device blocks alongside the network's.

One caveat worth knowing if you extend this: the backends are interchangeable at *solve* time but
not at *construction* time. `KluNative` and `Klu` factor numerically inside `new`, so they must be
handed real values from an actual fill, not a placeholder pattern.

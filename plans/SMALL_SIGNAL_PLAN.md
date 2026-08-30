# Sparse small-signal analysis, and eigenvalue sensitivities

Status: **implemented**, 2026-08-30, against `15b15ed`. Phases 1–6 landed; §9 records where this
plan was wrong, which is the part worth keeping.

> **What this is.** The two items `plans/RMS_PLAN.md` §22 left on its "What is left" list:
> *"sparse Arnoldi small-signal, and eigenvalue sensitivities"*. §22 had implemented a **refusal**
> instead — past two thousand states `smallsignal::analyze` said the dense method was wrong for the
> problem rather than leaving a caller waiting — and called that honest but not the feature. This is
> the feature.
>
> **Scope.** The RMS half only. `docs/src/reference/feature_comparison.md:66` stays ⚠️ afterwards,
> but for a different reason than before: the EMT half is a different simulation entirely, and is
> now the *only* thing that row is missing.

## 1. Why the dense method runs out

`smallsignal::analyze` forms the reduced state matrix `A = A_x − A_v·C_v⁻¹·C_x` and decomposes it
whole. That costs `O(n_x²·n_net)` in the reduction and `O(n_x³)` in the decomposition, and it is
the right method up to a couple of thousand states — which is most cases anyone has, and is why it
was built first.

A four-thousand-bus case has twenty thousand differential states. There the dense method is not
slow, it is hours; and almost none of what it computes is wanted. The question a stability study
asks is narrower — *the modes near this frequency*, or *the least damped ones*, six or ten of them.
That narrower question has a method, and it is a different one rather than a faster version of the
same one.

## 2. The formulation

### The pencil

Write the linearized DAE as a generalized eigenproblem rather than a reduced matrix:

```text
 ⎡ A_x  A_v ⎤ ⎡x⎤       ⎡ I  0 ⎤ ⎡x⎤
 ⎣ C_x  C_v ⎦ ⎣v⎦ = λ   ⎣ 0  0 ⎦ ⎣v⎦
        J                    E
```

Its finite eigenvalues are exactly those of the reduced `A`. `E`'s lower block is zero, so the
bottom block-row carries no `λ` and reads `C_x x + C_v v = 0` whatever `λ` is; that gives
`v = −C_v⁻¹C_x x`, and the top row becomes `A x = λ x`. Nothing is approximated — this is the
elimination *written down* rather than *performed*.

### The operator

For `b` in the state space,

```text
 S(b) = the leading n_x entries of (J − σE)⁻¹ [b; 0]
```

is `(A − σI)⁻¹ b` **exactly**, by the Schur-complement identity `(K⁻¹)₁₁ = (P − QT⁻¹R)⁻¹`. The
Krylov space lives in `C^{n_x}` — short vectors — and each application is one sparse solve on the
bordered `(n_x + 2·n_bus)` system against a factorization computed once per shift.

This is the choice the whole plan rests on, and it is worth naming what it avoids. Running Arnoldi
on the *pencil* directly would mean an operator with a large null space (every direction `E`
annihilates), infinite eigenvalues to filter, and `E`-orthogonality to maintain. Restricting to the
state space removes all three: `S` is nonsingular, its eigenvalues are `1/(λ − σ)` and nothing else,
and its Ritz vectors *are* the eigenvectors the participation and shape code already expects.

### The shift is free, and complex

`DaePattern::fill` at `h·a = 1` writes `M = [I − A_x, −A_v; C_x, C_v]`, and `J − σE` is that with
the top `n_x` rows **negated** and `(1 − σ)` added on the state diagonal:
`−(I − A_x) + (1 − σ)I = A_x − σI`, `−(−A_v) = A_v`, bottom rows already correct. The state diagonal
is structurally present, because `DaePattern::analyze` emits a dense `∂f/∂x` block per device — so a
shift adds **no fill-in** and changes no pattern. The same property that made every event in a run
value-only makes every shift value-only here.

Complex, because the question is. A real shift can only aim at a point on the real axis; naming a
frequency band needs `σ = α + jβ`, and targeting a region of the complex plane is the whole reason
to use this method. The price is that the five pluggable `LinearSolver` backends are real-only and
do not apply; `sparse::ComplexSparseSystem` serves, whose factorize-once/solve-many contract is
exactly what an Arnoldi iteration wants. That is the position the Y-bus's own complex solves are
already in.

### Sensitivities, in the same pencil

\\[ \frac{d\lambda}{dp} = \frac{W^H (\partial J/\partial p)\, U}{W^H E\, U}
                        = \frac{W^H (\partial J/\partial p)\, U}{w_x^H u} \\]

The pencil is what makes this tractable. Differentiating the reduced `A` means differentiating
`C_v⁻¹`; differentiating `J` does not, and `∂J/∂p` touches only the rows of the device that owns `p`.
The two extra halves are one solve each against the network factorization the dense reduction builds
anyway: `v = −C_v⁻¹C_x u` and `w_vᴴ = −w_xᴴA_v C_v⁻¹`.

`∂J/∂p` is a **central difference of the assembly** — set the parameter, refill the pattern,
subtract. The pattern does not move, because sparsity depends on structure and not on values, so
the difference is a value array over the same `(row, col)` pairs. Nothing is re-derived, and nothing
can fall out of step with the model library as it grows.

## 3. What already existed, and what it bought

- **`DaePattern::analyze`/`fill`/`to_triplets`** — the four blocks, analytically exact and
  oracle-checked per model, with the state diagonal already in the pattern. Both methods read from
  one `fill`, which is what guarantees they linearize the *same* system.
- **`sparse::ComplexSparseSystem`** — complex sparse LU, factorize once, solve many. It needed one
  addition: `solve_adjoint`, for the left eigenvectors.
- **`sparse::RealFactorization`** — one factorization, many right-hand sides, with a transpose solve.
  The dense reduction should have been using it all along (§9).
- **`Mat::eigen()`** — for the small `m × m` Hessenberg, the same call the dense path makes on `A`.
- **`examples/dynamics_scale.rs`'s ring** — an arbitrarily large dynamics case, already built for
  gate G10, and now shared verbatim with the tests.
- **The `opf-highs`/`opf-ipopt` arrangement** — the template for an optional external cross-check.

## 4. Phases, as built

1. **The operator.** `src/dynamics/shift.rs`, `ComplexSparseSystem::solve_adjoint`, and three fixes
   to the dense path (§9). Gate: `apply(b)` inverts the matrix `state_matrix` forms, to 1e-10, at
   three shifts including two well off the real axis.
2. **The iteration.** `src/dynamics/arnoldi.rs` — Arnoldi with full reorthogonalization, dimension
   growth on restart, and residuals measured rather than estimated.
3. **The reporting.** `analyze_near`, `SmallSignalOptions`, `Method`, and per-mode `residual`,
   `eigenvector`, `left_eigenvector`. Participation from a second Arnoldi pass on the adjoint
   operator. `analyze` falls over to the sparse method past its limit instead of refusing. CLI and
   Python.
4. **Scale.** `tests/dynamics_ring/`, shared with `examples/smallsignal_scale.rs`.
5. **Sensitivities.** `smallsignal::sensitivities`, and the `Machine`/`DynamicModel` parameter
   surface beneath it.
6. **The cross-check, the docs, the row.** `src/dynamics/arpack.rs` behind `smallsignal-arpack`.

## 5. What it costs, measured

A ring of alternating generator and load buses, every generator a sixth-order machine with an
exciter and a governor — the same case G10 timed. Eight modes near 1 Hz:

| buses | units | states | build | analyze | worst residual | restarts | with participation |
|---|---|---|---|---|---|---|---|
| 64 | 32 | 320 | 0.86 ms | 44 ms | 2.7e-14 | 1 | 8/8 |
| 256 | 128 | 1 280 | 0.65 ms | 143 ms | 1.9e-14 | 1 | 8/8 |
| 1 024 | 512 | 5 120 | 2.9 ms | 1.93 s | 1.7e-14 | 2 | 8/8 |
| 4 096 | 2 048 | 20 480 | 12 ms | 14.8 s | 2.9e-14 | 3 | 8/8 |

The dense method cannot start on the last two rows.

Read the **restart** column beside the time, because it is what the time is. Each restart doubles
the Krylov dimension, and the cost of a cycle is one solve per dimension plus an orthogonalization
quadratic in it — so three restarts is a dimension of 288, and that, not the sparse solve, is where
14 s goes. The factorization is near-linear like every other in this crate. What drives the
dimension is how tightly the spectrum is packed: this ring puts two thousand electromechanical modes
into a tenth of a hertz, which is close to the worst case a real system could present, and is
deliberately so — it is the same fixture the degeneracy gate uses.

## 6. Validation

Six gates, and the point of the list is that they fail for different reasons:

| | What it checks | Against |
|---|---|---|
| The operator | `(A − σI)·apply(b) = b` to 1e-10 | The dense reduction, which shares only the `fill` |
| The subspace | eigenvalues to 1e-7, and that they are the *nearest* ones | The dense decomposition, on 320 states with a Krylov dimension of 32 — a genuine projection |
| The shift | the same eigenvalue from three shifts, to 1e-9 | Itself — a mode belongs to the system, not to the aim |
| The closed form | `±j√(Ω_b·K_s/2H)` to 1e-9 relative | Theory, on the one case that has one |
| The trajectory | period to 2e-3, decay to 2% | A time-domain run of the same ring, exciting one mode along its eigenvector |
| ARPACK | eigenvalues to 1e-9, residuals, eigenvector directions | The field's own implementation, on the **identical** operator |

Plus, for the sensitivities: `dλ/dH = −λ/2H` — the closed form differentiated — and a central
difference of two whole analyses, to 1e-6 relative.

Two gates are about honesty rather than accuracy. A **degenerate** ring (identical machines) is
asserted to return distinct, genuine, converged modes — *not* that it finds the multiplicity, which
a Krylov method cannot: the failure would otherwise be silent, since every eigenvalue returned is
real and nothing about them says a second copy went unreported. And the "same `Mode` from both
paths" gate compares participation, shape and eigenvector *direction*, not just eigenvalues, because
a difference in convention between the two would be invisible in an eigenvalue comparison and
glaring to anyone reading the output.

## 7. The surfaces

`gridoxide dynamics <case> --modes <n>` gains `--modes-freq <Hz>` with `--modes-damping <ζ>`,
`--modes-near <re>,<im>`, and `--modes-sensitivity`. Naming any of them selects the sparse method at
**any** size — asking about one band of a small system is a legitimate question — and the output
says which method answered and that a partial list is partial.

Python's `small_signal` grew from returning a list to returning a `SmallSignalResult` carrying
`method`, `shift`, `converged`, `complete` and `sensitivities`. It keeps `__len__` and
`__getitem__`, so every caller that treated it as a sequence still does; the object exists to carry
the one distinction a list cannot — whether "no unstable modes" is a statement about the system or
about a neighbourhood of a shift.

## 8. Deliberately out of scope

- The **EMT half** of the row.
- **Block Arnoldi** for degenerate spectra. Gated as a limitation instead.
- **Krylov–Schur** restarting. Growth is what these spectra actually needed (§9), and the
  Schur-with-reordering that Krylov–Schur wants is not exposed by `faer`.
- **Sensitivities to network parameters** — a different perturbation path, through the Y-bus rather
  than through a device's own fill.
- **Sensitivities to anything the equilibrium depends on**, which is most parameters. See §9.

## 9. Where this plan was wrong

### Restarting had to mean *growing*

The plan proposed restarting from a combination of the wanted Ritz vectors at a fixed Krylov
dimension, with Krylov–Schur named as the upgrade. Measured on the 1 024-bus ring, that restart is
**nearly useless**: five restarts at dimension 32 moved the worst residual from `6e-2` to `7e-3`,
taking 0.74 s. Dimension 128 in a single cycle reached `1.9e-19` in 0.82 s, and dimension 256
reached `1.6e-76`.

The diagnosis is the spectrum, not the restart. With two thousand modes packed into a tenth of a
hertz, the eighth-nearest eigenvalue to the shift and the ninth are almost equidistant, and no
Krylov space of a few dozen vectors can separate them however it is seeded. So a restart now
**doubles the dimension** and re-aims the start vector, capped by memory and by the cost of the
dense projected problem. Krylov–Schur would still converge in fewer applications; it is no longer
the thing standing between this and working.

### The textbook residual estimate is unusable here

The plan assumed the standard `|h_{m+1,m}|·|eₘᵀy|`, which is free. Measured against an explicit
application of the operator, it reported **`8e-52` where the truth was `1.3e-14`** — the last Krylov
component underflows once a Ritz vector is essentially contained in the early part of the space, and
the estimate collapses with it. It is now measured by applying the operator to each returned Ritz
vector: `count` extra solves against a factorization that already exists, against the hundreds the
iteration itself spends.

The follow-on finding is the one worth keeping. **A residual is not an error bar.** On the 4 096-bus
ring the forward and adjoint passes disagree about the eigenvalues at the `1e-4` level with
residuals at `3e-14` — because the two are related by the eigenvalue's condition `1/|wᴴu|`, and a
densely clustered non-normal spectrum is badly conditioned. Neither pass is wrong; that is the
accuracy available for modes that ill-conditioned. It also forced the left/right pairing rule from
an absolute threshold (`1e-6` relative, which rejected every correct pairing at that size) to a
**clear-winner** rule: the best candidate must stand a factor of ten clear of the second, which on
that ring it does by a factor of a hundred.

### Most parameters move the equilibrium

The plan named `h`, `d`, four reactances, two time constants, four AVR parameters and four governor
parameters. **Only `h` and `d` can be answered.** The sensitivity formula holds the operating point
fixed, and initialization picks `δ`, `e'_q` and the flux states *from* the reactances — so changing
one moves the point being linearized about, and the true derivative carries a `∂J/∂x·dx/dp` term
that would need the initialization differentiated too. `H` appears only as `1/2H` in the swing
equation and `D` only multiplying `(ω − 1)`, which is zero at rest; those two are exactly the
parameters the equilibrium does not see.

Offering the rest would produce a number that looks like an answer and is a fraction of one, so
`Machine::tunable` lists two names and `sensitivities` refuses the others by name. The control
parameters are a further question again — a rebuilt control loses what `initialize` latched into it,
so they would need that carried across even before the equilibrium term was faced.

### A rebuilt model loses what initialization gave it

Related, and the sharpest failure of the six phases. `with_parameter` rebuilds a machine through its
ordinary constructor — right, and the plan said why: `GenRound::new` stores `h · (mbase/s_base)`, so
patching the field directly would skip the base conversion. What the plan missed is the other half:
a constructed machine has **not been initialized**, and `GenCls` keeps its internal EMF magnitude as
a latched constant rather than a state.

A copy without it is a machine with no excitation. Its electrical torque and every derivative of it
are zero — which does not fail, it quietly answers about a different machine. The symptom was
`dλ/dH` coming back as *exactly zero* on an undamped case, and on a damped one as the `D/4H²` term
alone with the whole synchronizing part missing: two failures that both look like physics.
`Machine::carry_latched_from` exists for this, and the higher-order machines implement nothing
because they carry their EMF as states.

### Exact-zero tests do not survive complex arithmetic

`shape_of` and the mode's derived quantities tested `lambda.im == 0.0`. A real eigenvalue of a real
matrix comes out of the dense decomposition with an imaginary part of exactly zero; the same
eigenvalue reached through `λ = σ + 1/θ` comes out at `1e-17j`. So the sparse path reported a **mode
shape** — a set of relative phases — for a mode that does not oscillate.

Caught by the gate that compares a `Mode` from each path field by field, which is what that gate was
for. The fix is a relative tolerance in the one shared constructor both paths go through, which is
also why there is one.

### The output was wrong at scale, and only a real transcript showed it

The docs' example of a large-case run was going to be reconstructed from the numbers the scaling
example prints. Generating it for real instead — dumping the 4 096-bus ring to JSON and running the
CLI on it — found two display defects that no test had, because every fixture in the suite is small:

- **Participation printed `0%`.** A mode spread across two thousand machines belongs to each at
  about 0.14%, and `{:.0}%` rounds that to zero — which reads as *no* participation when the truth
  is the opposite. Two decimals below one per cent now.
- **The mode shape printed the first four rotors**, not the largest four. The shape is scaled so the
  largest component is `1∠0`; on a ring of two thousand rotors, `G0` is not it, so every line read
  `G0.delta 0.00∠-10°`. Sorted by magnitude now.

With both fixed the transcript says something: equal magnitudes with the phase advancing 21° per
rotor, which is a **travelling wave** going round the ring — the right answer for a near-circulant
system, and one participation factors could not have expressed at all.

One unrelated defect surfaced while producing that file, and is **not** fixed here:
`DynamicsDocument` serializes to JSON its own reader rejects. `serde_json` writes `null` for a
`None` field and for a non-finite `f64` — `q_min`/`q_max` default to infinity — and `json::read`
refuses both. Round-tripping a document is not something anything in the tree does, which is why it
has gone unnoticed; the file for the transcript above was post-processed by hand.

### Three things the dense path had wrong, found on the way

- The reduction called `LinearSolver::factor_and_solve_values` once per state column, **numerically
  refactorizing `C_v` `n_x` times**, under a comment claiming one factorization served every column.
  `RealFactorization` is the type that delivers what the comment promised.
- `TooLarge` was checked before `NotAnEquilibrium`, so a large mid-transient system reported the less
  actionable of its two problems.
- The CLI printed a mode shape only when `eigenvalue.im > 0.0`, which depends on which half of a
  conjugate pair sorts first.

## 10. What is left

- The **EMT half** of the small-signal row.
- **Block Arnoldi**, for spectra with genuine multiplicity.
- **Sensitivities beyond `h` and `d`**, which need the equilibrium's own derivative — a separate
  piece of machinery, and the honest prerequisite for offering a reactance or a control gain.

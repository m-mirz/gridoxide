# RMS simulation in gridoxide

Status: **complete except G6**, 2026-08-29, against `6f212eb`. Phases 1–4 and 6 done, phase 5
half done — Dynawo yes, ANDES dropped. §11–§19 record what was actually done, including the
places this plan was wrong. §17 onward is follow-on work beyond the plan's own scope.

## Context

`docs/src/reference/feature_comparison.md:64` carries a ❌ for **RMS / transient stability**, and
line 283 already names it as the largest of the remaining gaps with the clearest path. Everything
gridoxide computes today answers a question about one *instant*: a power flow, a short circuit, a
state estimate, an optimum, or — with continuation — a curve of instants parameterized by loading.
None of them can say what happens *over time* after a fault: whether the machines stay in step,
how far the voltage dips and how fast it recovers, whether the governors arrest the frequency.

That is a different machinery, not a refinement of an existing one. It needs a differential-algebraic
system, an integrator, discrete event handling, and a library of machine/exciter/governor models —
none of which the tree has any of. What it does have, and what makes this affordable, is listed in
§1.

The intended outcome: `gridoxide dynamics <network> --dyn <models>` runs a phasor-domain time-domain
simulation over a published multi-machine case, with a fault applied and cleared, and produces
trajectories that agree with ANDES and Dynawo to a stated tolerance. The comparison row goes to ✅.

---

## 1. Why this is affordable now

Four pieces already exist and are load-bearing:

- **`solver::LinearSolver`** (`src/solver.rs:35`) is exactly the contract a DAE Newton wants:
  *pattern fixed, values change every call*, with `factor_and_solve_values` taking only the nonzero
  values positionally. Five backends implement it and all five come along for free.
- **`jacobian::JacobianPattern`** (`src/jacobian.rs`) is the analyze-once / refill-every-iteration
  shape the DAE Jacobian needs, and the precedent for how to write one: a flat `Entry` recipe array
  built from topology, then `fill` writing only `f64`s into a caller-owned buffer.
- **`network::build_ybus` / `YBusSparse`** already assembles the admittance matrix from
  `Line`/`Transformer`, with `build_ybus_with_outages` for the topology changes an event needs.
- **`types::Bus::zip_terms`** (`src/types.rs:41`) already carries ZIP load composition, which is the
  static load model an RMS run needs on day one.

And the initial condition is a solved power flow, which is the thing this crate is best at:
`run_power_flow_analysis` with its outer loops produces exactly the terminal `(V, S)` per machine
that initialization consumes.

What is genuinely new: the DAE residual and its cross-derivatives, the integrator, event handling,
the model library, and the readers. That is the cost, and it is real — but it is *new numerics on
existing infrastructure*, which is the same position `src/continuation/` was in.

---

## 2. The formulation

**Simultaneous-implicit, rectangular current-balance.** One Newton per step over the whole
system — differential and algebraic together — with no interface iteration between a device solver
and a network solver.

Unknowns, per step:

```
z = [ x            device differential states, grouped by device
    ; v_re, v_im ] network algebraic variables, interleaved per bus
```

The system:

\\[ \dot{x} = f(x, V), \qquad \mathbf{Y}V - I_{inj}(x, V) = 0 \\]

Trapezoidal rule on the differential half, the algebraic half enforced at the new point:

\\[
F_{diff} = x_{k+1} - x_k - \tfrac{h}{2}\big(f(x_k,V_k) + f(x_{k+1},V_{k+1})\big) = 0
\\]
\\[
F_{alg} = \mathbf{Y}V_{k+1} - I_{inj}(x_{k+1},V_{k+1}) = 0
\\]

Newton matrix:

```
      [ I − (h/2)·∂f/∂x       −(h/2)·∂f/∂V ]
  J = [                                     ]
      [    −∂I_inj/∂x        Y − ∂I_inj/∂V  ]
```

**Why rectangular, and why current balance.** In rectangular coordinates the network block is the
real form of the admittance matrix,

```
Y_real = [ G  −B ]        (I = YV  ⇒  i_re = G v_re − B v_im,  i_im = B v_re + G v_im)
         [ B   G ]
```

which is **constant** for the lifetime of a topology. Nothing about it depends on `x` or on `V`.
Only the device stamps move between Newton iterations, and every one of them is local: `∂f/∂x` is
block-diagonal per device, `∂f/∂V` and `∂I_inj/∂x` touch one device block against its own bus's two
columns/rows, and `∂I_inj/∂V` is 2×2 per bus. That is a strictly better-conditioned and sparser
object than a polar power-mismatch Jacobian, and it is why the DAE literature (Sauer & Pai ch. 9,
Dynawo, PSAT) writes the network this way rather than as power balance.

It also means the power-flow Jacobian is **not** reused. `jacobian::JacobianPattern` is the
*template* for the new `dae::DaePattern`, not a component of it. Attempting to reuse the polar
H/N/M/L blocks would force a coordinate change on every device model for no gain.

**Per-unit.** The network is on `s_base`; machine parameters are conventionally on the machine's own
MVA rating. `types::Bus` carries no machine base, so every reader must supply `mbase` per machine
and `dynamics::init` converts once, at build. This is a classic silent-bug source — see §8.9.

**The Norton stamp.** Each machine's transient (or subtransient) admittance `1/(r_a + j x'_d)` is
stamped into `Y` once, at build time, and the model's `I_inj` returns the corresponding Norton
current. This is constant, so it does not disturb the "network block is constant" property, and it
keeps `Y` diagonally dominant at generator buses. A salient machine (`x'_q ≠ x'_d`) leaves a residual
saliency term in `I_inj` that depends on `δ`; that is expected and is handled in the model's own
analytic Jacobian, not by touching `Y`.

---

## 3. Structure

New module `src/dynamics/`, behind a new default-off `dynamics` Cargo feature — pure Rust, no
system dependency, so it is built and tested in CI alongside `opf`, `ucte` and `capi`. The feature
gate is about build cost, not dependencies, exactly as `opf`'s is.

| File | Contents |
|---|---|
| `src/dynamics/mod.rs` | `DynamicsOptions`, `run_dynamics`, `Trajectory`, `DynamicsReport`, `DynamicsStatus` |
| `src/dynamics/dae.rs` | `DaePattern` (analyze/fill, after `jacobian::JacobianPattern`), the residual, the real-form Y expansion |
| `src/dynamics/integrator.rs` | Trapezoidal, backward-Euler damping after events, step control, the per-step Newton |
| `src/dynamics/init.rs` | Equilibrium initialization from a `PowerFlowReport` |
| `src/dynamics/events.rs` | `Event`, `EventKind`, the schedule, the algebraic re-solve at a discontinuity |
| `src/dynamics/models/mod.rs` | The `DynamicModel` trait, `GeneratingUnit`, `ModelJacobian`, the finite-difference oracle |
| `src/dynamics/models/machine.rs` | `GenCls` (2nd order), `GenTransient` (4th), `GenRound` (6th) |
| `src/dynamics/models/avr.rs` | `Sexs`, `IeeeT1`/`ExdC2`, `ExsT1` |
| `src/dynamics/models/gov.rs` | `Tgov1`, a reduced `IeeeG1` |
| `src/dynamics/models/pss.rs` | `Ieeest` / `Stab1` |
| `src/dynamics/models/load.rs` | ZIP from `types::ZipTerm`, low-voltage cutoff |
| `src/dynamics/json.rs` | The native `dynamics` section (extends `src/json.rs`, currently 8 lines) |
| `src/dynamics/dyr.rs` | PSS/E `.dyr` reader — fixed-ish column text, pure `std` |
| `src/dynamics/dyd.rs` | Dynawo `.dyd`/`.par`/`.crv` reader, gated on `iidm` for `quick-xml` |

Plus: a `dynamics` arm in `src/main.rs`, bindings in `src/python.rs`, and a book chapter set.

### The model boundary

```rust
pub trait DynamicModel {
    fn n_states(&self) -> usize;
    fn state_names(&self) -> &[&'static str];
    /// Stamped into Y once, at build. Constant.
    fn norton_admittance(&self) -> Option<Complex<f64>>;
    fn derivatives(&self, x: &[f64], v: Complex<f64>, out: &mut [f64]);
    fn injection(&self, x: &[f64], v: Complex<f64>) -> Complex<f64>;
    /// ∂f/∂x, ∂f/∂V, ∂I/∂x, ∂I/∂V — analytic.
    fn jacobian(&self, x: &[f64], v: Complex<f64>, out: &mut ModelJacobian);
    /// Choose x so that derivatives(x, v) == 0 at this terminal condition.
    fn initialize(&mut self, v: Complex<f64>, s: Complex<f64>) -> Result<Vec<f64>, InitError>;
}
```

**A machine and its controls form one composite device.** `GeneratingUnit { machine, avr, gov, pss }`
owns one contiguous state block and implements `DynamicModel` once. The couplings — the AVR writes
`E_fd` into the machine, the governor writes `P_m`, the PSS writes `v_s` into the AVR, and all three
read machine states or terminal voltage — stay *inside* that block, where they are ordinary partial
derivatives rather than cross-device pattern entries. This makes the analytic Jacobian tractable and
keeps `DaePattern` simple: one dense-ish diagonal block per unit, plus its two bus columns.

The cost is that models cannot be mixed across units arbitrarily at runtime; the composite is built
from a declared combination. That is what PSS/E does and it is adequate here. §8.10 records it as a
limitation.

### Ordering and the pattern

`z = [unit 0 states, unit 1 states, …, load states, …, v_re/v_im per bus]`. The fill-reducing
ordering inside each `LinearSolver` backend does the rest — no hand-tuned permutation. `DaePattern`
is built from a **topology superset**: every entry that any reachable configuration needs is present
from the start, with zeros where a branch is currently out. Value-only events (a fault admittance, a
branch trip, a load step) then become a refill against the cached symbolic factorization; only
structural events (a unit tripped off entirely) force a re-analyze plus a fresh `LinearSolver::new`.
This is the same arrangement `src/continuation/` uses for Q-limit switching.

---

## 4. Initialization, and the gate that makes it survivable

Every RMS run starts from a converged power flow, then works **backwards** through each device:

1. `run_power_flow_analysis` gives the terminal `(V, S)` at each machine bus.
2. The machine's stator equations give `δ`, `e'_q`, `e'_d` (and the subtransient states) from
   `(V, S)`, and hence the `E_fd` required to hold them.
3. The AVR's states are set so its *output* is exactly that `E_fd` and its derivatives are zero —
   which fixes `V_ref` rather than reading it from the file.
4. The governor's states are set so its output is exactly the scheduled `P_m`, fixing the load
   reference.
5. The PSS initializes to zero output by construction (washout filters).

This is the most bug-prone code in any RMS simulator, and it has a decisive self-gate:

> **Run the simulation with no disturbance at all and assert every state is constant.**

If initialization is right, every derivative is zero at `t=0` and stays zero; the trajectory is a
horizontal line to machine precision. If any device is initialized inconsistently — a sign flipped,
a per-unit base missed, a saturation term forgotten — the state drifts immediately and visibly. This
is gate **G1** and it is run for every model in the library, not once.

---

## 5. Events and discontinuities

`EventKind`:

| Kind | Effect | Pattern |
|---|---|---|
| `BusFault { bus, y_fault }` | shunt admittance added at a bus | value-only |
| `ClearFault { bus }` | removed | value-only |
| `BranchTrip { branch }` / `BranchClose` | branch admittance zeroed / restored | value-only |
| `LoadStep { bus, dp, dq }` | injection changed | value-only |
| `UnitTrip { unit }` | device removed from the DAE | **re-analyze** |
| `Reference { unit, kind, value }` | `V_ref` / `P_ref` step | value-only |

**The algebraic variables jump; the differential ones do not.** At an event time the trapezoidal
rule must *not* be applied across the discontinuity. The handling:

1. Step exactly onto `t_event` (the step is truncated so events always land on a step boundary — no
   root-finding is needed because every event in scope is time-scheduled, not state-triggered).
2. Apply the topology/value change.
3. Hold `x` fixed and solve the **algebraic block alone** for the new `V` — a Newton on
   `Y V − I_inj(x, V) = 0` with `x` frozen.
4. Resume integration from `(x, V_new)`.

Gate **G9** asserts that after step 3 the algebraic residual is at machine zero and `x` is
bit-identical across the jump.

**Trapezoidal ringing.** The trapezoidal rule is A-stable but not L-stable, so a discontinuity
excites a numerical oscillation at the fastest mode that decays only as `(-1)^k`. This is the
best-known artifact in RMS simulation and the standard fix is cheap: take **two backward-Euler steps
immediately after each event**, then return to trapezoidal. Backward Euler is L-stable and kills the
mode in one step, at the cost of `O(h)` accuracy for two steps out of thousands.

**State-triggered events** (relay trips on an under-voltage or over-frequency threshold) are
deliberately out of scope for this plan; they need a root-finder in the step, and the Illinois
locator in `src/continuation/events.rs` is the obvious thing to reuse when they arrive.

---

## 6. Readers

All three requested, in this order of dependence:

1. **Native JSON** (`src/dynamics/json.rs`). Extends `NetworkData` with a `dynamics` object. This is
   the internal representation every other reader targets, so it is written first and is the only
   one the core tests need.

   ```json
   { "buses": [...], "lines": [...],
     "dynamics": {
       "units": [ { "at_bus": 0, "mbase": 100.0,
                    "machine": { "model": "gentransient", "h": 6.4, "d": 0.0,
                                 "ra": 0.0, "xd": 1.8, "xd_p": 0.30, "td0_p": 8.0,
                                 "xq": 1.7, "xq_p": 0.55, "tq0_p": 0.4 },
                    "avr": { "model": "sexs", "k": 200.0, "ta": 0.05, ... },
                    "gov": { "model": "tgov1", "r": 0.05, "t1": 0.4, ... } } ],
       "events": [ { "t": 1.0, "kind": "bus_fault", "bus": 4, "x_fault": 0.001 },
                   { "t": 1.1, "kind": "clear_fault", "bus": 4 } ],
       "observe": ["unit.*.delta", "unit.*.omega", "bus.*.vmag"]
     } }
   ```

2. **PSS/E `.dyr`** (`src/dynamics/dyr.rs`). Pure `std`, no dependency. This is what unlocks the
   published IEEE 14/39/118 dynamic cases, and it is the format ANDES itself reads — so the same
   file drives both gridoxide and the reference, which removes an entire class of "the two tools
   were given different data" disagreement.

3. **Dynawo `.dyd`/`.par`** (`src/dynamics/dyd.rs`). `src/iidm.rs` already reads the `.xiidm` half of
   every Dynawo case, including `tests/data/iidm/nordic32.xiidm`, so this closes the loop on a
   corpus that is already half-imported. Gated on `iidm` for `quick-xml`, following that feature's
   deliberately version-tolerant posture. `.crv` is read too, to pick the observed variables.

---

## 7. Validation

All three references, because each covers a different failure and none subsumes another.

**The primary gate is analytic and needs nothing installed.** A classical machine against an
infinite bus through a lossless reactance, with a three-phase fault at the machine terminal
(`P_e = 0` during the fault) and full recovery on clearing. The equal-area criterion gives the
critical clearing angle in closed form:

\\[ \delta_0 = \arcsin\!\frac{P_m}{P_{max}}, \qquad \delta_{max} = \pi - \delta_0 \\]
\\[ \cos\delta_{cc} = \cos\delta_{max} + \frac{P_m}{P_{max}}(\delta_{max} - \delta_0) \\]

and with `P_e = 0` the swing equation integrates exactly during the fault, `δ(t) = δ_0 + \frac{\Omega_b P_m}{4H}t^2`, so

\\[ t_{cc} = \sqrt{\frac{4H(\delta_{cc} - \delta_0)}{\Omega_b P_m}} \\]

This is a closed-form answer for the exact quantity the simulator exists to compute, it exercises
the integrator and the event handling precisely where they matter, and it has no reference
dependency at all. It is the analogue of `CONTINUATION_PLAN.md` §7's two-bus nose.

| Gate | Property |
|---|---|
| **G1** | **Equilibrium invariance.** No disturbance ⇒ every state constant to `1e-12` over 10 s. Run for *every* model in the library, and for every case fixture |
| **G2** | **Analytic CCT.** Simulated critical clearing time matches the closed form above to < 1 ms, bisected over clearing times; stable just below, unstable just above |
| **G3** | **Order of accuracy.** Halving `h` quarters the error against a reference run at `h/16` — trapezoidal is `O(h²)`. Falls to `O(h)` if the BE damping steps are miscounted, which is the point |
| **G4** | **Analytic vs. numerical Jacobian.** Every model's `jacobian` matches a central-difference oracle to `1e-7` relative, over randomized states. Same arrangement as `klu_native::ffi_oracle` and `tests/jacobian_pattern_test.rs` |
| **G5** | **Backend agreement.** `Scalar` / `KluNative` / `Klu` / `Pardiso` agree on the full trajectory to `1e-9`; `Block` is refused (§8.8) |
| **G6** | **ANDES.** IEEE 14 with GENROU + EXDC2 + TGOV1, driven from the *same* `.dyr`, fault-and-clear. Per-variable curve comparison over the run; tolerance stated per variable, not globally |
| **G7** | **Dynawo.** `nordic32.xiidm` + its `.dyd`/`.par`, via `pypowsybl.dynamic`. Same comparison shape as `scripts/bench/iidm_reference.py`, producing committed `tests/data/dynamics/<case>.dynawo.json` so the Rust suite needs neither Python nor Dynawo |
| **G8** | **Event consistency.** After the algebraic re-solve, `‖Y V − I_inj‖∞ < 1e-12` and `x` is bit-identical across the jump |
| **G9** | **Islanding.** A trip that splits the network is detected via `network::connected_components`; an island with no machine is reported, not silently solved |
| **G10** | **Scale.** A large case runs to completion; wall-clock, step count, Newton iterations and re-analyze count recorded in `scripts/bench/` |

**On disagreements with the references.** `SHORT_CIRCUIT_PLAN.md` §6 is the precedent to follow.
PSS/E, Dynawo and ANDES differ in saturation representation, in sign conventions on `v_s`, and in
whether the network is on a per-phase or three-phase base. Where a curve differs for one of those
reasons rather than a bug, the plan is to document the reason and the per-fixture tolerance
individually — not to loosen a global tolerance until everything passes.

---

## 8. Phases

0. **Prerequisites, before any Rust.** Check Dynawo out under `references/` (MPL-2.0), install its
   release binary and set `DYNAWO_HOME` — `pypowsybl 1.16.1` is already in `.venv-pypowsybl` but
   `DYNAWO_HOME` is unset, so *nothing can run today*. Create `.venv-andes`. Confirm both references
   produce a trajectory for a case before committing to compare against them.
1. **Core DAE, no events.** `dae.rs`, `integrator.rs`, `init.rs`, `models/machine.rs::GenCls`,
   fixed-step trapezoidal. Gates G1, G3, G4, G5.
2. **Events.** `events.rs`, the algebraic re-solve, BE damping, the topology-superset pattern,
   re-analyze on structural events. Gate **G2** — the headline result — plus G8, G9.
3. **The model library.** `GenTransient`, `GenRound`, the AVRs, the governors, the PSS, ZIP loads
   with a low-voltage cutoff. G1 and G4 per model, which is what makes this phase mechanical rather
   than risky.
4. **Readers.** Native JSON, `.dyr`, `.dyd`/`.par`/`.crv`.
5. **External validation.** G6 (ANDES, IEEE 14) then G7 (Dynawo, nordic32), with
   `scripts/bench/andes_reference.py` and `scripts/bench/dynawo_reference.py` following
   `iidm_reference.py`'s shape and committing their JSON.
6. **Surfaces.** `gridoxide dynamics` in `src/main.rs`, `gridoxide.dynamics` in `src/python.rs`,
   the book chapters, `docs/src/SUMMARY.md`, the feature-comparison row (❌ → ✅), a benchmark
   entry. G10.

Phases 1–2 alone are a usable transient-stability tool; the plan is not front-loaded with
infrastructure that pays off only at the end.

---

## 9. Risks and limitations

1. **Initialization is the hardest part**, by a distance. Mitigated by G1 being run on every model
   and every fixture, which converts a subtle physics bug into an immediate, loud failure.
2. **Constant-power loads at low voltage** are singular — `I = S*/V*` diverges as `V → 0`, and a
   nearby fault will drive it there. A low-voltage cutoff (below `~0.5 pu`, transition to constant
   impedance) is required; it *changes answers* and must be documented and configurable, as Dynawo
   documents its own.
3. **Trapezoidal ringing after events** — addressed by the BE damping steps in §5, and detected by
   G3 if the count is wrong.
4. **Machine saliency and the Norton stamp** — `x'_q ≠ x'_d` leaves a `δ`-dependent term in
   `I_inj`. Correct, but easy to get wrong in the analytic Jacobian; G4 is the guard.
5. **Per-unit base conversion** between machine MVA and `s_base`. A wrong `H` or `x'_d` produces a
   plausible-looking but wrong swing frequency — the worst failure mode, because it looks right.
   G2's closed form pins it exactly.
6. **Reference availability.** Dynawo is a ~1 GB download that is not installed. Phase 0 exists so
   this is discovered before code depends on it, not after.
7. **Convention mismatches between the three references** will produce disagreements that are not
   bugs (§7).
8. **`Block` backend unsupported**, by construction: it assumes a uniform 2×2 block per bus, and a
   DAE has variable-size device blocks. Refused explicitly, as continuation refuses it.
9. **Islanding after a trip** — an island with no machine has no voltage reference and no meaningful
   dynamic answer. Reported via G9, not solved.
10. **Composite generating units** (§3) mean model combinations are declared, not assembled freely
    at runtime.
11. **No state-triggered events** in this plan (§5), so no relay/protection modelling. That is the
    natural follow-up, and `continuation/events.rs`'s Illinois locator is the piece to reuse.
12. **No EMT, no small-signal.** The small-signal row becomes cheap *after* this lands — it is the
    same `∂f/∂x`, `∂f/∂V`, `∂I/∂x`, `∂I/∂V` blocks reduced and eigen-decomposed — but it is a
    separate plan.

---

## 10. Verification

End to end, after Phase 6:

```bash
# Unit + integration suite, including the analytic CCT gate
cargo test --features dynamics
cargo test --features "dynamics,iidm" --test dynamics_dyd_test

# The headline: a fault on a multi-machine case, from the CLI
cargo run --release --features dynamics -- dynamics \
    tests/data/dynamics/ieee14.json --dyn tests/data/dynamics/ieee14.dyr \
    --stop 10 --step 0.005 --csv /tmp/ieee14.csv

# Backend agreement (G5)
cargo test --features "dynamics,klu" --test dynamics_backend_test

# Regenerate the external references (needs the Phase 0 installs), then re-run
.venv-andes/bin/python  scripts/bench/andes_reference.py
.venv-pypowsybl/bin/python scripts/bench/dynawo_reference.py
cargo test --features dynamics --test dynamics_reference_test

# Python surface
VIRTUAL_ENV=$PWD/.venv-pypowsybl maturin develop --features "python,dynamics"
.venv-pypowsybl/bin/python -m pytest python/tests/test_dynamics.py

# The book builds and the new chapters are linked
mdbook build docs
```

The single most informative check while developing is G1: run any fixture with an empty event list
and confirm the trajectory is flat. It is fast, it needs no reference, and it fails loudly for
almost every mistake in initialization, per-unit conversion, model derivatives and Jacobian
assembly alike.


---

## 11. What happened: phase 1

Landed: `src/dynamics/` with the DAE (`dae.rs`), the integrator
(`integrator.rs`), initialization (`init.rs`), the model boundary
(`models/mod.rs`) and the classical machine (`models/machine.rs`), behind the
`dynamics` feature and built in CI. Seven gates in `tests/dynamics_test.rs`,
all passing; the full suite is unaffected.

Phase 0's cheap half is done — `.venv-andes` holds ANDES 2.0.0. Dynawo is
**not** installed and `DYNAWO_HOME` is still unset; that stays a phase-5
prerequisite, since nothing in phases 1–4 depends on it.

### The formulation held

§2 needed no correction. The network block really is constant, the device
stamps really are local, and the identity that `(a, b) = (½, ½)` versus
`(1, 0)` is the only difference between trapezoidal and backward Euler meant
one residual and one assembly serve both — which will matter in phase 2, where
the rule switches mid-run.

The one structural decision the plan did not name: `∂I/∂V` is **folded onto**
its bus's diagonal group at fill time rather than emitted as its own triplet.
Emitting it would put two entries at the same `(row, col)`, and the
positional-values contract `LinearSolver::factor_and_solve_values` is written
against cannot survive duplicates. Folding is one line and keeps the triplet
list injective.

### Two things this plan was wrong about

**1. `LinearSolver::new` needs real values, not a placeholder pattern.**
`DaePattern` originally handed `S::new` a triplet list with every value set to
`1.0`, on the reasoning that only the `(row, col)` halves are read there. That
is true of `RealSparseSystem`, which does symbolic analysis only — and false of
`KluNative` and `Klu`, which *factor numerically* inside `new` and correctly
refused a matrix of all ones as singular. Gate G5 caught it: `Scalar` passed
and `KluNative` returned `Singular` at `t = 0`.

The fix is what `solver::newton_raphson_cached` already does and what this plan
should have copied outright: fill first, analyze against those values, and
build the backend lazily on the first iteration that has them. `to_triplets`
takes a values slice for exactly this reason.

Worth stating plainly because it generalizes: **the backends are not
interchangeable at construction time**, only at solve time. Any future code
that builds a `LinearSolver` must have real values in hand.

**2. §4's equilibrium check does not catch a mis-declared device/load split.**
The plan claimed the check would catch "a device whose declared terminal power
does not match what the power flow put at its bus". It does not, and cannot: a
wrong `DeviceSpec::s` is *self-consistent*. The machine initializes to an
equilibrium at the power it was told it makes, the remainder is absorbed into
the bus's constant admittance, and both residuals are exactly zero. What comes
out is a different machine — smaller output, smaller internal EMF, smaller
rotor angle — swinging against a network that makes the difference up from
something inert.

So G1 is a **code-correctness** gate, not an input-validation one. It catches
sign errors in derivatives, missed per-unit conversions, a Norton stamp that
disagrees with the impedance its own model initialized against, and mistakes in
residual assembly — all of which are otherwise silent. It cannot catch bad
input that happens to be consistent. `init.rs`'s module doc says so, and
`tests/dynamics_test.rs::a_mis_declared_split_is_silent_and_changes_the_machine`
pins the behaviour so it stays a known limitation rather than a surprise.

That is the strongest argument for reaching phase 4's readers sooner rather
than later: a `.dyr` or `.dyd` file states the split, so the caller stops being
the place it can go wrong.

### Gates, as built

| Gate | What it does | Result |
|---|---|---|
| G1 | Undisturbed SMIB, 10 s: angle, speed and terminal voltage constant | drift < 1e-12 |
| G4 | Analytic vs. central-difference Jacobian, five probes off-equilibrium | < 1e-6 relative |
| — | Small-signal period vs. `2π/√(Ω_b·P_max·cos δ₀ / 2H)` | within 0.2% |
| — | Undamped swing keeps amplitude; damped swing decays | both hold |
| G3 | Order of accuracy from three step sizes | 2.0 ± 0.2 |
| G5 | `Scalar` vs. `KluNative` trajectories | < 1e-9 |
| — | A mis-declared split builds, and gives a different machine | pinned |

The small-signal period gate was not in §7's list and turned out to be the most
useful one in phase 1. G2's critical clearing time needs events and is the
headline for phase 2; the oscillation period is its little brother, needs no
event machinery, and exercises the same three things — the swing equation, the
network coupling and the integrator — against a closed form. It found nothing,
which is the point: it was passing before the two bugs above were found, and it
would not have been if either had touched the physics.

### Not done, and deliberately

No events, so no fault, no trip, no `t_cc`. No model library beyond `GenCls`.
No readers — a system is assembled from a Rust `SystemSpec`, which is why the
mis-declared-split hazard is currently reachable at all. No CLI, no Python, no
book chapter. Loads are constant impedance, converted at the solved voltage.


---

## 12. What happened: phase 2

Landed: `src/dynamics/events.rs`, event-aware stepping in `integrator.rs`, and
structure-preserving Y-bus reassembly on `DynamicSystem`. Seven more gates in
`tests/dynamics_events_test.rs`; fourteen in total, all passing, and the rest of
the suite is unaffected.

### G2, the headline

| | |
|---|---|
| Equal-area criterion | **0.309335 s** |
| Simulated (bisected, `h` = 2 ms) | **0.309348 s** |
| Difference | **12 µs** |

against the 1 ms this plan asked for. The setup is the one the closed form is
written for: `P_max` = 2.094, `P_m` = 0.800, `δ₀` = 0.3919 rad, a bolted fault
at the machine terminal so `P_e` = 0 during it, and a post-fault network
identical to the pre-fault one.

### Nothing needed re-analysis

§5's claim that every event in scope is value-only held exactly, and
`network::build_ybus_with_outages` turned out to already provide the
topology-superset property this needs — it re-stamps an out-of-service branch's
positions at zero rather than dropping them, for reasons of its own. Combined
with stamping every bus's diagonal unconditionally at build (so a fault can
land anywhere), `LinearSolver` analyzes **once for the whole run**: a fault, a
clearing and a trip are all numeric refactorizations against one symbolic
factorization. The `reanalyses` counter this plan implied would be needed does
not exist, because nothing increments it.

### Where this plan was wrong again

**§5 said the differential states are continuous across an event. The code did
not make them so.** With `h·a = 0` the Jacobian is block lower triangular —
`[I, 0; −∂I/∂x, Y − ∂I/∂V]` — so forward substitution gives `Δx = 0` and the
plan reasoned no further. But no backend does forward substitution in that
order: each applies its own fill-reducing permutation and partial pivoting, and
roundoff leaks across the block boundary. Measured at ~7e-11 rad on a faulted
network, whose conditioning the large fault admittance dominates.

Small, but wrong in kind and cumulative over a run with many events. A rotor
angle is *defined* to be continuous across a discontinuity, so `newton` now
takes a `pin_states` flag and simply does not apply `Δx` during an algebraic
re-solve. That makes the property exact rather than approximate, and the gate
asserts bit-identity rather than a tolerance.

### The test that had to be rewritten twice

Checking that a terminal fault removes the electrical power looked trivial:
with `P_e` gone the acceleration is constant, so the angle is exactly quadratic
and the trapezoidal rule integrates it with **no error at all**. The observed
departure was 2.5e-5 rad.

Two explanations were offered and each was rejected by an experiment before the
right answer appeared — which is that *both* were true, of different
configurations:

- with the damping steps **off**, the departure is the residual `P_e` that a
  finite fault admittance still lets through. It scales as `1/y_fault`:
  2.66e-5 rad at `1e6`, 2.66e-7 at `1e8`.
- with the damping steps **on**, it is backward Euler's own overshoot. Each
  damping step evaluates `δ̇` at the end of the step, so on a linearly growing
  speed it overshoots by exactly `Ω_b·a·h²/2` with `a = P_m/2H`. Two steps
  leave a permanent offset of `Ω_b·a·h²` = 2.513e-5 rad at `h` = 1 ms,
  independent of the fault, and matching the closed form to better than 1%.

At the default settings those are 2.7e-5 and 2.5e-5 with **opposite signs**.
Measuring their combination says nothing; the first attribution attempt varied
`y_fault` with the damping left on and saw no change at all, which looked like
a refutation and was actually the other term dominating. The gate now measures
each against its own closed form, and also checks that the damping offset falls
as `h²` — which it does, despite backward Euler being first-order, because only
a fixed number of steps ever use it.

This is the most useful thing phase 2 produced. It is a complete, quantitative
account of every departure from an exactly-known trajectory, which is a much
stronger position than a passing tolerance.

### The damping steps have nothing to damp

`plans/RMS_PLAN.md` §5 justified backward-Euler damping by the trapezoidal
rule's ringing after a discontinuity. That justification is sound and the
mechanism is wired in, but **it cannot bite yet**: ringing needs a mode fast
enough that `h·λ` is large, and the only differential mode in the present
library is a classical machine's swing at about 1 Hz. A gate pins that turning
the damping on and off does not move the critical clearing time, which is the
honest claim to make about it today. The exciters and governors of phase 3 have
time constants of 20–50 ms, which is where it starts to matter, and the offset
quantified above is what it will cost.

### Also landed

Branch trip and close, load steps (as an admittance change, matching the load
model `init` already chose), and a de-energized-island warning: a switching
event that leaves a group of buses with no machine and no fixed bus is
reported rather than presented as a trajectory. Events land bit-exactly on
their own times — the step is truncated and the time snapped rather than
accumulated — so an event time need not be a multiple of the step. An event
naming a bus or branch that does not exist is skipped and named, not fatal.

### Deferred from §5, deliberately

`UnitTrip` is the one event kind in §5's table that is **not** value-only: a
tripped unit's states leave the system, which changes the variable layout and
so needs a genuine re-analysis. It is deferred to phase 3, where the model
library gives it something worth tripping. `EventKind` does not name it, so
nothing pretends to support it.

State-triggered events — a relay on an under-voltage or over-frequency
threshold — remain out of scope as §5 said, and
`continuation::events`'s Illinois locator remains the piece to reuse.


---

## 13. What happened: phase 3

Landed: the model library, and the composite that holds it together.
`GeneratingUnit` (`models/unit.rs`), `GenTransient` and `GenRound`
(`models/machine.rs`), `Sexs`, `Tgov1`, `Stab1`, `ZipLoad`, and unit
trip/close. Fourteen gates in `tests/dynamics_models_test.rs`; twenty-eight
dynamics gates in total, all passing, and the rest of the suite is unaffected.

### The composite, and why the chain rule rather than enumeration

§3 said a machine and its controls form one device with one contiguous state
block. What it did not say is how the couplings get written, and writing them
per combination would be combinatorial — three optional controls is eight
combinations before any second model exists.

Instead each part declares its derivatives with respect to its own states and
its own **scalar** input, and the unit composes them by the chain rule over the
signal graph. The graph turns out to be acyclic and shallow, and no block's
output depends on its own input through another block, so one forward pass
evaluates it and one sweep differentiates it. That is what made eight
combinations cost the same as one.

The governor's direct feedthrough from `Δω` to `P_m` looks like a loop and is
not: `P_m` enters `ω̇`, and `ω̇` does not enter `P_m` — only `ω` does.

### The two identities that pin the rotor-frame algebra

Neither was in this plan, and they turned out to be the most valuable gates in
the phase, because both are **exact** rather than tolerance-bounded:

1. **A machine reproduces its own terminal current.** After initialization,
   `injection(x₀, V₀) − y_norton·V₀` must equal `conj(S/V)` to 1e-12. A
   transposed sine and cosine, a `q` axis defined the other way round, or a
   sign slip in the stator solve all survive a plausibility reading; none
   survives this.
2. **A machine with no subtransient saliency cancels its own Norton stamp
   exactly.** When `x''_d = x''_q` the machine really is an impedance in the
   network frame, the stamp is that impedance, and `∂I/∂V` comes out zero to
   the last bit. This also makes visible what §2 claimed about the stamp — that
   it is a conditioning device changing no answer — as an identity rather than
   an assertion.

### Deriving the sixth-order model instead of citing it

§7 predicted that convention mismatches between references would produce
disagreements that are not bugs. That prediction landed earlier than expected
and inside a single model: published statements of the subtransient machine
disagree with each other about the sign of `ψ_2q` and of `e'_d`, and a
formulation copied from one source and checked against another is internally
consistent and physically wrong.

So the conventions were **derived**, from two requirements the code can be held
to:

- the steady state must reduce to `GenTransient`'s, term for term — which pins
  `e''` and the damper flux equations;
- the two axes must map onto each other under
  `(e'_q, ψ_1d, i_d) ↔ (e'_d, ψ_2q, −i_q)` — which pins the sign of the
  damper-coupling correction in `ė'_d`, something the first requirement cannot
  see because that correction vanishes at steady state.

A pleasant confirmation fell out: both damper equations reduce to `−Δ/T''` for
the same `Δ` the correction terms use. One expression serving both is itself
evidence the signs agree.

`the_subtransient_machine_reduces_to_the_transient_one` is the gate: take `x''`
up to `x'` and the sixth-order machine must trace the fourth-order machine's
trajectory through a fault. Since `GenTransient` is independently pinned by the
two identities above, that transfers the confidence. **What it cannot confirm is
the magnitude of the damper coupling in the regime where it matters** — that
waits for phase 5.

Saturation is not implemented, and deliberately: it is the single largest
convention divergence between PSS/E, Dynawo and ANDES, and choosing a
representation is better done with the reference comparison actually running
than from a reading of three disagreeing sources.

### Unit trip is value-only after all

§12 recorded `UnitTrip` as the one event kind that is genuinely structural,
because a tripped unit's states leave the system. That was a failure of
imagination. **Freezing** the unit rather than removing it keeps the variable
layout and the sparsity pattern identical — its rows become `x₁ − x₀ = 0`, its
Jacobian block the identity the implicit rule contributes — so a trip is a
refill like every other event and nothing re-analyzes. The run's single
symbolic factorization still serves.

Freezing is also the better model. Nothing in the network can observe a
disconnected machine's rotor, so integrating it would be tracking a quantity no
result depends on, and reconnecting it properly would need synchronization,
which is not modelled.

### The gate that took four attempts

`a_stabilizer_leaves_no_steady_signal`. The claim — a washout contributes
nothing at steady state — is right; every way of asserting it was wrong first.

A washout's **state does not go to zero, it goes to the input**: `ẋ = (u − x)/T_w`,
so `x → u` and it is the *output* `u − x` that vanishes. That is the mechanism,
not an artifact. With no governor the frequency stays permanently low at
`(P_m − P_e)/D`, and the washout state settles on exactly that. Then: the state
trails the input by `T_w·dΔω/dt` while the input is still creeping, so a fixed
tolerance was really asserting the frequency had stopped moving, which on an
asymptotic approach it never has. And everything downstream is that same
residual scaled — the first lead-lag's leftover is exactly `K(1 − T₁/T₂)·y₁`.

The gate now predicts each residual from the observed drift rate rather than
bounding it. That is a better test than the one intended, and the same pattern
as §12's: a complete quantitative account of every departure beats a passing
tolerance.

### Limits are absent, on purpose

No exciter ceiling, no governor valve limit, no stabilizer output clamp. A hard
clamp makes the right-hand side non-smooth, so the analytic Jacobian acquires a
discontinuity the step's Newton solve can chatter against, and doing it properly
needs non-windup logic plus limiter state. Half-implemented limits would be
worse than none, because they would look present. A model with no limits is at
least honestly unlimited, and its `E_fd` can be read to see whether a study
would have hit one.

### A limitation worth recording

An islanded machine running off-nominal accumulates rotor angle without bound —
`δ` reached −10⁴ rad in a 400-second gate, which is physical, since `δ` is
measured against a reference rotating at nominal frequency. `sin δ` and `cos δ`
stay accurate at that magnitude, but the argument's relative precision degrades
with `|δ|`, and a much longer run or a much larger frequency excursion would
start to feel it. Not a phase-3 problem; worth knowing before anyone runs an
hour of simulated time.


---

## 14. What happened: phase 4

Landed: all three readers. `src/dynamics/json.rs` (the native format),
`src/dynamics/dyr.rs` (PSS/E records), `src/dynamics/dyd.rs` (Dynawo, gated on
`iidm` for its XML reader). Twenty-one more gates; forty-nine dynamics gates in
total.

### The format defines itself

The native format's model blocks **are** the models' own parameter structs —
`GenRoundParams` and friends derive `Deserialize` — rather than a parallel set
of file structs mapped across. The two therefore cannot drift: a renamed field
is a missing-field parse error at the point of use, not a silently defaulted
zero somewhere downstream.

### §11's hazard is closed for the ordinary case

A wrong device/load split is self-consistent and no gate can see it. A *file*
removes the need to state it: a device may give its own `p` and `q`, and if it
does not, it takes what its bus's solved injection has left. One device omitting
it is unambiguous; two at the same bus is refused by name.

The hazard is not gone — a file that states the split wrongly is still silent —
but the common case no longer requires anyone to state it at all.

### The `.dyr` reads half a pair of files, and says so

Two things every machine model needs are simply absent from a `.dyr`: the MVA
rating and the armature resistance, which are the `.raw`'s `MBASE` and `ZSORCE`.
So is the bus numbering. All three are demanded from the caller, and the error
message says where they live. That is a real limitation of the format, not of
the reader, and stating it beats inventing defaults.

Two parsing details earned their gates. A record ends at the `/` and not at the
newline, so a three-line `GENROU` is one record with fourteen parameters rather
than three truncated ones. And a leading `/` is a *comment* only when no record
is open — mid-record the same character is the terminator, and files do put it
on a line of its own. Nothing but "is anything pending" distinguishes the two.

`SEXS`'s first field is the ratio `T_a/T_b` and not a time constant. Misreading
it gives a plausible exciter an order of magnitude too fast.

### Dynawo: the fixtures are real, and that was the point

`.par` is addressed by parameter **name**, so the only real risk is looking up a
name no file uses — and a hand-written fixture would have agreed with whatever
the reader happened to expect. The fixtures under
`tests/data/dynamics/dynawo/` are therefore copied verbatim from Dynawo's own
repository (MPL-2.0, provenance recorded beside them), and the gates assert
against the file's own literal values.

Getting them cost a sparse shallow clone, not the 1 GB install §8 phase 0
assumed.

**Two findings changed the plan.**

First, `GeneratorSynchronousFourWindings*` maps onto `GenRound` parameter for
parameter, with `generator_SNom` as the machine base — no approximation
anywhere. But `...ProportionalRegulations` carries a purely *proportional*
voltage regulator and governor, which are **not** approximations of `Sexs` and
`Tgov1` — they are different devices with no states at all. Rather than invent
time constants a Dynawo file never stated, the library grew `VrProportional` and
`GoverProportional`: zero-state controls, pure feedthrough, exact
correspondences. They cost about eighty lines each and they are also what
Kundur's worked examples use.

That a zero-state control drops into the chain rule with no special case is
worth noting — it is the same property that let `ZipLoad` be a device with no
states.

Second, and larger: **Dynawo's repository ships its own reference outputs.**
`examples/…/reference/outputs/curves/curves.csv` holds the trajectories its
solver produced, committed. So phase 5's Dynawo comparison needs **no Dynawo
install at all** — the same arrangement `tests/data/ucte/` and
`tests/data/iidm/` already use for pypowsybl. §8's phase-0 prerequisite was
wrong about the cost of the most expensive gate in the plan.

That vendored case also records `PMIN : activation` in its own timeline: its
governors hit their power limits, which this library does not model. So it is
known *in advance* to diverge, for a stated reason — which is a much better
position than discovering it during the comparison.

### One conversion is inferred rather than read

`governor_KGover` is a gain on the machine's own `governor_PNom`, and
`GoverProportional` wants one on the network base, so the reader applies
`k = KGover · PNom / s_base`. Every other quantity is read as stated. This one
rests on a reading of Dynawo's base convention rather than on anything in the
file, and it is flagged in the module doc so phase 5 knows where to look first
if the frequencies disagree.

### Saturation, twice, incompatibly

PSS/E states it as `S(1.0)`/`S(1.2)`, two points on a curve. Dynawo states it as
`md`/`mq`/`nd`/`nq`, an exponential characteristic. They are not convertible
without committing to a curve shape. Both readers parse it, neither uses it, and
both report a nonzero value — which is exactly the divergence §7 predicted and
exactly why §13 declined to pick a representation before a reference was
running.


---

## 15. What happened: phase 5

**G7 (Dynawo) is done. G6 (ANDES) is not.** Five gates in
`tests/dynamics_reference_test.rs`; fifty-four dynamics gates in total.

### The case, and why this one

Kundur's Example 13.2 as Dynawo ships it (`examples/DynaSwing/Kundur_Example13`,
the `SetPoint` variant): a sixth-order machine with **no regulators**, an
infinite bus, two parallel lines, a fixed-ratio transformer, a bolted fault at
the line junction, and a line trip on clearing.

It was chosen because every element is something gridoxide models *exactly* —
no tap changers, no limits, no saturation, no fifth-order machine. Nothing had
to be excused before the comparison started, which is what makes a
disagreement mean something. §14's finding held: Dynawo's committed
`reference/outputs/curves/curves.csv` **is** the comparison data, so no Dynawo
install was needed.

### What agreed, and how well

| | |
|---|---|
| Terminal angle from the power flow | matches the file's `UPhase0` to 1e-4 |
| `δ₀` against Dynawo's `theta(0)` | **6e-5 rad**, from two independent derivations |
| Air-gap power convention | `P_m` exceeds terminal power by exactly the copper loss, in both |
| Rotor angle through the fault | **9.1e-4 rad** |
| Rotor speed through the fault | **6.6e-5 pu** |
| Terminal voltage during the fault | **9.0e-4 pu** |
| Rotor angle over the first swing | 1.9e-2 rad |
| The published 70 ms clearing time | survivable in both; 500 ms in neither |

The `δ₀` agreement is the one worth dwelling on. Two implementations, deriving
the same rotor angle from the same terminal condition by independent routes, is
much stronger evidence for the dq conventions than any self-consistency check
could be — and it is the thing §13 built two internal identities to protect.

### The disagreement, chased down rather than absorbed

After clearing the two separate steadily. Excluded, each by experiment:
step size (gridoxide's answer is converged to 1e-4 rad from `h = 4 ms` to
`h = 0.0625 ms`); the nominal frequency (at 60 Hz the machine loses synchronism
outright); the tripped branch (tripping the other, or neither, gives a wholly
different trajectory); and **the sign of the `q`-axis damper coupling §13 left
open** — flipping it moves the answer by under 1e-3 rad, which also means *this
case does not settle that question either way*.

Localizing it took separating the power-angle relation from the accumulated
angle: at matched rotor angle in the first tenth of a second after clearing,
gridoxide's terminal power is **0.6% below** Dynawo's, and the voltage error
*steps* at the clearing instant rather than growing smoothly. So it is
algebraic, not numerical.

**Reading Dynawo's own Modelica settled it.** From
`Electrical/Machines/OmegaRef/BaseClasses/BaseGeneratorSynchronous.mo`:

```text
udPu = (Ra + RTfo)·idPu − omegaPu·lambdaqPu
uqPu = (Ra + RTfo)·iqPu + omegaPu·lambdadPu
2·H·der(omegaPu) = cmPu·PNomTurb/SNom − cePu − DPu·(omegaPu − omegaRefPu)
PePu = cePu·omegaPu
```

Dynawo keeps `ω` on the speed-voltage terms and writes the swing equation in
**torque**. gridoxide makes the classical RMS approximation in both places.
The two differ by exactly a factor of `ω`, so they agree at synchronous speed
and part in proportion to the speed deviation — which is exactly the observed
pattern: nil at `t = 0`, `0.04%` early in the fault at `ω − 1 = 0.0015`, `0.6%`
after clearing at `ω − 1 = 0.009`.

Neither form is wrong. Kundur §13.3 states the approximation explicitly; Sauer
& Pai keep the terms. What changed is that the cost is now **measured** —
0.6% of terminal power per 0.9% of speed deviation — recorded in
`models/machine.rs` where the equations are, and pinned by a gate so that
adopting the full form would visibly drive it to zero rather than pass
unnoticed.

This is the most valuable thing phase 5 produced, and it is exactly what §7
said an external gate was for: a difference that is invisible to every
self-consistency check, because both formulations are internally perfect.

### G6 (ANDES) — attempted, not achieved

ANDES 2.0.0 is installed and its power flow reproduces the case correctly
(bus-3 angle 0.494496 against Dynawo's 0.49445) once the lines are given
matching `Vn1`/`Vn2` — ANDES defaults them to 110 kV, which silently rescales
every impedance by `(110/400)²` and was worth an hour on its own.

But ANDES's **own initialization fails** on the hand-built case, with residuals
of 0.16 in the bus-3 angle equation and 0.065 in its voltage, and the machine
then loses synchronism where both other tools keep it. Its `δ₀` comes out
1.2208 against 1.2240 for the other two. That is a setup problem in how the case
was declared to ANDES, not a finding about anything — and diagnosing it is ANDES
work rather than gridoxide work.

Recorded as outstanding rather than reported as a result. The one datum worth
keeping: gridoxide and Dynawo agree with each other on `δ₀` roughly fifty times
more closely than either agrees with the ANDES figure, and the ANDES figure
comes from a failed initialization.

### Still outstanding

G6, and phase 6 entirely — the CLI, the Python bindings, the book chapters and
the feature-comparison row.


---

## 16. What happened: phase 6

The surfaces. `gridoxide dynamics` in `src/main.rs`, `gridoxide.dynamics` in `src/python.rs`, seven
book chapters, the feature-comparison row, and the scale measurement §7 called G10. Twelve more
gates — five for the CLI, seven in Python — bringing the total to **sixty-six**.

### G10, measured

A ring of alternating generator and load buses, every generator a sixth-order machine with an
exciter and a governor, through a bolted fault at a 5 ms step:

| buses | units | unknowns | build | 3 s run | per step | Newton/step |
|---|---|---|---|---|---|---|
| 16 | 8 | 112 | 0.7 ms | 30 ms | 0.050 ms | 1.34 |
| 256 | 128 | 1 792 | 0.5 ms | 682 ms | 1.14 ms | 1.34 |
| 4 096 | 2 048 | 28 672 | 9.2 ms | 16.4 s | 27.3 ms | 1.34 |

**Time per step is near-linear in system size** — 256× the unknowns costs 546×, an exponent of
1.14 — which is what §2's formulation was chosen for. And **Newton iterations per step are flat at
1.34** across four orders of magnitude, which is the signature of an exact analytic Jacobian.

§8's phase 6 also asked for a "re-analyze count". There is no such counter, because nothing
re-analyzes: §12 and §13 between them made every event value-only, so one symbolic factorization
serves a whole run however long and whatever happens in it.

The first sweep stopped at 1024 buses — not on the dynamics but on the *power flow*, because a
4096-bus ring carrying heavy flow is a hard base case. Lowering the injections fixed it. Worth
recording because the ceiling looked like a dynamics limit and was not; a synthetic benchmark can
put a wall in front of the thing it is trying to measure.

### The surfaces

The CLI summary is deliberately about **machines** rather than states: a trajectory has hundreds of
columns and almost none is the answer to anything, while "did each machine stay in step, how far did
the frequency go, how deep did the voltage dip" are. It also prints the initial state derivative,
because a nonzero one invalidates everything after it and a reader should not have to ask.

The Python binding takes an optional `events` list in the **same vocabulary the document uses**, so
a caller sweeping clearing times has one event language rather than two. It returns plain lists
rather than numpy arrays, keeping the extension free of a numpy dependency.

### The row

`docs/src/reference/feature_comparison.md`'s RMS row goes ❌ → ✅. The small-signal row's note
changes too: it used to say the row "presupposes the RMS row", and now records that `∂f/∂x`,
`∂f/∂V`, `∂I/∂x` and `∂I/∂V` are already assembled analytically every step and oracle-checked, so
what remains is the reduction and the eigen-decomposition rather than the modelling.

### What the whole plan got right, and what it did not

Right: the formulation (§2 needed no correction), the choice of Dynawo as the reference, and the
estimate that the corpus would come close to free — it was freer than expected, since Dynawo ships
its own solver's outputs.

Wrong, and each recorded where it happened: the placeholder-pattern assumption about
`LinearSolver::new` (§11); the claim that the equilibrium gate catches a mis-declared split (§11);
the claim that the states were already continuous across an event (§12); that `UnitTrip` was
structural (§13); and that the Dynawo gate needed a Dynawo install (§14).

And the thing no part of this plan anticipated: that the largest real risk was not the integrator or
the DAE but the **conventions** — the subtransient signs, saturation's two incompatible
representations, and the `ω ≈ 1` stator approximation. None of those is a bug in anyone's code, and
none of them is visible to a self-consistency check. Two were settled by declining to copy a source
and deriving instead; the third was settled by reading the reference implementation's own equations
after its trajectory diverged from ours.

### Still outstanding

- **G6 (ANDES)** — see §15.
- Regulator limits, saturation, state-triggered events, a fifth-order machine.
- Coupling the Dynawo reader to `src/iidm.rs` so a full IIDM-plus-`.dyd` case loads in one step.
- Small-signal analysis, which is now much closer than the plan assumed.


---

## 17. Beyond the plan: closing the `ω ≈ 1` finding

§15 measured a 0.6% disagreement with Dynawo, attributed it by reading Dynawo's Modelica, and
stopped there — the attribution was an argument, not a demonstration. This closes it.

Both machine formulations are now available on every machine, selected by
`with_speed_voltages(true)` in Rust, `"speed_voltages": true` in a document, `--speed-voltages` on
the command line, or `speed_voltages=True` in Python:

```text
approximate (default):  v_d = −r_a·i_d − λ_q            2H·ω̇ = P_m   − P_e − D·Δω
full (Dynawo, S&P):     v_d = −r_a·i_d − ω·λ_q          2H·ω̇ = P_m/ω − c_e − D·Δω
```

Turning the full form on:

| | approximate | full form |
|---|---|---|
| Post-fault power offset at matched angle | −0.611% | **−0.026%** |
| Rotor angle vs Dynawo, first swing | 1.89e-2 rad | **1.76e-3 rad** |
| Rotor angle vs Dynawo, whole 5 s | 2.06e-1 rad | **3.81e-2 rad** |

A factor of twenty-four on the offset. Had the approximation not been the cause, switching it off
would have moved the number somewhere arbitrary rather than to zero. That is the difference between
attributing a discrepancy and proving the attribution.

The residual 0.026% is genuinely unexplained and is bounded by a gate so it stays visible.

### It stays the default

Two reasons, both stated in the code. The `ω ≈ 1` assumption is what makes the phasor formulation
coherent in the first place. And every closed-form gate in this crate is derived from the power
form — the equal-area criterion above all, whose critical clearing time gridoxide reproduces to
12 µs. Changing the default would have meant either losing that gate or rewriting the closed form it
checks against.

### What it cost, and what made it cheap

Three machines and their analytic Jacobians. The change is one speed factor `w` — `ω` in the full
form, `1` in the approximate one — threaded through each stator solve, plus its derivative `dw`,
which is `0` in the approximate form and therefore leaves every existing column untouched. One code
path, and the two forms are the same equations at `dw = 0`.

The `ω` column is the one place it is not mechanical: the speed factor enters the stator
coefficients **and** the determinant of the 2×2 solve, so that column is a product rule rather than
the fixed inverse every other column goes through. The finite-difference oracle caught nothing here,
which is the point of having had it all along — it was probed deliberately at `ω = 1.03`, since at
synchronous speed the new terms vanish and the oracle would have been re-checking the old form.

Two gates hold it down: undisturbed, the two forms are **bit-identical** (at `ω = 1` they are the
same equations); disturbed, they differ by the order of the speed deviation and no more.

### Still outstanding

- Regulator limits, saturation, state-triggered events, a fifth-order machine.
- Coupling the Dynawo reader to `src/iidm.rs` so a full IIDM-plus-`.dyd` case loads in one step.
- Small-signal analysis.
- G6 (ANDES) — dropped, not deferred.


---

## 18. Beyond the plan: limits

§13 declined to implement regulator limits, and gave a reason that was right at the time: a hard
clamp makes the right-hand side non-smooth, doing it properly needs non-windup logic plus limiter
state, and half-implemented limits would be worse than none. This does it properly.

Non-windup limits on the exciter's field voltage and the governor's valve, a clamp on the
stabilizer's output, and both readers carrying them through — `SEXS`'s `EMIN`/`EMAX`, `TGOV1`'s
`VMIN`/`VMAX`, Dynawo's `voltageRegulator_EfdMinPu`/`MaxPu` and `governor_PMin`/`PMax`. Five more
gates; **seventy-one** dynamics gates in total.

### Three failures, three pieces

The design was not obvious in advance and the failures found it.

**The first attempt chattered.** Recomputing the active set from each Newton iterate makes the
residual non-smooth *inside* the solve: an iterate landing just above a boundary sees a zeroed
derivative, the next lands just below and sees the full one, and they alternate. An ordinary exciter
ceiling made the step fail outright — `NewtonFailed at t = 1.015`, on a case that completes without
limits. So the active set is **latched once per step**, before any iteration. The step is then
smooth and its Jacobian is exact for what is actually being solved; a limit engages one step late,
which at five milliseconds is not worth avoiding.

**The second attempt overshot.** The step that *crosses* a boundary still carries the state past it,
because the trapezoidal rule averages a start-of-step derivative that was still driving hard with an
end-of-step one that has been zeroed — and a latched derivative cannot then bring it back. Measured
at 0.1 pu past a 2.6 pu ceiling. So states are **projected** onto their limits after each accepted
step. The projection is exact rather than a correction: a non-windup state has no legitimate value
outside its limits. Locating the crossing time instead would be the state-triggered-event machinery
this library deliberately does not have.

**The third failure was a wrong test.** Non-windup was asserted as "comes off the ceiling within
100 ms of clearing", and it does not — it stays there for seconds, because the field flux decayed
during the fault and recovers with `T'_d0 = 8 s`, so the error the exciter is answering is genuinely
still positive. That is physics, not windup. The gate now asserts the property that actually
distinguishes the two: **the state never exceeds the ceiling at all**, where the same fault drives
an unlimited exciter to 18 pu. A wound-up state would sit at the boundary for as long as it took to
fall back through fifteen per unit.

### A base question settled on the way

Dynawo's regulator limits are stated in per-unit field voltage, so carrying them is only sound if
the two tools mean the same thing by that. They do: gridoxide's initialization derives 2.420747 for
the Kundur machine and the reference file's `efdPu` at `t = 0` is 2.420747. That is now a gate of
its own — a third independent check on the machine model, alongside the rotor angle and the air-gap
power — and it is what the `.dyd` reader carries the ceiling on the strength of.

Dynawo's governor limits are in MW and divide by the network base; `TGOV1` states `VMAX` **before**
`VMIN`, and reading them in field order gives a valve limited upside-down that therefore never
moves. Both are gated.

### Still outstanding

- Saturation; valve **rate** limits; state-triggered events; a fifth-order machine.
- Coupling the Dynawo reader to `src/iidm.rs`.
- Small-signal analysis.


---

## 19. Beyond the plan: the fifth-order machine, and small-signal analysis

Two of the items §18 left outstanding.

### The fifth-order machine

A salient-pole rotor: a field winding and a damper on the `d` axis, one damper on the `q` axis.
Dynawo's `ThreeWindings`, PSS/E's `GENSAL`. Its two axes are borrowed wholesale from the models
either side of it — the `d` axis is `GenRound`'s, the `q` axis is `GenTransient`'s — so it cost
little beyond the Jacobian bookkeeping.

There is no `x'_q` and no `T'_q0`, and that is the physics rather than a simplification: a salient
rotor has no `q`-axis field for a transient to live in. The `.dyd` reader therefore keys on the
**library name** rather than on which parameters happen to be present — a three-windings set simply
has no `XpqPu` to find, and reading its absence as a modelling decision would be right only by
coincidence. All five generators in the vendored IEEE 14 case now map, where three did before.

Gated the way its neighbours were, including a reduction: take the sixth-order model's `q`-axis
transient reactance down to its subtransient one and it must trace the fifth-order model's
trajectory, which it does to `1e-5` rad.

### Small-signal analysis

The feature-comparison row that used to say "presupposes the RMS row". §16 already noted the
modelling was done; this is the reduction and the eigen-decomposition.

`A = A_x − A_v·C_v⁻¹·C_x`, then eigenvalues, damping ratios, modal frequencies and **participation
factors**. The four blocks are **not re-derived**: they are exactly what `DaePattern::fill` already
assembles for every Newton iteration of every step, read out at `h·a = 1`. That is worth more than
the saved code — two hand-written derivations of one Jacobian would be two things to keep in step,
and a modal analysis that had quietly drifted from the simulation it describes would be worse than
none.

The gates go through the closed form the time-domain run is *independently* checked against: for an
undamped classical machine the swing eigenvalue is exactly `±j√(Ω_b·K_s/2H)`, and both halves are
asserted — the frequency, and that the real part is zero because nothing dissipates. Two more
compare against the run itself, through almost no shared code: the predicted period matches the
observed one to `2e-3`, and the real part predicts the peak-to-peak decay to 2%.

Two test bugs of my own on the way, and both the same mistake: measuring an oscillation about the
*perturbed* starting value rather than about the equilibrium. That makes every zero crossing a
tangency, and leaves a constant offset in the envelope that never decays. Worth recording because
the symptom — "no crossings found", "decays too slowly" — reads exactly like a physics failure.

Analysing a point the system is not sitting at is refused. A linearization about a mid-transient
state describes nothing in particular, and its eigenvalues would look entirely plausible.

The output is where the value is. On the smallest fixture it separates three distinct mechanisms and
names each by the states it belongs to: an electromechanical swing at 1.27 Hz that is 77% the
rotor's, an excitation-and-field-flux mode at 0.36 Hz on the AVR's lead stage and `e'_q`, and a fast
mode on the damper winding and the stabilizer. None of that has to be inferred from a trajectory.

### Still outstanding

- Saturation; valve **rate** limits; state-triggered events.
- Coupling the Dynawo reader to `src/iidm.rs`.
- Sparse (Arnoldi) small-signal for systems of thousands of states; eigenvalue sensitivities.

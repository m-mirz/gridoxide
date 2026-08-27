# RMS simulation in gridoxide

Status: **phases 1–2 implemented**, 2026-08-27, against `6f212eb`. Phases 3–6 outstanding.
§11 and §12 record what was actually done, including the places this plan was wrong.

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

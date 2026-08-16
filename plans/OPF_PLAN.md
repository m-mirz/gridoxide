# Optimal power flow in gridoxide

Status: **proposal**, not implemented. Written 2026-08-15 against `5f8bd49`.

## 1. What this is

Power flow answers *what is the state, given these injections*. Sensitivity answers *what
changes if an injection moves*. OPF answers the question those two set up but neither poses:
**what should the injections be?**

\\[ \min_{u}\ f(x, u) \quad \text{s.t.}\quad g(x,u) = 0,\quad h(x,u) \le 0 \\]

with \\(g\\) the power flow equations, \\(h\\) the operating limits, and \\(u\\) the controls.
It is the basis of economic dispatch, market clearing, congestion pricing, and the "is this
dispatch even feasible on the wires" check that sits under most operational planning.

## 2. Decisions taken

Confirmed before writing this:

- **DC-OPF first, then AC-OPF.** DC-OPF is convex, so *optimal* is provable rather than merely
  *converged*, and it exercises the whole pipeline — cost data, limits, optimizer, results — at
  a fraction of the risk. Phase 3 is shippable on its own.
- **Optimizer: HiGHS first through our own binding, then an in-house QP — deliberately both.**
  HiGHS for the convex DC problem, reached through bindgen against its C API rather than a
  third-party binding crate; then an in-house convex QP interior-point method behind the same
  boundary. Keeping both is not indecision: the pair cross-checks (§7.4), and the in-house solver
  removes the system-install burden once it exists. §5.2a has the reasoning. The AC choice stays
  deferred until the Hessian phase makes the problem's shape concrete (§6).
- **All four control families in scope**: generator active power, generator reactive
  power/voltage, transformer taps and phase shifters, and load shedding. Not all are meaningful
  in DC — see §5.3.
- **Validation by KKT certificates, published objective values, and analytic small cases.**
  Deliberately *not* by generating reference solutions from the pypower/pandapower installs on
  this machine. §7 takes that constraint seriously, including the one place it bites.

## 3. What already exists, and what it buys

More than one might expect, because OPF is largely made of parts this crate already has.

| Need | Already there |
|---|---|
| \\(g(x,u) = 0\\) | `solver::newton_raphson` and its residual |
| \\(\partial g/\partial x\\) | `jacobian::JacobianPattern` — the constraint Jacobian *is* the power-flow Jacobian |
| \\(\partial h/\partial x\\) for branch flows | `branch_flow::terminal_flow_derivs` |
| \\(\partial g/\partial u\\) for taps | `ac_sensitivity` — derived and validated last commit |
| DC network and branch-flow map | `linear::btheta`, `linear::sensitivity::DcSensitivity` |
| Sparse factorization | `sparse::RealFactorization`, five LU backends |
| Cost data, limits, an optimizer | **nothing** |

The AC sensitivity work is the direct predecessor: an OPF gradient is exactly the quantity
`function_row` computes, and the tap derivatives it validated are the same ones an AC-OPF needs
for tap controls. What is genuinely missing is the *second* derivatives (phase 4) and an
optimizer (phase 2).

**None of the four vendored references implements OPF.** power-grid-model, lightsim2grid and
powsybl-open-loadflow have none; VeraGrid and pandapower do but are not vendored. So unlike
short-circuit, there is no reference implementation in-tree to transcribe from. The formulation
below is standard and well-documented, but it is being written from the literature rather than
translated from a checkout — which raises the value of the validation in §7 considerably.

## 4. Data layer (phase 1)

Nothing in the model carries cost or active-power limits today: `Bus` has `q_min`/`q_max` and
that is all. `PgmLine` has `i_n` (rated *current*, not MVA). So this is genuinely new data.

**Input.** `python/gridoxide/matpower.py` already parses MATPOWER numeric matrix literals for
`mpc.bus`/`gen`/`branch`; `mpc.gencost` is the same shape and is a small extension. All 12 cases
in `tests/data/benchmark-grids/matpower/` carry `gencost` and generator limits.

### 4.1 The fixture set is not the one already committed

An earlier draft assumed those 12 cases would serve as the OPF fixtures. They will not, and the
reason only shows up when you count:

| Case | Branches | `rateA = 0` (unlimited) |
|---|---|---|
| `case14` | 20 | **20** |
| `case118` | 186 | **186** |
| `case300` | 411 | **411** |
| `case1354pegase` | 1991 | 559 |
| `case3120sp` | 3693 | 12 |
| `case_ACTIVSg200` | 245 | **0** |

The three smallest — precisely the ones anyone reaches for first — have **no branch limits at
all**. On those, no flow constraint ever binds, every LMP is identical, and the congestion half
of DC-OPF is simply not exercised. A fixture set that cannot make a constraint bind cannot test
an optimizer. These are power-*flow* benchmarks that happen to carry cost data, not OPF
benchmarks.

Two of the committed cases are usable — `case_ACTIVSg200` (245 branches, fully rated) and
`case3120sp` (12 of 3693 unrated) — and they should stay in the suite. But the primary fixture
source should be **[pglib-opf](https://github.com/power-grid-lib/pglib-opf)**, which exists to
solve exactly this problem:

- Curated *for* OPF by the IEEE PES Task Force, with meaningful generation limits, line ratings
  and costs — it supersedes NESTA.
- **`BASELINE.md` ships reference objective values** per case, produced by PowerModels.jl with
  IPOPT. That is §7's third gate, already tabulated and independent of anything here.
- Plain MATPOWER `.m` files, so the `gencost` extension above reads them with no extra work.
- **CC-BY-4.0** — permissive, attribution required, which
  `docs/src/reference/provenance.md` already does for every other vendored dataset.

Vendor a small subset (`case14_ieee`, `case118_ieee`, `case200_activ`, one Pegase case) the way
`tests/data/pgm/` holds its fixtures, or add the repo as a submodule beside
`benchmark-grids/`. Note that pglib's `case14_ieee` is *not* the committed `case14` — same
network, deliberately different limits — so the two must not be conflated.

This came from `docs/src/reference/resources.md`, whose own recommendation is exactly
"AC-OPF/SCOPF work → pglib-opf (primary benchmark)". It should have been consulted before the
first draft rather than after.

**Format.** PGM JSON has nowhere to put any of this, and inventing unofficial fields in someone
else's format is how a converter becomes a liability. Instead emit a **companion OPF document**
alongside the PGM one, keyed by the same component ids:

```json
{ "version": "1.0", "type": "opf_input",
  "generator": [ {"id": 12, "p_min": 0.0, "p_max": 3.324,
                  "cost": {"kind": "polynomial", "coefficients": [0.043, 20.0, 0.0]}} ],
  "branch_limit": [ {"id": 7, "rate_a": 0.0} ],
  "load": [ {"id": 31, "sheddable": true, "penalty": 1000.0} ] }
```

Rust side: `src/opf/model.rs` with `OpfData` and an `OpfNetwork` pairing it with the existing
`Bus`/`Line`/`Transformer` lists. A rate of `0.0` means *unlimited*, which is MATPOWER's own
convention and worth honouring rather than silently treating as a binding zero — several of the
committed cases use it.

## 5. Optimizer (phase 2) and DC-OPF (phase 3)

### 5.1 The DC problem

\\[ \min_{P_g,\alpha,s}\ \sum_g \left(c_{2g}P_g^2 + c_{1g}P_g + c_{0g}\right) + \sum_d \pi_d s_d \\]

subject to \\(B\theta = P_{inj}(P_g, s) + \Gamma\alpha\\), the generation box
\\(P_g^{min} \le P_g \le P_g^{max}\\), shedding \\(0 \le s_d \le P_d\\), and branch limits
\\(|P_{branch}(\theta)| \le \text{rate}\\). Convex quadratic objective, linear constraints — a
QP, and a sparse one.

Branch limits can be written through \\(\theta\\) or through PTDF. **Use \\(\theta\\)**: the
PTDF form eliminates the angles but is dense (1.19 GB on `case9241pegase`, as
`docs/src/powerflow/dc.md` records), whereas the \\(\theta\\) form keeps the sparsity the whole
crate is built around. `DcSensitivity` stays what it is — a screening tool — rather than becoming
a constraint builder.

### 5.2 The solver

**HiGHS**, via the `highs` crate (2.4.0, MIT — so `Cargo.toml`'s license field is unchanged).

An earlier draft of this plan proposed `clarabel` on the grounds that it is pure Rust and would
keep the AC choice open. The second half of that reasoning was wrong and is worth recording,
because it nearly locked in a limitation:

> **Neither a conic solver nor an LP/QP solver survives to phase 5.** clarabel is convex-only;
> HiGHS is LP/QP/MIP-only; AC-OPF is a nonconvex NLP. Whatever is chosen here is a *DC-only*
> choice, and framing one convex solver as more future-proof than another was simply confused.
> The choice should be made on DC-OPF merits alone — and on those, HiGHS is the better fit.

What HiGHS brings that clarabel does not:

- **Piecewise-linear costs become free.** HiGHS solves LPs natively, so MATPOWER's model-1 `pwl`
  cost curves are just additional rows rather than a special case. §10 previously listed pwl
  costs as out of scope partly *because* the problem had been framed as a QP — a framing artifact
  rather than a real limitation. Real market data uses pwl costs, so this matters.
- **A basic solution from simplex.** DC-OPF is routinely degenerate: many limits bind at once and
  the dual is non-unique. A simplex basis gives a vertex solution with well-defined duals, where
  an interior-point method lands in the middle of the optimal face and returns an interior dual.
  Since the duals *are* the LMPs, that difference is visible to anyone using the output.
- **It is the standard open solver in this domain**, which makes results comparable to the wider
  open power-systems stack rather than idiosyncratic.

The quadratic objective is supported through the safe wrapper — `Model::pass_hessian` with
`HessianFormat::{Triangular, Square}` — so quadratic `gencost` needs no workaround, and HiGHS
carries both simplex and interior-point methods for whichever form the objective takes.

What it costs: a C++ toolchain at build time and libstdc++ at runtime. Real, but precedented —
the existing `klu` feature already needs a C compiler and libclang, and `highs-sys` offers the
same vendored-static-build versus link-a-system-install split (`discover`) that `klu`/
`klu-dynamic` already offers. clarabel's genuine advantage is being pure Rust with no toolchain
at all; that is worth something, and less than the items above.

An ADMM solver (OSQP) would be the wrong trade regardless: first-order methods reach modest
accuracy quickly, which is fine for control loops and poor for the residual-based optimality
certificate §7 depends on.

Two things keep the phase-5 decision genuinely open:

- The dependency sits behind a **default-off `opf` cargo feature**, like `cgmes`/`klu`/`pardiso`
  already do. A user who wants power flow pays nothing.
- Everything above the solver talks to a small in-crate `LinearProgram`/`Solution` boundary, so
  phase 5 can bring an entirely different solver without the DC path noticing.

### 5.2a How HiGHS would actually be reached — and whether it should be

**Not through the `highs`/`highs-sys` crates.** They are not in the category the IPOPT bindings
are — `highs` 2.4.0 and `highs-sys` 1.15.0 are MIT, have ~1M downloads each, were updated within
the last two months, and come from the same `rust-or` org as `good_lp`. Maintenance risk is not
the objection. The objections are that `highs-sys` vendors HiGHS as a submodule and builds it
with CMake, which drags a large C++ build into the crate, and that this crate already has two
precedents for owning its FFI rather than delegating it.

HiGHS exposes a **stable C API** (`highs/interfaces/highs_c_api.h`, 173 functions). Everything
needed is there and the surface actually used is about seven calls: `Highs_passLp` /
`Highs_passHessian` to build, `Highs_run` to solve, `Highs_getSolution` for primal *and dual*
values, `Highs_getBasis` for the simplex basis §5.2 leans on, plus `Highs_addRows` /
`Highs_changeColsCost` for incremental work later.

That is precisely the shape of the existing **`pardiso` feature**: bindgen against a
system install's own header, discovered by env var, linking the shared library, with nothing
vendored. `klu` is the same idea against vendored C. So the consistent option is a bindgen-based
`opf` feature plus a safe wrapper of a few hundred lines that this crate owns.

**But that comparison also exposes the weakness.** `pardiso` is documented as a
"local/manual-verification-only backend" precisely *because* it needs a system install — no CI
runner has MKL. HiGHS is not commonly preinstalled either, so an `opf` feature built the same way
inherits the same problem, and DC-OPF is supposed to be a shippable feature rather than a
local-only one. Vendoring HiGHS to avoid that means adopting its CMake build, which is the thing
the `highs-sys` objection was about.

Three ways out, and the choice is a genuine preference call:

| | Dependency | Toolchain | Ships easily | Gets |
|---|---|---|---|---|
| **A** — `highs` crate | 2 crates (well maintained) | C++ / CMake, vendored | ✅ | Simplex basis, battle-tested numerics, least code |
| **B** — own bindgen, system HiGHS | none | bindgen + system install | ❌ local-only, like `pardiso` | Same as A, and we own the binding |
| **C** — in-house convex QP IPM | **none** | none — pure Rust | ✅ everywhere | pwl-as-LP survives; **no simplex basis**; we own the numerics |

Option C deserves more weight than the first draft gave it. The pwl-as-LP argument in §5.2
survives intact — piecewise-linear costs are just extra rows, and any LP/QP solver takes them,
an interior-point method included. What C gives up is the *simplex basis* (so degenerate duals
are interior rather than vertex — LMPs at a degenerate optimum become one valid dual among many
rather than the vertex one a market would publish) and the robustness of a mature implementation.
What it buys is exactly what this crate has repeatedly chosen before: no dependency, no
toolchain, no system install, and it works wherever Rust does. A convex QP interior-point method
is well-understood, and §7's KKT gate is a proof of correctness for one, not merely a smoke test.

**Decided: B first, then C.** Not A — the `rust-or` org is unfamiliar and unvetted, and "it has a
lot of downloads" is not a substitute for trusting the people behind a load-bearing dependency.
Owning the binding removes that question entirely.

An earlier draft called B "the worst of both". That was wrong, because it judged B as a
*destination* when it is a *stage*. Taken as B→C the sequence is better than either alone:

- **B gives an oracle before C exists.** An in-house QP interior-point method needs something to
  be checked against while it is being written, and KKT residuals alone will not catch a
  formulation error that both the model builder and the solver share. HiGHS is that
  independent second opinion — and unlike the pypower route declined in §2, it checks *our*
  matrices rather than a re-derived model.
- **C then removes B's deployment burden.** Once the in-house solver exists it becomes the
  portable default, working wherever Rust does; HiGHS demotes to an optional reference backend.
  The local-only limitation is temporary rather than structural.
- **The pair is itself a validation gate** — see §7.4. Two independent solvers agreeing on a
  convex problem is a strong result, and disagreement localizes immediately.

The build follows `pardiso`'s pattern: bindgen against the system install's own
`highs_c_api.h`, discovered via **pkg-config** (HiGHS ships `highs.pc.in`, so an installed copy
provides `highs.pc`) with an `HIGHS_ROOT` environment-variable fallback in the `MKLROOT` mould.
Nothing vendored, nothing built from source. HiGHS is C++ internally, so the link needs the C++
standard library even though the API crossed is pure C.

Feature layout, mirroring `klu`/`klu-dynamic`:

- `opf` — the OPF machinery and the `LinearProgram` boundary. Until phase 4 it requires a
  backend, so it implies `opf-highs`; from phase 4 it defaults to the in-house solver.
- `opf-highs` — the HiGHS backend. Needs the system install, and like `pardiso` will not be
  exercised by CI, which is a cost worth stating plainly rather than discovering later.

### 5.3 Which controls DC actually admits

Not all four, and the plan should say so rather than promise them and quietly deliver three:

| Control | In DC-OPF | Why |
|---|---|---|
| Generator \\(P\\) | ✅ | The classic variable |
| Load shedding | ✅ | Linear, and what keeps an infeasible case solvable |
| Phase shifter \\(\alpha\\) | ✅ | Enters \\(B\theta = P + \Gamma\alpha\\) linearly — a genuine linear control |
| Tap ratio \\(k\\) | ❌ deferred | DC uses \\(b/k\\), so it is *not* linear in \\(k\\); making it a variable makes the problem nonconvex, which defeats the point of doing DC first |
| Generator \\(Q\\), \\(|V|\\) | ❌ n/a | DC has no reactive power and asserts \\(\vert V\vert = 1\\) |

Tap ratio and reactive controls arrive in phase 5, where they are no harder than anything else
already nonlinear.

### 5.4 Outputs

Dispatch, objective, and — because the duals come free — **LMPs** and the list of binding
constraints. Congestion rent and the shadow price of each limit fall out of the same vector.

## 6. Toward AC-OPF (phases 4–5)

**Phase 4: second derivatives.** Any AC-OPF interior-point method needs the Hessian of the
Lagrangian, which means \\(\partial^2 g/\partial x^2\\) — the second derivatives of the power
injections. This is self-contained, independently useful, and validated the same way the AC
sensitivities were: finite-difference the *existing* Jacobian, which shares no code with the new
Hessian. Landing it separately keeps phase 5 from mixing two kinds of risk.

**Phase 5: AC-OPF.** Nonconvex, so HiGHS does not apply and the optimizer question genuinely
reopens. Controls: generator \\(P\\) and \\(Q\\)/\\(|V|\\), taps and phase shifters (derivatives
already available and validated by the AC sensitivity work), load shedding.

Two credible paths, and the trade-off between them is sharper than it first looks:

- **IPOPT.** The standard answer — MATPOWER, PowerModels.jl and pandapower-via-pyomo all use it
  for AC-OPF, so choosing it means inheriting a great deal of hard-won numerical robustness
  (inertia correction, filter line search, restoration) rather than rediscovering it. Two real
  concerns. The Rust bindings (`ipopt` 0.6.0, `ipopt-src`) have ~23k downloads and have not been
  updated since **December 2024**, which is thin for a load-bearing dependency. And IPOPT itself
  is **EPL-2.0** — weak copyleft, so `Cargo.toml`'s license field would have to change — and
  wants Fortran plus MUMPS or HSL underneath. This crate went to the trouble of *translating KLU
  into Rust* (`src/klu_native/`) specifically to keep LGPL out of default builds, so it has a
  demonstrated aversion to exactly this shape of dependency, even behind a feature gate.
- **A bespoke primal-dual IPM**, in the MATPOWER-MIPS mould, on top of the sparse factorization
  stack that already exists. Matches the KLU-translation precedent and keeps the dependency
  surface clean, at the cost of writing and validating the parts IPOPT already gets right.

Worth noting for completeness that IPOPT is the one option that could serve *both* phases — a QP
is a trivial NLP. That is not a reason to use it for phase 3: it would forgo the simplex basis
and the pwl-as-LP path in §5.2, and front-load the heaviest dependency into the lowest-risk
phase.

This phase should get its own plan revision once phase 4 lands and the sparsity and conditioning
of the KKT system are measurable rather than guessed at. Committing to either path now would be
guessing.

## 7. Validation

Three gates, none of which depends on any tool installed on this machine.

**KKT certificates — the primary gate for DC-OPF.** Check that the returned point satisfies
stationarity, primal and dual feasibility, and complementary slackness, to a stated tolerance.
For a *convex* problem this is not a weaker check than agreeing with another solver — it is a
**proof of optimality**, and it cannot be fooled by a shared convention error the way a
tool-to-tool comparison can. This is the strongest validation any phase of this project has had
available.

**Analytic small cases.** Two- and three-bus networks with one binding limit, where the optimum
is derivable in closed form. These are what localize a failure to a specific term; KKT tells you
the answer is wrong, an analytic case tells you which part.

**Published objective values.** MATPOWER and pglib-opf publish reference objectives per case, and
they are independent of anything here.

**Correction to an earlier draft.** This section previously claimed that published objectives
are "overwhelmingly AC-OPF numbers", that "there is no comparable published table for DC-OPF",
and concluded that phase 3 would ship without an external number to point at. That was wrong,
and checking pglib rather than recalling it settled it: **`BASELINE.md` publishes a DC column
beside the AC one**, per case, produced by PowerModels.jl with IPOPT.

| Case | DC ($/h) | AC ($/h) |
|---|---|---|
| `case3_lmbd` | 5.6959e+03 | 5.8126e+03 |
| `case5_pjm` | 1.7480e+04 | 1.7552e+04 |
| `case14_ieee` | 2.0515e+03 | 2.1781e+03 |
| `case30_ieee` | 7.4728e+03 | 8.2085e+03 |
| `case118_ieee` | 9.3101e+04 | 9.7214e+04 |

So **phase 3 has an external published reference after all**, and all three gates apply to it
rather than two. The pypower fallback the earlier draft held in reserve is unnecessary.

Two caveats remain, and they are about interpretation rather than availability. PowerModels' DC
formulation need not share every convention with this one — reference-bus handling and whether
line limits apply to the DC approximation are both places implementations differ — so a small
gap is a question to investigate, not an immediate failure. And the AC column is a *local*
optimum, so at phase 6 a disagreement may mean a different local solution; that comparison has
to report feasibility alongside the objective (§9 risk 2).

### 7.4 Two independent solvers — a fourth gate, from §5.2a

Carrying both HiGHS and an in-house QP (§5.2a) buys a check that neither provides alone, and it
substantially answers the caveat above.

On a convex problem the optimum is unique in objective value, and in the primal too where the
objective is strictly convex — which quadratic `gencost` with all \(c_2 > 0\) makes it. So two
independent solvers on the same model **must** agree, and any disagreement is a bug in one of
them rather than a modelling convention or a different local optimum. That is a far sharper
signal than the tool-to-tool comparisons this project has relied on elsewhere, where a shared
convention error can hide.

It also converts a theoretical caveat into a measurement. §5.2 notes that a simplex basis gives
vertex duals where an interior-point method gives interior ones, so LMPs may differ at a
degenerate optimum. With both solvers in hand that difference can be *observed* on real cases —
how often degeneracy actually arises, and how far apart the two duals land — rather than left as
a warning in the documentation.

One thing this gate cannot do: catch an error in the model *builder*, since both solvers receive
the same matrices. That is what the analytic small cases are for.

## 8. Phases

| # | Deliverable | Gate |
|---|---|---|
| 1 | Cost/limit data layer; `matpower.py` reads `gencost`, limits, ratings; pglib-opf fixtures vendored | Round-trips every fixture without loss |
| 2 | `LinearProgram` boundary; `opf-highs` backend via own bindgen against `highs_c_api.h` | Solves a hand-built LP and QP with correct duals |
| 3 | **DC-OPF** — dispatch, objective, LMPs, binding constraints; CLI and Python | KKT residuals; analytic cases; existing suites unmoved |
| 4 | In-house convex QP interior-point method, second backend behind the same boundary | KKT residuals; **agrees with HiGHS** on every fixture (§7.4) |
| 5 | Injection Hessians | Finite-difference against the existing Jacobian |
| 6 | **AC-OPF** — all four control families | KKT; published pglib/MATPOWER objectives |

Phases 1–3 are the first shippable unit, though `opf-highs` needs a system HiGHS install to build.
**Phase 4 is what makes DC-OPF portable** — it becomes the default backend and drops the install
requirement, so it is not optional polish. 5 is independently useful. 6 needs its own revision
first.

## 9. Risks

1. **Conventions differ between DC-OPF formulations.** pglib does publish DC reference
   objectives (§7), so the external gate exists — but PowerModels' DC model need not match this
   one on reference-bus handling or on whether limits apply to the linearized flows. A gap of a
   fraction of a percent is a convention question; a large one is a bug. KKT remains the check
   that decides which, since for a convex problem it proves optimality rather than corroborating
   it.
2. **AC-OPF is nonconvex.** A KKT point is locally optimal; disagreeing with a published
   objective may mean a different local optimum rather than a bug. Any phase-5 comparison has to
   report the objective *and* whether the point is feasible, not just the gap.
3. **`opf-highs` will not be exercised by CI**, because it needs a system HiGHS install — the
   same limitation `pardiso` carries and documents. Phase 4 is the mitigation rather than a
   workaround: once the in-house QP is the default backend, the portable path is the tested one
   and HiGHS becomes an optional reference. Until then, phase 3's gate runs only where HiGHS is
   installed, which should be stated wherever the feature is described.
4. **Neither convex solver survives phase 6.** No LP/QP method takes a nonconvex NLP. That is
   understood rather than a risk to mitigate, but it means the AC decision in §6 is genuinely
   independent of everything chosen for DC.
5. **Relaxed taps are not implementable.** Real tap changers are discrete. A continuous
   relaxation is standard and gives a bound, but the answer needs rounding and a re-solve before
   anyone acts on it. Say so in the output, not only in the docs.
6. **Scope creep toward security-constrained OPF.** N-1 constraints inside the optimization
   multiply the problem by the contingency count. Explicitly out of scope (§10).

## 10. Deliberately out of scope

- **Unit commitment** — integer on/off decisions; a different class of problem needing a MILP
  solver.
- **Security-constrained OPF** — N-1 embedded in the optimization. `BatchSolver::solve_contingencies`
  and the DC screening factors remain the answer for contingency analysis.
- **Multi-period, storage, ramping** — needs a time-coupled formulation and a time-series input
  path that does not exist.
- **Market mechanisms beyond LMP** — reserves, capacity, bid formats.

**Piecewise-linear cost curves** (MATPOWER's model 1) were listed here in an earlier draft. They
are **in scope** now: choosing an LP-capable solver in §5.2 makes them ordinary rows rather than a
special case. All 12 committed cases use model 2 (polynomial), so model 1 needs a fixture of its
own before it can be claimed — another reason to pull in pglib-opf (§4.1), which carries both.

### 10.1 Other sources considered

The rest of `docs/src/reference/resources.md`, assessed against this plan:

| Source | Bearing on OPF |
|---|---|
| **pglib-opf** | **Adopted** as the primary fixture set — §4.1 |
| **pglib-opf-hvdc** | Genuinely interesting later: gridoxide already has a real DC-side network (`src/dc.rs`, `VsConverter`/`CsConverter`), so OPF over AC/DC is closer than it would be for most tools. MatACDC format is a new reader, and the reference implementation is Julia (PowerModelsACDC.jl). Out of scope for phases 1–5, worth revisiting if MTDC work continues. |
| **PyPSA/technology-data** | Cost and efficiency assumptions by technology and year. Relevant to capacity-expansion studies rather than dispatch — `gencost` already gives per-generator curves, which is what OPF needs. Not required. |
| **simbench** | MV/LV benchmarks with full time series. The natural input if multi-period OPF ever comes into scope; irrelevant while it is out (§10). |
| **Sienna PowerSystemsTestData** | TAMU ACTIVSg synthetic grids — overlaps pglib's `case200_activ`/`case500_goc` and the committed `case_ACTIVSg200`. No additional coverage for OPF. |
| **VeraGrid `Grids_and_profiles`** | Bundles PGLib cases, so redundant with taking pglib directly. Its OPF implementation is a cross-check that §2 deliberately declined. |
| **IEEE/EPRI feeders, CGMES conformity, Dynawo** | Distribution, parser and dynamics validation respectively. No OPF bearing. |

One point of tension worth naming: that file's recommendation reads "AC-OPF/**SCOPF** work →
pglib-opf", and pglib bundles ARPA-E GO Competition cases that are security-constrained by
design. This plan puts SCOPF out of scope (§10). Vendoring pglib does not commit to SCOPF — the
cases solve perfectly well as ordinary OPF — but it does mean the data is already there if that
scope ever changes.

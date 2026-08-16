# The Optimal Power Flow Problem

## Motivation

A power flow answers *what is the state of the network, given who is generating what?* Optimal
power flow turns that around: **who should generate what, so that demand is met at least cost
without overloading anything?**

That inversion changes the character of the problem. A power flow has one answer, and Newton
either finds it or does not converge. An OPF has a *feasible set* — every dispatch that serves
demand within limits — and asks for the cheapest point in it. The output is not just a state
but a decision, and alongside it a set of prices that say what each constraint is costing.

Those prices are usually the reason to run it:

- **Locational marginal price** — what one more MW of demand at this bus would cost. Uniform
  across the network when nothing is congested; the spread *is* the congestion, in dollars.
- **Shadow price of a branch limit** — what one more MW of capacity on that corridor would
  save per hour. This is the number that justifies a reinforcement.

gridoxide implements both **DC-OPF** and **AC-OPF**, and the difference between them is not
just accuracy. DC is a convex quadratic program, so its answer is *provably* optimal. AC is
nonconvex, so its answer is a local optimum satisfying the first-order conditions — which is
the state of the art, and what every published AC-OPF objective means. That distinction shapes
how each is validated.

## The formulation

Minimize generation cost over dispatch \\(P_g\\), bus angles \\(\theta\\), and load shedding
\\(s\\):

\\[ \min_{P_g,\ \theta,\ s}\ \sum_g \left(c_{2g}P_g^2 + c_{1g}P_g + c_{0g}\right)
   + \sum_d \pi\, s_d \\]

subject to, at every bus \\(i\\), the DC power balance

\\[ \sum_{g \in i} P_g + s_i - (B\theta)_i = P^{load}_i - c_i \\]

plus the generation box \\(P^{min}_g \le P_g \le P^{max}_g\\), the shedding box
\\(0 \le s_d \le P^{load}_d\\), the branch limits

\\[ -\text{rate} \le b\,(\theta_f - \theta_t - \alpha) \le \text{rate} \\]

and the reference angle \\(\theta_{ref} = 0\\). Here \\(c_i\\) collects the constant
contribution of any fixed phase shift \\(\alpha\\).

With quadratic costs this is a convex QP; with piecewise-linear costs it is an LP. Both go to
the same solver through the same interface.

### Piecewise-linear costs enter through an epigraph

MATPOWER's cost model 1 gives a curve the objective cannot express directly. A convex
piecewise-linear function is the upper envelope of its segments, so \\(\min f(p)\\) becomes
\\(\min z\\) subject to \\(z \ge m_i p + k_i\\) for every segment — ordinary rows, which is
exactly why an LP-capable solver was a requirement rather than a preference.

Exactness needs convexity, which is checked and enforced: the epigraph of a **non-convex** curve
is its convex envelope, which charges less than the curve does, so such a curve is refused
rather than silently relaxed.

AC-OPF uses the same reformulation for a different reason. `CostCurve::evaluate` handles a
piecewise curve correctly, so feeding it to the objective gives the right *values* — but the
objective is then only \\(C^0\\), and a Newton method assumes \\(C^2\\). The gradient jumps at
every breakpoint, and an optimum very often sits exactly on one, since a breakpoint is where
marginal cost changes. The epigraph makes the objective linear and moves the kinks into
constraints, which the solver handles exactly.

> This was a real bug, not a hypothetical. Both formulations read costs with
> `if let Some(CostCurve::Polynomial { .. })`, so a piecewise curve fell through in silence and
> left the generator's objective coefficient at zero — it looked **free**, and the optimizer
> dispatched it first. Wrong dispatch, wrong cost, no error, reachable straight from the
> documented converter. What hid it was that every committed fixture used model 2; the plan had
> already flagged that as the reason the feature could not be claimed. `case5_pjm_pwl.m` is now
> committed as an *exactly equivalent* rewrite of `case5_pjm` — every generator there has
> \\(c_2 = 0\\), so three collinear points reproduce the cost precisely while still producing two
> segments — which makes the test a question with a known answer.

### The balance row is written generation-minus-outflow

The balance constraint could equally be written with load on the left. Putting it on the
**right** is a deliberate choice about the duals rather than cosmetics: with load on the
right-hand side, the row's dual is \\(\partial(\text{cost})/\partial(\text{load})\\) directly
— which *is* the locational marginal price, positive, in the sign anyone reading it expects.
The opposite orientation yields the negated quantity, and the resulting sign error would appear
in exactly the output most likely to be published, looking plausible everywhere and being wrong
everywhere.

### Why \\(\theta\\) and not PTDF

Branch limits could be expressed against injections through a PTDF matrix, eliminating the
angle variables entirely. That formulation is standard in the literature and tempting here,
because gridoxide already computes PTDF for
[DC sensitivity](../powerflow/dc.md#sensitivity-factors-ptdf-and-lodf).

It is rejected on density. PTDF is a full \\(n_{branch} \times n_{bus}\\) dense matrix — 1.19
GB on `case9241pegase`, as [DC power flow](../powerflow/dc.md) records — whereas the
\\(\theta\\) formulation keeps the sparsity the rest of the crate is built around. The extra
angle variables are cheap; a dense constraint matrix is not. `DcSensitivity` stays a screening
tool rather than becoming a constraint builder.

## Which susceptance — and why it is not a detail

DC has two defensible ways to form branch susceptance from \\(r\\) and \\(x\\):

| | Formula | Who computes it |
|---|---|---|
| `IgnoreR` | \\(b = 1/x\\) | MATPOWER `makeBdc`, pandapower, lightsim2grid |
| `IgnoreG` | \\(b = x/(r^2 + x^2)\\) | PowerModels — i.e. \\(-\mathrm{Im}(y_{series})\\) |

For power flow the choice is minor and gridoxide defaults to `1/x`, matching the tools people
cross-check against. **For OPF it is not minor**, because susceptance enters the *constraint
set* and not merely the reported flows: it decides which branch reaches its limit first, and
therefore which constraint binds and what everything is priced at.

This surfaced as a concrete discrepancy worth recounting, because the symptom pointed the wrong
way. Measured against pglib-opf's published DC objectives:

| Case | \\(b = 1/x\\) | \\(b = x/(r^2+x^2)\\) |
|---|---|---|
| `case3_lmbd` | −0.037% | −0.0001% |
| `case5_pjm` | −0.0006% | −0.0006% |
| `case14_ieee` | +0.0013% | +0.0013% |
| `case30_ieee` | **+0.423%** | −0.025% |
| `case118_ieee` | +0.034% | −0.013% |

Four cases agreed to better than 0.04% and one was ten times worse. The natural reading — a bad
limit, a missing constraint, a converter bug specific to that case — was wrong. Direct
comparison against the MATPOWER file ruled out all 41 susceptances, all 41 branch limits, every
per-bus load, and every generator box; the KKT certificate (below) ruled out our own solver.

The cause was the formula. `case30_ieee`'s branch 1→2 has \\(r = 0.0192,\ x = 0.0575\\) — an
r/x ratio high enough that \\(1/x\\) overstates susceptance by 10% — and that branch is
congested at the optimum, so the error landed squarely on a binding constraint. Adopting the
series form brought `case30` inside 0.03% **and improved all four other cases**.

> The transferable lesson: a discrepancy isolated to one case is not evidence that the *fault*
> is isolated to one case. The formula was wrong on every network; only `case30` was congested
> on a branch resistive enough to reveal it. Matching a published objective to 0.4% was not
> reassurance — it was the signal.

So DC-OPF defaults to `IgnoreG` while DC power flow defaults to `IgnoreR`. The two defaults are
opposite on purpose, under one consistent rule: **each matches what the reference
implementations in its own domain compute.** Both remain selectable in either.

## Which controls DC admits

Of the four control families an OPF might optimize, DC admits two:

| Control | Status | Why |
|---|---|---|
| Generator active power | ✅ | The decision variable. |
| Load shedding | ✅ | Linear, priced at `shed_price`. |
| Phase-shifter angle | ⚠️ read, held fixed | Linear in \\(\theta\\), so this is a small extension the module is shaped for — but not yet taken. Shifts enter as constants. |
| Transformer tap ratio | ❌ | DC uses \\(b/k\\), so making \\(k\\) a variable is **nonconvex** and would forfeit the guarantee that is the entire reason to do DC first. |
| Reactive power / voltage | ❌ | Meaningless in DC, which has no reactive power and holds \\(|V| = 1\\). |

Taps and reactive controls arrive with AC-OPF.

### Load shedding

Shedding is offered by default, priced at \\$10,000/MWh — far above any plausible generator
marginal cost, so it is a last resort rather than an economic choice, but *finite*, so an
otherwise-infeasible case still returns an answer that says **where** demand could not be
served. That is almost always more useful than the word "infeasible".

Turning it off (`--no-shedding`) makes such a case genuinely infeasible, which is sometimes the
question being asked.

## The solver boundary

OPF modelling never touches an optimizer directly. Everything crosses a small in-crate
interface:

```rust
pub struct LinearProgram {
    pub n_vars: usize,
    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
    pub col_cost: Vec<f64>,
    pub offset: f64,
    /// Lower triangle of the objective Hessian, `(i, j, value)` with `i >= j`.
    /// `None` for a pure LP.
    pub hessian: Option<Vec<(usize, usize, f64)>>,
    /// Constraint matrix as `(row, col, value)` triplets.
    pub rows: Vec<(usize, usize, f64)>,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
}
```

Two-sided row bounds express everything DC-OPF needs uniformly: an equality is
`lower == upper` (the balance rows), a range is `-rate ≤ flow ≤ rate`, and a one-sided limit
sets the other side infinite. Triplets rather than a compressed format because that is the
shape the rest of the crate already speaks; the backend converts internally, where the
conversion is tested once.

There are **two backends** behind that interface, and having two is deliberate.

**The default is gridoxide's own interior-point method** (`opf::ipm`) — Mehrotra
predictor-corrector on the sparse factorization this crate already owns. Pure Rust, no system
libraries, nothing to install, so it is built and tested in CI along with everything else.

**HiGHS is the reference**, reached through gridoxide's own bindgen-generated FFI rather than a
third-party wrapper crate — the same approach the [KLU and PARDISO backends](../solvers/backends.md)
take, and for the same reason. It needs the `opf-highs` feature and a local HiGHS install
(`libhighs-dev` on Debian/Ubuntu), so it is not exercised by CI. Nothing depends on it; it is
selected with `--highs` or `solver="highs"`.

One deliberate non-choice worth recording: `build.rs` does **not** use pkg-config, even though
HiGHS ships a `.pc` file. Ubuntu's is broken — every path in it carries a doubled `/usr`
prefix. Discovery uses standard paths with a `HIGHS_ROOT` override instead.

### Why keep both

Because **two independent solvers are a validation gate**. On a convex problem the optimal
objective is unique, so a disagreement between them is a bug in one — not a modelling
convention, not a different local optimum, not a tolerance question. Across the pglib fixtures
they agree to 1.1e-11 relative; `tests/opf_cross_test.rs` also runs 300 randomly generated
convex QPs through both.

That randomized comparison earned its keep immediately. It found a real defect in the
interior-point method that no fixture exposed: the solver was taking **different primal and
dual step lengths**, which is standard and strictly better for a linear program but silently
breaks a quadratic one. Substituting the Newton step into the dual residual after a split step
leaves

\\[ r_d \leftarrow (1 - \alpha_d) r_d + (\alpha_p - \alpha_d) Q \Delta u \\]

The first term is the contraction the method depends on; the second is pure error, vanishing
only when \\(Q = 0\\) or the step lengths agree. Left in, the iterates settle into a limit
cycle — one generated QP repeated with period 8 until the iteration limit while HiGHS solved it
without trouble. Forcing a common step length whenever the objective is quadratic fixes it.

**What the cross-check does not do is establish correctness.** Both backends receive the same
`LinearProgram`. If the model is assembled wrongly, both agree perfectly on the wrong answer —
which is exactly what would have happened with the susceptance bug above. Cross-validation
tests the *solvers*; the published objectives test the *model*.

## Validation

DC-OPF is checked three ways, deliberately independent:

**1. Analytic cases.** Small networks whose optimum is derivable on paper — an uncongested
two-bus case that must dispatch the cheap unit only, a congested one whose price split is known
in closed form, and shedding cases with and without the option enabled.

**2. KKT certificates and the second solver.** For a convex problem the KKT conditions are a
*proof* of optimality, not a comparison. Every pglib case asserts stationarity
(\\(\text{col\\_dual} = Qx + c - A^{\mathsf T}y\\)), primal and dual feasibility, and
complementary slackness. This is what separates *"our model differs from theirs"* from *"our
answer is wrong"* — and in the `case30` investigation above, it is what made the difference
diagnosable at all.

**3. Published objectives.** Against pglib-opf's own DC baseline, all five cases now agree to
better than 0.03% — about as close as figures published to five significant digits can resolve.
The measured per-case gaps are pinned in `tests/opf_dc_test.rs` so a regression shows up as a
*change* rather than requiring someone to re-derive what "close enough" means.

Note what is deliberately absent: no reference solutions generated from an installed
pypower or pandapower. Comparing against another implementation's output tests agreement, not
correctness, and inherits its bugs silently.

## Using it

### CLI

```bash
gridoxide opf network.json
```

Costs and limits come from a companion document defaulting to `<network>.opf.json` — the pair
`gridoxide-matpower` writes when converting a MATPOWER case. On `case5_pjm`:

```
5 bus(es), 5 generator(s), 1000.0 MW of demand
total cost: 17479.90 $/h

dispatch (MW):
  generator    0:     40.000 (at max)
  generator    1:    170.000 (at max)
  generator    2:    323.495
  generator    3:      0.000 (at min)
  generator    4:    466.505

locational marginal price ($/MWh):
  bus    0:    16.9774
  bus    1:    26.3845
  bus    2:    30.0000
  bus    3:    39.9427
  bus    4:    10.0000
  spread: 29.9427 (the congestion)

binding branch limits:
  branch    5: flow   -240.000 of   240.000 MW, worth   62.3220 $/MWh to relieve
```

Every number that makes this an OPF rather than a power flow is in the bottom two blocks: the
price spread, and the single branch causing it.

`--data <path>` names the companion document explicitly, `--no-shedding` removes the shedding
option, `--shed-price` re-prices it, `--ignore-r` selects the textbook \\(b = 1/x\\), and
`--highs` swaps in the reference backend.

### Rust

```rust
use gridoxide::linear::DcApproximation;
use gridoxide::opf::dc::{DcOpf, DcOpfNetwork, DcOpfOptions};
use gridoxide::opf::ipm::IpmSolver;
use gridoxide::opf::Solver;

let network = DcOpfNetwork::from_pgm(input, &data, 50.0, DcApproximation::IgnoreG)?;
let opf = DcOpf::build(network, DcOpfOptions::default())?;

let mut solver = IpmSolver::new();
let result = opf.interpret(&solver.solve(opf.problem())?);

println!("{:.2} $/h", result.objective);
for b in &result.binding {
    println!("branch {} at {:.1} MW, worth {:.2} $/MWh", b.branch, b.flow, b.price);
}
```

`build` and `solve` are separate steps because the assembled `LinearProgram` is worth
inspecting on its own — and because that is the seam a second backend plugs into.

### Python

```python
import gridoxide

r = gridoxide.dc_opf("network.json")

print(f"{r.objective:.2f} $/h")
print(f"congestion: {max(r.lmp) - min(r.lmp):.2f} $/MWh")

for b in r.binding:
    print(f"branch {b.branch}: {b.flow:.1f} of {b.rate:.1f} MW, {b.price:.2f} $/MWh to relieve")
```

`dc_opf` is a function rather than a class because — unlike
[`AcSensitivityModel`](../sensitivity/ac.md) — there is nothing worth keeping between calls:
one solve answers one question.

## AC-OPF

The full problem: real voltage magnitudes, reactive power and losses, optimizing generator
active *and* reactive output together.

```bash
gridoxide opf network.json --ac
```

### What changes

Everything that made DC a convex QP goes away. The balance equations are trigonometric, so the
feasible set is curved and generally nonconvex. In exchange, the two control families DC cannot
express arrive: generator reactive power, and bus voltage magnitude.

| | DC-OPF | AC-OPF |
|---|---|---|
| Problem class | Convex QP | Nonconvex NLP |
| Guarantee | Provably global optimum | Local optimum (first-order conditions) |
| Variables | \\(P_g, \theta, s\\) | \\(P_g, Q_g, \theta, \|V\|\\) |
| Balance rows | \\(n\\), linear | \\(2n\\), trigonometric |
| Branch limits | \\(\|b\,\Delta\theta\| \le \text{rate}\\) | \\(P^2 + Q^2 \le \text{rate}^2\\) |
| Solver | `opf::ipm` (Mehrotra) | `opf::nlp` (line search + regularization) |

Limits are squared rather than written as \\(\sqrt{P^2+Q^2}\\) so that no square root — whose
derivative is undefined at zero flow — ever enters the model.

### What nonconvexity costs the solver

`opf::nlp` is a second interior-point method, not a reuse of the convex one, because three
things change:

- **A line search.** The step is a direction, not a destination — a full Newton step may
  increase the objective or worsen feasibility, so it is accepted only if it reduces a merit
  function trading the two off.
- **Regularization driven by failure rather than structure.** In the QP, \\(\gamma\\) is a tiny
  fixed value that only makes a rank-deficient matrix factorizable. Here the Hessian of the
  Lagrangian can be genuinely indefinite and the Newton direction then points at a saddle, so
  \\(\gamma\\) is raised by orders of magnitude until the direction is usable.
- **A gradual barrier.** The QP drives \\(\mu \to 0\\) as fast as Mehrotra's heuristic allows,
  because its model is exact. Here each barrier subproblem is solved loosely before \\(\mu\\)
  drops; going faster produces a step that is accurate for a problem nobody asked about.

### Where the derivatives come from

The balance rows' first and second derivatives are
[`injection_hessian`](../sensitivity/ac.md#second-derivatives)'s — that module exists for this.
The branch limits need their own, and the structure turns out to be identical: a terminal flow

\\[ P_{ft} = \|V_f\|^2 g_{self} + \|V_f\|\|V_t\| (g_{mut}\cos\theta_{ft} + b_{mut}\sin\theta_{ft}) \\]

is algebraically **a bus injection with exactly one neighbour**, with \\(y_{ff}\\) as the
self-admittance and \\(y_{ft}\\) the mutual one. The same trigonometry and the same derivative
rotations apply.

### Validation, and two bugs the objective alone would not have found

All five pglib cases match the published AC objectives to **0.001%**, at constraint violations
of 1e-9 or better. Getting there took two corrections, and in both cases what identified them
was not the size of the gap but *which cases had one*:

1. **Per-bus voltage limits.** Defaulting to `[0.9, 1.1]` matched `case3` and `case5` exactly
   and left the other three low by 0.4–0.5%. Those three specify `[0.94, 1.06]` — the looser
   default was buying a cheaper answer that was infeasible for the real case.
2. **Bus shunts.** With voltage limits fixed, exactly the three cases carrying shunt capacitors
   still disagreed, and the two with none matched to 0.001%. A shunt supplies reactive power for
   free; dropping it makes the generators supply it instead, at a cost.

> The same lesson as the susceptance bug above, in a different key: when some cases agree and
> others do not, the question worth asking is what the disagreeing ones have in common — not
> how big the gap is.

Beyond the published objectives, three further checks run: the model's gradient, Jacobian and
Hessian against central differences (a wrong derivative does not make a solver fail, it makes it
converge confidently to the wrong point); feasibility re-derived from `network::power_injections`
rather than read out of the solver; and prices checked against a numerical
\\(\partial\text{cost}/\partial\text{load}\\) — which caught a sign error that produced
prices of exactly the right magnitude, negated, on every bus of every case.

### Starting points matter

A nonconvex solver returns the optimum in whichever basin it starts. On these networks every
start tested reaches the same objective to 1e-6, but that is an *empirical finding about these
cases*, not a property of the method.

It is also load-bearing for convergence. Starting each generator at its box midpoint — the
obvious choice — leaves the dispatch about 23% short of demand on every pglib fixture, because
`p_min` is usually zero. `case5_pjm` and `case118_ieee` then never recovered, stalling at the
iteration limit. Distributing demand proportionally across the generators' ranges instead costs
nothing and fixed both.

### Reading the answer

```python
import gridoxide

r = gridoxide.ac_opf("network.json")

print(f"{r.objective:.2f} $/h in {r.iterations} iterations")
print(f"largest violation: {r.violation:.2e} pu")   # read this with the objective
print(f"voltage range: {min(r.magnitudes):.4f} – {max(r.magnitudes):.4f} pu")
```

`violation` is part of the answer rather than diagnostics. On a nonconvex problem a lower cost
at an infeasible point is not a better result, so an objective reported without it can be badly
misleading.

## What is not here yet

**Transformer taps and phase shifters as decision variables.** Both are read and held fixed.
Taps are genuinely discrete, so a continuous relaxation gives a bound that needs rounding and a
re-solve before anyone acts on it.

**Unit commitment.** On/off decisions make this a mixed-integer program. HiGHS solves MIPs, so
the backend would carry it, but nothing above the solver boundary models it.

**Security constraints.** N-1 constrained OPF needs contingency cases inside the optimization.
The [DC outage factors](../powerflow/dc.md#sensitivity-factors-ptdf-and-lodf) are the screening
half of this and already exist; the constrained optimization is not built on them.

# DC (Bθ) Power Flow

## Motivation

Newton-Raphson answers the full question — every voltage magnitude, every angle, every
reactive flow — and pays for it with an initial guess, an iteration count, and the
possibility of not converging at all. A great many operational questions do not need
that. *How does this transfer divide between the two corridors? Which line does the
outage overload? Is this dispatch feasible on the wires?* Those are questions about
**active power**, and active power divides between paths in a way that barely depends on
voltage magnitude at all.

The DC approximation makes three assumptions and gets a linear problem out:

- every voltage magnitude is exactly 1 p.u.,
- every branch resistance is zero,
- every angle difference is small, so \\(\sin\delta \approx \delta\\) and \\(\cos\delta \approx 1\\).

What remains is one sparse symmetric system, solved once by direct factorization. There
is no initial guess, no iteration count, no convergence to fail — a DC power flow either
has a reference bus or it does not. That unconditional robustness is why DC survives as
a first-class mode in every serious tool despite being, on its face, a crude
approximation: it is the method you can run a hundred thousand times inside a contingency
screen and trust to come back.

What it gives up is everything reactive. No voltage magnitudes, no Q flows, no losses,
no way to tell a PV bus from a PQ bus.

## What changes in the equation system

Starting from the AC branch flow at \\(|V| = 1\\), with \\(\delta = \theta_f - \theta_t - \alpha\\)
and a transformer ratio \\(k\\), the three assumptions collapse the flow to a linear
function of angle:

\\[ P_{from} = b\,(\theta_{from} - \theta_{to} - \alpha), \qquad P_{to} = -P_{from} \\]

Two decisions remain, and the tools genuinely disagree on both.

### 1. Which susceptance?

\\[ \text{ignore } r:\quad b = \frac{1}{x} \qquad\qquad
   \text{ignore } g:\quad b = \frac{x}{r^2 + x^2} = -\operatorname{Im}(y_{series}) \\]

The first is the textbook form and what MATPOWER, pandapower and lightsim2grid all
compute. The second is strictly the better linearization — it is the *exact* coefficient
of \\(\sin\delta\\) in the AC flow equation, whereas \\(1/x\\) is that coefficient only in
the limit \\(r \to 0\\). On transmission the two differ by \\(1/(1 + (r/x)^2)\\), about
1% at \\(r/x = 0.1\\); on a distribution feeder where \\(r/x\\) exceeds 1, they differ by
more than half.

There is a neat consequence worth knowing: where \\(r/x\\) is *uniform* across the
network, the two choices differ by a constant factor on every branch, which rescales
every angle and leaves every flow identical. The choice can only change an answer on a
network whose \\(r/x\\) varies from branch to branch.

### 2. Where does the phase shift go?

A phase-shifting transformer contributes a constant \\(-b\alpha\\) to its branch flow.
Being constant, it moves to the right-hand side rather than into the matrix, which is
what keeps \\(B\\) symmetric even on a network full of phase shifters:

\\[ B\,\theta = P + \varphi, \qquad
   \varphi_i = \sum_{from_k = i} b_k \alpha_k \;-\; \sum_{to_k = i} b_k \alpha_k \\]

MATPOWER stores \\(-\varphi\\) and calls it `Pbusinj`. The symmetry this preserves is
not merely tidy — the sensitivity factors below solve against \\(B^\mathsf{T}\\) without
ever forming it.

### 3. Which reference?

DC fixes angles, not voltages, so a component needs exactly one angle reference to be
determined — but unlike AC, *two* references do not over-determine it. Each simply
supplies whatever its own incident branches demand. The system stays well posed; only
the split of slack pickup between the references becomes arbitrary.

## Where this fits in gridoxide today

The solver is `linear::btheta::dc_power_flow`, reached through
`PowerFlowMethod::Dc`. It runs in five steps:

1. **`dc_branches`** reduces every `Line` and `Transformer` to a `DcBranch` — a
   susceptance, a phase shift, and the crate's flat branch index (lines first, then
   transformers, the same index space `branch_flow::branch_params` uses). Branches with
   no series path — half-open self-loop lines, transformers with an open terminal —
   produce nothing and are listed in `DcSolution::ignored_branches`.
2. **`dc_components`** partitions the buses with `topology::UnionFind`. This is
   deliberately *not* `network::connected_components`, which needs a `&YBusSparse` that
   a branch-list solver has no reason to build.
3. **`network::classify` and `network::mark_unreferenced_islands`** — reused unchanged
   from the AC path, so a sourceless island is pinned to a placeholder rather than given
   a fabricated reference, exactly as documented under
   [Multi-Island Power Flow](./multi_island.md).
4. **One factorization per island**, through `sparse::RealFactorization`. Per-island
   rather than one shared system because the LU of a block-diagonal matrix *is* the sum
   of the block LUs — it costs nothing, and it means a singular island can be named
   rather than merely suspected.
5. **Flows, pickup and residual** are derived from the solved angles. `DcSolution`
   carries per-branch active flows, per-island slack pickup, and a `max_residual` that
   is a factorization-quality check rather than a convergence check — there is nothing
   to converge.

Two details of gridoxide's own data model needed explicit handling.

**Purely resistive branches.** `IgnoreR` asserts \\(r = 0\\), so a branch with \\(x = 0\\)
has no impedance left at all in that approximation's world — it is a short circuit, and
gets the crate's standard zero-impedance treatment
(`topology::clamp_branch_impedance`) rather than an infinite susceptance. This is not
hypothetical: power-grid-model's `link/dummy-test` fixture contains a line at
\\(r = 10\,\Omega,\ x = 0\\), and dropping it would silently split that network into a
live island and a sourceless one. Under `IgnoreG` the same branch correctly comes out at
\\(b = 0\\), because a purely resistive branch really does transmit no angle-driven
active power — so that network genuinely islands under `IgnoreG`. The two approximations
disagree about such a branch by construction, not by accident.

**PGM links.** A `link` carries `topology::IDEAL_CONNECTION_Y = 2e5 + j2e5`, whose
positive imaginary part is inherited from power-grid-model's own `1e8 + j1e8` and is a
regularization choice, not a claim that the element is capacitive. Inverting it gives
\\(x = -2.5\times10^{-6}\\), and a negative susceptance would flip that branch's coupling
sign and make \\(B\\) indefinite. AC never noticed, because only \\(|y|\\) and the link's
own reported Q depend on that sign. `dc_branches` matches on the constant — never on
\\(x < 0\\) in general, since a genuinely negative reactance is a series capacitor and
must pass through with its sign intact.

### Scope: what DC here deliberately does not do

**Shunt admittance is ignored entirely**, both \\(b\\) and \\(g\\), matching
powsybl-open-loadflow (whose `DcEquationSystemCreator` has no shunt term). MATPOWER and
pandapower additionally fold \\(G_s\\) in as a constant real load at \\(|V| = 1\\);
gridoxide does not, so on data where \\(g_{shunt} \neq 0\\) its slack pickup differs from
theirs by exactly \\(\sum g_{ii}\\). gridoxide has three separate shunt sources
(`Line::g_shunt`, `Transformer::y_shunt`, and the `network::ShuntAdm` list the DC entry
point is not even handed); covering one and not the others would be worse than covering
none.

**Bus type is ignored beyond `Slack`** in the *solve*: PV and PQ buses are both unknowns,
because the distinction between them is entirely about reactive power. It is not ignored on
write-back, though. A `PV` bus's `voltage_mag` is a setpoint — an input the AC solver holds
fixed and never updates — so DC leaves it alone, along with the slack's, and normalizes only
`PQ` magnitudes to its own \\(|V| = 1\\) assumption. Overwriting a setpoint would silently
change the result of any AC solve seeded from that state.

**Distributed slack is not supported**, here or in the sensitivity factors. Each island
has one reference, or several whose pickup split is arbitrary.

### Validated against real data

`tests/dc_powerflow_test.rs`. The load-bearing test is
`dc_flows_match_the_lossless_ac_limit`, which checks DC against gridoxide's *own* AC
branch-flow code (`branch_flow::terminal_flow`, built on `network::branch_calc_param`) on
a lossless, meshed network carrying both an off-nominal transformer and a phase shifter.
That is a genuinely independent path — complex π-model arithmetic against real Bθ
assembly — and the two agree to \\(10^{-6}\\), the cubic term the linearization drops. A
flipped \\(\varphi\\) sign or a \\(1/k^2\\) where \\(1/k\\) belongs fails it by several
orders of magnitude.

Alongside it: exact power conservation on `symmetric/transmission-case` (\\(10^{-9}\\)),
the link-sign regression on `link/dummy-test`, and the flat-branch-index contract against
`branch_flow::branch_params`.

`dc_angle_error_tracks_the_voltage_assumption_it_rests_on` is worth reading for what it
measures. On `symmetric/transmission-case` the worst DC-versus-AC angle gap is 0.041 rad
— about half the network's entire 0.087 rad angle span. That is not an implementation
error: this fixture's AC voltages run to **1.19 p.u.**, and since flow goes as
\\(V_i V_j b \sin\delta\\), a bus at 1.19 p.u. carries its power at a visibly smaller
angle than DC predicts. Restricted to the buses whose AC voltage actually lands near
1 p.u. — where DC's own assumption holds — the same comparison comes in at
\\(2.6\times10^{-3}\\) rad, two orders better. DC's accuracy is exactly as good as the
assumption it rests on, and this test measures both sides of that.

## Sensitivity factors: PTDF and LODF

Because DC is *exactly* linear, the derivative of a branch flow with respect to a bus
injection is not a local slope that drifts as the operating point moves — it is a
constant, and the true global answer. That is what makes these factors worth
precomputing, and why they exist for DC and not for AC.

**PTDF** answers *if I inject one more unit at bus \\(j\\) and let the reference absorb
it, how much shows up on branch \\(k\\)?* **LODF** answers *if branch \\(l\\) trips, what
fraction of what it was carrying lands on branch \\(k\\)?* — the workhorse of contingency
screening, since it replaces a re-solve per outage with a matrix column.

\\[ \mathrm{PTDF}[:, j] = B_f\,\theta^{(j)}, \qquad B_{NN}\,\theta^{(j)} = e_j \\]
\\[ \mathrm{PTDF}[k, :] = x, \qquad B_{NN}\,x = B_f[k, N]^\mathsf{T} \\]
\\[ \mathrm{LODF}[:, l] = \frac{\mathrm{PTDF}[:, from_l] - \mathrm{PTDF}[:, to_l]}{1 - d_l},
   \qquad d_l = \mathrm{PTDF}[l, from_l] - \mathrm{PTDF}[l, to_l] \\]

Two economies fall out of the structure. A PTDF *row* is one solve rather than one per
bus, because \\(B_{NN}\\) is symmetric — which is true only because the phase shift was
kept out of the matrix. And a LODF column's numerator is one solve against
\\(e_{from} - e_{to}\\), which yields \\(d_l\\) as a by-product, so a LODF column costs
exactly what a PTDF column does.

`linear::sensitivity::DcSensitivity` factorizes each island's \\(B\\) once in `new` and
every accessor is a solve against that — the reason `sparse::RealFactorization` exists
rather than reusing the Newton path's `solver::LinearSolver`, whose contract
refactorizes on every call and would pay for a factorization once per bus.

**Radial branches have no LODF column.** \\(d_l \to 1\\) exactly when branch \\(l\\) is a
bridge: removing it disconnects the network, so its power has nowhere to redistribute to
and no finite factor exists. `lodf_column` returns `None` and `is_radial` reports it —
a structural fact, not a numerical accident, which is why it is not papered over with a
large number.

**Prefer columns and rows to matrices.** A dense PTDF on `case9241pegase` is 1.19 GB and
a dense LODF 2.06 GB. `ptdf_dense`/`lodf_dense` exist for small networks and for tests,
and their doc comments carry those numbers.

### Validated against real data

`tests/dc_sensitivity_test.rs`, against oracles that share no code with the sensitivity
module. PTDF is checked by perturbing an injection and re-solving — exact to round-off,
since DC is linear, so this is the derivative rather than an approximation of it. LODF is
checked by actually opening the branch and re-solving, then confirming each surviving
branch picked up the predicted fraction. Both hold to \\(10^{-9}\\).

## Tool reference

| Tool | Susceptance | Phase shift | Where |
|---|---|---|---|
| **gridoxide** | both, selectable (`DcApproximation`) | RHS injection | `linear::btheta::dc_power_flow` |
| lightsim2grid | \\(1/x\\), precomputed per branch | RHS injection | `BaseDCAlgo.tpp`; `ydc_*` in `TwoSidesContainer_rxh_A.hpp` |
| powsybl-open-loadflow | both (`DcApproximationType`) | RHS injection, opposite \\(\alpha\\) sign convention | `dc/equations/DcEquationSystemCreator` |
| pandapower | \\(1/x\\) | `Pbusinj` (MATPOWER-derived) | `rundcpp`, `makeBdc` |
| MATPOWER | \\(1/x\\) | `Pbusinj` | `dcpf`, `makeBdc` |
| power-grid-model | — (no Bθ mode; its `linear` method is the [constant-admittance](./linear_impedance.md) one) | — | — |

Note the last row. "DC / linear power flow" names two different algorithms across these
tools, and power-grid-model's `CalculationMethod.linear` is not this one — see
[The Constant-Admittance Linearization](./linear_impedance.md).

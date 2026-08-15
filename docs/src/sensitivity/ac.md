# AC Sensitivity Analysis

## Motivation

A power flow answers *what is the state of the network?* Most operational questions are one
step past that: *what happens if something changes?*

- This generator is about to ramp 50 MW. Which line picks it up?
- This corridor is at 98% of its limit. Which injection relieves it fastest, and by how much
  per MW?
- The voltage at this bus is sagging. Would a tap change help, or does it need reactive
  support?

Each is a partial derivative of the solved operating point with respect to an input. You could
answer any of them by re-solving — perturb, solve again, subtract — and for one question that
is perfectly reasonable. It stops being reasonable at a hundred questions, and it is
unnecessary: the derivative is available in closed form from a factorization the solve already
computed.

## The formulation

A converged power flow satisfies

\\[ g(x, p) = s_{calc}(x) - s_{spec}(p) = 0 \\]

where \\(x = [\theta_{non\text{-}slack};\ |V|_{PQ}]\\) is exactly the unknown vector Newton
iterates on, and \\(p\\) is whatever input we are differentiating with respect to.
Differentiating the identity and rearranging:

\\[ \frac{dx}{dp} = -J^{-1} \frac{\partial g}{\partial p} \\]

with \\(J = \partial g / \partial x\\) — **the same Jacobian Newton already builds**, filled at
the converged state. Nothing new is assembled.

For an injection the right-hand side is especially simple. A specified injection enters exactly
one equation, so \\(\partial g/\partial p = -e_i\\) and

\\[ \frac{dx}{dp_i} = J^{-1} e_i \\]

— a single triangular solve with a unit right-hand side. That is the entire method for the
common case.

Any quantity of interest \\(f\\) then follows by the chain rule:

\\[ \frac{df}{dp} = \left(\frac{\partial f}{\partial x}\right)^{\mathsf T} \frac{dx}{dp}
                  + \left.\frac{\partial f}{\partial p}\right|_{x} \\]

For a branch flow, \\(\partial f/\partial x\\) is already computed by
`branch_flow::terminal_flow_derivs` — written originally for the state estimator, reused here
unchanged. The second term is zero unless \\(p\\) is the branch's own tap; see
[below](#the-direct-term).

## Two directions

The expression above can be bracketed either way, and the two groupings cost very differently:

| | Solve | One solve gives you | Ask when |
|---|---|---|---|
| **Forward** | \\(J z = e_i\\), per *variable* | every function's response to that variable | few things move, many are watched |
| **Adjoint** | \\(J^{\mathsf T} w = \partial f/\partial x\\), per *function* | that function's response to every variable | one thing is watched, many could move |

Forward is *"this generator ramps — what happens to all 5,000 branches?"* Adjoint is *"this
line is overloaded — which of 500 injections relieves it?"* Neither is derivable cheaply from
the other, so both are offered: `state_response`/`branch_response` and `function_row`.

Both run against **one** factorization, taken once in `AcSensitivity::new`.
`sparse::RealFactorization::solve_transpose` reuses the same LU for the transposed direction
rather than factorizing \\(J^{\mathsf T}\\) separately — an `LU = PAQ` decomposition solves the
transposed system by running the same triangular factors in the opposite order.

This mirrors the `ptdf_column`/`ptdf_row` pair in
[DC power flow](../powerflow/dc.md#sensitivity-factors-ptdf-and-lodf), which is the same
duality on the DC side.

## Variables

| Variable | Meaning | Zero when |
|---|---|---|
| `ActiveInjection(bus)` | \\(P\\) injected at a bus | the bus is slack |
| `ReactiveInjection(bus)` | \\(Q\\) injected at a bus | the voltage magnitude is held (PV or slack) |
| `TransformerRatio(branch)` | off-nominal ratio \\(k\\) | the branch is a line |
| `PhaseShift(branch)` | phase-shifter angle \\(\alpha\\) | the branch is a line |

The zeros are answers, not gaps. A slack bus's injection is not an input at all — it is what
the solve determines — so perturbing it moves nothing, exactly as a DC PTDF column is zero at
its reference bus. Likewise reactive power at a PV bus is an output of the voltage controller.

Branch indices are the crate's flat order: lines first, then transformers.

## Taps and phase shifters

A tap is a different shape of variable from an injection, and gets two things wrong if treated
casually.

**The sign flips.** An injection enters through \\(s_{spec}\\), so
\\(\partial g/\partial p = -\partial s_{spec}/\partial p\\). A tap changes the branch's
admittance and therefore the *calculated* injections at both its ends, so
\\(\partial g/\partial p = +\partial s_{calc}/\partial p\\). The response picks up the opposite
sign.

**The direct term.** <a name="the-direct-term"></a>A tapped branch's own flow depends on the
tap *directly*, not only through the state — that \\(\partial f/\partial p|_{x}\\) term above.
It is zero for every branch except the tapped one, where it is large.

This is not a rounding-level concern. On `distribution-case`, dropping it changes
\\(dP_8/dk\\) from \\(-0.276\\) to \\(+0.286\\): a sign flip, on the branch whose tap is being
adjusted, which is the branch a user is most likely to read. Every *other* branch stays
correct, which is exactly what makes the omission dangerous — the answer looks entirely
plausible.

The admittance derivatives themselves are unusually tidy. With \\(tap = k\,e^{j\alpha}\\), and
given that \\(y_{ff}\\) always carries \\(1/k^2\\), \\(y_{ft}\\) and \\(y_{tf}\\) always carry
\\(1/k\\), and \\(y_{tt}\\) never depends on the tap at all:

\\[ \frac{\partial y_{ff}}{\partial k} = \frac{-2\,y_{ff}}{k}, \quad
    \frac{\partial y_{ft}}{\partial k} = \frac{-y_{ft}}{k}, \quad
    \frac{\partial y_{tf}}{\partial k} = \frac{-y_{tf}}{k}, \quad
    \frac{\partial y_{tt}}{\partial k} = 0 \\]

\\[ \frac{\partial y_{ft}}{\partial \alpha} = j\,y_{ft}, \quad
    \frac{\partial y_{tf}}{\partial \alpha} = -j\,y_{tf}, \quad
    \frac{\partial y_{ff}}{\partial \alpha} = \frac{\partial y_{tt}}{\partial \alpha} = 0 \\]

Every entry is a scaling of itself, so nothing has to be rebuilt from nameplate data — and the
relations hold unchanged for the half-open terminal states, where the affected entries are
already zero.

Taps come out of the adjoint at no extra solve, alongside the injection sensitivities. So
*"which tap relieves this overload?"* costs exactly what *"which injection relieves it?"* does.

## How this differs from the DC factors

[PTDF and LODF](../powerflow/dc.md#sensitivity-factors-ptdf-and-lodf) are **exact** for the
model they describe, because the DC model is linear — a PTDF is not an approximation of DC, it
*is* DC. These are different. The AC problem is nonlinear, so these are first-order derivatives:
accurate for small perturbations and degrading as the step grows.

What they buy in exchange is everything DC discards — voltage magnitudes, reactive flows,
losses. A DC PTDF cannot tell you that an injection will collapse a voltage; an AC sensitivity
can.

The usual division of labour: screen thousands of cases in DC, answer precisely about the few
that matter in AC.

## Islands

None, and none is needed. Every island carries its own slack, so the assembled Jacobian is
block-diagonal across islands and non-singular as a whole. A sourceless island contributes no
rows at all, because `network::mark_unreferenced_islands` has already pinned its buses to
`Slack` — so it is excluded rather than making the system singular.

This is simpler than the DC side, where `DcSensitivity` factorizes each island's \\(B\\)
separately because the reduced \\(B\\) is built per island.

## What is not here

**No contingency sensitivities.** An outage is a finite change in topology, not an
infinitesimal change in an input, so it is not a derivative at all. The DC side handles
outages with a Woodbury update (`DcSensitivity::outage_flows`) that has no equally cheap AC
analogue; `batch::BatchSolver::solve_contingencies` is the AC answer, and it re-solves.

**No sensitivities through the outer loops.** These differentiate the power-flow equations as
posed. If PV→PQ switching would trip at the perturbed point, or a tap controller would respond,
the derivative does not know it — it is the derivative of the *current* configuration.

## Validation

`tests/ac_sensitivity_test.rs`, against an oracle that shares no code with the sensitivity
module: perturb an input, run the ordinary Newton solver to convergence, take a **central**
difference of the solved state.

Because AC is nonlinear the finite difference is not exact, so the two constants are chosen
together against each other:

- the solve's residual noise is divided by \\(2h\\), so a *smaller* step amplifies it;
- the central difference's own truncation error is \\(O(h^2)\\), so a *larger* step grows it.

At \\(h = 10^{-4}\\) and a solve tolerance of \\(10^{-10}\\) both land near \\(10^{-7}\\).
Tightening the solve further does not help: the test case's residual floors at
\\(1.2 \times 10^{-11}\\) in double precision, so asking for \\(10^{-12}\\) simply never
converges.

Tap sensitivities are checked the same way, by actually changing a `Transformer`'s tap and
re-solving, so the whole downstream chain sees it as a real tap change. Forward and adjoint are
additionally checked against each other to \\(10^{-9}\\) — the same scalar bracketed the other
way, so any disagreement is a transpose or indexing error rather than a modelling choice.

## Using it

From the CLI — forward, then adjoint:

```bash
gridoxide sensitivity network.json --dp 2
gridoxide sensitivity network.json --watch 8
```

`--dp`/`--dq` take a bus, `--dk`/`--dalpha` a branch, `--watch` a branch to watch;
`--terminal from|to` picks the end. It refuses to differentiate a power flow that did not
converge, rather than printing derivatives of nothing.

From Rust:

```rust
use gridoxide::ac_sensitivity::{AcSensitivity, Function, Variable};
use gridoxide::branch_flow::Terminal;

let sens = AcSensitivity::new(&solved_buses, &ybus, &lines, &transformers)
    .expect("nonsingular at a converged point");

// Forward: bus 2 ramps — what does every branch do?
let flows = sens.branch_response(Variable::ActiveInjection(2), Terminal::From).unwrap();

// Adjoint: branch 8 is loaded — what moves it?
let row = sens
    .function_row(Function::BranchActivePower { branch: 8, terminal: Terminal::From })
    .unwrap();
let (best_bus, _) = row
    .d_active
    .iter()
    .enumerate()
    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
    .unwrap();
```

From Python:

```python
import gridoxide

s = gridoxide.AcSensitivityModel("network.json")

col = s.column(active_injection=2)   # bus 2 ramps — what responds?
row = s.row(branch=8)                # branch 8 is loaded — what moves it?

# Which tap has the most leverage on branch 8?
best = max(range(s.n_branches), key=lambda b: abs(row.d_transformer_ratio[b]))
```

`AcSensitivityModel` is a class rather than a function because there is genuinely something
worth keeping between calls: one factorization, arbitrarily many questions.

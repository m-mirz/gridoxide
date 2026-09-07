# The Constant-Admittance Linearization

## Motivation

"Linear power flow" names two unrelated algorithms in this field, and the difference
matters enough to be worth its own page.

The [DC approximation](./dc.md) discards resistance and voltage magnitude to get a real
system in angles alone. That is the right trade on transmission, where flows divide by
reactance and magnitudes barely move. It is the wrong trade on a distribution feeder,
where the interesting quantity *is* the voltage drop along the feeder — precisely what
DC throws away by construction.

The constant-admittance linearization makes a different sacrifice. It keeps the full
complex network and keeps resistance; what it linearizes is the **load model**. A
constant-power load is what makes power flow nonlinear — the current it draws depends on
the voltage that the current itself determines. Replace each load with the fixed
admittance that would draw its rated power at \\(|V| = 1\\), and the circuit becomes an
ordinary linear one:

\\[ S_i = U_i \overline{I_i} = -U_i\overline{U_i Y_{load}}
   \;\implies\; Y_{load} = -\overline{S_i} \quad\text{at } |U| = 1 \\]

Fold that into the Y-bus diagonal and one complex factorization gives every voltage,
magnitude and angle alike.

## What changes in the equation system

\\[ Y'\,U = I, \qquad Y' = Y + \operatorname{diag}(-\overline{S}) \\]

with fixed-voltage buses moved to the right-hand side via their known \\(U\\). Solved
once. No iteration, no convergence criterion.

The approximation is *exact* when the loads really are constant-impedance — which is not
a hypothetical, since gridoxide's `ZipTerm`/`ZipKind` model represents exactly that case
— and degrades as loading moves away from nominal. It is at its best where DC is at its
worst and vice versa:

| | DC (Bθ) | Constant-admittance |
|---|---|---|
| Arithmetic | real | complex |
| Keeps resistance | no | yes |
| Produces \\(\|V\|\\) | no | yes |
| Produces Q | no | yes |
| Exact when | angles are small and \\(r \to 0\\) | loads are constant-impedance |
| Natural home | transmission | distribution |

## Where this fits in gridoxide today

`linear::impedance::linear_power_flow`, reached through
`PowerFlowMethod::LinearImpedance`.

gridoxide has had this algorithm since before it was a solver: it was written as
Newton-Raphson's warm start, mirroring power-grid-model's own
`NewtonRaphsonPFSolver::initialize_derived_solver`, and it remains that
(`network::linear_initial_guess`). Both entry points share one implementation
(`linear::impedance::solve_constant_admittance`); what the standalone mode adds is the
island handling a warm start never needed — `connected_components`, `classify` and
`mark_unreferenced_islands`, so a sourceless island is pinned rather than solved and a
singular island is attributable to itself.

The two differ in one deliberate respect. `linear_initial_guess` runs *before* the
solver's own classification pass, so it must not mutate bus types, and a singular result
there is simply a warm start that did not happen rather than a failure to report.

This is mathematically power-grid-model's `CalculationMethod.linear`, down to the
\\(Y_{load} = -\overline{S}\\) step
(`power_grid_model/math_solver/linear_pf_solver.hpp`).

### Scope: only PQ buses are unknowns

`PV` buses are held at their present voltage, exactly as `Slack` buses are. The method
has no mechanism for "magnitude fixed, angle free" — that constraint is not linear in
\\(U\\).

On power-grid-model data this costs nothing, because PGM has no PV buses at all: its
sources are slacks. That is also why the comparison with PGM is exact. On data that does
carry PV buses — CGMES imports, typically — treating them as fully fixed is a second
approximation on top of the linearization, and it is the reason this mode is offered
alongside Newton-Raphson rather than instead of it.

### Validated against real data

In-module tests in `src/linear/impedance.rs`.
`a_constant_impedance_load_is_reproduced_exactly` checks the defining property: a load
that genuinely is constant-impedance is solved exactly, so the solved voltages reproduce
the specified power to round-off. `agrees_with_the_warm_start_entry_point` pins the
promotion as a refactor rather than a reimplementation, asserting the standalone mode and
`network::linear_initial_guess` agree to \\(10^{-15}\\) bus for bus.

## Tool reference

| Tool | Has it | Where |
|---|---|---|
| **gridoxide** | yes | `linear::impedance::linear_power_flow` |
| power-grid-model | yes, plus a `linear_current` variant | `CalculationMethod.linear`, `linear_pf_solver.hpp` |
| VeraGrid | yes, as a distinct linearization | `SolverType.LACPF` |
| lightsim2grid | no | — |
| powsybl-open-loadflow | no | — |
| pandapower | no | — |

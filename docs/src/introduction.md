# Introduction

`gridoxide` is an AC power flow library written in Rust. At its core it solves the power flow
equations with the Newton-Raphson method, using a sparse Jacobian throughout — assembly,
factorization, and solve — and the other analyses it offers are built on that same core rather than
beside it.

That reuse is the recurring theme, and it is worth naming up front. The state estimator's branch
flow derivatives are what the sensitivity analysis differentiates with. The Jacobian Newton
assembles each iteration is the matrix the sensitivities factorize. The sequence-domain parameters
the unbalanced power flow needs are what the short-circuit calculation assembles its fault network
from. Each capability cost far less than it would have standalone, and several of them found latent
bugs in the shared code on their way in.

This book covers both the *method* (what the equations are, how the sparse solve works, what each
modeling feature changes about the equation system) and the *tool* (how to build it, which linear
solver backends exist, how CGMES input is mapped onto the internal network model).

## Where to start

- **[Getting Started](./getting_started/building.md)** — build the Rust project, run a solve, or
  `pip install gridoxide` and drive it from Python.
- **[Power Flow](./powerflow/index.md)** — the Newton-Raphson formulation, the two linearizations
  (DC and constant-admittance), and the modeling features that change the equation system:
  reactive power limits, zero-impedance branches, and multiple islands.
- **[Sensitivity Analysis](./sensitivity/ac.md)** — differentiating a converged operating point:
  what responds when an injection or a tap moves, and what would move a quantity you are watching.
  (The DC factors, PTDF and LODF, live in the [DC chapter](./powerflow/dc.md) alongside the method
  they are exact for.)
- **[Short Circuit](./short_circuit/index.md)** — IEC 60909 fault currents in the phase domain, the
  four fault types and their boundary conditions.
- **[State Estimation](./state_estimation/index.md)** — the weighted-least-squares problem, what
  each measurement type means, observability, and bad-data detection.
- **[Sparse Linear Solvers](./solvers/backends.md)** — the five interchangeable linear-solver
  backends, and a step-by-step walkthrough of the KLU algorithm all of them are measured against.
- **[CGMES Data Model](./cgmes/index.md)** — reading ENTSO-E RDF/XML grid models, and how
  individual CIM classes map onto buses, branches, and injections.
- **[Reference](./reference/feature_comparison.md)** — how gridoxide compares against five other
  power flow tools, where the benchmark numbers live, and the licensing of every vendored and
  translated piece of third-party code.

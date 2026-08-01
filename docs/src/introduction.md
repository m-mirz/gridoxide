# Introduction

`gridoxide` is an AC power flow solver written in Rust. It solves the power flow equations for an
electrical grid with the Newton-Raphson method, using a sparse Jacobian throughout — assembly,
factorization, and solve.

This book covers both the *method* (what the equations are, how the sparse solve works, what each
modeling feature changes about the equation system) and the *tool* (how to build it, which linear
solver backends exist, how CGMES input is mapped onto the internal network model).

## Where to start

- **[Getting Started](./getting_started/building.md)** — build the Rust project, run a solve, or
  `pip install gridoxide` and drive it from Python.
- **[Power Flow](./powerflow/index.md)** — the Newton-Raphson formulation, and the three modeling
  features that change it: reactive power limits, zero-impedance branches, and multiple islands.
- **[Sparse Linear Solvers](./solvers/backends.md)** — the five interchangeable linear-solver
  backends, and a step-by-step walkthrough of the KLU algorithm all of them are measured against.
- **[CGMES Data Model](./cgmes/index.md)** — reading ENTSO-E RDF/XML grid models, and how
  individual CIM classes map onto buses, branches, and injections.
- **[Reference](./reference/feature_comparison.md)** — how gridoxide compares against five other
  power flow tools, where the benchmark numbers live, and the licensing of every vendored and
  translated piece of third-party code.

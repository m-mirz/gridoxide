# Summary

[Introduction](./introduction.md)

# Getting Started

- [Building and Running](./getting_started/building.md)
- [Python Bindings](./getting_started/python.md)

# Power Flow

- [The Power Flow Problem](./powerflow/index.md)
- [DC (Bθ) Power Flow](./powerflow/dc.md)
- [The Constant-Admittance Linearization](./powerflow/linear_impedance.md)
- [Reactive Power Limits (PV → PQ Switching)](./powerflow/q_limits.md)
- [Distributed Slack](./powerflow/distributed_slack.md)
- [Ideal Switches and Zero-Impedance Branches](./powerflow/zero_impedance_branches.md)
- [Multi-Island Power Flow](./powerflow/multi_island.md)

# Sensitivity Analysis

- [AC Sensitivity Analysis](./sensitivity/ac.md)

# Optimal Power Flow

- [The Optimal Power Flow Problem](./opf/index.md)

# Short Circuit

- [The Short-Circuit Problem](./short_circuit/index.md)
- [Fault Types and Their Boundary Conditions](./short_circuit/faults.md)

# State Estimation

- [The State Estimation Problem](./state_estimation/index.md)
- [Measurements and What They Mean](./state_estimation/measurements.md)
- [The Iterative-Linear Method](./state_estimation/iterative.md)
- [Observability and Bad Data](./state_estimation/diagnostics.md)

# Sparse Linear Solvers

- [Backends and Factorization Reuse](./solvers/backends.md)
- [Inside KLU: the Sparse Solve, Step by Step](./solvers/klu.md)

# CGMES Data Model

- [Reading CGMES Input](./cgmes/index.md)
- [Node-Breaker Topology](./cgmes/node_breaker.md)
- [StaticVarCompensator](./cgmes/static_var_compensator.md)
- [Line Shunt Conductance (`ACLineSegment.gch`)](./cgmes/shunt_conductance.md)
- [PhaseTapChangerLinear](./cgmes/phase_tap_changer_linear.md)
- [RatioTapChanger.RatioTapChangerTable](./cgmes/ratio_tap_changer_table.md)
- [ExternalNetworkInjection](./cgmes/external_network_injection.md)

# Reference

- [Feature Comparison](./reference/feature_comparison.md)
- [Benchmarking and Profiling](./reference/benchmarking.md)
- [Provenance and Licensing](./reference/provenance.md)

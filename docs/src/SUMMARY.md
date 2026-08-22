# Summary

[Introduction](./introduction.md)

# Getting Started

- [Building and Running](./getting_started/building.md)
- [Python Bindings](./getting_started/python.md)
- [C and C++ API](./getting_started/c_api.md)

# Power Flow

- [The Power Flow Problem](./powerflow/index.md)
- [DC (Bθ) Power Flow](./powerflow/dc.md)
- [The Constant-Admittance Linearization](./powerflow/linear_impedance.md)
- [Outer Loops](./powerflow/outer_loops.md)
- [Reactive Power Limits (PV → PQ Switching)](./powerflow/q_limits.md)
- [Distributed Slack](./powerflow/distributed_slack.md)
- [Transformer Tap Control](./powerflow/tap_control.md)
- [Ideal Switches and Zero-Impedance Branches](./powerflow/zero_impedance_branches.md)
- [Multi-Island Power Flow](./powerflow/multi_island.md)

# Sensitivity Analysis

- [AC Sensitivity Analysis](./sensitivity/ac.md)

# Optimal Power Flow

- [The Optimal Power Flow Problem](./opf/index.md)
- [Two Buses, One Congested Line](./opf/worked_example.md)

# Short Circuit

- [The Short-Circuit Problem](./short_circuit/index.md)
- [Fault Types and Their Boundary Conditions](./short_circuit/faults.md)
- [Symmetrical Components and the Fault Equations](./short_circuit/sequence.md)
- [A Fault Current, by Hand](./short_circuit/worked_example.md)

# State Estimation

- [The State Estimation Problem](./state_estimation/index.md)
- [Measurements and What They Mean](./state_estimation/measurements.md)
- [The Iterative-Linear Method](./state_estimation/iterative.md)
- [Observability and Bad Data](./state_estimation/diagnostics.md)
- [A Weighted Least Squares Estimate, Worked](./state_estimation/worked_example.md)

# Remedial Action Optimization

- [The Remedial Action Problem](./rao/index.md)
- [CNECs, Thresholds and Margins](./rao/margins.md)
- [The Linear Optimization of Range Actions](./rao/linear.md)
- [The Search Tree and the CASTOR Decomposition](./rao/search.md)

# Sparse Linear Solvers

- [Backends and Factorization Reuse](./solvers/backends.md)
- [Inside KLU: the Sparse Solve, Step by Step](./solvers/klu.md)

# Data Import

- [Reading UCTE-DEF Input](./import/ucte.md)
- [Reading IIDM Input](./import/iidm.md)

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
- [Datasets, Tools and Books](./reference/resources.md)
- [Benchmarking and Profiling](./reference/benchmarking.md)
- [Provenance and Licensing](./reference/provenance.md)

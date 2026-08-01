# Summary

[Introduction](./introduction.md)

# Getting Started

- [Building and Running](./getting_started/building.md)
- [Python Bindings](./getting_started/python.md)

# Power Flow

- [The Power Flow Problem](./powerflow/index.md)
- [Reactive Power Limits (PV → PQ Switching)](./powerflow/q_limits.md)
- [Ideal Switches and Zero-Impedance Branches](./powerflow/zero_impedance_branches.md)
- [Multi-Island Power Flow](./powerflow/multi_island.md)

# State Estimation

- [The State Estimation Problem](./state_estimation/index.md)
- [Measurements and What They Mean](./state_estimation/measurements.md)
- [Observability and Bad Data](./state_estimation/diagnostics.md)

# Sparse Linear Solvers

- [Backends and Factorization Reuse](./solvers/backends.md)
- [Inside KLU: the Sparse Solve, Step by Step](./solvers/klu.md)

# CGMES Data Model

- [Reading CGMES Input](./cgmes/index.md)
- [StaticVarCompensator](./cgmes/static_var_compensator.md)
- [Line Shunt Conductance (`ACLineSegment.gch`)](./cgmes/shunt_conductance.md)
- [PhaseTapChangerLinear](./cgmes/phase_tap_changer_linear.md)
- [RatioTapChanger.RatioTapChangerTable](./cgmes/ratio_tap_changer_table.md)
- [ExternalNetworkInjection](./cgmes/external_network_injection.md)

# Reference

- [Feature Comparison](./reference/feature_comparison.md)
- [Benchmarking and Profiling](./reference/benchmarking.md)
- [Provenance and Licensing](./reference/provenance.md)

# Dynamics autodiff: TODO

Starting points for a plan, taken from a comparison with PowerDynamics.jl
(NetworkDynamics.jl, ModelingToolkit, ForwardDiff). Every device Jacobian in
`src/dynamics/models/` is written by hand today and checked against
`finite_difference` by gate G4 (`tests/dynamics_jacobian_test.rs`).

## Approach

- [ ] Use forward-mode autodiff with dual numbers, on each device's own
  block. PowerDynamics.jl needs sparsity detection and coloring for the
  whole-network Jacobian. gridoxide does not: its pattern is fixed by
  construction (`docs/src/dynamics/dae.md`), and each device adds a small
  dense block of `∂f/∂x`, `∂f/∂V`, `∂I/∂x` and `∂I/∂V`. One pass with
  `n_states + 2` seeds gives all four blocks exactly.
- [ ] Choose between `num-dual` and an in-house `Dual<const N: usize>`.
  Check first whether `num-dual` pulls in `nalgebra`; the workspace uses
  faer. An in-house type needs `sin`, `cos`, `sqrt` and `exp`, and the
  `num-traits` traits so that `Complex<Dual>` works in `injection`.

## Fitting it into the code

- [ ] Keep the `dyn` boundary. `DynamicModel`, `Machine` and `Control` are
  used as `Box<dyn …>`, so a generic method on them would break that. Put
  the generic residual in a separate trait on each concrete model, and
  provide `jacobian` (and `Control::jacobian`, `output_jacobian`) through a
  blanket adapter that evaluates it with duals.
- [ ] Differentiate machines, controls and loads, not the whole `Unit`. The
  chain rule in `src/dynamics/models/unit.rs` is cheap and already written.
  Differentiating through the boxed sub-models would need a dual type whose
  size is only known at run time.

## Rollout

- [ ] Prototype on `GenRound`: write its residual generically, compare the
  dual-number Jacobian with the analytic one bit for bit, and time both in
  `examples/dynamics_scale.rs`.
- [ ] Make the autodiff Jacobian the G4 oracle in place of central
  differences, so the check becomes an exact match instead of a tolerance.
- [ ] Make it the default `jacobian`, so a new model needs only
  `derivatives` and `injection`.
- [ ] Keep hand-written Jacobians only where the benchmark shows they are
  worth it. `machine.rs` is the likely candidate.
- [ ] Replace the central difference in `smallsignal::sensitivities` by
  seeding the parameter. That removes the need for the `set_parameter`
  mutation it relies on.

## Out of scope for now

- Enzyme (`std::autodiff`): nightly only and needs an Enzyme-enabled
  toolchain, so it cannot go into a crates.io library on edition 2024.
  Indirect calls through `dyn` are also a weak spot. Look again once it
  is stable.
- Symbolic codegen in the style of ModelingToolkit. Its value is index
  reduction and composing models from equations; `unit.rs` already
  composes units, and the models are explicit ODEs plus an injection.

// The sparse LU backends live in their own crate; re-exporting them under
// their old module names keeps `crate::sparse::…` working inside this crate
// and `gridoxide::sparse::…` working for everyone else.
#[cfg(feature = "klu")]
pub use gridoxide_sparse::sparse_klu;
#[cfg(feature = "pardiso")]
pub use gridoxide_sparse::sparse_pardiso;
pub use gridoxide_sparse::{block_sparse, klu_native, lu, sparse};
// Likewise the solvers: the network model, power flow and the analyses built
// on it, state estimation and short circuit, plus `run_power_flow` and its
// types at the crate root.
pub use gridoxide_core::{
    OuterLoopOutcome, PowerFlowInit, PowerFlowMethod, PowerFlowOptions, PowerFlowReport,
    RemoteControlData, TapData, initialize_newton, run_power_flow, run_power_flow_analysis,
    run_power_flow_analysis_from_ybus, run_power_flow_with_remote,
};
pub use gridoxide_core::{
    ac_sensitivity, batch, bde, branch_flow, constrained, continuation, dc, dispatch,
    injection_hessian, jacobian, json, linear, measurement, network, outerloop, qv, ratings, roots,
    scenario, se, shortcircuit, solver, switches, topology, types,
};

#[cfg(feature = "capi")]
pub mod capi;
#[cfg(feature = "cgmes")]
pub mod cgmes;
#[cfg(feature = "dynamics")]
pub mod dynamics;
#[cfg(feature = "iidm")]
pub mod iidm;
#[cfg(any(
    feature = "ucte",
    feature = "iidm",
    feature = "cgmes",
    feature = "matpower"
))]
pub mod load;
#[cfg(feature = "matpower")]
pub mod matpower;
#[cfg(feature = "opf")]
pub mod opf;
pub mod pgm;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "rao")]
pub mod rao;
#[cfg(feature = "ucte")]
pub mod ucte;

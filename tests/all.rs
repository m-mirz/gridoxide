//! Every integration test, compiled as **one** test binary.
//!
//! Each file under `tests/` used to be its own binary, and each of those linked
//! the whole crate: 114 links after every library edit, and again for every
//! feature set CI builds. Here each file is a module instead, so a library edit
//! relinks once. A file still reads as before — `mod common;` and friends
//! resolve next to it, because a `#[path]` module owns its directory — and its
//! tests are named `<file>::<test>`, so `cargo test --test all rao_` still
//! selects by file.
//!
//! The `cfg`s are what `required-features` said in `Cargo.toml` per file.

#[path = "ac_contingency_test.rs"]
mod ac_contingency_test;
#[path = "ac_sensitivity_test.rs"]
mod ac_sensitivity_test;
#[path = "area_interchange_test.rs"]
mod area_interchange_test;
#[path = "batch_solver_test.rs"]
mod batch_solver_test;
#[path = "bde_test.rs"]
mod bde_test;
#[path = "block_jacobian_test.rs"]
mod block_jacobian_test;
#[path = "branch_flow_test.rs"]
mod branch_flow_test;
#[cfg(feature = "capi")]
#[path = "capi_test.rs"]
mod capi_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_area_test.rs"]
mod cgmes_area_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_dispatch_test.rs"]
mod cgmes_dispatch_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_fullgrid_test.rs"]
mod cgmes_fullgrid_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_identity_test.rs"]
mod cgmes_identity_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_limits_test.rs"]
mod cgmes_limits_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_microgrid_be_test.rs"]
mod cgmes_microgrid_be_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_microgrid_hvdc_test.rs"]
mod cgmes_microgrid_hvdc_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_microgrid_nl_test.rs"]
mod cgmes_microgrid_nl_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_microgrid_type1_test.rs"]
mod cgmes_microgrid_type1_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_microgrid_type2_test.rs"]
mod cgmes_microgrid_type2_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_microgrid_type3_test.rs"]
mod cgmes_microgrid_type3_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_minigrid_test.rs"]
mod cgmes_minigrid_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_node_breaker_se_test.rs"]
mod cgmes_node_breaker_se_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_node_breaker_solve_test.rs"]
mod cgmes_node_breaker_solve_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_node_breaker_test.rs"]
mod cgmes_node_breaker_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_powerflow_test.rs"]
mod cgmes_powerflow_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_pst_phase_tap_changer_linear_test.rs"]
mod cgmes_pst_phase_tap_changer_linear_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_pst_phase_tap_changer_table_test.rs"]
mod cgmes_pst_phase_tap_changer_table_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_realgrid_test.rs"]
mod cgmes_realgrid_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_remote_test.rs"]
mod cgmes_remote_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_smallgrid_test.rs"]
mod cgmes_smallgrid_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_smoke_test.rs"]
mod cgmes_smoke_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_svedala_test.rs"]
mod cgmes_svedala_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_tap_regulation_test.rs"]
mod cgmes_tap_regulation_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_tap_table_test.rs"]
mod cgmes_tap_table_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_voltage_base_test.rs"]
mod cgmes_voltage_base_test;
#[cfg(feature = "cgmes")]
#[path = "cgmes_voltage_control_test.rs"]
mod cgmes_voltage_control_test;
#[path = "cli_continuation_test.rs"]
mod cli_continuation_test;
#[cfg(feature = "dynamics")]
#[path = "cli_dynamics_test.rs"]
mod cli_dynamics_test;
#[path = "cli_estimate_test.rs"]
mod cli_estimate_test;
#[cfg(feature = "opf")]
#[path = "cli_opf_test.rs"]
mod cli_opf_test;
#[path = "cli_qv_test.rs"]
mod cli_qv_test;
#[cfg(all(feature = "rao", feature = "ucte", feature = "iidm"))]
#[path = "cli_rao_test.rs"]
mod cli_rao_test;
#[cfg(all(feature = "rao", feature = "ucte", feature = "iidm"))]
#[path = "cli_security_test.rs"]
mod cli_security_test;
#[path = "cli_sensitivity_test.rs"]
mod cli_sensitivity_test;
#[path = "cli_short_circuit_test.rs"]
mod cli_short_circuit_test;
#[cfg(all(feature = "cgmes", feature = "ucte"))]
#[path = "cli_solve_test.rs"]
mod cli_solve_test;
#[cfg(feature = "cgmes")]
#[path = "cli_switches_test.rs"]
mod cli_switches_test;
#[path = "constrained_switch_test.rs"]
mod constrained_switch_test;
#[path = "continuation_events_test.rs"]
mod continuation_events_test;
#[path = "continuation_test.rs"]
mod continuation_test;
#[path = "dc_batch_test.rs"]
mod dc_batch_test;
#[path = "dc_powerflow_test.rs"]
mod dc_powerflow_test;
#[path = "dc_sensitivity_test.rs"]
mod dc_sensitivity_test;
#[path = "distributed_slack_test.rs"]
mod distributed_slack_test;
#[cfg(all(feature = "dynamics", feature = "iidm"))]
#[path = "dynamics_dyd_test.rs"]
mod dynamics_dyd_test;
#[cfg(feature = "dynamics")]
#[path = "dynamics_dyr_test.rs"]
mod dynamics_dyr_test;
#[cfg(feature = "dynamics")]
#[path = "dynamics_events_test.rs"]
mod dynamics_events_test;
#[cfg(feature = "dynamics")]
#[path = "dynamics_json_test.rs"]
mod dynamics_json_test;
#[cfg(feature = "dynamics")]
#[path = "dynamics_models_test.rs"]
mod dynamics_models_test;
#[cfg(all(feature = "dynamics", feature = "iidm"))]
#[path = "dynamics_reference_test.rs"]
mod dynamics_reference_test;
#[cfg(feature = "dynamics")]
#[path = "dynamics_relay_test.rs"]
mod dynamics_relay_test;
#[cfg(feature = "smallsignal-arpack")]
#[path = "dynamics_smallsignal_arpack_test.rs"]
mod dynamics_smallsignal_arpack_test;
#[cfg(feature = "dynamics")]
#[path = "dynamics_smallsignal_test.rs"]
mod dynamics_smallsignal_test;
#[cfg(feature = "dynamics")]
#[path = "dynamics_test.rs"]
mod dynamics_test;
#[cfg(all(feature = "iidm", feature = "ucte"))]
#[path = "iidm_test.rs"]
mod iidm_test;
#[path = "injection_hessian_test.rs"]
mod injection_hessian_test;
#[cfg(feature = "klu")]
#[path = "klu_jacobian_test.rs"]
mod klu_jacobian_test;
#[path = "klu_native_jacobian_test.rs"]
mod klu_native_jacobian_test;
#[path = "link_test.rs"]
mod link_test;
#[path = "measurement_residual_test.rs"]
mod measurement_residual_test;
#[path = "multi_island_test.rs"]
mod multi_island_test;
#[cfg(feature = "opf")]
#[path = "opf_ac_test.rs"]
mod opf_ac_test;
#[cfg(feature = "opf-highs")]
#[path = "opf_cross_test.rs"]
mod opf_cross_test;
#[cfg(feature = "opf")]
#[path = "opf_data_test.rs"]
mod opf_data_test;
#[cfg(feature = "opf")]
#[path = "opf_dc_test.rs"]
mod opf_dc_test;
#[cfg(feature = "opf-highs")]
#[path = "opf_highs_test.rs"]
mod opf_highs_test;
#[cfg(feature = "opf")]
#[path = "opf_ipm_test.rs"]
mod opf_ipm_test;
#[cfg(feature = "opf-ipopt")]
#[path = "opf_ipopt_test.rs"]
mod opf_ipopt_test;
#[cfg(feature = "opf")]
#[path = "opf_milp_test.rs"]
mod opf_milp_test;
#[cfg(feature = "opf")]
#[path = "opf_nlp_test.rs"]
mod opf_nlp_test;
#[cfg(feature = "opf")]
#[path = "opf_pwl_test.rs"]
mod opf_pwl_test;
#[path = "outer_loop_test.rs"]
mod outer_loop_test;
#[cfg(feature = "pardiso")]
#[path = "pardiso_jacobian_test.rs"]
mod pardiso_jacobian_test;
#[path = "persistent_solver_test.rs"]
mod persistent_solver_test;
#[path = "pgm_asym_line_test.rs"]
mod pgm_asym_line_test;
#[path = "pgm_asym_three_winding_test.rs"]
mod pgm_asym_three_winding_test;
#[path = "pgm_asym_transformer_test.rs"]
mod pgm_asym_transformer_test;
#[path = "pgm_component_coverage_test.rs"]
mod pgm_component_coverage_test;
#[path = "pgm_gen_test.rs"]
mod pgm_gen_test;
#[path = "pgm_line_test.rs"]
mod pgm_line_test;
#[path = "pgm_load_test.rs"]
mod pgm_load_test;
#[path = "pgm_network_test.rs"]
mod pgm_network_test;
#[path = "pgm_node_test.rs"]
mod pgm_node_test;
#[path = "pgm_short_circuit_test.rs"]
mod pgm_short_circuit_test;
#[path = "pgm_shunt_test.rs"]
mod pgm_shunt_test;
#[path = "pgm_source_test.rs"]
mod pgm_source_test;
#[path = "pgm_three_winding_transformer_test.rs"]
mod pgm_three_winding_transformer_test;
#[path = "pgm_transformer_test.rs"]
mod pgm_transformer_test;
#[path = "power_flow_methods_test.rs"]
mod power_flow_methods_test;
#[path = "powerflow_test.rs"]
mod powerflow_test;
#[path = "q_limits_test.rs"]
mod q_limits_test;
#[path = "qv_test.rs"]
mod qv_test;
#[cfg(all(feature = "rao", feature = "ucte"))]
#[path = "rao_castor_test.rs"]
mod rao_castor_test;
#[cfg(feature = "rao")]
#[path = "rao_crac_test.rs"]
mod rao_crac_test;
#[cfg(all(feature = "rao", feature = "ucte"))]
#[path = "rao_cucumber_test.rs"]
mod rao_cucumber_test;
#[cfg(all(feature = "rao", feature = "ucte"))]
#[path = "rao_evaluate_test.rs"]
mod rao_evaluate_test;
#[cfg(all(feature = "rao", feature = "ucte"))]
#[path = "rao_linear_test.rs"]
mod rao_linear_test;
#[cfg(all(feature = "rao", feature = "ucte"))]
#[path = "rao_search_test.rs"]
mod rao_search_test;
#[cfg(all(feature = "rao", feature = "ucte", feature = "opf"))]
#[path = "rao_validate_test.rs"]
mod rao_validate_test;
#[path = "remote_voltage_test.rs"]
mod remote_voltage_test;
#[path = "se_batch_test.rs"]
mod se_batch_test;
#[path = "se_three_phase_test.rs"]
mod se_three_phase_test;
#[path = "state_estimation_test.rs"]
mod state_estimation_test;
#[cfg(feature = "cgmes")]
#[path = "tap_control_test.rs"]
mod tap_control_test;
#[cfg(feature = "ucte")]
#[path = "ucte_test.rs"]
mod ucte_test;

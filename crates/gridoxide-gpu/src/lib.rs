//! Batched AC power flow for NVIDIA GPUs, after `plans/GPU_BATCH_PLAN.md`.
//!
//! Two layers, the second written against the first so that it is built and
//! tested on the CPU before any device code exists:
//!
//! - [`uniform`]: a uniform batch of sparse linear systems sharing one
//!   pattern, with masked factorization and solve — cuDSS's uniform-batch API
//!   as a trait, and [`CpuUniformBatch`], a CPU implementation that enforces
//!   its documented rules.
//! - [`newton`]: the batched Newton-Raphson loop over it, every scenario
//!   stopped at its own convergence rather than after a fixed step count.
//!
//! The device side is behind the `cuda` feature, off by default, so this
//! crate builds and tests without CUDA:
//! [`CudssUniformBatch`](cudss::CudssUniformBatch), the uniform batch on
//! cuDSS, and [`DeviceBatch`](device::DeviceBatch), the network and every
//! scenario's Newton state on the device with the step's kernels.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub mod cudss;
#[cfg(feature = "cuda")]
pub mod device;
pub mod newton;
pub mod uniform;

#[cfg(feature = "cuda")]
pub use cudss::CudssUniformBatch;
#[cfg(feature = "cuda")]
pub use device::DeviceBatch;

pub use newton::{
    BatchError, BatchOptions, BatchReport, Outcome, ScenarioResult, StepPolicy, csr_jacobian,
    solve_batch, solve_contingency_batch,
};
pub use uniform::{
    CpuUniformBatch, CpuUniformBatch08, CpuUniformBatchWith, CsrPattern, LinearCounts, LinearError,
    UniformBatch,
};

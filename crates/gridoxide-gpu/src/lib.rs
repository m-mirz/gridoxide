//! Batched AC power flow for NVIDIA GPUs; what is left to make it faster is
//! in `plans/GPU_BATCH_TODO.md`.
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
//! scenario's Newton state on the device with the step's kernels, and
//! [`GpuBatchSolver`](gpu::GpuBatchSolver), the adaptive loop over both: a
//! session per network that keeps its cuDSS analysis. And
//! [`GpuDcSolver`](dc::GpuDcSolver), DC N-1: one factorization, every
//! outage's angles and flows from one multi-column solve. And
//! [`CudssSolver`](kkt::CudssSolver), the KKT system of `opf::nlp`'s
//! interior point on cuDSS, and [`DeviceAcModel`](ac::DeviceAcModel), the
//! AC-OPF model it solves, evaluated on the device.

#[cfg(feature = "cuda")]
pub mod ac;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub mod cudss;
#[cfg(feature = "cuda")]
pub mod dc;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod kkt;
pub mod newton;
pub mod uniform;

#[cfg(feature = "cuda")]
pub use cudss::CudssUniformBatch;
#[cfg(feature = "cuda")]
pub use dc::GpuDcSolver;
#[cfg(feature = "cuda")]
pub use device::DeviceBatch;
#[cfg(feature = "cuda")]
pub use gpu::GpuBatchSolver;
#[cfg(feature = "cuda")]
pub use kkt::{
    CudssSolver, cudss_ldlt_solver, cudss_ldlt_solver_fast, cudss_llt_solver, cudss_llt_solver_fast,
};

pub use newton::{
    BatchError, BatchOptions, BatchReport, Outcome, ScenarioResult, StepPolicy, csr_jacobian,
    solve_batch, solve_contingency_batch,
};
pub use uniform::{CpuUniformBatch, CsrPattern, LinearCounts, LinearError, UniformBatch};

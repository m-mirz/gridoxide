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
//! The device backend, [`CudssUniformBatch`](cudss::CudssUniformBatch), is
//! behind the `cuda` feature, off by default, so this crate builds and tests
//! without CUDA.

#[cfg(feature = "cuda")]
pub mod cudss;
pub mod newton;
pub mod uniform;

#[cfg(feature = "cuda")]
pub use cudss::CudssUniformBatch;

pub use newton::{BatchOptions, BatchReport, Outcome, ScenarioResult, StepPolicy, solve_batch};
pub use uniform::{CpuUniformBatch, CsrPattern, LinearCounts, LinearError, UniformBatch};

//! GPU Jacobian assembly via CubeCL — `plans/GPU_PLAN.md` Phase 2.
//!
//! [`crate::jacobian::JacobianPattern`] already reduced assembly to the shape a
//! GPU wants: a flat array of independent [`Entry`](crate::jacobian::Entry)
//! recipes, each producing one value at a precomputed offset, with all reads
//! gathered from small per-bus arrays. This module is the transliteration of
//! [`JacobianPattern::fill`](crate::jacobian::JacobianPattern::fill) into a
//! `#[cube]` kernel, extended across a batch: **one thread per (scenario,
//! entry) pair**, fully coalesced writes, no branching beyond the eight-way
//! dispatch on entry kind. That is §3 property 4 ("assembly becomes one flat
//! kernel").
//!
//! # Precision, stated up front
//!
//! This module assembles in **f64**, on CubeCL's CUDA backend
//! (`DefaultRuntime`). The kernel logic and index arithmetic were first
//! proven out in f32 against `wgpu` (WGSL has no f64) — see git history for
//! that version — and this is the same kernel with the dtype and runtime
//! switched, per `scripts/GPU_RUNBOOK.md` Phase 2. §4.4 of the plan already
//! rejected double-single emulation for the wgpu path as costing 10-20x per
//! operation for ~f48 precision; going to a real f64 backend instead of
//! emulating it is exactly that tradeoff resolved correctly.
//!
//! What this module establishes, on whatever CUDA-capable GPU is present:
//!
//! - the kernel's arithmetic and eight-way dispatch match the CPU reference
//!   to f64 exactness,
//! - the (scenario, entry) index arithmetic and the batch stride are right,
//! - the host/device plumbing — buffer layout, upload, launch, readback —
//!   works end to end.
//!
//! [`GpuAssembler`] stays generic over `R: Runtime` — only [`DefaultRuntime`]
//! and this module's dtype changed, not its structure. The dtype is fixed at
//! f64 in the kernel and host buffers below, though, so `R = WgpuRuntime`
//! (WGSL, no f64) will not run correctly any more; reverting to the f32/wgpu
//! path for AMD-iGPU development per §5 means reverting the dtype too — see
//! git history for the last commit before this switch.

use cubecl::prelude::*;

use crate::jacobian::{EntryKind, JacobianPattern};
use crate::types::Bus;

/// Entry-kind discriminants, duplicated as plain `u32` constants because the
/// kernel cannot branch on a Rust enum. Kept in lockstep with
/// [`EntryKind`] by [`kind_code`], which is exhaustive over it — adding a
/// variant there will fail to compile here rather than silently mis-assemble.
const K_HII: u32 = 0;
const K_NII: u32 = 1;
const K_HIK: u32 = 2;
const K_NIK: u32 = 3;
const K_MII: u32 = 4;
const K_LII: u32 = 5;
const K_MIK: u32 = 6;
const K_LIK: u32 = 7;

fn kind_code(k: EntryKind) -> u32 {
    match k {
        EntryKind::Hii => K_HII,
        EntryKind::Nii => K_NII,
        EntryKind::Hik => K_HIK,
        EntryKind::Nik => K_NIK,
        EntryKind::Mii => K_MII,
        EntryKind::Lii => K_LII,
        EntryKind::Mik => K_MIK,
        EntryKind::Lik => K_LIK,
    }
}

/// One thread per `(scenario, entry)` pair.
///
/// Thread `t` handles scenario `t / nnz`, entry `t % nnz`. Per-bus state is
/// indexed `scenario * n_buses + bus`, so each scenario's slice is contiguous
/// and neighbouring threads read neighbouring entries.
///
/// The formulas are exactly `JacobianPattern::fill`'s; see that function for
/// the H/N/M/L derivations. `tests/gpu_assembly_test.rs` asserts the two agree.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn assemble_kernel(
    kinds: &Array<u32>,
    bus_i: &Array<u32>,
    bus_k: &Array<u32>,
    y_re: &Array<f64>,
    y_im: &Array<f64>,
    vm: &Array<f64>,
    va: &Array<f64>,
    p_calc: &Array<f64>,
    q_calc: &Array<f64>,
    values: &mut Array<f64>,
    nnz: u32,
    n_buses: u32,
) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    let pos = ABSOLUTE_POS as u32;
    let scenario = pos / nnz;
    let e = pos % nnz;

    let kind = kinds[e as usize];
    let base = scenario * n_buses;
    let i = base + bus_i[e as usize];
    let k = base + bus_k[e as usize];

    let g = y_re[e as usize];
    let b = y_im[e as usize];
    let vm_i = vm[i as usize];

    let mut out = 0.0f64;

    if kind == K_HII {
        out = -q_calc[i as usize] - vm_i * vm_i * b;
    } else if kind == K_NII {
        out = p_calc[i as usize] / vm_i + vm_i * g;
    } else if kind == K_MII {
        out = p_calc[i as usize] - vm_i * vm_i * g;
    } else if kind == K_LII {
        out = q_calc[i as usize] / vm_i - vm_i * b;
    } else {
        let vm_k = vm[k as usize];
        let ang = va[i as usize] - va[k as usize];
        let sin = f64::sin(ang);
        let cos = f64::cos(ang);

        if kind == K_HIK {
            out = vm_i * vm_k * (g * sin - b * cos);
        } else if kind == K_NIK {
            out = vm_i * (g * cos + b * sin);
        } else if kind == K_MIK {
            out = -vm_i * vm_k * (g * cos + b * sin);
        } else {
            out = vm_i * (g * sin - b * cos);
        }
    }

    values[ABSOLUTE_POS] = out;
}

/// Device-side buffers for one topology, uploaded once and reused across
/// iterations.
///
/// The per-entry recipe arrays (`kinds`, `bus_i`, `bus_k`, `y_re`, `y_im`)
/// depend only on the topology, so they are uploaded at construction and never
/// touched again — exactly the property that makes the pattern worth
/// precomputing in the first place. Only the per-scenario bus state moves per
/// iteration.
pub struct GpuAssembler<R: Runtime> {
    client: ComputeClient<R>,
    kinds: cubecl::server::Handle,
    bus_i: cubecl::server::Handle,
    bus_k: cubecl::server::Handle,
    y_re: cubecl::server::Handle,
    y_im: cubecl::server::Handle,
    nnz: usize,
    n_buses: usize,
}

impl<R: Runtime> GpuAssembler<R> {
    /// Uploads a topology's per-entry recipe. `n_buses` is the bus count the
    /// per-scenario state arrays will be strided by.
    pub fn new(client: ComputeClient<R>, pattern: &JacobianPattern, n_buses: usize) -> Self {
        let entries = pattern.entries();
        let kinds: Vec<u32> = entries.iter().map(|e| kind_code(e.kind)).collect();
        let bus_i: Vec<u32> = entries.iter().map(|e| e.i).collect();
        let bus_k: Vec<u32> = entries.iter().map(|e| e.k).collect();
        let y_re: Vec<f64> = entries.iter().map(|e| e.y.re).collect();
        let y_im: Vec<f64> = entries.iter().map(|e| e.y.im).collect();

        Self {
            kinds: client.create_from_slice(u32::as_bytes(&kinds)),
            bus_i: client.create_from_slice(u32::as_bytes(&bus_i)),
            bus_k: client.create_from_slice(u32::as_bytes(&bus_k)),
            y_re: client.create_from_slice(f64::as_bytes(&y_re)),
            y_im: client.create_from_slice(f64::as_bytes(&y_im)),
            nnz: entries.len(),
            n_buses,
            client,
        }
    }

    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Assembles every scenario's Jacobian values in one launch, returning the
    /// batch's flat array laid out scenario-major (`scenario * nnz + entry`) —
    /// the same layout [`crate::bde::BlockDiagonal::fill`] produces on the CPU.
    ///
    /// `states`, `p_calc` and `q_calc` are per scenario, each inner slice
    /// `n_buses` long.
    pub fn assemble_batch(
        &self,
        states: &[Vec<Bus>],
        p_calc: &[Vec<f64>],
        q_calc: &[Vec<f64>],
    ) -> Vec<f64> {
        let nb = states.len();
        let n = self.n_buses;
        assert!(
            states.iter().all(|s| s.len() == n),
            "every scenario's bus array must be n_buses long"
        );

        let mut vm = Vec::with_capacity(nb * n);
        let mut va = Vec::with_capacity(nb * n);
        let mut p = Vec::with_capacity(nb * n);
        let mut q = Vec::with_capacity(nb * n);
        for s in 0..nb {
            vm.extend(states[s].iter().map(|b| b.voltage_mag));
            va.extend(states[s].iter().map(|b| b.voltage_ang));
            p.extend(p_calc[s].iter().copied());
            q.extend(q_calc[s].iter().copied());
        }

        let vm_h = self.client.create_from_slice(f64::as_bytes(&vm));
        let va_h = self.client.create_from_slice(f64::as_bytes(&va));
        let p_h = self.client.create_from_slice(f64::as_bytes(&p));
        let q_h = self.client.create_from_slice(f64::as_bytes(&q));

        let total = nb * self.nnz;
        let out = self.client.empty(total * size_of::<f64>());

        // One thread per (scenario, entry); round the grid up and let the
        // kernel's bounds check drop the tail.
        let block = 256u32;
        let blocks = total.div_ceil(block as usize) as u32;

        unsafe {
            assemble_kernel::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(blocks),
                CubeDim::new_1d(block),
                ArrayArg::from_raw_parts(self.kinds.clone(), self.nnz),
                ArrayArg::from_raw_parts(self.bus_i.clone(), self.nnz),
                ArrayArg::from_raw_parts(self.bus_k.clone(), self.nnz),
                ArrayArg::from_raw_parts(self.y_re.clone(), self.nnz),
                ArrayArg::from_raw_parts(self.y_im.clone(), self.nnz),
                ArrayArg::from_raw_parts(vm_h, nb * n),
                ArrayArg::from_raw_parts(va_h, nb * n),
                ArrayArg::from_raw_parts(p_h, nb * n),
                ArrayArg::from_raw_parts(q_h, nb * n),
                ArrayArg::from_raw_parts(out.clone(), total),
                self.nnz as u32,
                n as u32,
            );
        }

        let bytes = self.client.read_one_unchecked(out);
        f64::from_bytes(&bytes).to_vec()
    }
}

/// The default runtime for this build: CubeCL's CUDA backend, in **f64** —
/// `scripts/GPU_RUNBOOK.md` Phase 2. NVIDIA-only; see this module's doc
/// comment for the portable f32/wgpu alternative used for AMD-iGPU
/// development.
pub type DefaultRuntime = cubecl::cuda::CudaRuntime;

/// Convenience constructor for the default (CUDA) runtime.
pub fn default_assembler(pattern: &JacobianPattern, n_buses: usize) -> GpuAssembler<DefaultRuntime> {
    let client = <DefaultRuntime as Runtime>::client(&Default::default());
    GpuAssembler::new(client, pattern, n_buses)
}

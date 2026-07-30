// gridoxide's batched AC power flow kernels — CUDA, f64, NVIDIA-only.
//
// This file replaces the CubeCL assembly kernel that lived in `src/gpu.rs`
// through commit ff92b66. The move to hand-written CUDA is not about kernel
// performance (the assembly kernel is <1% of iteration time either way); it
// is about owning the three things CubeCL's runtime hid: which *stream* work
// runs on, when device memory is allocated (CubeCL allocated fresh buffers
// per launch), and stable raw device pointers to bind cuDSS's batched matrix
// against. See `plans/GPU_PLAN.md` §5 for the portability that was
// deliberately given up here, and ff92b66 for the portable kernel itself.
//
// Every kernel below is a transliteration of Rust that already exists in this
// repo and is already cross-validated against five independent solvers. The
// CPU original is named in each kernel's comment; `src/device_layout.rs`
// holds the host-side flattening these kernels index into, with unit tests
// that run on machines with no GPU.
//
// Layout conventions, shared by every kernel:
//
//   * per-bus state    -> `scenario * n_buses + bus`
//   * per-unknown data -> `scenario * blk + r`, blk = n_angle + n_pq,
//                         r < n_angle is angle unknown `non_slack[r]`,
//                         r >= n_angle is magnitude unknown `pq[r - n_angle]`
//   * Jacobian values  -> `scenario * nnz + scatter[entry]`
//
// The launchers are `extern "C"` and take `void* stream` rather than
// `cudaStream_t` so the Rust side needs no bindgen'd CUDA type to call them;
// they return `cudaGetLastError()` as an `int` (0 == cudaSuccess).

#include <cuda_runtime.h>
#include <math.h>

// Entry-kind discriminants. Must stay in lockstep with
// `jacobian::EntryKind`'s `#[repr(u8)]` discriminants and `gpu::kind_code`.
#define K_HII 0u
#define K_NII 1u
#define K_HIK 2u
#define K_NIK 3u
#define K_MII 4u
#define K_LII 5u
#define K_MIK 6u
#define K_LIK 7u

#define THREADS 256

static inline int grid_for(size_t total, int threads) {
    // Cap the grid and let the kernels' grid-stride loops cover the rest, so
    // a 35,040-scenario QSTS batch can't overflow gridDim.x.
    size_t blocks = (total + (size_t)threads - 1) / (size_t)threads;
    const size_t kMaxBlocks = 65535 * 16;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    if (blocks == 0) blocks = 1;
    return (int)blocks;
}

// ---------------------------------------------------------------------------
// Jacobian assembly — transliteration of `jacobian::JacobianPattern::fill_into`
// (and of the CubeCL `assemble_kernel` it replaces), extended across a batch.
//
// One thread per (scenario, entry). `active[s] == 0` writes the precomputed
// identity block instead of the real Newton values, replicating
// `JacobianPattern::fill_identity_into` on-device. That masking is a
// correctness requirement, not an optimization: a converged-or-diverged
// scenario's block would otherwise be free to go singular and fail the whole
// batch's factorization.
// ---------------------------------------------------------------------------
__global__ void k_assemble_jacobian(
    const unsigned* __restrict__ kinds,
    const unsigned* __restrict__ bus_i,
    const unsigned* __restrict__ bus_k,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
    const double* __restrict__ identity_value,
    const unsigned* __restrict__ scatter,
    const unsigned* __restrict__ active,
    const double* __restrict__ vm,
    const double* __restrict__ va,
    const double* __restrict__ p_calc,
    const double* __restrict__ q_calc,
    double* __restrict__ values,
    unsigned nnz,
    unsigned n_buses,
    size_t total)
{
    for (size_t pos = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         pos < total;
         pos += (size_t)blockDim.x * gridDim.x) {

        size_t scenario = pos / nnz;
        unsigned e = (unsigned)(pos - scenario * nnz);

        double out;

        if (active[scenario] == 0u) {
            out = identity_value[e];
        } else {
            unsigned kind = kinds[e];
            size_t base = scenario * n_buses;
            size_t i = base + bus_i[e];
            size_t k = base + bus_k[e];

            double g = y_re[e];
            double b = y_im[e];
            double vm_i = vm[i];

            if (kind == K_HII) {
                out = -q_calc[i] - vm_i * vm_i * b;
            } else if (kind == K_NII) {
                out = p_calc[i] / vm_i + vm_i * g;
            } else if (kind == K_MII) {
                out = p_calc[i] - vm_i * vm_i * g;
            } else if (kind == K_LII) {
                out = q_calc[i] / vm_i - vm_i * b;
            } else {
                double vm_k = vm[k];
                double ang = va[i] - va[k];
                double sn, cs;
                sincos(ang, &sn, &cs);

                if (kind == K_HIK) {
                    out = vm_i * vm_k * (g * sn - b * cs);
                } else if (kind == K_NIK) {
                    out = vm_i * (g * cs + b * sn);
                } else if (kind == K_MIK) {
                    out = -vm_i * vm_k * (g * cs + b * sn);
                } else {
                    out = vm_i * (g * sn - b * cs);
                }
            }
        }

        values[scenario * nnz + scatter[e]] = out;
    }
}

extern "C" int go_assemble_jacobian(
    const unsigned* kinds, const unsigned* bus_i, const unsigned* bus_k,
    const double* y_re, const double* y_im, const double* identity_value,
    const unsigned* scatter, const unsigned* active,
    const double* vm, const double* va, const double* p_calc, const double* q_calc,
    double* values, unsigned nnz, unsigned n_buses, unsigned n_scenarios,
    void* stream)
{
    size_t total = (size_t)nnz * (size_t)n_scenarios;
    if (total == 0) return 0;
    k_assemble_jacobian<<<grid_for(total, THREADS), THREADS, 0, (cudaStream_t)stream>>>(
        kinds, bus_i, bus_k, y_re, y_im, identity_value, scatter, active,
        vm, va, p_calc, q_calc, values, nnz, n_buses, total);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Power injections — transliteration of `network::power_injections`.
//
//   I = Y·V,  S = V ⊙ conj(I)
//
// One thread per (scenario, bus). The Y-bus CSR is *shared* by every scenario
// (block-diagonal embedding's whole premise is one topology), so only V is
// strided; see `device_layout::YbusCsr`, whose `power_injections_reference`
// is this function in Rust and is unit-tested against `power_injections`.
//
// Summation order differs from `faer`'s column-major SpMV, so agreement with
// the CPU is to rounding (~1e-14 relative), not bit-for-bit. That is expected
// and is what `tests/gpu_assembly_test.rs` asserts.
// ---------------------------------------------------------------------------
__global__ void k_power_injections(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
    const double* __restrict__ vm,
    const double* __restrict__ va,
    double* __restrict__ p_calc,
    double* __restrict__ q_calc,
    unsigned n_buses,
    size_t total)
{
    for (size_t pos = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         pos < total;
         pos += (size_t)blockDim.x * gridDim.x) {

        size_t scenario = pos / n_buses;
        unsigned k = (unsigned)(pos - scenario * n_buses);
        size_t base = scenario * n_buses;

        double i_re = 0.0;
        double i_im = 0.0;
        int start = row_ptr[k];
        int end = row_ptr[k + 1];
        for (int slot = start; slot < end; ++slot) {
            unsigned j = (unsigned)col_idx[slot];
            double sn, cs;
            sincos(va[base + j], &sn, &cs);
            double v_re = vm[base + j] * cs;
            double v_im = vm[base + j] * sn;
            i_re += y_re[slot] * v_re - y_im[slot] * v_im;
            i_im += y_re[slot] * v_im + y_im[slot] * v_re;
        }

        double sn, cs;
        sincos(va[base + k], &sn, &cs);
        double v_re = vm[base + k] * cs;
        double v_im = vm[base + k] * sn;

        // S = V * conj(I)
        p_calc[base + k] = v_re * i_re + v_im * i_im;
        q_calc[base + k] = v_im * i_re - v_re * i_im;
    }
}

extern "C" int go_power_injections(
    const int* row_ptr, const int* col_idx, const double* y_re, const double* y_im,
    const double* vm, const double* va, double* p_calc, double* q_calc,
    unsigned n_buses, unsigned n_scenarios, void* stream)
{
    size_t total = (size_t)n_buses * (size_t)n_scenarios;
    if (total == 0) return 0;
    k_power_injections<<<grid_for(total, THREADS), THREADS, 0, (cudaStream_t)stream>>>(
        row_ptr, col_idx, y_re, y_im, vm, va, p_calc, q_calc, n_buses, total);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Mismatch + per-scenario convergence norm — transliteration of `bde.rs`'s
// per-scenario right-hand-side loop plus `network::effective_injection`.
//
// One CUDA *block* per scenario, so the max-|mismatch| reduction that decides
// convergence stays entirely inside one block and needs no second pass. The
// host reads back only `max_mis` (one f64 per scenario) — the single
// per-iteration device-to-host transfer in the whole Newton loop.
//
// ZIP loads are evaluated from `device_layout::ZipCoeffs`' three per-bus
// coefficients rather than a per-bus term list:
//   S_eff = S_spec + S_const + S_curr·|V| + S_imp·|V|²
// which that module unit-tests against `effective_injection` directly.
//
// Injections are computed for masked scenarios too, exactly as the CPU loop
// does — `go_zero_masked_rhs` (run after the host updates the mask) is what
// forces their Δx to zero.
// ---------------------------------------------------------------------------
__global__ void k_mismatch(
    const unsigned* __restrict__ non_slack,
    const unsigned* __restrict__ pq,
    const double* __restrict__ p_spec,
    const double* __restrict__ q_spec,
    const double* __restrict__ zip_p_const,
    const double* __restrict__ zip_q_const,
    const double* __restrict__ zip_p_curr,
    const double* __restrict__ zip_q_curr,
    const double* __restrict__ zip_p_imp,
    const double* __restrict__ zip_q_imp,
    const double* __restrict__ vm,
    const double* __restrict__ p_calc,
    const double* __restrict__ q_calc,
    double* __restrict__ rhs,
    double* __restrict__ max_mis,
    unsigned n_angle,
    unsigned n_pq,
    unsigned n_buses)
{
    unsigned s = blockIdx.x;
    unsigned blk = n_angle + n_pq;
    size_t bus_base = (size_t)s * n_buses;
    size_t unk_base = (size_t)s * blk;

    double local = 0.0;

    for (unsigned r = threadIdx.x; r < blk; r += blockDim.x) {
        double v;
        if (r < n_angle) {
            unsigned i = non_slack[r];
            double vmi = vm[bus_base + i];
            double p_eff = p_spec[bus_base + i] + zip_p_const[i]
                         + zip_p_curr[i] * vmi + zip_p_imp[i] * vmi * vmi;
            v = p_eff - p_calc[bus_base + i];
        } else {
            unsigned i = pq[r - n_angle];
            double vmi = vm[bus_base + i];
            double q_eff = q_spec[bus_base + i] + zip_q_const[i]
                         + zip_q_curr[i] * vmi + zip_q_imp[i] * vmi * vmi;
            v = q_eff - q_calc[bus_base + i];
        }
        rhs[unk_base + r] = v;
        local = fmax(local, fabs(v));
    }

    // Warp reduction, then one value per warp through shared memory. 32 is
    // written literally rather than via `warpSize`: `__shfl_down_sync`'s mask
    // already hard-codes a 32-lane warp, so deriving the loop bound from the
    // runtime variable would only make the two disagree if they ever diverged.
    for (int offset = 16; offset > 0; offset >>= 1) {
        local = fmax(local, __shfl_down_sync(0xffffffffu, local, offset));
    }

    const unsigned kWarps = THREADS / 32;
    __shared__ double warp_max[kWarps];
    unsigned lane = threadIdx.x % 32u;
    unsigned warp = threadIdx.x / 32u;
    if (lane == 0) warp_max[warp] = local;
    __syncthreads();

    if (threadIdx.x == 0) {
        double m = warp_max[0];
        for (unsigned w = 1; w < kWarps; ++w) m = fmax(m, warp_max[w]);
        max_mis[s] = m;
    }
}

extern "C" int go_mismatch(
    const unsigned* non_slack, const unsigned* pq,
    const double* p_spec, const double* q_spec,
    const double* zip_p_const, const double* zip_q_const,
    const double* zip_p_curr, const double* zip_q_curr,
    const double* zip_p_imp, const double* zip_q_imp,
    const double* vm, const double* p_calc, const double* q_calc,
    double* rhs, double* max_mis,
    unsigned n_angle, unsigned n_pq, unsigned n_buses, unsigned n_scenarios,
    void* stream)
{
    if (n_scenarios == 0) return 0;
    k_mismatch<<<n_scenarios, THREADS, 0, (cudaStream_t)stream>>>(
        non_slack, pq, p_spec, q_spec,
        zip_p_const, zip_q_const, zip_p_curr, zip_q_curr, zip_p_imp, zip_q_imp,
        vm, p_calc, q_calc, rhs, max_mis, n_angle, n_pq, n_buses);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Zero a masked scenario's right-hand side — `bde.rs`'s
// `rhs[base..base + blk] = 0` on convergence.
//
// Paired with the identity block `k_assemble_jacobian` writes for the same
// scenario, this makes Δx exactly zero: the scenario stops moving while the
// stacked pattern its cached symbolic factorization was built for stays
// untouched. Run *after* the host has updated `active` from `max_mis`, so it
// catches scenarios that converged this very iteration.
// ---------------------------------------------------------------------------
__global__ void k_zero_masked_rhs(
    const unsigned* __restrict__ active,
    double* __restrict__ rhs,
    unsigned blk,
    size_t total)
{
    for (size_t pos = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         pos < total;
         pos += (size_t)blockDim.x * gridDim.x) {
        size_t scenario = pos / blk;
        if (active[scenario] == 0u) rhs[pos] = 0.0;
    }
}

extern "C" int go_zero_masked_rhs(
    const unsigned* active, double* rhs, unsigned blk, unsigned n_scenarios, void* stream)
{
    size_t total = (size_t)blk * (size_t)n_scenarios;
    if (total == 0) return 0;
    k_zero_masked_rhs<<<grid_for(total, THREADS), THREADS, 0, (cudaStream_t)stream>>>(
        active, rhs, blk, total);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Apply the Newton step — `bde.rs`'s `voltage_ang += dx[...]` /
// `voltage_mag += dx[...]` loop, on-device so the batch's voltages never
// round-trip to the host between iterations.
//
// Masked scenarios are skipped rather than relying on their Δx being zero.
// It should be zero (identity block, zeroed rhs), but skipping makes that a
// belt-and-braces property instead of a load-bearing one.
// ---------------------------------------------------------------------------
__global__ void k_apply_update(
    const unsigned* __restrict__ non_slack,
    const unsigned* __restrict__ pq,
    const unsigned* __restrict__ active,
    const double* __restrict__ dx,
    double* __restrict__ vm,
    double* __restrict__ va,
    unsigned n_angle,
    unsigned n_pq,
    unsigned n_buses,
    size_t total)
{
    unsigned blk = n_angle + n_pq;
    for (size_t pos = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         pos < total;
         pos += (size_t)blockDim.x * gridDim.x) {

        size_t scenario = pos / blk;
        if (active[scenario] == 0u) continue;

        unsigned r = (unsigned)(pos - scenario * blk);
        size_t bus_base = scenario * n_buses;
        if (r < n_angle) {
            va[bus_base + non_slack[r]] += dx[pos];
        } else {
            vm[bus_base + pq[r - n_angle]] += dx[pos];
        }
    }
}

extern "C" int go_apply_update(
    const unsigned* non_slack, const unsigned* pq, const unsigned* active,
    const double* dx, double* vm, double* va,
    unsigned n_angle, unsigned n_pq, unsigned n_buses, unsigned n_scenarios,
    void* stream)
{
    size_t total = (size_t)(n_angle + n_pq) * (size_t)n_scenarios;
    if (total == 0) return 0;
    k_apply_update<<<grid_for(total, THREADS), THREADS, 0, (cudaStream_t)stream>>>(
        non_slack, pq, active, dx, vm, va, n_angle, n_pq, n_buses, total);
    return (int)cudaGetLastError();
}

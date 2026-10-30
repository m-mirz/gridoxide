// The Newton step's device kernels; see kernels.h for the layout and the
// launchers' contract.
//
// Each kernel transliterates the CPU code named in its comment, so the device
// takes the same steps the CPU stand-in does. They agree to rounding, not to
// the bit: the device's sincos and its fused multiply-adds round differently.
//
// One thread per (scenario, item), over a grid-stride loop, except the
// mismatch, which reduces each scenario's norm in one block.

#include <cuda_runtime.h>
#include <math.h>

#include "kernels.h"

namespace {

constexpr int kThreads = 256;

int grid_for(size_t total) {
    // Capped; the grid-stride loops cover the rest.
    size_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks > 65535u * 16u) blocks = 65535u * 16u;
    return blocks == 0 ? 1 : (int)blocks;
}

int launched() { return (int)cudaGetLastError(); }

// The largest |v| seen, NaN once any is NaN: a NaN must flag its scenario as
// diverged, however many finite entries follow it.
__device__ double max_abs(double m, double v) {
    return (isnan(m) || isnan(v)) ? NAN : fmax(m, fabs(v));
}

// gridoxide-core's `BusVoltages::update`.
__global__ void k_voltages(const double* __restrict__ va, double* __restrict__ sin_va,
                           double* __restrict__ cos_va, unsigned n_bus, size_t total,
                           const int* __restrict__ active) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)blockDim.x * gridDim.x) {
        if (!active[t / n_bus]) continue;
        sincos(va[t], &sin_va[t], &cos_va[t]);
    }
}

// gridoxide-core's `power_injections_from`: I = Y V, then S = V conj(I).
__global__ void k_injections(const unsigned* __restrict__ row_ptr,
                             const unsigned* __restrict__ col, const double* __restrict__ y,
                             const double* __restrict__ vm, const double* __restrict__ sin_va,
                             const double* __restrict__ cos_va, double* __restrict__ p,
                             double* __restrict__ q, unsigned n_bus, size_t total,
                             const int* __restrict__ active) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)blockDim.x * gridDim.x) {
        size_t s = t / n_bus;
        if (!active[s]) continue;
        unsigned i = (unsigned)(t - s * n_bus);
        size_t base = s * n_bus;
        double i_re = 0.0, i_im = 0.0;
        for (unsigned e = row_ptr[i]; e < row_ptr[i + 1]; ++e) {
            size_t k = base + col[e];
            double v_re = vm[k] * cos_va[k], v_im = vm[k] * sin_va[k];
            double g = y[2 * e], b = y[2 * e + 1];
            i_re += g * v_re - b * v_im;
            i_im += g * v_im + b * v_re;
        }
        double v_re = vm[t] * cos_va[t], v_im = vm[t] * sin_va[t];
        // V conj(I)
        p[t] = v_re * i_re + v_im * i_im;
        q[t] = v_im * i_re - v_re * i_im;
    }
}

// The mismatch of gridoxide-gpu's `newton::run`, with gridoxide-core's
// `effective_injection`: one block per scenario, its threads over the
// unknowns, then a reduction of |F| in shared memory.
__global__ void k_mismatch(const unsigned* __restrict__ non_slack, const unsigned* __restrict__ pq,
                           unsigned n_ns, unsigned n_pq, const double* __restrict__ p_spec,
                           const double* __restrict__ q_spec, const double* __restrict__ zip,
                           const double* __restrict__ vm, const double* __restrict__ p,
                           const double* __restrict__ q, double* __restrict__ rhs,
                           double* __restrict__ norm, unsigned n_bus, unsigned batch,
                           const int* __restrict__ active) {
    __shared__ double partial[kThreads];
    unsigned n = n_ns + n_pq;
    for (unsigned s = blockIdx.x; s < batch; s += gridDim.x) {
        if (!active[s]) continue;  // uniform across the block
        size_t base = (size_t)s * n_bus;
        double m = 0.0;
        for (unsigned r = threadIdx.x; r < n; r += blockDim.x) {
            bool angle = r < n_ns;
            unsigned i = angle ? non_slack[r] : pq[r - n_ns];
            size_t t = base + i;
            const double* z = zip + 6 * (size_t)i;
            double v = vm[t];
            double f = angle ? p_spec[t] + z[0] + z[2] * v * v + z[4] * v - p[t]
                             : q_spec[t] + z[1] + z[3] * v * v + z[5] * v - q[t];
            rhs[(size_t)s * n + r] = f;
            m = max_abs(m, f);
        }
        partial[threadIdx.x] = m;
        __syncthreads();
        for (unsigned half = blockDim.x / 2; half > 0; half /= 2) {
            if (threadIdx.x < half) {
                double a = partial[threadIdx.x], b = partial[threadIdx.x + half];
                partial[threadIdx.x] = (isnan(a) || isnan(b)) ? NAN : fmax(a, b);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) norm[s] = partial[0];
        __syncthreads();
    }
}

// gridoxide-core's `JacobianPattern::fill_from`, first pass, for one
// diagonal owner (bus i, admittance g + jb) and one off-diagonal pair.
__device__ void diag_slots(double* out, size_t i, double g, double b, const double* vm,
                           const double* p, const double* q) {
    double vm_i = vm[i];
    out[0] = -q[i] - vm_i * vm_i * b;  // H_ii
    out[1] = p[i] / vm_i + vm_i * g;   // N_ii
    out[2] = p[i] - vm_i * vm_i * g;   // M_ii
    out[3] = q[i] / vm_i - vm_i * b;   // L_ii
}

__device__ void pair_slots(double* out, size_t i, size_t k, double g, double b, const double* vm,
                           const double* sin_va, const double* cos_va) {
    double vm_i = vm[i], vm_k = vm[k];
    double sn = sin_va[i] * cos_va[k] - cos_va[i] * sin_va[k];
    double cs = cos_va[i] * cos_va[k] + sin_va[i] * sin_va[k];
    out[0] = vm_i * vm_k * (g * sn - b * cs);   // H_ik
    out[1] = vm_i * (g * cs + b * sn);          // N_ik
    out[2] = -vm_i * vm_k * (g * cs + b * sn);  // M_ik
    out[3] = vm_i * (g * sn - b * cs);          // L_ik
}

__global__ void k_jacobian_slots(const unsigned* __restrict__ diag_bus,
                                 const double* __restrict__ diag_y, unsigned n_diag,
                                 const unsigned* __restrict__ pair_bus,
                                 const double* __restrict__ pair_y, unsigned n_pairs,
                                 const double* __restrict__ vm, const double* __restrict__ sin_va,
                                 const double* __restrict__ cos_va, const double* __restrict__ p,
                                 const double* __restrict__ q, double* __restrict__ slots,
                                 unsigned n_bus, size_t total, const int* __restrict__ active) {
    unsigned owners = n_diag + n_pairs;
    size_t n_slots = 4 * (size_t)owners;
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)blockDim.x * gridDim.x) {
        size_t s = t / owners;
        if (!active[s]) continue;
        unsigned o = (unsigned)(t - s * owners);
        size_t base = s * n_bus;
        double* out = slots + s * n_slots + 4 * (size_t)o;
        if (o < n_diag) {
            diag_slots(out, base + diag_bus[o], diag_y[2 * o], diag_y[2 * o + 1], vm, p, q);
        } else {
            unsigned pr = o - n_diag;
            pair_slots(out, base + pair_bus[2 * pr], base + pair_bus[2 * pr + 1], pair_y[2 * pr],
                       pair_y[2 * pr + 1], vm, sin_va, cos_va);
        }
    }
}

// N-1: one (scenario, row)'s injection again, the row summed in column
// order with the scenario's outaged values in place of the shared ones, as
// the CPU sums the Y-bus it has changed in place.
__global__ void k_injections_delta(const unsigned* __restrict__ row_ptr,
                                   const unsigned* __restrict__ col, const double* __restrict__ y,
                                   const unsigned* __restrict__ ovr_ptr,
                                   const unsigned* __restrict__ ovr_pos,
                                   const double* __restrict__ ovr_y,
                                   const unsigned* __restrict__ row_scn,
                                   const unsigned* __restrict__ row_bus, unsigned n_rows,
                                   const double* __restrict__ vm,
                                   const double* __restrict__ sin_va,
                                   const double* __restrict__ cos_va, double* __restrict__ p,
                                   double* __restrict__ q, unsigned n_bus,
                                   const int* __restrict__ active) {
    for (unsigned t = blockIdx.x * blockDim.x + threadIdx.x; t < n_rows;
         t += blockDim.x * gridDim.x) {
        unsigned s = row_scn[t];
        if (!active[s]) continue;
        unsigned i = row_bus[t];
        size_t base = (size_t)s * n_bus;
        unsigned o = ovr_ptr[s], o_end = ovr_ptr[s + 1];
        while (o < o_end && ovr_pos[o] < row_ptr[i]) ++o;
        double i_re = 0.0, i_im = 0.0;
        for (unsigned e = row_ptr[i]; e < row_ptr[i + 1]; ++e) {
            double g = y[2 * e], b = y[2 * e + 1];
            if (o < o_end && ovr_pos[o] == e) {
                g = ovr_y[2 * o];
                b = ovr_y[2 * o + 1];
                ++o;
            }
            size_t k = base + col[e];
            double v_re = vm[k] * cos_va[k], v_im = vm[k] * sin_va[k];
            i_re += g * v_re - b * v_im;
            i_im += g * v_im + b * v_re;
        }
        size_t t_i = base + i;
        double v_re = vm[t_i] * cos_va[t_i], v_im = vm[t_i] * sin_va[t_i];
        p[t_i] = v_re * i_re + v_im * i_im;
        q[t_i] = v_im * i_re - v_re * i_im;
    }
}

// N-1: one (scenario, owner)'s four slots again, with its outaged admittance.
__global__ void k_jacobian_slots_delta(const unsigned* __restrict__ diag_bus, unsigned n_diag,
                                       const unsigned* __restrict__ pair_bus, unsigned n_pairs,
                                       const unsigned* __restrict__ slot_scn,
                                       const unsigned* __restrict__ slot_owner,
                                       const double* __restrict__ slot_y, unsigned n_items,
                                       const double* __restrict__ vm,
                                       const double* __restrict__ sin_va,
                                       const double* __restrict__ cos_va,
                                       const double* __restrict__ p, const double* __restrict__ q,
                                       double* __restrict__ slots, unsigned n_bus,
                                       const int* __restrict__ active) {
    size_t n_slots = 4 * ((size_t)n_diag + n_pairs);
    for (unsigned t = blockIdx.x * blockDim.x + threadIdx.x; t < n_items;
         t += blockDim.x * gridDim.x) {
        unsigned s = slot_scn[t];
        if (!active[s]) continue;
        unsigned o = slot_owner[t];
        size_t base = (size_t)s * n_bus;
        double g = slot_y[2 * t], b = slot_y[2 * t + 1];
        double* out = slots + s * n_slots + 4 * (size_t)o;
        if (o < n_diag) {
            diag_slots(out, base + diag_bus[o], g, b, vm, p, q);
        } else {
            unsigned pr = o - n_diag;
            pair_slots(out, base + pair_bus[2 * pr], base + pair_bus[2 * pr + 1], g, b, vm,
                       sin_va, cos_va);
        }
    }
}

// gridoxide-core's `JacobianPattern::fill_from`, second pass.
__global__ void k_jacobian_gather(const unsigned* __restrict__ src, const double* __restrict__ slots,
                                  double* __restrict__ values, unsigned nnz, unsigned n_slots,
                                  size_t total, const int* __restrict__ active) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)blockDim.x * gridDim.x) {
        size_t s = t / nnz;
        if (!active[s]) continue;
        unsigned e = (unsigned)(t - s * nnz);
        values[t] = slots[s * n_slots + src[e]];
    }
}

// The update of gridoxide-gpu's `newton::run`.
__global__ void k_update(const unsigned* __restrict__ non_slack, const unsigned* __restrict__ pq,
                         unsigned n_ns, unsigned n_pq, const double* __restrict__ dx,
                         double* __restrict__ vm, double* __restrict__ va, unsigned n_bus,
                         size_t total, const int* __restrict__ active) {
    unsigned n = n_ns + n_pq;
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)blockDim.x * gridDim.x) {
        size_t s = t / n;
        if (!active[s]) continue;
        unsigned r = (unsigned)(t - s * n);
        if (r < n_ns)
            va[s * n_bus + non_slack[r]] += dx[t];
        else
            vm[s * n_bus + pq[r - n_ns]] += dx[t];
    }
}

}  // namespace

extern "C" int go_voltages(const double* va, double* sin_va, double* cos_va, unsigned n_bus,
                           unsigned batch, const int* active, void* stream) {
    size_t total = (size_t)n_bus * batch;
    if (total == 0) return 0;
    k_voltages<<<grid_for(total), kThreads, 0, (cudaStream_t)stream>>>(va, sin_va, cos_va, n_bus,
                                                                         total, active);
    return launched();
}

extern "C" int go_injections(const unsigned* row_ptr, const unsigned* col, const double* y,
                             const double* vm, const double* sin_va, const double* cos_va,
                             double* p, double* q, unsigned n_bus, unsigned batch,
                             const int* active, void* stream) {
    size_t total = (size_t)n_bus * batch;
    if (total == 0) return 0;
    k_injections<<<grid_for(total), kThreads, 0, (cudaStream_t)stream>>>(
        row_ptr, col, y, vm, sin_va, cos_va, p, q, n_bus, total, active);
    return launched();
}

extern "C" int go_mismatch(const unsigned* non_slack, const unsigned* pq, unsigned n_ns,
                           unsigned n_pq, const double* p_spec, const double* q_spec,
                           const double* zip, const double* vm, const double* p, const double* q,
                           double* rhs, double* norm, unsigned n_bus, unsigned batch,
                           const int* active, void* stream) {
    if (batch == 0) return 0;
    unsigned blocks = batch < 65535u ? batch : 65535u;
    k_mismatch<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(non_slack, pq, n_ns, n_pq, p_spec,
                                                              q_spec, zip, vm, p, q, rhs, norm,
                                                              n_bus, batch, active);
    return launched();
}

extern "C" int go_jacobian_slots(const unsigned* diag_bus, const double* diag_y, unsigned n_diag,
                                 const unsigned* pair_bus, const double* pair_y, unsigned n_pairs,
                                 const double* vm, const double* sin_va, const double* cos_va,
                                 const double* p, const double* q, double* slots, unsigned n_bus,
                                 unsigned batch, const int* active, void* stream) {
    size_t total = ((size_t)n_diag + n_pairs) * batch;
    if (total == 0) return 0;
    k_jacobian_slots<<<grid_for(total), kThreads, 0, (cudaStream_t)stream>>>(
        diag_bus, diag_y, n_diag, pair_bus, pair_y, n_pairs, vm, sin_va, cos_va, p, q, slots,
        n_bus, total, active);
    return launched();
}

extern "C" int go_jacobian_gather(const unsigned* src, const double* slots, double* values,
                                  unsigned nnz, unsigned n_slots, unsigned batch,
                                  const int* active, void* stream) {
    size_t total = (size_t)nnz * batch;
    if (total == 0) return 0;
    k_jacobian_gather<<<grid_for(total), kThreads, 0, (cudaStream_t)stream>>>(
        src, slots, values, nnz, n_slots, total, active);
    return launched();
}

extern "C" int go_injections_delta(const unsigned* row_ptr, const unsigned* col, const double* y,
                                   const unsigned* ovr_ptr, const unsigned* ovr_pos,
                                   const double* ovr_y, const unsigned* row_scn,
                                   const unsigned* row_bus, unsigned n_rows, const double* vm,
                                   const double* sin_va, const double* cos_va, double* p,
                                   double* q, unsigned n_bus, const int* active, void* stream) {
    if (n_rows == 0) return 0;
    k_injections_delta<<<grid_for(n_rows), kThreads, 0, (cudaStream_t)stream>>>(
        row_ptr, col, y, ovr_ptr, ovr_pos, ovr_y, row_scn, row_bus, n_rows, vm, sin_va, cos_va,
        p, q, n_bus, active);
    return launched();
}

extern "C" int go_jacobian_slots_delta(const unsigned* diag_bus, unsigned n_diag,
                                       const unsigned* pair_bus, unsigned n_pairs,
                                       const unsigned* slot_scn, const unsigned* slot_owner,
                                       const double* slot_y, unsigned n_items, const double* vm,
                                       const double* sin_va, const double* cos_va,
                                       const double* p, const double* q, double* slots,
                                       unsigned n_bus, const int* active, void* stream) {
    if (n_items == 0) return 0;
    k_jacobian_slots_delta<<<grid_for(n_items), kThreads, 0, (cudaStream_t)stream>>>(
        diag_bus, n_diag, pair_bus, n_pairs, slot_scn, slot_owner, slot_y, n_items, vm, sin_va,
        cos_va, p, q, slots, n_bus, active);
    return launched();
}

extern "C" int go_update(const unsigned* non_slack, const unsigned* pq, unsigned n_ns,
                         unsigned n_pq, const double* dx, double* vm, double* va, unsigned n_bus,
                         unsigned batch, const int* active, void* stream) {
    size_t total = ((size_t)n_ns + n_pq) * batch;
    if (total == 0) return 0;
    k_update<<<grid_for(total), kThreads, 0, (cudaStream_t)stream>>>(non_slack, pq, n_ns, n_pq,
                                                                     dx, vm, va, n_bus, total,
                                                                     active);
    return launched();
}

// DC N-1 (src/dc.rs): gridoxide-core's `DcSensitivity::contingencies`, one
// column per outage. See kernels.h for the layout.
namespace {

// The angle response at a bus: theta's row, or 0 at a reference bus.
__device__ double response(const double* theta_col, const int* pos, unsigned bus) {
    int r = pos[bus];
    return r < 0 ? 0.0 : theta_col[r];
}

__global__ void k_dc_unit_rhs(const unsigned* __restrict__ out_br,
                              const unsigned* __restrict__ br_from,
                              const unsigned* __restrict__ br_to, const int* __restrict__ pos,
                              double* __restrict__ rhs, unsigned n, size_t total) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)blockDim.x * gridDim.x) {
        size_t col = t / n;
        int r = (int)(t % n);
        unsigned l = out_br[col];
        rhs[t] = (pos[br_from[l]] == r ? 1.0 : 0.0) - (pos[br_to[l]] == r ? 1.0 : 0.0);
    }
}

__global__ void k_dc_transfer(const double* __restrict__ theta, const unsigned* __restrict__ out_br,
                              const unsigned* __restrict__ br_from,
                              const unsigned* __restrict__ br_to, const double* __restrict__ br_b,
                              const int* __restrict__ pos, const double* __restrict__ base_flows,
                              double tol, double* __restrict__ c, int* __restrict__ solved,
                              unsigned n, unsigned k) {
    for (unsigned col = blockIdx.x * blockDim.x + threadIdx.x; col < k;
         col += blockDim.x * gridDim.x) {
        unsigned l = out_br[col];
        unsigned f = br_from[l], to = br_to[l];
        const double* th = theta + (size_t)col * n;
        double d = br_b[l] * (response(th, pos, f) - response(th, pos, to));
        bool ok = br_b[l] != 0.0 && (pos[f] >= 0 || pos[to] >= 0) && fabs(1.0 - d) >= tol;
        solved[col] = ok ? 1 : 0;
        c[col] = ok ? base_flows[l] / (1.0 - d) : 0.0;
    }
}

__global__ void k_dc_angles(const double* __restrict__ theta, const int* __restrict__ pos,
                            const double* __restrict__ base_angles, const double* __restrict__ c,
                            const int* __restrict__ solved, double* __restrict__ angles,
                            unsigned n, unsigned n_bus, size_t total) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)blockDim.x * gridDim.x) {
        size_t col = t / n_bus;
        unsigned bus = (unsigned)(t % n_bus);
        angles[t] = solved[col]
                        ? base_angles[bus] + c[col] * response(theta + col * n, pos, bus)
                        : NAN;
    }
}

__global__ void k_dc_flows(const double* __restrict__ theta, const unsigned* __restrict__ out_br,
                           const unsigned* __restrict__ br_from,
                           const unsigned* __restrict__ br_to, const double* __restrict__ br_b,
                           const int* __restrict__ pos, const double* __restrict__ base_flows,
                           const double* __restrict__ c, const int* __restrict__ solved,
                           double* __restrict__ flows, unsigned n, unsigned n_br, size_t total) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)blockDim.x * gridDim.x) {
        size_t col = t / n_br;
        unsigned j = (unsigned)(t % n_br);
        if (!solved[col]) {
            flows[t] = NAN;
        } else if (j == out_br[col]) {
            flows[t] = 0.0;
        } else {
            const double* th = theta + col * n;
            double psi = br_b[j] * (response(th, pos, br_from[j]) - response(th, pos, br_to[j]));
            flows[t] = base_flows[j] + c[col] * psi;
        }
    }
}

}  // namespace

extern "C" int go_dc_unit_rhs(const unsigned* out_br, const unsigned* br_from,
                              const unsigned* br_to, const int* pos, double* rhs, unsigned n,
                              unsigned k, void* stream) {
    size_t total = (size_t)n * k;
    if (total == 0) return 0;
    k_dc_unit_rhs<<<grid_for(total), kThreads, 0, (cudaStream_t)stream>>>(out_br, br_from, br_to,
                                                                         pos, rhs, n, total);
    return launched();
}

extern "C" int go_dc_transfer(const double* theta, const unsigned* out_br, const unsigned* br_from,
                              const unsigned* br_to, const double* br_b, const int* pos,
                              const double* base_flows, double tol, double* c, int* solved,
                              unsigned n, unsigned k, void* stream) {
    if (k == 0) return 0;
    k_dc_transfer<<<grid_for(k), kThreads, 0, (cudaStream_t)stream>>>(
        theta, out_br, br_from, br_to, br_b, pos, base_flows, tol, c, solved, n, k);
    return launched();
}

extern "C" int go_dc_angles(const double* theta, const int* pos, const double* base_angles,
                            const double* c, const int* solved, double* angles, unsigned n,
                            unsigned n_bus, unsigned k, void* stream) {
    size_t total = (size_t)n_bus * k;
    if (total == 0) return 0;
    k_dc_angles<<<grid_for(total), kThreads, 0, (cudaStream_t)stream>>>(
        theta, pos, base_angles, c, solved, angles, n, n_bus, total);
    return launched();
}

extern "C" int go_dc_flows(const double* theta, const unsigned* out_br, const unsigned* br_from,
                           const unsigned* br_to, const double* br_b, const int* pos,
                           const double* base_flows, const double* c, const int* solved,
                           double* flows, unsigned n, unsigned n_br, unsigned k, void* stream) {
    size_t total = (size_t)n_br * k;
    if (total == 0) return 0;
    k_dc_flows<<<grid_for(total), kThreads, 0, (cudaStream_t)stream>>>(
        theta, out_br, br_from, br_to, br_b, pos, base_flows, c, solved, flows, n, n_br, total);
    return launched();
}

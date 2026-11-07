// The interior point's vector work on the device; see ip.h.

#include <cuda_runtime.h>
#include <math.h>

#include "ip.h"

namespace {

constexpr int kThreads = 256;

int grid_for(size_t total) {
    size_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks > 65535) blocks = 65535;
    return (int)(blocks == 0 ? 1 : blocks);
}

int launched() { return (int)cudaGetLastError(); }

#define FOR_EACH(i, count)                                                          \
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < (count);           \
         i += blockDim.x * gridDim.x)

#define LAUNCH(kernel, count, stream, ...)                                          \
    do {                                                                            \
        if ((count) == 0) return 0;                                                 \
        kernel<<<grid_for(count), kThreads, 0, (cudaStream_t)(stream)>>>(__VA_ARGS__); \
        return launched();                                                          \
    } while (0)

__device__ bool has_lower(const go_ip& ip, unsigned k) { return isfinite(ip.lower[k]); }
__device__ bool has_upper(const go_ip& ip, unsigned k) { return isfinite(ip.upper[k]); }

// nlp::damping.
__device__ double damping(const go_ip& ip, unsigned k) {
    const double kappa_d = 1e-5;
    bool lo = has_lower(ip, k), hi = has_upper(ip, k);
    if (lo && !hi) return kappa_d;
    if (hi && !lo) return -kappa_d;
    return 0.0;
}

// Fixed in shape whatever the data, so a sum comes out the same every run.
template <int Kind>
__device__ double identity() {
    return Kind == GO_IP_MIN ? INFINITY : 0.0;
}

template <int Kind>
__device__ double term(double v) {
    switch (Kind) {
        case GO_IP_SUM_ABS:
        case GO_IP_MAX_ABS:
            return fabs(v);
        case GO_IP_NONFINITE:
            return isfinite(v) ? 0.0 : 1.0;
        default:
            return v;
    }
}

template <int Kind>
__device__ double combine(double a, double b) {
    switch (Kind) {
        case GO_IP_SUM:
        case GO_IP_SUM_ABS:
            return a + b;
        case GO_IP_MIN:
            return fmin(a, b);
        default:
            return fmax(a, b);
    }
}

template <int Kind>
__device__ double block_reduce(double v) {
    __shared__ double shared[kThreads];
    shared[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if ((int)threadIdx.x < s)
            shared[threadIdx.x] = combine<Kind>(shared[threadIdx.x], shared[threadIdx.x + s]);
        __syncthreads();
    }
    return shared[0];
}

template <int Kind>
__global__ void k_partial(const double* __restrict__ x, unsigned count,
                          double* __restrict__ partial) {
    double v = identity<Kind>();
    FOR_EACH(i, count) v = combine<Kind>(v, term<Kind>(x[i]));
    double total = block_reduce<Kind>(v);
    if (threadIdx.x == 0) partial[blockIdx.x] = total;
}

template <int Kind>
__global__ void k_final(const double* __restrict__ partial, double* __restrict__ out) {
    double v = identity<Kind>();
    for (unsigned i = threadIdx.x; i < GO_IP_PARTIALS; i += blockDim.x)
        v = combine<Kind>(v, partial[i]);
    double total = block_reduce<Kind>(v);
    if (threadIdx.x == 0) out[0] = total;
}

template <int Kind>
int reduce(const double* x, unsigned count, double* partial, double* out, cudaStream_t stream) {
    k_partial<Kind><<<GO_IP_PARTIALS, kThreads, 0, stream>>>(x, count, partial);
    k_final<Kind><<<1, kThreads, 0, stream>>>(partial, out);
    return launched();
}

__global__ void k_primal(go_ip ip, const double* __restrict__ u, double* __restrict__ r) {
    FOR_EACH(i, ip.m) {
        if (i < ip.rows) {
            int slack = ip.slack_of[i];
            r[i] = ip.c[i] * ip.row_scale[i] - (slack >= 0 ? u[slack] : ip.c_lower[i]);
        } else {
            r[i] = u[ip.pin_col[i - ip.rows]] - ip.pin_value[i - ip.rows];
        }
    }
}

__global__ void k_jacobian(go_ip ip) {
    FOR_EACH(e, ip.n_jac) ip.a_val[e] = ip.jac[e] * ip.row_scale[ip.a_row[e]];
}

__global__ void k_slacks(go_ip ip) {
    FOR_EACH(k, ip.n) {
        ip.g[k] = has_lower(ip, k) ? ip.u[k] - ip.lower[k] : 1.0;
        ip.t[k] = has_upper(ip, k) ? ip.upper[k] - ip.u[k] : 1.0;
    }
}

__global__ void k_dual(go_ip ip) {
    FOR_EACH(k, ip.n) {
        double aty = 0.0;
        for (unsigned p = ip.a_start[k]; p < ip.a_start[k + 1]; ++p) {
            unsigned e = ip.a_entries[p];
            aty += ip.a_val[e] * ip.y[ip.a_row[e]];
        }
        ip.r_d[k] = ip.grad[k] - aty - ip.z_l[k] + ip.z_u[k];
    }
}

__global__ void k_complementarity(go_ip ip, double target) {
    FOR_EACH(k, ip.n) {
        double lo = has_lower(ip, k) ? fabs(ip.g[k] * ip.z_l[k] - target) : 0.0;
        double hi = has_upper(ip, k) ? fabs(ip.t[k] * ip.z_u[k] - target) : 0.0;
        ip.w[k] = fmax(lo, hi);
    }
}

__global__ void k_sigma(go_ip ip) {
    FOR_EACH(k, ip.n) {
        ip.sigma[k] = (has_lower(ip, k) ? ip.z_l[k] / ip.g[k] : 0.0) +
                      (has_upper(ip, k) ? ip.z_u[k] / ip.t[k] : 0.0);
    }
}

__global__ void k_model_y(go_ip ip) {
    FOR_EACH(r, ip.rows) ip.y_model[r] = ip.y[r] * ip.row_scale[r] / ip.objective;
}

__global__ void k_diagonal(go_ip ip, double dw) {
    FOR_EACH(k, ip.n) ip.diag[k] = ip.sigma[k] + dw;
}

__global__ void k_identity(go_ip ip) {
    FOR_EACH(k, ip.n) ip.diag[k] = 1.0;
}

__global__ void k_bad_slacks(go_ip ip) {
    FOR_EACH(r, ip.rows) {
        int slack = ip.slack_of[r];
        ip.w[r] = 0.0;
        if (slack >= 0) {
            double d = ip.diag[slack];
            if (isnan(d) || d <= 0.0) ip.w[r] = 1.0;
        }
    }
}

__global__ void k_kkt(go_ip ip, int hessian, double delta_c) {
    double* h = ip.trip;
    double* d = h + ip.h_kept;
    double* a = d + ip.n_x;
    double* rows = a + ip.a_kept;
    FOR_EACH(i, ip.h_kept) h[i] = hessian ? ip.hess[ip.h_index[i]] * ip.objective : 0.0;
    FOR_EACH(k, ip.n_x) d[k] = ip.diag[k];
    FOR_EACH(i, ip.a_kept) a[i] = ip.a_val[ip.a_index[i]];
    FOR_EACH(r, ip.m) {
        int slack = r < ip.rows ? ip.slack_of[r] : -1;
        rows[r] = slack >= 0 ? -1.0 / ip.diag[slack] : -delta_c;
    }
}

__global__ void k_least_squares_rhs(go_ip ip, double* __restrict__ rhs) {
    FOR_EACH(i, ip.n + ip.m) {
        rhs[i] = i < ip.n ? ip.grad[i] - ip.z_l[i] + ip.z_u[i] : 0.0;
    }
}

__global__ void k_rhs(go_ip ip, double mu, double* __restrict__ rhs) {
    FOR_EACH(k, ip.n) {
        double rho = -ip.r_d[k] - mu * damping(ip, k);
        if (has_lower(ip, k)) rho -= (ip.g[k] * ip.z_l[k] - mu) / ip.g[k];
        if (has_upper(ip, k)) rho += (ip.t[k] * ip.z_u[k] - mu) / ip.t[k];
        rhs[k] = rho;
    }
    FOR_EACH(r, ip.m) rhs[ip.n + r] = -ip.r_p[r];
}

__global__ void k_correct(go_ip ip, double alpha, double* __restrict__ rhs) {
    FOR_EACH(r, ip.m) {
        ip.c_soc[r] = alpha * ip.c_soc[r] + ip.trial_r[r];
        rhs[ip.n + r] = -ip.c_soc[r];
    }
}

__global__ void k_condense(go_ip ip, const double* __restrict__ rhs,
                           double* __restrict__ reduced) {
    FOR_EACH(k, ip.n_x) reduced[k] = rhs[k];
    FOR_EACH(r, ip.m) {
        double v = rhs[ip.n + r];
        int slack = r < ip.rows ? ip.slack_of[r] : -1;
        if (slack >= 0) v += rhs[slack] / ip.diag[slack];
        reduced[ip.n_x + r] = v;
    }
}

__global__ void k_expand(go_ip ip, const double* __restrict__ rhs,
                         const double* __restrict__ solution, double* __restrict__ d) {
    FOR_EACH(k, ip.n_x) d[k] = solution[k];
    FOR_EACH(r, ip.m) d[ip.n + r] = solution[ip.n_x + r];
    FOR_EACH(s, ip.n - ip.n_x) {
        unsigned k = ip.n_x + s;
        d[k] = (rhs[k] + solution[ip.n_x + ip.slack_row[s]]) / ip.diag[k];
    }
}

__global__ void k_split(go_ip ip, const double* __restrict__ d, double* __restrict__ primal,
                        double* __restrict__ dual) {
    FOR_EACH(k, ip.n_x) {
        primal[k] = d[k];
        dual[k] = 0.0;
    }
    FOR_EACH(r, ip.m) {
        primal[ip.n_x + r] = 0.0;
        dual[ip.n_x + r] = d[ip.n + r];
    }
}

// Augmented::residual.
__global__ void k_residual(go_ip ip, const double* __restrict__ rhs,
                           const double* __restrict__ d, const double* __restrict__ primal,
                           const double* __restrict__ dual, double* __restrict__ r) {
    FOR_EACH(k, ip.n_x) r[k] = rhs[k] - (primal[k] + dual[k]);
    FOR_EACH(s, ip.n - ip.n_x) {
        unsigned k = ip.n_x + s;
        r[k] = rhs[k] - (ip.diag[k] * d[k] - d[ip.n + ip.slack_row[s]]);
    }
    FOR_EACH(i, ip.m) {
        double v = rhs[ip.n + i] - primal[ip.n_x + i];
        int slack = i < ip.rows ? ip.slack_of[i] : -1;
        if (slack >= 0) v += d[slack];
        r[ip.n + i] = v;
    }
}

__global__ void k_add(double* __restrict__ d, const double* __restrict__ correction,
                      unsigned count) {
    FOR_EACH(i, count) d[i] += correction[i];
}

// A column's terms of nlp::barrier_objective; infinite outside the bounds.
__global__ void k_barrier(go_ip ip, const double* __restrict__ u, double mu) {
    FOR_EACH(k, ip.n) {
        double value = mu * damping(ip, k) * u[k];
        bool outside = false;
        if (has_lower(ip, k)) {
            double s = u[k] - ip.lower[k];
            if (s <= 0.0) outside = true;
            value -= mu * log(s);
        }
        if (has_upper(ip, k)) {
            double s = ip.upper[k] - u[k];
            if (s <= 0.0) outside = true;
            value -= mu * log(s);
        }
        ip.w[k] = outside ? INFINITY : value;
    }
}

__global__ void k_slope(go_ip ip, const double* __restrict__ du, double mu) {
    FOR_EACH(k, ip.n) {
        double term = ip.grad[k] + mu * damping(ip, k);
        if (has_lower(ip, k)) term -= mu / ip.g[k];
        if (has_upper(ip, k)) term += mu / ip.t[k];
        ip.w[k] = term * du[k];
    }
}

__global__ void k_boundary(go_ip ip, const double* __restrict__ du) {
    FOR_EACH(k, ip.n) {
        double a = INFINITY;
        if (has_lower(ip, k) && du[k] < 0.0) a = fmin(a, -ip.g[k] / du[k]);
        if (has_upper(ip, k) && du[k] > 0.0) a = fmin(a, ip.t[k] / du[k]);
        ip.w[k] = a;
    }
}

__global__ void k_trial(go_ip ip, const double* __restrict__ du, double alpha) {
    FOR_EACH(k, ip.n) ip.trial_u[k] = ip.u[k] + alpha * du[k];
}

__global__ void k_multipliers(go_ip ip, const double* __restrict__ du, double mu, double tau) {
    FOR_EACH(k, ip.n) {
        double a = INFINITY;
        double dz_l = 0.0, dz_u = 0.0;
        if (has_lower(ip, k)) {
            dz_l = (mu - ip.g[k] * ip.z_l[k] - ip.z_l[k] * du[k]) / ip.g[k];
            if (dz_l < 0.0) a = fmin(a, -tau * ip.z_l[k] / dz_l);
        }
        if (has_upper(ip, k)) {
            dz_u = (mu - ip.t[k] * ip.z_u[k] + ip.z_u[k] * du[k]) / ip.t[k];
            if (dz_u < 0.0) a = fmin(a, -tau * ip.z_u[k] / dz_u);
        }
        ip.dz_l[k] = dz_l;
        ip.dz_u[k] = dz_u;
        ip.w[k] = a;
    }
}

__global__ void k_update(go_ip ip, const double* __restrict__ d, double alpha, double alpha_z) {
    FOR_EACH(k, ip.n) {
        ip.u[k] += alpha * d[k];
        ip.z_l[k] += alpha_z * ip.dz_l[k];
        ip.z_u[k] += alpha_z * ip.dz_u[k];
    }
    FOR_EACH(r, ip.m) ip.y[r] += alpha * (-d[ip.n + r]);
}

__device__ double clamped(double z, double center) {
    return fmin(fmax(z, center / 1e10), center * 1e10);
}

__global__ void k_band(go_ip ip, double mu) {
    FOR_EACH(k, ip.n) {
        if (has_lower(ip, k)) ip.z_l[k] = clamped(ip.z_l[k], mu / (ip.u[k] - ip.lower[k]));
        if (has_upper(ip, k)) ip.z_u[k] = clamped(ip.z_u[k], mu / (ip.upper[k] - ip.u[k]));
    }
}

__global__ void k_zero(double* __restrict__ x, unsigned count) {
    FOR_EACH(i, count) x[i] = 0.0;
}

__global__ void k_reset_bounds(go_ip ip) {
    FOR_EACH(k, ip.n) {
        ip.z_l[k] = has_lower(ip, k) ? 1.0 : 0.0;
        ip.z_u[k] = has_upper(ip, k) ? 1.0 : 0.0;
    }
}

}  // namespace

extern "C" {

int go_ip_reduce(int kind, const double* x, unsigned count, double* partial, double* out,
                 void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    switch (kind) {
        case GO_IP_SUM:
            return reduce<GO_IP_SUM>(x, count, partial, out, s);
        case GO_IP_SUM_ABS:
            return reduce<GO_IP_SUM_ABS>(x, count, partial, out, s);
        case GO_IP_MAX_ABS:
            return reduce<GO_IP_MAX_ABS>(x, count, partial, out, s);
        case GO_IP_MIN:
            return reduce<GO_IP_MIN>(x, count, partial, out, s);
        default:
            return reduce<GO_IP_NONFINITE>(x, count, partial, out, s);
    }
}

int go_ip_primal(const go_ip* ip, const double* u, double* r, void* s) {
    LAUNCH(k_primal, ip->m, s, *ip, u, r);
}
int go_ip_jacobian(const go_ip* ip, void* s) { LAUNCH(k_jacobian, ip->n_jac, s, *ip); }
int go_ip_slacks(const go_ip* ip, void* s) { LAUNCH(k_slacks, ip->n, s, *ip); }
int go_ip_dual(const go_ip* ip, void* s) { LAUNCH(k_dual, ip->n, s, *ip); }
int go_ip_complementarity(const go_ip* ip, double target, void* s) {
    LAUNCH(k_complementarity, ip->n, s, *ip, target);
}
int go_ip_sigma(const go_ip* ip, void* s) { LAUNCH(k_sigma, ip->n, s, *ip); }
int go_ip_model_y(const go_ip* ip, void* s) { LAUNCH(k_model_y, ip->rows, s, *ip); }
int go_ip_diagonal(const go_ip* ip, double dw, void* s) { LAUNCH(k_diagonal, ip->n, s, *ip, dw); }
int go_ip_identity(const go_ip* ip, void* s) { LAUNCH(k_identity, ip->n, s, *ip); }
int go_ip_bad_slacks(const go_ip* ip, void* s) { LAUNCH(k_bad_slacks, ip->rows, s, *ip); }
int go_ip_kkt(const go_ip* ip, int hessian, double delta_c, void* s) {
    size_t most = ip->h_kept > ip->a_kept ? ip->h_kept : ip->a_kept;
    if (ip->m > most) most = ip->m;
    if (ip->n_x > most) most = ip->n_x;
    LAUNCH(k_kkt, most, s, *ip, hessian, delta_c);
}
int go_ip_least_squares_rhs(const go_ip* ip, double* rhs, void* s) {
    LAUNCH(k_least_squares_rhs, ip->n + ip->m, s, *ip, rhs);
}
int go_ip_rhs(const go_ip* ip, double mu, double* rhs, void* s) {
    LAUNCH(k_rhs, ip->n + ip->m, s, *ip, mu, rhs);
}
int go_ip_correct(const go_ip* ip, double alpha, double* rhs, void* s) {
    LAUNCH(k_correct, ip->m, s, *ip, alpha, rhs);
}
int go_ip_condense(const go_ip* ip, const double* rhs, double* reduced, void* s) {
    LAUNCH(k_condense, ip->n + ip->m, s, *ip, rhs, reduced);
}
int go_ip_expand(const go_ip* ip, const double* rhs, const double* solution, double* d, void* s) {
    LAUNCH(k_expand, ip->n + ip->m, s, *ip, rhs, solution, d);
}
int go_ip_split(const go_ip* ip, const double* d, double* primal, double* dual, void* s) {
    LAUNCH(k_split, ip->n + ip->m, s, *ip, d, primal, dual);
}
int go_ip_residual(const go_ip* ip, const double* rhs, const double* d, const double* primal,
                   const double* dual, double* r, void* s) {
    LAUNCH(k_residual, ip->n + ip->m, s, *ip, rhs, d, primal, dual, r);
}
int go_ip_add(double* d, const double* correction, unsigned count, void* s) {
    LAUNCH(k_add, count, s, d, correction, count);
}
int go_ip_barrier(const go_ip* ip, const double* u, double mu, void* s) {
    LAUNCH(k_barrier, ip->n, s, *ip, u, mu);
}
int go_ip_slope(const go_ip* ip, const double* du, double mu, void* s) {
    LAUNCH(k_slope, ip->n, s, *ip, du, mu);
}
int go_ip_boundary(const go_ip* ip, const double* du, void* s) {
    LAUNCH(k_boundary, ip->n, s, *ip, du);
}
int go_ip_trial(const go_ip* ip, const double* du, double alpha, void* s) {
    LAUNCH(k_trial, ip->n, s, *ip, du, alpha);
}
int go_ip_multipliers(const go_ip* ip, const double* du, double mu, double tau, void* s) {
    LAUNCH(k_multipliers, ip->n, s, *ip, du, mu, tau);
}
int go_ip_update(const go_ip* ip, const double* d, double alpha, double alpha_z, void* s) {
    LAUNCH(k_update, ip->n + ip->m, s, *ip, d, alpha, alpha_z);
}
int go_ip_band(const go_ip* ip, double mu, void* s) { LAUNCH(k_band, ip->n, s, *ip, mu); }
int go_ip_zero(double* x, unsigned count, void* s) { LAUNCH(k_zero, count, s, x, count); }
int go_ip_reset_bounds(const go_ip* ip, void* s) { LAUNCH(k_reset_bounds, ip->n, s, *ip); }

}  // extern "C"

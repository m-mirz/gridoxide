// The AC-OPF model's evaluation on the device; see opf.h. Each function
// mirrors its namesake in src/opf/ac_kernels.rs, expression for expression,
// and this file is compiled without FMA contraction (build.rs), so every
// value that involves no sine or cosine rounds as the CPU's does.

#include <cuda_runtime.h>
#include <math.h>

#include "opf.h"

namespace {

constexpr int kThreads = 256;

int grid_for(size_t total) {
    size_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks > 65535) blocks = 65535;
    return (int)(blocks == 0 ? 1 : blocks);
}

int launched() { return (int)cudaGetLastError(); }

#define FOR_EACH(i, count)                                                        \
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (count); \
         i += (size_t)blockDim.x * gridDim.x)

enum { TH_NEAR = 0, TH_FAR = 1, V_NEAR = 2, V_FAR = 3 };

struct Terminal {
    double p, q;
    double dp[4], dq[4];
    double d2p[4][4], d2q[4][4];
};

// ac_kernels::terminal.
__device__ Terminal terminal(double ys_re, double ys_im, double ym_re, double ym_im, double v_n,
                             double v_f, double angle) {
    double sn, cs;
    sincos(angle, &sn, &cs);
    double c = ym_re * cs + ym_im * sn;
    double s = ym_re * sn - ym_im * cs;
    Terminal d;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) d.d2p[a][b] = d.d2q[a][b] = 0.0;
    d.p = v_n * v_n * ys_re + v_n * v_f * c;
    d.q = -v_n * v_n * ys_im + v_n * v_f * s;

    d.dp[TH_NEAR] = -v_n * v_f * s;
    d.dp[TH_FAR] = v_n * v_f * s;
    d.dp[V_NEAR] = 2.0 * v_n * ys_re + v_f * c;
    d.dp[V_FAR] = v_n * c;

    d.dq[TH_NEAR] = v_n * v_f * c;
    d.dq[TH_FAR] = -v_n * v_f * c;
    d.dq[V_NEAR] = -2.0 * v_n * ys_im + v_f * s;
    d.dq[V_FAR] = v_n * s;

    double pc = v_n * v_f * c;
    double ps = v_n * v_f * s;
    d.d2p[TH_NEAR][TH_NEAR] = -pc;
    d.d2p[TH_NEAR][TH_FAR] = pc;
    d.d2p[TH_FAR][TH_NEAR] = pc;
    d.d2p[TH_FAR][TH_FAR] = -pc;
    d.d2q[TH_NEAR][TH_NEAR] = -ps;
    d.d2q[TH_NEAR][TH_FAR] = ps;
    d.d2q[TH_FAR][TH_NEAR] = ps;
    d.d2q[TH_FAR][TH_FAR] = -ps;

    d.d2p[V_NEAR][V_NEAR] = 2.0 * ys_re;
    d.d2p[V_NEAR][V_FAR] = c;
    d.d2p[V_FAR][V_NEAR] = c;
    d.d2q[V_NEAR][V_NEAR] = -2.0 * ys_im;
    d.d2q[V_NEAR][V_FAR] = s;
    d.d2q[V_FAR][V_NEAR] = s;

    const int v_index[2] = {V_NEAR, V_FAR};
    const double scale[2] = {v_f, v_n};
    for (int k = 0; k < 2; ++k) {
        int v = v_index[k];
        double sc = scale[k];
        d.d2p[v][TH_NEAR] = -sc * s;
        d.d2p[TH_NEAR][v] = -sc * s;
        d.d2p[v][TH_FAR] = sc * s;
        d.d2p[TH_FAR][v] = sc * s;
        d.d2q[v][TH_NEAR] = sc * c;
        d.d2q[TH_NEAR][v] = sc * c;
        d.d2q[v][TH_FAR] = -sc * c;
        d.d2q[TH_FAR][v] = -sc * c;
    }
    return d;
}

// AcKernels::evaluate.
__device__ Terminal evaluate(const go_ac_model& m, unsigned t, const double* x) {
    unsigned i = m.near[t], j = m.far[t];
    return terminal(m.y_self[2 * t], m.y_self[2 * t + 1], m.y_mut[2 * t], m.y_mut[2 * t + 1],
                    x[m.vm + i], x[m.vm + j], x[m.theta + i] - x[m.theta + j]);
}

__device__ unsigned limit_row(const go_ac_model& m, unsigned r) { return 2 * m.n + r; }
__device__ unsigned angle_row(const go_ac_model& m, unsigned r) {
    return 2 * m.n + m.n_limit + r;
}
__device__ unsigned cost_row(const go_ac_model& m, unsigned r) {
    return 2 * m.n + m.n_limit + m.n_angle + r;
}

// ---- constraints (AcKernels::constraints) ----------------------------------

__global__ void k_flows(go_ac_model m, const double* __restrict__ x, double* __restrict__ flows) {
    FOR_EACH(t, m.n_term) {
        unsigned i = m.near[t], j = m.far[t];
        double v_n = x[m.vm + i], v_f = x[m.vm + j];
        double sn, cs;
        sincos(x[m.theta + i] - x[m.theta + j], &sn, &cs);
        double ys_re = m.y_self[2 * t], ys_im = m.y_self[2 * t + 1];
        double ym_re = m.y_mut[2 * t], ym_im = m.y_mut[2 * t + 1];
        double c = ym_re * cs + ym_im * sn;
        double s = ym_re * sn - ym_im * cs;
        flows[2 * t] = v_n * v_n * ys_re + v_n * v_f * c;
        flows[2 * t + 1] = -v_n * v_n * ys_im + v_n * v_f * s;
    }
}

__global__ void k_balance(go_ac_model m, const double* __restrict__ x,
                          const double* __restrict__ flows, double* __restrict__ out) {
    FOR_EACH(i, m.n) {
        double v2 = x[m.vm + i] * x[m.vm + i];
        double p = v2 * m.shunt[2 * i] + m.p_load[i];
        double q = -v2 * m.shunt[2 * i + 1] + m.q_load[i];
        for (unsigned k = m.term_ptr[i]; k < m.term_ptr[i + 1]; ++k) {
            unsigned t = m.term_idx[k];
            if (m.has_flow) {
                p += x[m.flow_col + 2 * t];
                q += x[m.flow_col + 2 * t + 1];
            } else {
                p += flows[2 * t];
                q += flows[2 * t + 1];
            }
        }
        for (unsigned k = m.gen_ptr[i]; k < m.gen_ptr[i + 1]; ++k) {
            unsigned g = m.gen_idx[k];
            p -= x[m.pg + g];
            q -= x[m.qg + g];
        }
        out[i] = p;
        out[m.n + i] = q;
    }
}

__global__ void k_rows(go_ac_model m, const double* __restrict__ x,
                       const double* __restrict__ flows, double* __restrict__ out) {
    size_t total = (size_t)m.n_limit + m.n_angle + m.n_cost + (m.has_flow ? m.n_term : 0);
    FOR_EACH(e, total) {
        unsigned r = (unsigned)e;
        if (r < m.n_limit) {
            unsigned t = m.limit_term[r];
            double p, q;
            if (m.has_flow) {
                p = x[m.flow_col + 2 * t];
                q = x[m.flow_col + 2 * t + 1];
            } else {
                p = flows[2 * t];
                q = flows[2 * t + 1];
            }
            out[limit_row(m, r)] = p * p + q * q;
            continue;
        }
        r -= m.n_limit;
        if (r < m.n_angle) {
            out[angle_row(m, r)] = x[m.theta + m.angle_from[r]] - x[m.theta + m.angle_to[r]];
            continue;
        }
        r -= m.n_angle;
        if (r < m.n_cost) {
            out[cost_row(m, r)] = x[m.cost_epi[r]] - m.cost_slope[r] * x[m.cost_col[r]];
            continue;
        }
        unsigned t = r - m.n_cost;
        out[m.flow_row + 2 * t] = x[m.flow_col + 2 * t] - flows[2 * t];
        out[m.flow_row + 2 * t + 1] = x[m.flow_col + 2 * t + 1] - flows[2 * t + 1];
    }
}

// ---- Jacobian slots (AcKernels::jacobian_slots) ------------------------------

__global__ void k_jac_terminals(go_ac_model m, const double* __restrict__ x,
                                double* __restrict__ slots) {
    FOR_EACH(t, m.n_term) {
        Terminal d = evaluate(m, (unsigned)t, x);
        if (m.has_flow) {
            double* o = slots + 12 * t;
            o[0] = 1.0;
            o[1] = 1.0;
            o[2] = 1.0;
            o[3] = 1.0;
            for (int k = 0; k < 4; ++k) {
                o[4 + k] = -d.dp[k];
                o[8 + k] = -d.dq[k];
            }
        } else {
            double* o = slots + 8 * t;
            for (int k = 0; k < 4; ++k) {
                o[k] = d.dp[k];
                o[4 + k] = d.dq[k];
            }
        }
    }
}

// Shunts, generators, limits, angles and costs, after the terminals' block.
__global__ void k_jac_rest(go_ac_model m, const double* __restrict__ x,
                           double* __restrict__ slots) {
    size_t total = (size_t)m.n + m.n_gen + m.n_limit + m.n_angle + m.n_cost;
    size_t base_shunt = (size_t)m.n_term * (m.has_flow ? 12 : 8);
    size_t base_gen = base_shunt + 2 * (size_t)m.n;
    size_t base_limit = base_gen + 2 * (size_t)m.n_gen;
    size_t limit_k = m.has_flow ? 2 : 4;
    size_t base_angle = base_limit + limit_k * m.n_limit;
    size_t base_cost = base_angle + 2 * (size_t)m.n_angle;
    FOR_EACH(e, total) {
        unsigned r = (unsigned)e;
        if (r < m.n) {
            double v = x[m.vm + r];
            slots[base_shunt + 2 * r] = 2.0 * v * m.shunt[2 * r];
            slots[base_shunt + 2 * r + 1] = -2.0 * v * m.shunt[2 * r + 1];
            continue;
        }
        r -= m.n;
        if (r < m.n_gen) {
            slots[base_gen + 2 * r] = -1.0;
            slots[base_gen + 2 * r + 1] = -1.0;
            continue;
        }
        r -= m.n_gen;
        if (r < m.n_limit) {
            unsigned t = m.limit_term[r];
            if (m.has_flow) {
                unsigned pc = m.flow_col + 2 * t, qc = pc + 1;
                slots[base_limit + 2 * r] = 2.0 * x[pc];
                slots[base_limit + 2 * r + 1] = 2.0 * x[qc];
            } else {
                Terminal d = evaluate(m, t, x);
                for (int k = 0; k < 4; ++k)
                    slots[base_limit + 4 * r + k] = 2.0 * (d.p * d.dp[k] + d.q * d.dq[k]);
            }
            continue;
        }
        r -= m.n_limit;
        if (r < m.n_angle) {
            slots[base_angle + 2 * r] = 1.0;
            slots[base_angle + 2 * r + 1] = -1.0;
            continue;
        }
        r -= m.n_angle;
        slots[base_cost + 2 * r] = 1.0;
        slots[base_cost + 2 * r + 1] = -m.cost_slope[r];
    }
}

// ---- Hessian slots (AcKernels::hessian_slots) --------------------------------

__global__ void k_hess_terminals(go_ac_model m, const double* __restrict__ x,
                                 const double* __restrict__ y, double* __restrict__ slots) {
    FOR_EACH(t, m.n_term) {
        Terminal d = evaluate(m, (unsigned)t, x);
        double wp, wq;
        if (m.has_flow) {
            wp = y[m.flow_row + 2 * t];
            wq = y[m.flow_row + 2 * t + 1];
        } else {
            wp = -y[m.near[t]];
            wq = -y[m.n + m.near[t]];
        }
        double* o = slots + m.n_quad + 16 * t;
        for (int k = 0; k < 16; ++k) {
            int a = k / 4, b = k % 4;
            o[k] = wp * d.d2p[a][b] + wq * d.d2q[a][b];
        }
    }
}

// Quadratic costs, shunts and limits, around the terminals' block.
__global__ void k_hess_rest(go_ac_model m, const double* __restrict__ x,
                            const double* __restrict__ y, double* __restrict__ slots) {
    size_t total = (size_t)m.n_quad + m.n + m.n_limit;
    size_t base_shunt = (size_t)m.n_quad + 16 * (size_t)m.n_term;
    size_t base_limit = base_shunt + m.n;
    FOR_EACH(e, total) {
        unsigned r = (unsigned)e;
        if (r < m.n_quad) {
            slots[r] = m.quad_h[r];
            continue;
        }
        r -= m.n_quad;
        if (r < m.n) {
            slots[base_shunt + r] =
                -y[r] * 2.0 * m.shunt[2 * r] + y[m.n + r] * 2.0 * m.shunt[2 * r + 1];
            continue;
        }
        r -= m.n;
        unsigned t = m.limit_term[r];
        if (m.has_flow) {
            double weight = -2.0 * y[limit_row(m, r)];
            slots[base_limit + 2 * r] = weight;
            slots[base_limit + 2 * r + 1] = weight;
        } else {
            Terminal d = evaluate(m, t, x);
            double weight = -y[limit_row(m, r)];
            for (int k = 0; k < 16; ++k) {
                int a = k / 4, b = k % 4;
                double value = 2.0 * (d.dp[a] * d.dp[b] + d.p * d.d2p[a][b] +
                                      d.dq[a] * d.dq[b] + d.q * d.d2q[a][b]);
                slots[base_limit + 16 * r + k] = weight * value;
            }
        }
    }
}

}  // namespace

extern "C" int go_ac_flows(const go_ac_model* m, const double* x, double* flows, void* stream) {
    if (m->n_term == 0) return 0;
    k_flows<<<grid_for(m->n_term), kThreads, 0, (cudaStream_t)stream>>>(*m, x, flows);
    return launched();
}

extern "C" int go_ac_constraints(const go_ac_model* m, const double* x, const double* flows,
                                 double* out, void* stream) {
    if (m->n > 0)
        k_balance<<<grid_for(m->n), kThreads, 0, (cudaStream_t)stream>>>(*m, x, flows, out);
    size_t rows = (size_t)m->n_limit + m->n_angle + m->n_cost + (m->has_flow ? m->n_term : 0);
    if (rows > 0)
        k_rows<<<grid_for(rows), kThreads, 0, (cudaStream_t)stream>>>(*m, x, flows, out);
    return launched();
}

extern "C" int go_ac_jacobian_slots(const go_ac_model* m, const double* x, double* slots,
                                    void* stream) {
    if (m->n_term > 0)
        k_jac_terminals<<<grid_for(m->n_term), kThreads, 0, (cudaStream_t)stream>>>(*m, x, slots);
    size_t rest = (size_t)m->n + m->n_gen + m->n_limit + m->n_angle + m->n_cost;
    if (rest > 0)
        k_jac_rest<<<grid_for(rest), kThreads, 0, (cudaStream_t)stream>>>(*m, x, slots);
    return launched();
}

extern "C" int go_ac_hessian_slots(const go_ac_model* m, const double* x, const double* y,
                                   double* slots, void* stream) {
    if (m->n_term > 0)
        k_hess_terminals<<<grid_for(m->n_term), kThreads, 0, (cudaStream_t)stream>>>(*m, x, y,
                                                                                    slots);
    size_t rest = (size_t)m->n_quad + m->n + m->n_limit;
    if (rest > 0)
        k_hess_rest<<<grid_for(rest), kThreads, 0, (cudaStream_t)stream>>>(*m, x, y, slots);
    return launched();
}

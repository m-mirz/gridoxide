/* The AC-OPF model's evaluation on the device: gridoxide's
 * `opf::ac_kernels` (src/opf/ac_kernels.rs) as CUDA kernels, one thread per
 * element, each writing its element's slot values in the CPU's slot order,
 * so the CPU's merge map sums them (go_kkt_gather, kernels.h).
 *
 * Layout, as the CPU's: columns [theta][|V|][P_g][Q_g][epigraph][flows],
 * rows [P balance][Q balance][limits][angles][costs][flow definitions].
 * Terminal t is branch t/2's from end (t even) or to end; with branch flows
 * (has_flow) its flow columns are flow_col + 2t and + 1, its definition rows
 * flow_row + 2t and + 1.
 *
 * Every pointer is device memory. Each launcher queues its kernels on
 * `stream` and returns cudaGetLastError() (0 is cudaSuccess).
 *
 * Plain C, without CUDA's headers, so bindgen reads it alone. */
#ifndef GRIDOXIDE_GPU_OPF_H
#define GRIDOXIDE_GPU_OPF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct go_ac_model {
    unsigned n, theta, vm, pg, qg;
    unsigned n_term, n_gen, n_quad, n_limit, n_angle, n_cost;
    int has_flow;
    unsigned flow_col, flow_row;
    /* Per terminal: its bus, the far bus, y_self and y_mut as re/im pairs. */
    const unsigned* near;
    const unsigned* far;
    const double* y_self;
    const double* y_mut;
    /* Per bus: the shunt as a re/im pair, the loads, and the terminals and
     * generators at the bus (CSR). */
    const double* shunt;
    const double* p_load;
    const double* q_load;
    const unsigned* term_ptr;
    const unsigned* term_idx;
    const unsigned* gen_ptr;
    const unsigned* gen_idx;
    /* Per generator. */
    const unsigned* gen_bus;
    /* Per quadratic cost term: its second derivative. */
    const double* quad_h;
    /* Per limit row, its terminal; per angle row, its buses; per cost row,
     * its epigraph column, generator column and slope. */
    const unsigned* limit_term;
    const unsigned* angle_from;
    const unsigned* angle_to;
    const unsigned* cost_epi;
    const unsigned* cost_col;
    const double* cost_slope;
} go_ac_model;

/* Every terminal's flow: flows[2t] = P, flows[2t + 1] = Q. */
int go_ac_flows(const go_ac_model* m, const double* x, double* flows, void* stream);

/* Every row's value into `out`, from the flows go_ac_flows wrote. */
int go_ac_constraints(const go_ac_model* m, const double* x, const double* flows, double* out,
                      void* stream);

/* The Jacobian's slot values, element block by element block: terminals
 * (8, or 12 with branch flows), shunts (2), generators (2), limits (4, or
 * 2), angles (2), costs (2). */
int go_ac_jacobian_slots(const go_ac_model* m, const double* x, double* slots, void* stream);

/* The Lagrangian Hessian's slot values: quadratic costs (1), terminals
 * (16), shunts (1), limits (16, or 2). */
int go_ac_hessian_slots(const go_ac_model* m, const double* x, const double* y, double* slots,
                        void* stream);

#ifdef __cplusplus
}
#endif

#endif

/* The Newton step's device kernels, one launcher each: what
 * `src/device.rs` calls, and what build.rs generates its bindings from, so
 * no signature is written twice by hand.
 *
 * Layout, shared by every launcher, for B scenarios of n_bus buses, n
 * unknowns, nnz Jacobian entries and n_slots fill slots:
 *
 *   per bus      scenario * n_bus + bus
 *   per unknown  scenario * n + r; r < n_ns is the angle of non_slack[r],
 *                r >= n_ns the magnitude of pq[r - n_ns]
 *   Jacobian     scenario * nnz + entry, entries in CSR order
 *   fill slots   scenario * n_slots + slot (JacobianPattern's first pass)
 *
 * `active` holds one int per scenario; a scenario with 0 is skipped and its
 * buffers are left as they were. Each launcher queues its kernel on
 * `stream` and returns cudaGetLastError() (0 is cudaSuccess).
 *
 * Plain C, without CUDA's headers, so bindgen reads it alone. */
#ifndef GRIDOXIDE_GPU_KERNELS_H
#define GRIDOXIDE_GPU_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/* sin and cos of every bus angle. */
int go_voltages(const double* va, double* sin_va, double* cos_va, unsigned n_bus,
                unsigned batch, const int* active, void* stream);

/* P + jQ = V conj(Y V) per bus, Y in CSR form (row_ptr, col, y as re/im
 * pairs), each row summed in column order as the CPU's mat-vec does. */
int go_injections(const unsigned* row_ptr, const unsigned* col, const double* y,
                  const double* vm, const double* sin_va, const double* cos_va, double* p,
                  double* q, unsigned n_bus, unsigned batch, const int* active, void* stream);

/* F = S_spec - S_calc over the unknowns, and ||F||_inf per scenario into
 * `norm` (NaN if any entry is NaN). S_spec is the bus's p_spec + j q_spec
 * plus its ZIP terms, summed per kind into `zip` (six per bus: constant
 * power re/im, constant impedance re/im, constant current re/im). */
int go_mismatch(const unsigned* non_slack, const unsigned* pq, unsigned n_ns, unsigned n_pq,
                const double* p_spec, const double* q_spec, const double* zip, const double* vm,
                const double* p, const double* q, double* rhs, double* norm, unsigned n_bus,
                unsigned batch, const int* active, void* stream);

/* The fill's first pass: H, N, M, L of every non-slack bus (diag_bus with
 * diag_y as re/im pairs) and of every off-diagonal pair (pair_bus as (i, k),
 * pair_y as re/im pairs), four slots each. */
int go_jacobian_slots(const unsigned* diag_bus, const double* diag_y, unsigned n_diag,
                      const unsigned* pair_bus, const double* pair_y, unsigned n_pairs,
                      const double* vm, const double* sin_va, const double* cos_va,
                      const double* p, const double* q, double* slots, unsigned n_bus,
                      unsigned batch, const int* active, void* stream);

/* The fill's second pass: values[entry] = slots[src[entry]]. */
int go_jacobian_gather(const unsigned* src, const double* slots, double* values, unsigned nnz,
                       unsigned n_slots, unsigned batch, const int* active, void* stream);

/* N-1: an outage changes a few Y-bus values of its scenario. The two
 * launchers below run after go_injections and go_jacobian_slots and redo
 * only what those values touch.
 *
 * Each scenario's changed values, `ovr_ptr[s]..ovr_ptr[s + 1]`, ascending by
 * their position in the shared Y-bus's CSR arrays (ovr_pos) with the
 * outaged value (ovr_y, re/im pairs). Each (scenario, row) to recompute is an
 * item: row_scn and row_bus. */
int go_injections_delta(const unsigned* row_ptr, const unsigned* col, const double* y,
                        const unsigned* ovr_ptr, const unsigned* ovr_pos, const double* ovr_y,
                        const unsigned* row_scn, const unsigned* row_bus, unsigned n_rows,
                        const double* vm, const double* sin_va, const double* cos_va, double* p,
                        double* q, unsigned n_bus, const int* active, void* stream);

/* Each (scenario, fill-slot owner) an outage changes, with the owner's
 * outaged admittance (slot_y, re/im pairs): owners below n_diag are
 * diag_bus's, the rest pair_bus's, as in go_jacobian_slots. */
int go_jacobian_slots_delta(const unsigned* diag_bus, unsigned n_diag, const unsigned* pair_bus,
                            unsigned n_pairs, const unsigned* slot_scn,
                            const unsigned* slot_owner, const double* slot_y, unsigned n_items,
                            const double* vm, const double* sin_va, const double* cos_va,
                            const double* p, const double* q, double* slots, unsigned n_bus,
                            const int* active, void* stream);

/* The Newton update: angles and magnitudes += dx. */
int go_update(const unsigned* non_slack, const unsigned* pq, unsigned n_ns, unsigned n_pq,
              const double* dx, double* vm, double* va, unsigned n_bus, unsigned batch,
              const int* active, void* stream);

/* DC N-1 (src/dc.rs), for k outages at a time on a network of n_bus buses,
 * n_br flat branches and n unknowns (the reduced B's rows: every non-reference
 * bus of a referenced island). `pos[bus]` is the bus's row, or -1 for a
 * reference bus or one outside every referenced island; per flat branch,
 * `br_from`, `br_to` its buses and `br_b` its susceptance (0 for a branch
 * that carries no DC flow). `out_br[col]` is column col's outaged branch.
 *
 * Layout: the unit-transfer right-hand sides and the angle responses
 * `theta` are column-major n x k (column col at col * n), the outages'
 * angles and flows outage-major (row col at col * n_bus, col * n_br). */

/* rhs[:, col] = e_from - e_to of the outaged branch, reference rows left out. */
int go_dc_unit_rhs(const unsigned* out_br, const unsigned* br_from, const unsigned* br_to,
                   const int* pos, double* rhs, unsigned n, unsigned k, void* stream);

/* Per column: d = b (theta[from] - theta[to]), the transfer
 * c = base_flows[l] / (1 - d), and solved[col] = 0 where there is no
 * post-outage state (no DC flow on the branch, no unknown at either end, or
 * |1 - d| < tol: the branch is radial). */
int go_dc_transfer(const double* theta, const unsigned* out_br, const unsigned* br_from,
                   const unsigned* br_to, const double* br_b, const int* pos,
                   const double* base_flows, double tol, double* c, int* solved, unsigned n,
                   unsigned k, void* stream);

/* angles[col, bus] = base_angles[bus] + c[col] theta[pos[bus], col]; NaN rows
 * where unsolved. */
int go_dc_angles(const double* theta, const int* pos, const double* base_angles,
                 const double* c, const int* solved, double* angles, unsigned n,
                 unsigned n_bus, unsigned k, void* stream);

/* flows[col, j] = base_flows[j] + c[col] b_j (theta[from_j] - theta[to_j]),
 * 0 at the outaged branch; NaN rows where unsolved. */
int go_dc_flows(const double* theta, const unsigned* out_br, const unsigned* br_from,
                const unsigned* br_to, const double* br_b, const int* pos,
                const double* base_flows, const double* c, const int* solved, double* flows,
                unsigned n, unsigned n_br, unsigned k, void* stream);

/* The interior point's KKT matrix (src/kkt.rs), as gridoxide-sparse's
 * LowerPattern assembles and equilibrates it on the CPU, to the same bits.
 *
 * Assembly: values[s] = sum of trip[src[t]] for t in slot_ptr[s] ..
 * slot_ptr[s+1], in triplet order, from 0.0. */
int go_kkt_gather(const unsigned* slot_ptr, const unsigned* src, const double* trip,
                  double* values, unsigned nnz, void* stream);

/* Ruiz, the maxima: row_max[i] = max |values[inc[k]]| over k in inc_ptr[i]
 * .. inc_ptr[i+1], the slots in row or column i. */
int go_kkt_row_max(const unsigned* inc_ptr, const unsigned* inc, const double* values,
                   double* row_max, unsigned n, void* stream);

/* Ruiz, the factors: factor[i] = 1/sqrt(row_max[i]) (1 where it is 0 or not
 * finite), scale[i] *= factor[i], or = factor[i] in the `first` pass. */
int go_kkt_factor(const double* row_max, double* factor, double* scale, int first, unsigned n,
                  void* stream);

/* Ruiz, the scaling: values[s] *= factor[row[s]] * factor[col[s]]; with
 * `bad` not null, *bad = 1 where the result is not finite. */
int go_kkt_scale(const unsigned* row, const unsigned* col, const double* factor,
                 double* values, int* bad, unsigned nnz, void* stream);

/* x[i] *= d[i]. */
int go_scale_vec(double* x, const double* d, unsigned n, void* stream);

/* y = A x, A symmetric with its upper triangle in CSR (up_ptr, up_col,
 * values; the diagonal first in each row, as LowerPattern's columns have it)
 * and its strictly lower part as a CSR of positions into `values` (lo_ptr,
 * lo_col, lo_src). One thread per row, no atomics: the same sums every
 * time. */
int go_kkt_symv(const int* up_ptr, const int* up_col, const unsigned* lo_ptr,
                const unsigned* lo_col, const unsigned* lo_src, const double* values,
                const double* x, double* y, unsigned n, void* stream);

/* An NVTX range on this thread, for Nsight Systems. NVTX is header-only and
 * nothing to link: without a tool attached these return at once. */
void go_nvtx_push(const char* name);
void go_nvtx_pop(void);

/* DC sweeps (src/dc.rs), for k injection scenarios at a time, on the same
 * network tables and layout as DC N-1 above. */

/* rhs = 0 (n x k), then rhs[rows[e], cols[e]] = vals[e] for each of the
 * `count` entries: each scenario's injection change at its unknowns, at most
 * one entry per row and column. */
int go_dc_injection_rhs(const unsigned* cols, const unsigned* rows, const double* vals,
                        unsigned count, double* rhs, unsigned n, unsigned k, void* stream);

/* angles[col, bus] = base_angles[bus] + theta[pos[bus], col]. */
int go_dc_sweep_angles(const double* theta, const int* pos, const double* base_angles,
                       double* angles, unsigned n, unsigned n_bus, unsigned k, void* stream);

/* flows[col, j] = base_flows[j] + b_j (theta[from_j, col] - theta[to_j, col]). */
int go_dc_sweep_flows(const double* theta, const unsigned* br_from, const unsigned* br_to,
                      const double* br_b, const int* pos, const double* base_flows,
                      double* flows, unsigned n, unsigned n_br, unsigned k, void* stream);

#ifdef __cplusplus
}
#endif

#endif

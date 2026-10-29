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

#ifdef __cplusplus
}
#endif

#endif

// opf::nlp's interior point on the device: its vector work, one launcher per
// step of `Engine` (src/opf/nlp.rs), behind src/ip.rs. Each kernel mirrors
// HostEngine's expression for expression, and this file is compiled without
// FMA contraction (build.rs), so an element rounds as the host's does; only
// sums differ, a reduction's order not being the host's.
#ifndef GRIDOXIDE_GPU_IP_H
#define GRIDOXIDE_GPU_IP_H

#ifdef __cplusplus
extern "C" {
#endif

// The problem and the iterate, every pointer a device buffer src/ip.rs owns
// (or, for `c`, `jac` and `hess`, the model's).
typedef struct go_ip {
    unsigned n_x;   // the problem's variables
    unsigned n;     // with the inequality slacks, columns n_x..n
    unsigned m;     // internal rows
    unsigned rows;  // the problem's rows; rows..m pin fixed variables
    unsigned n_jac; // the problem's Jacobian entries, first in `a_val`
    unsigned n_a;   // the internal Jacobian's entries
    unsigned h_kept, a_kept;
    double objective;  // the objective's scale

    const double* lower;
    const double* upper;
    const double* c_lower;    // per problem row
    const double* row_scale;  // per problem row
    const double* pin_value;  // per pinning row
    const unsigned* pin_col;  // per pinning row
    const int* slack_of;      // per problem row, -1 for none
    const unsigned* slack_row;  // per slack column
    const unsigned* a_row;      // per internal Jacobian entry
    const unsigned* a_start;    // per column, into a_entries
    const unsigned* a_entries;  // a column's entries, ascending
    const unsigned* h_index;    // the KKT matrix's Hessian entries
    const unsigned* a_index;    // the KKT matrix's Jacobian entries

    const double* c;     // the model's rows
    const double* jac;   // the model's Jacobian values
    const double* hess;  // the model's Hessian values
    double* trip;        // the KKT solver's triplet values

    double* u;
    double* y;
    double* z_l;
    double* z_u;
    double* g;
    double* t;
    double* grad;
    double* r_d;
    double* r_p;
    double* sigma;
    double* diag;
    double* a_val;
    double* y_model;
    double* trial_u;
    double* trial_r;
    double* c_soc;
    double* dz_l;
    double* dz_u;
    double* w;  // a reduction's terms
} go_ip;

// The reductions: into out[0], in a fixed order, from `partial` (GO_IP_PARTIALS
// values of scratch). A sum is 0, a maximum 0 and a minimum infinity when
// empty; the maximum and minimum skip NaN, as f64::max and f64::min do.
enum { GO_IP_PARTIALS = 128 };
enum {
    GO_IP_SUM = 0,
    GO_IP_SUM_ABS = 1,
    GO_IP_MAX_ABS = 2,
    GO_IP_MIN = 3,
    GO_IP_NONFINITE = 4  // 1 if any value is NaN or infinite, else 0
};
int go_ip_reduce(int kind, const double* x, unsigned count, double* partial, double* out,
                 void* stream);

// r = c(x) - s or the row's bound, or the pinned variable less its value.
int go_ip_primal(const go_ip* ip, const double* u, double* r, void* stream);
int go_ip_jacobian(const go_ip* ip, void* stream);
int go_ip_slacks(const go_ip* ip, void* stream);
int go_ip_dual(const go_ip* ip, void* stream);
int go_ip_complementarity(const go_ip* ip, double target, void* stream);
int go_ip_sigma(const go_ip* ip, void* stream);
int go_ip_model_y(const go_ip* ip, void* stream);
int go_ip_diagonal(const go_ip* ip, double dw, void* stream);
int go_ip_identity(const go_ip* ip, void* stream);
// w[r] = 1 for a slack whose diagonal is NaN or not positive.
int go_ip_bad_slacks(const go_ip* ip, void* stream);
// The KKT matrix's values, in the order of its pattern: the Hessian's
// (`hessian` false: zeros), the diagonal, the Jacobian's and the rows'.
int go_ip_kkt(const go_ip* ip, int hessian, double delta_c, void* stream);
int go_ip_least_squares_rhs(const go_ip* ip, double* rhs, void* stream);
int go_ip_rhs(const go_ip* ip, double mu, double* rhs, void* stream);
int go_ip_correct(const go_ip* ip, double alpha, double* rhs, void* stream);
int go_ip_condense(const go_ip* ip, const double* rhs, double* reduced, void* stream);
int go_ip_expand(const go_ip* ip, const double* rhs, const double* solution, double* d,
                 void* stream);
// [d_x, 0] and [0, w], the two products of the refinement's residual.
int go_ip_split(const go_ip* ip, const double* d, double* primal, double* dual, void* stream);
int go_ip_residual(const go_ip* ip, const double* rhs, const double* d, const double* primal,
                   const double* dual, double* r, void* stream);
int go_ip_add(double* d, const double* correction, unsigned count, void* stream);
int go_ip_barrier(const go_ip* ip, const double* u, double mu, void* stream);
int go_ip_slope(const go_ip* ip, const double* du, double mu, void* stream);
int go_ip_boundary(const go_ip* ip, const double* du, void* stream);
int go_ip_trial(const go_ip* ip, const double* du, double alpha, void* stream);
int go_ip_multipliers(const go_ip* ip, const double* du, double mu, double tau, void* stream);
// `d` is the full step, `w` its rows' part: y moves by -w.
int go_ip_update(const go_ip* ip, const double* d, double alpha, double alpha_z, void* stream);
int go_ip_band(const go_ip* ip, double mu, void* stream);
int go_ip_zero(double* x, unsigned count, void* stream);
int go_ip_reset_bounds(const go_ip* ip, void* stream);

#ifdef __cplusplus
}
#endif

#endif

//! IPOPT as a reference backend for [`nlp`](super::nlp).
//!
//! The nonlinear counterpart of what [`highs`](super::highs) is for
//! [`ipm`](super::ipm): a second, independently written solver to hold ours
//! against. Nothing depends on it — `opf::nlp` is the default and needs
//! nothing installed — and it is not built in CI.
//!
//! Reached through gridoxide's own bindgen-generated FFI against a system
//! install's `IpStdCInterface.h`, not through a wrapper crate. The same
//! approach [`highs`](super::highs) and the KLU/PARDISO backends take.
//!
//! # What the comparison is worth, and what it is not
//!
//! Weaker than the convex one, and the difference matters. On a convex problem
//! the optimum is unique, so any disagreement between two solvers is a bug in
//! one of them. AC-OPF is **nonconvex**: two correct solvers may converge to
//! genuinely different local optima, and neither is wrong. So a disagreement
//! here is a question, not a verdict — and agreement is correspondingly
//! stronger evidence, since two independent methods landing on the same point
//! from the same start is not something a shared bug would produce.
//!
//! What it does test unambiguously is the **model**. IPOPT consumes the same
//! [`NonlinearProblem`] — the same objective, Jacobian and Hessian — so if it
//! reaches a point ours also finds feasible and no better, the derivative
//! algebra and constraint set are corroborated by something outside this
//! crate.
//!
//! # Two conventions that must be translated
//!
//! Both are silent if got wrong, so both are converted explicitly here rather
//! than assumed to line up:
//!
//! - **Hessian sign and scaling.** [`NonlinearProblem::lagrangian_hessian`]
//!   returns \\(\nabla^2 f - \sum_i y_i \nabla^2 c_i\\); IPOPT asks for
//!   \\(\sigma \nabla^2 f + \sum_i \lambda_i \nabla^2 c_i\\). So the
//!   multipliers are negated on the way in, and when \\(\sigma \ne 1\\) — which
//!   happens in IPOPT's restoration phase, where it is 0 — the objective's own
//!   Hessian has to be separated out and rescaled.
//! - **Triangle.** Ours is full symmetric; IPOPT wants the lower triangle
//!   only, with each off-diagonal appearing once. Handing it both halves
//!   doubles every off-diagonal term, which converges to the wrong point
//!   rather than failing.

#![allow(non_upper_case_globals)]

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

use crate::opf::nlp::{NlpSolution, NonlinearProblem};
use crate::opf::{OpfError, OptStatus};

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::all
)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/ipopt_bindings.rs"));
}

/// Tuning passed straight through to IPOPT.
#[derive(Clone, Debug)]
pub struct IpoptOptions {
    pub tolerance: f64,
    pub max_iterations: i32,
    /// IPOPT's own `print_level`, 0–12. Zero keeps it silent, which is the
    /// only sane default for a library.
    pub print_level: i32,
    /// Extra `(name, value)` string options, applied after the above.
    pub string_options: Vec<(String, String)>,
}

impl Default for IpoptOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 3000,
            print_level: 0,
            string_options: Vec::new(),
        }
    }
}

/// Everything the C callbacks need, reached through IPOPT's `user_data`.
///
/// The sparsity patterns are captured once at the initial point and reused,
/// because IPOPT asks for structure and values in separate calls and requires
/// them to agree. That is a real constraint on the problem, not an
/// optimization: a `NonlinearProblem` whose Jacobian pattern changes with `x`
/// cannot be handed to IPOPT at all, and would silently write values into the
/// wrong entries if this pretended otherwise.
struct Context<'a> {
    problem: &'a dyn NonlinearProblem,
    jacobian_pattern: Vec<(usize, usize)>,
    /// Lower-triangle Hessian pattern.
    hessian_pattern: Vec<(usize, usize)>,
    /// Set when a callback saw a pattern it could not honour.
    pattern_mismatch: bool,
}

fn lower_triangle(triplets: &[(usize, usize, f64)]) -> Vec<(usize, usize)> {
    let mut seen: Vec<(usize, usize)> =
        triplets.iter().filter(|(r, c, _)| r >= c).map(|(r, c, _)| (*r, *c)).collect();
    seen.sort_unstable();
    seen.dedup();
    seen
}

/// `σ∇²f + Σ λᵢ∇²cᵢ`, folded into the caller's lower-triangle pattern.
///
/// The `σ ≠ 1` branch costs a second Hessian evaluation and is worth it:
/// IPOPT sets `σ = 0` in its restoration phase, and quietly using the
/// objective's curvature there would corrupt exactly the steps taken when the
/// solve is already in trouble.
fn scaled_hessian(
    problem: &dyn NonlinearProblem,
    x: &[f64],
    lambda: &[f64],
    sigma: f64,
    pattern: &[(usize, usize)],
) -> Vec<f64> {
    let negated: Vec<f64> = lambda.iter().map(|v| -v).collect();
    let mut dense: std::collections::HashMap<(usize, usize), f64> =
        std::collections::HashMap::new();
    for (r, c, v) in problem.lagrangian_hessian(x, &negated) {
        if r >= c {
            *dense.entry((r, c)).or_insert(0.0) += v;
        }
    }
    if sigma != 1.0 {
        // `lagrangian_hessian(x, 0)` is ∇²f alone, so `(σ − 1)∇²f` corrects
        // the implicit factor of one above.
        let zero = vec![0.0; lambda.len()];
        for (r, c, v) in problem.lagrangian_hessian(x, &zero) {
            if r >= c {
                *dense.entry((r, c)).or_insert(0.0) += (sigma - 1.0) * v;
            }
        }
    }
    pattern.iter().map(|key| dense.get(key).copied().unwrap_or(0.0)).collect()
}

unsafe extern "C" fn eval_f(
    n: bindings::ipindex,
    x: *mut bindings::ipnumber,
    _new_x: bool,
    obj_value: *mut bindings::ipnumber,
    user_data: bindings::UserDataPtr,
) -> bool {
    let context = unsafe { &mut *(user_data as *mut Context) };
    let x = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let value = context.problem.objective(x);
    unsafe { *obj_value = value };
    value.is_finite()
}

unsafe extern "C" fn eval_grad_f(
    n: bindings::ipindex,
    x: *mut bindings::ipnumber,
    _new_x: bool,
    grad_f: *mut bindings::ipnumber,
    user_data: bindings::UserDataPtr,
) -> bool {
    let context = unsafe { &mut *(user_data as *mut Context) };
    let x = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let gradient = context.problem.gradient(x);
    let out = unsafe { std::slice::from_raw_parts_mut(grad_f, n as usize) };
    out.copy_from_slice(&gradient);
    true
}

unsafe extern "C" fn eval_g(
    n: bindings::ipindex,
    x: *mut bindings::ipnumber,
    _new_x: bool,
    m: bindings::ipindex,
    g: *mut bindings::ipnumber,
    user_data: bindings::UserDataPtr,
) -> bool {
    let context = unsafe { &mut *(user_data as *mut Context) };
    let x = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let values = context.problem.constraints(x);
    if m > 0 {
        let out = unsafe { std::slice::from_raw_parts_mut(g, m as usize) };
        out.copy_from_slice(&values);
    }
    true
}

unsafe extern "C" fn eval_jac_g(
    n: bindings::ipindex,
    x: *mut bindings::ipnumber,
    _new_x: bool,
    _m: bindings::ipindex,
    nele_jac: bindings::ipindex,
    i_row: *mut bindings::ipindex,
    j_col: *mut bindings::ipindex,
    values: *mut bindings::ipnumber,
    user_data: bindings::UserDataPtr,
) -> bool {
    let context = unsafe { &mut *(user_data as *mut Context) };
    let count = nele_jac as usize;

    if values.is_null() {
        // Structure pass.
        let rows = unsafe { std::slice::from_raw_parts_mut(i_row, count) };
        let cols = unsafe { std::slice::from_raw_parts_mut(j_col, count) };
        for (k, (r, c)) in context.jacobian_pattern.iter().enumerate() {
            rows[k] = *r as bindings::ipindex;
            cols[k] = *c as bindings::ipindex;
        }
        return true;
    }

    // Value pass. Duplicates in the caller's triplets sum, so accumulate into
    // the captured pattern rather than assuming a one-to-one correspondence.
    let x = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let mut folded = vec![0.0; count];
    let mut index: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::with_capacity(count);
    for (k, key) in context.jacobian_pattern.iter().enumerate() {
        index.insert(*key, k);
    }
    for (r, c, v) in context.problem.jacobian(x) {
        match index.get(&(r, c)) {
            Some(&k) => folded[k] += v,
            None => {
                // A pattern that grew since the structure pass cannot be
                // honoured — IPOPT allocated for the original one.
                context.pattern_mismatch = true;
                return false;
            }
        }
    }
    let out = unsafe { std::slice::from_raw_parts_mut(values, count) };
    out.copy_from_slice(&folded);
    true
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn eval_h(
    n: bindings::ipindex,
    x: *mut bindings::ipnumber,
    _new_x: bool,
    obj_factor: bindings::ipnumber,
    m: bindings::ipindex,
    lambda: *mut bindings::ipnumber,
    _new_lambda: bool,
    nele_hess: bindings::ipindex,
    i_row: *mut bindings::ipindex,
    j_col: *mut bindings::ipindex,
    values: *mut bindings::ipnumber,
    user_data: bindings::UserDataPtr,
) -> bool {
    let context = unsafe { &mut *(user_data as *mut Context) };
    let count = nele_hess as usize;

    if values.is_null() {
        let rows = unsafe { std::slice::from_raw_parts_mut(i_row, count) };
        let cols = unsafe { std::slice::from_raw_parts_mut(j_col, count) };
        for (k, (r, c)) in context.hessian_pattern.iter().enumerate() {
            rows[k] = *r as bindings::ipindex;
            cols[k] = *c as bindings::ipindex;
        }
        return true;
    }

    let x = unsafe { std::slice::from_raw_parts(x, n as usize) };
    let lambda = if m > 0 {
        unsafe { std::slice::from_raw_parts(lambda, m as usize) }.to_vec()
    } else {
        Vec::new()
    };
    let folded =
        scaled_hessian(context.problem, x, &lambda, obj_factor, &context.hessian_pattern);
    let out = unsafe { std::slice::from_raw_parts_mut(values, count) };
    out.copy_from_slice(&folded);
    true
}

fn map_status(status: bindings::ApplicationReturnStatus) -> OptStatus {
    use bindings::*;
    match status {
        ApplicationReturnStatus_Solve_Succeeded
        | ApplicationReturnStatus_Solved_To_Acceptable_Level => OptStatus::Optimal,
        ApplicationReturnStatus_Infeasible_Problem_Detected => OptStatus::Infeasible,
        ApplicationReturnStatus_Diverging_Iterates => OptStatus::Unbounded,
        ApplicationReturnStatus_Maximum_Iterations_Exceeded => OptStatus::Other("iteration limit"),
        ApplicationReturnStatus_Maximum_CpuTime_Exceeded => OptStatus::Other("time limit"),
        ApplicationReturnStatus_Restoration_Failed => OptStatus::Other("restoration failed"),
        ApplicationReturnStatus_Search_Direction_Becomes_Too_Small => {
            OptStatus::Other("search direction collapsed")
        }
        ApplicationReturnStatus_User_Requested_Stop => OptStatus::Other("stopped by user"),
        ApplicationReturnStatus_Invalid_Number_Detected => {
            OptStatus::Other("invalid number in a callback")
        }
        _ => OptStatus::Other("IPOPT reported a failure"),
    }
}

fn set_string_option(problem: bindings::IpoptProblem, name: &str, value: &str) -> Result<(), OpfError> {
    let name = CString::new(name).map_err(|e| OpfError::Backend(e.to_string()))?;
    let value = CString::new(value).map_err(|e| OpfError::Backend(e.to_string()))?;
    // SAFETY: both strings outlive the call; IPOPT copies what it keeps.
    let ok = unsafe {
        bindings::AddIpoptStrOption(
            problem,
            name.as_ptr() as *mut c_char,
            value.as_ptr() as *mut c_char,
        )
    };
    if ok {
        Ok(())
    } else {
        Err(OpfError::Backend(format!("IPOPT rejected option {name:?} = {value:?}")))
    }
}

/// Solves a [`NonlinearProblem`] with IPOPT.
///
/// # Requirements on the problem
///
/// The Jacobian and Hessian sparsity patterns must not depend on `x`. IPOPT
/// asks for structure once and values many times, and writes values into the
/// slots the structure pass declared. This captures both patterns at the
/// initial point; a later evaluation producing an entry outside them is
/// refused rather than written somewhere wrong.
///
/// `opf::nlp` has no such requirement — it rebuilds the KKT matrix from
/// triplets every iteration — so a problem can be valid for the in-house
/// solver and not for this one. Both AC-OPF and DC-OPF satisfy it.
pub fn solve(
    problem: &dyn NonlinearProblem,
    options: &IpoptOptions,
) -> Result<NlpSolution, OpfError> {
    let n = problem.n_vars();
    let m = problem.n_constraints();
    let (mut x_lower, mut x_upper) = problem.var_bounds();
    let (mut g_lower, mut g_upper) = problem.constraint_bounds();
    let start = problem.initial_point();

    // IPOPT reads anything beyond ±2e19 as infinite by default. Our
    // formulations use f64 infinities, which it would otherwise take
    // literally in arithmetic.
    const IPOPT_INFINITY: f64 = 2.0e19;
    for value in x_lower.iter_mut().chain(g_lower.iter_mut()) {
        if *value == f64::NEG_INFINITY {
            *value = -IPOPT_INFINITY;
        }
    }
    for value in x_upper.iter_mut().chain(g_upper.iter_mut()) {
        if *value == f64::INFINITY {
            *value = IPOPT_INFINITY;
        }
    }

    let jacobian_pattern = {
        let mut keys: Vec<(usize, usize)> =
            problem.jacobian(&start).into_iter().map(|(r, c, _)| (r, c)).collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    };
    let hessian_pattern = {
        let zero = vec![0.0; m];
        let mut both = problem.lagrangian_hessian(&start, &zero);
        // Union with a nonzero-multiplier evaluation: the constraint Hessians
        // contribute entries the objective's alone does not, and a pattern
        // missing them would be silently truncated at every later iteration.
        let ones = vec![1.0; m];
        both.extend(problem.lagrangian_hessian(&start, &ones));
        lower_triangle(&both)
    };

    let mut context = Context {
        problem,
        jacobian_pattern,
        hessian_pattern,
        pattern_mismatch: false,
    };

    // SAFETY: every pointer below is to a live local that outlives the call;
    // IPOPT copies the bound arrays, as its header documents.
    let handle = unsafe {
        bindings::CreateIpoptProblem(
            n as bindings::ipindex,
            x_lower.as_mut_ptr(),
            x_upper.as_mut_ptr(),
            m as bindings::ipindex,
            g_lower.as_mut_ptr(),
            g_upper.as_mut_ptr(),
            context.jacobian_pattern.len() as bindings::ipindex,
            context.hessian_pattern.len() as bindings::ipindex,
            0, // C-style indexing, matching our own 0-based triplets.
            Some(eval_f),
            Some(eval_g),
            Some(eval_grad_f),
            Some(eval_jac_g),
            Some(eval_h),
        )
    };
    if handle.is_null() {
        return Err(OpfError::Backend("CreateIpoptProblem returned null".into()));
    }

    // A guard so every early return below still frees the handle.
    struct Handle(bindings::IpoptProblem);
    impl Drop for Handle {
        fn drop(&mut self) {
            // SAFETY: constructed from a non-null CreateIpoptProblem result
            // and freed exactly once.
            unsafe { bindings::FreeIpoptProblem(self.0) };
        }
    }
    let guard = Handle(handle);

    set_string_option(handle, "sb", "yes")?;
    let tol = CString::new("tol").unwrap();
    let max_iter = CString::new("max_iter").unwrap();
    let print_level = CString::new("print_level").unwrap();
    // SAFETY: as above.
    unsafe {
        bindings::AddIpoptNumOption(handle, tol.as_ptr() as *mut c_char, options.tolerance);
        bindings::AddIpoptIntOption(
            handle,
            max_iter.as_ptr() as *mut c_char,
            options.max_iterations as c_int,
        );
        bindings::AddIpoptIntOption(
            handle,
            print_level.as_ptr() as *mut c_char,
            options.print_level as c_int,
        );
    }
    for (name, value) in &options.string_options {
        set_string_option(handle, name, value)?;
    }

    let mut x = start.clone();
    let mut g = vec![0.0; m.max(1)];
    let mut objective = 0.0f64;
    let mut mult_g = vec![0.0; m.max(1)];
    let mut mult_x_lower = vec![0.0; n.max(1)];
    let mut mult_x_upper = vec![0.0; n.max(1)];

    // SAFETY: `context` outlives the call; IPOPT hands the pointer straight
    // back to the callbacks and does not retain it.
    let status = unsafe {
        bindings::IpoptSolve(
            handle,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut objective,
            mult_g.as_mut_ptr(),
            mult_x_lower.as_mut_ptr(),
            mult_x_upper.as_mut_ptr(),
            &mut context as *mut Context as bindings::UserDataPtr,
        )
    };
    drop(guard);

    if context.pattern_mismatch {
        return Err(OpfError::Backend(
            "the problem's Jacobian sparsity pattern changed during the solve; IPOPT \
             requires it to be fixed"
                .into(),
        ));
    }

    let status = map_status(status);
    let constraints = problem.constraints(&x);
    let (c_lower, c_upper) = problem.constraint_bounds();
    let mut violation = 0.0f64;
    for r in 0..m {
        violation = violation
            .max((c_lower[r] - constraints[r]).max(0.0))
            .max((constraints[r] - c_upper[r]).max(0.0));
    }

    Ok(NlpSolution {
        status,
        objective,
        x,
        // Negated back into this crate's convention — see the module docs.
        y: mult_g[..m].iter().map(|v| -v).collect(),
        z: (0..n).map(|k| mult_x_lower[k] - mult_x_upper[k]).collect(),
        // IPOPT's C API does not expose the iteration count through
        // `IpoptSolve`; reporting zero is honest, where a guess would not be.
        iterations: 0,
        violation,
    })
}

/// The linked IPOPT's version, for a test or a bug report to quote.
pub fn version() -> String {
    format!("{}.{}.{}", bindings::IPOPT_VERSION_MAJOR, bindings::IPOPT_VERSION_MINOR, bindings::IPOPT_VERSION_RELEASE)
}

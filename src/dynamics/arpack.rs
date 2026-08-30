//! ARPACK, as the reference [`arnoldi`](super::arnoldi) is checked against.
//!
//! Not a dependency and not a faster path: the in-house iteration is the one
//! that runs. This exists for the same reason `opf::highs` and `opf::ipopt` do
//! — an implicitly restarted Arnoldi method is the easiest thing in this crate
//! to get subtly and silently wrong, and the strongest evidence that it is not
//! wrong is that the field's own implementation, given the same problem,
//! returns the same numbers.
//!
//! **The same problem, exactly.** ARPACK is driven in *mode 1* on
//! [`ShiftedDae::apply`](super::shift::ShiftedDae::apply) — it never sees the
//! DAE, the shift, or the factorization, only a closure that maps a vector to a
//! vector. So a disagreement isolates the eigensolver and nothing else: the
//! operator, the reduction and the model library are shared, and cannot cancel
//! an error out.
//!
//! # Reverse communication
//!
//! ARPACK does not take a callback. It returns to the caller with `ido = ±1`
//! and two offsets into a workspace, meaning *apply the operator to this vector
//! and put the answer there*, and is called again. That is what lets one
//! Fortran routine serve every operator anyone has ever had, and it is why this
//! module is a loop rather than a call.
//!
//! # Why the bindings are written out
//!
//! `build.rs` explains it from the build side: arpack-ng exports ISO_C_BINDING
//! wrappers, so the surface used here is two ordinary C functions rather than
//! Fortran symbols with hidden string-length arguments. Declaring them costs
//! eighteen argument types once and removes the need for a header whose
//! location distributions disagree about. They are checked by the cross-check
//! itself: get one argument wrong and the comparison fails loudly.

use num_complex::Complex;

use super::arnoldi::{ArnoldiError, ArnoldiOptions, ArnoldiResult, RitzPair};

/// arpack-ng's integer, 32-bit unless it was built `ILP64`.
#[allow(non_camel_case_types)]
type a_int = i32;

// `a_dcomplex` is `struct { double real, imag; }`, which is exactly
// `num_complex::Complex<f64>`'s `#[repr(C)]` layout.
unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn znaupd_c(
        ido: *mut a_int,
        bmat: *const std::ffi::c_char,
        n: a_int,
        which: *const std::ffi::c_char,
        nev: a_int,
        tol: f64,
        resid: *mut Complex<f64>,
        ncv: a_int,
        v: *mut Complex<f64>,
        ldv: a_int,
        iparam: *mut a_int,
        ipntr: *mut a_int,
        workd: *mut Complex<f64>,
        workl: *mut Complex<f64>,
        lworkl: a_int,
        rwork: *mut f64,
        info: *mut a_int,
    );

    #[allow(clippy::too_many_arguments)]
    fn zneupd_c(
        rvec: a_int,
        howmny: *const std::ffi::c_char,
        select: *const a_int,
        d: *mut Complex<f64>,
        z: *mut Complex<f64>,
        ldz: a_int,
        sigma: Complex<f64>,
        workev: *mut Complex<f64>,
        bmat: *const std::ffi::c_char,
        n: a_int,
        which: *const std::ffi::c_char,
        nev: a_int,
        tol: f64,
        resid: *mut Complex<f64>,
        ncv: a_int,
        v: *mut Complex<f64>,
        ldv: a_int,
        iparam: *mut a_int,
        ipntr: *mut a_int,
        workd: *mut Complex<f64>,
        workl: *mut Complex<f64>,
        lworkl: a_int,
        rwork: *mut f64,
        info: *mut a_int,
    );
}

/// ARPACK said no, and what it said.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArpackError {
    /// `znaupd` returned a negative `info`.
    Iteration(i32),
    /// `zneupd` returned a negative `info` while extracting the eigenvectors.
    Extraction(i32),
    /// The operator refused a vector.
    OperatorFailed,
    /// It ran out of iterations with fewer converged than were asked for.
    NotConverged { wanted: usize, converged: usize },
    /// Nothing to iterate on, or more wanted than the problem has.
    Sizes,
}

impl std::fmt::Display for ArpackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArpackError::Iteration(info) => write!(f, "znaupd failed with info = {info}"),
            ArpackError::Extraction(info) => write!(f, "zneupd failed with info = {info}"),
            ArpackError::OperatorFailed => write!(f, "the operator could not be applied"),
            ArpackError::NotConverged { wanted, converged } => {
                write!(f, "ARPACK converged {converged} of {wanted} eigenvalues")
            }
            ArpackError::Sizes => write!(f, "nothing to iterate on"),
        }
    }
}

impl std::error::Error for ArpackError {}

/// The same question [`arnoldi::eigenpairs`](super::arnoldi::eigenpairs)
/// answers, asked of ARPACK.
///
/// The signature matches deliberately, so the cross-check is one substitution
/// and the two cannot drift apart in what they are given. `opts.max_restarts`
/// becomes ARPACK's iteration limit; `opts.max_dim` becomes `ncv`, the size of
/// the Arnoldi basis, which ARPACK keeps fixed rather than growing.
pub fn eigenpairs<F>(
    n: usize,
    apply: F,
    opts: &ArnoldiOptions,
) -> Result<ArnoldiResult, ArpackError>
where
    F: Fn(&[Complex<f64>]) -> Option<Vec<Complex<f64>>>,
{
    if n < 2 || opts.count == 0 || opts.count >= n {
        return Err(ArpackError::Sizes);
    }
    let nev = opts.count;
    // ARPACK requires `nev + 2 <= ncv <= n`, and recommends `ncv >= 2·nev`.
    let requested = if opts.max_dim == 0 { (2 * nev + 20).max(30) } else { opts.max_dim };
    let ncv = requested.clamp(nev + 2, n);

    let (ni, nevi, ncvi) = (n as a_int, nev as a_int, ncv as a_int);
    let lworkl = 3 * ncv * ncv + 5 * ncv;

    let zero = Complex::new(0.0, 0.0);
    let mut resid = vec![zero; n];
    let mut v = vec![zero; n * ncv];
    let mut workd = vec![zero; 3 * n];
    let mut workl = vec![zero; lworkl];
    let mut rwork = vec![0.0f64; ncv];
    let mut iparam = [0 as a_int; 11];
    let mut ipntr = [0 as a_int; 14];

    iparam[0] = 1; // exact shifts, chosen by ARPACK itself
    iparam[2] = ((opts.max_restarts + 1) * 300).max(300) as a_int; // max iterations
    iparam[6] = 1; // mode 1: we hand it OP·x directly

    let bmat = c"I";
    let which = c"LM"; // largest |θ| — nearest the shift, after the transform
    let howmny = c"A";

    let mut ido: a_int = 0;
    let mut info: a_int = 0; // 0: ARPACK picks its own random start vector

    loop {
        unsafe {
            znaupd_c(
                &mut ido,
                bmat.as_ptr(),
                ni,
                which.as_ptr(),
                nevi,
                opts.tol,
                resid.as_mut_ptr(),
                ncvi,
                v.as_mut_ptr(),
                ni,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr(),
                workl.as_mut_ptr(),
                lworkl as a_int,
                rwork.as_mut_ptr(),
                &mut info,
            );
        }

        match ido {
            // "Apply the operator to the vector at ipntr[0], and write the
            // result at ipntr[1]." The offsets are one-based, as Fortran's are.
            -1 | 1 => {
                let from = (ipntr[0] - 1) as usize;
                let to = (ipntr[1] - 1) as usize;
                let x: Vec<Complex<f64>> = workd[from..from + n].to_vec();
                let y = apply(&x).ok_or(ArpackError::OperatorFailed)?;
                workd[to..to + n].copy_from_slice(&y);
            }
            _ => break,
        }
    }

    if info < 0 {
        return Err(ArpackError::Iteration(info));
    }
    // `info = 1` means the iteration limit was reached; `iparam[4]` says how
    // many converged anyway, and returning those is more useful than nothing.
    let converged = iparam[4] as usize;
    if converged < nev {
        return Err(ArpackError::NotConverged { wanted: nev, converged });
    }

    let mut d = vec![zero; nev + 1];
    let mut z = vec![zero; n * nev];
    let mut workev = vec![zero; 2 * ncv];
    let select = vec![0 as a_int; ncv];
    let mut info_eupd: a_int = 0;

    unsafe {
        zneupd_c(
            1, // rvec: yes, we want the eigenvectors
            howmny.as_ptr(),
            select.as_ptr(),
            d.as_mut_ptr(),
            z.as_mut_ptr(),
            ni,
            zero, // sigma: unused in mode 1
            workev.as_mut_ptr(),
            bmat.as_ptr(),
            ni,
            which.as_ptr(),
            nevi,
            opts.tol,
            resid.as_mut_ptr(),
            ncvi,
            v.as_mut_ptr(),
            ni,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr(),
            workl.as_mut_ptr(),
            lworkl as a_int,
            rwork.as_mut_ptr(),
            &mut info_eupd,
        );
    }
    if info_eupd < 0 {
        return Err(ArpackError::Extraction(info_eupd));
    }

    // Residuals measured the same way `arnoldi` measures its own — by applying
    // the operator — so the two results are comparable field for field and not
    // only in their eigenvalues.
    let mut pairs = Vec::with_capacity(nev);
    for i in 0..nev {
        let theta = d[i];
        let vector: Vec<Complex<f64>> = z[i * n..(i + 1) * n].to_vec();
        let image = apply(&vector).ok_or(ArpackError::OperatorFailed)?;
        let raw = image
            .iter()
            .zip(&vector)
            .map(|(a, b)| (a - theta * b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let magnitude = theta.norm();
        pairs.push(RitzPair {
            theta,
            vector,
            residual: if magnitude > 0.0 { raw / (magnitude * magnitude) } else { f64::INFINITY },
            converged: magnitude > 0.0 && raw <= opts.tol.max(1e-14) * magnitude,
        });
    }
    pairs.sort_by(|a, b| {
        b.theta.norm().partial_cmp(&a.theta.norm()).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ArnoldiResult { pairs, restarts: 0, dim: ncv })
}

impl From<ArpackError> for ArnoldiError {
    fn from(e: ArpackError) -> Self {
        match e {
            ArpackError::OperatorFailed => ArnoldiError::OperatorFailed,
            ArpackError::Sizes => ArnoldiError::Empty,
            _ => ArnoldiError::Eigen,
        }
    }
}

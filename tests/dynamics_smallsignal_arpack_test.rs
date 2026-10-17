//! The in-house Arnoldi against ARPACK's, on the same operator.
//!
//! Every other gate on the sparse eigensolver compares it against something
//! *structurally* different: the dense decomposition of an explicitly formed
//! `A`, a closed form, a trajectory. Each of those is strong evidence about the
//! whole pipeline and weak evidence about the iteration in the middle of it,
//! because a Krylov method that converged to slightly the wrong subspace would
//! still be caught only where the tolerances are loose.
//!
//! This is the opposite shape. ARPACK is handed the *identical* closure —
//! `ShiftedDae::apply`, in its mode 1 — so the operator, the factorization, the
//! reduction and the model library are all shared and none of them can cancel
//! an error out. What is left in the comparison is the eigensolver alone.
//!
//! Needs a system ARPACK; not run in CI. See `Cargo.toml`'s
//! `smallsignal-arpack` feature.
//!
//! ```text
//! ARPACK_ROOT=/usr cargo test --features smallsignal-arpack \
//!     --test dynamics_smallsignal_arpack_test
//! ```

mod dynamics_ring;

use num_complex::Complex;

use gridoxide::dynamics::arnoldi::{self, ArnoldiOptions};
use gridoxide::dynamics::arpack;
use gridoxide::dynamics::shift::ShiftedDae;
use gridoxide::dynamics::DynamicSystem;

/// Both solvers on one operator, returning `(ours, theirs)` as eigenvalues of
/// the *original* problem — the shift undone the same way for both.
fn both(system: &DynamicSystem, sigma: Complex<f64>, count: usize) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let op = ShiftedDae::new(system, sigma).expect("a shift off the spectrum");
    let opts = ArnoldiOptions { count, tol: 1e-12, ..Default::default() };

    let ours = arnoldi::eigenpairs(op.n_states(), |b| op.apply(b), &opts)
        .expect("our own iteration runs");
    assert!(ours.converged(), "ours did not converge: {:?}", ours.pairs);

    let theirs = arpack::eigenpairs(op.n_states(), |b| op.apply(b), &opts)
        .expect("ARPACK runs");

    let undo = |pairs: &[arnoldi::RitzPair]| -> Vec<Complex<f64>> {
        let mut out: Vec<Complex<f64>> = pairs.iter().map(|p| sigma + 1.0 / p.theta).collect();
        // Sorted so the comparison is between sets, not between orderings —
        // which are ARPACK's business and not a claim either method makes.
        out.sort_by(|a, b| {
            (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    };
    (undo(&ours.pairs), undo(&theirs.pairs))
}

/// The two agree on a small case.
#[test]
fn arpack_and_the_in_house_iteration_agree_on_a_small_case() {
    let (system, _) = dynamics_ring::ring(16, 0.4).build().expect("the ring initializes");
    let sigma = Complex::new(-0.4, std::f64::consts::TAU);

    let (ours, theirs) = both(&system, sigma, 4);
    assert_eq!(ours.len(), theirs.len());
    for (a, b) in ours.iter().zip(&theirs) {
        assert!((a - b).norm() < 1e-10, "{a} against ARPACK's {b}");
    }
}

/// And on one where the Krylov space is a genuine projection.
///
/// 320 states against a Krylov dimension of a few dozen, so both methods are
/// really iterating rather than exhausting the space — which is the case the
/// comparison is worth making on.
#[test]
fn arpack_and_the_in_house_iteration_agree_on_a_real_subspace() {
    let (system, _) = dynamics_ring::ring(64, 0.4).build().unwrap();
    assert_eq!(system.n_states(), 320);

    for sigma in [
        Complex::new(-0.4, std::f64::consts::TAU),
        Complex::new(-2.0, 0.5 * std::f64::consts::TAU),
        Complex::new(-0.1, 3.0 * std::f64::consts::TAU),
    ] {
        let (ours, theirs) = both(&system, sigma, 6);
        for (a, b) in ours.iter().zip(&theirs) {
            assert!((a - b).norm() < 1e-9, "at σ={sigma}: {a} against ARPACK's {b}");
        }
    }
}

/// The residuals agree too, not just the eigenvalues.
///
/// Both measure it the same way — by applying the operator to the returned
/// vector — so this compares the *vectors*, which an eigenvalue comparison
/// does not. Two methods can agree on λ and disagree on which invariant
/// direction they found in a cluster.
#[test]
fn the_two_agree_on_their_residuals() {
    let (system, _) = dynamics_ring::ring(32, 0.4).build().unwrap();
    let sigma = Complex::new(-0.4, std::f64::consts::TAU);
    let op = ShiftedDae::new(&system, sigma).unwrap();
    let opts = ArnoldiOptions { count: 4, tol: 1e-12, ..Default::default() };

    let ours = arnoldi::eigenpairs(op.n_states(), |b| op.apply(b), &opts).unwrap();
    let theirs = arpack::eigenpairs(op.n_states(), |b| op.apply(b), &opts).unwrap();

    for pair in ours.pairs.iter().chain(&theirs.pairs) {
        assert!(pair.converged, "{} did not converge: {}", pair.theta, pair.residual);
        assert!(pair.residual < 1e-10, "{}: residual {}", pair.theta, pair.residual);
    }

    // Same eigenvector, up to the phase an eigenvector has no opinion about.
    for ours in &ours.pairs {
        let twin = theirs
            .pairs
            .iter()
            .min_by(|a, b| {
                (a.theta - ours.theta)
                    .norm()
                    .partial_cmp(&(b.theta - ours.theta).norm())
                    .unwrap()
            })
            .unwrap();
        let overlap: Complex<f64> =
            ours.vector.iter().zip(&twin.vector).map(|(a, b)| a.conj() * b).sum();
        let scale = twin.vector.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            (overlap.norm() / scale - 1.0).abs() < 1e-8,
            "{}: eigenvectors are not parallel, |⟨u,v⟩| = {}",
            ours.theta,
            overlap.norm() / scale
        );
    }
}

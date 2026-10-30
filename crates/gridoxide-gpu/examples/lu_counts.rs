//! The size of the Newton Jacobian's LU factors, on the CPU and on the
//! device, for one case: what the factorization costs per system.
//!
//! ```text
//! cargo run --release -p gridoxide-gpu --features cuda --example lu_counts -- <case.m>
//! ```
//!
//! The Jacobian is the solved base case's. KLU (gridoxide's port, BTF and
//! AMD, partial pivoting) reports its factors' nonzeros and its
//! factorization's floating-point operations; cuDSS reports the nonzeros its
//! analysis planned (`CUDSS_DATA_LU_NNZ`), for one system and for a batch.

use gridoxide::network::{build_ybus, linear_initial_guess, power_injections, stamp_shunts};
use gridoxide::solver::{JacobianBackend, PersistentSolver};
use gridoxide_gpu::{CudssUniformBatch, UniformBatch, csr_jacobian};
use gridoxide_sparse::klu_native::KluNativeSystem;

fn main() {
    let path = std::env::args().nth(1).expect("usage: lu_counts <case.m>");
    let net = gridoxide::matpower::read(&path).unwrap();
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let mut base = net.buses.clone();
    linear_initial_guess(&mut base, &ybus);
    PersistentSolver::new(JacobianBackend::KluNative).solve(&mut base, &ybus, 1e-10, 30);

    let (pattern, csr) = csr_jacobian(&base, &ybus);
    let (p, q) = power_injections(&base, &ybus);
    let mut values = Vec::new();
    pattern.fill(&base, &p, &q, &mut values);
    let n = csr.n;
    let nnz = csr.nnz();

    let klu = KluNativeSystem::new(n, &pattern.to_triplets(&values)).expect("singular Jacobian");
    let (klu_nnz, klu_flops) = klu.factor_stats();
    println!(
        "{path}: {} buses, Jacobian n = {n}, nnz = {nnz}",
        net.buses.len()
    );
    println!(
        "  klu:   nnz(L+U) = {klu_nnz} ({:.2}x the Jacobian's), factorization {:.3} Mflop",
        klu_nnz as f64 / nnz as f64,
        klu_flops / 1e6
    );
    for batch in [1, 4] {
        let lin = CudssUniformBatch::analyze(&csr, batch).unwrap();
        let lu = lin.lu_nnz().unwrap();
        println!(
            "  cudss: B = {batch}, LU_NNZ = {lu} ({:.2}x the Jacobian's per system)",
            lu as f64 / batch as f64 / nnz as f64
        );
    }
}

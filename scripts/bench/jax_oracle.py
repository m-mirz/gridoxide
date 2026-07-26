#!/usr/bin/env python3
"""JAX oracle for the batched AC power flow formulation.

`plans/GPU_PLAN.md` Phase 1 asks for "a numerical oracle for block-diagonal
embedding, validated against the existing accuracy suite". This is it. It is
**not** a performance prototype and must never be quoted as one — it runs on
CPU, uses dense linear algebra, and is deliberately the slowest power flow in
this repository. Its only job is to answer two questions with independent code:

1. **Is block-diagonal embedding actually equivalent to independent solves?**
   `plans/GPU_PLAN.md` §3 property 2 claims that stacking B scenarios into one
   block-diagonal matrix and taking a single LU is mathematically identical to
   B separate solves — which is what lets the AMD path work without a batched
   refactorization API. That claim is the architectural load-bearing wall of
   Phases 3-5. Here it is checked numerically rather than asserted:
   `solve_batch_bde` and `solve_batch_vmap` must agree to machine precision.

2. **Does the batched Newton formulation reach gridoxide's answer?** The
   oracle is written from the H/N/M/L equations directly, in a different
   language, on a different linear-algebra stack, so agreement with the `klu`
   backend is meaningful evidence that neither has a formulation bug.

It consumes gridoxide's *own* Y-bus and bus arrays (`ybus_triplets`,
`bus_spec`, `initial_guess`) rather than re-deriving them from the input file.
That is deliberate: if the oracle rebuilt the model itself, a disagreement
could equally be a tap-ratio or shunt-stamping difference in the converter as
a solver bug, and the comparison would prove nothing about the solver.

Usage:
    python3 jax_oracle.py <case.json> [n_scenarios]

Needs its own virtualenv (JAX is not a dependency of anything else here):
    python3 -m venv .venv-jax
    .venv-jax/bin/pip install jax numpy maturin
    VIRTUAL_ENV=$PWD/.venv-jax .venv-jax/bin/maturin develop --release --features python,klu

Scope limits, stated rather than discovered later:
  - Constant-power injections only. ZIP terms are asserted absent, not handled.
  - Dense Jacobian. Fine to a few thousand buses; do not point it at
    case9241pegase and expect it to finish.
  - No Q-limit enforcement and no island partitioning, matching what
    `PersistentSolver::solve` does by default.
"""
import sys
import time

import jax

# Non-negotiable for this workload: the whole point is agreeing with a solver
# that matches five other implementations to 4+ decimals. JAX defaults to f32.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (must follow the x64 config)
import numpy as np  # noqa: E402

import gridoxide  # noqa: E402

SLACK, PV, PQ = 0, 1, 2


def load_case(path):
    """Pulls the Y-bus and bus arrays straight out of gridoxide."""
    model = gridoxide.PowerFlowModel.from_pgm_json(path, backend="klu")

    zips = model.zip_term_counts()
    if any(z != 0 for z in zips):
        raise SystemExit(
            f"{sum(1 for z in zips if z)} bus(es) carry ZIP terms; this oracle models "
            "constant-power injections only and would silently give a wrong answer"
        )

    rows, cols, g, b = model.ybus_triplets()
    n = model.n_nodes
    ybus = np.zeros((n, n), dtype=np.complex128)
    ybus[np.asarray(rows), np.asarray(cols)] = np.asarray(g) + 1j * np.asarray(b)

    kinds, p_spec, q_spec = model.bus_spec()
    kinds = np.asarray(kinds, dtype=np.int64)
    # Guard against the binding handing back something that collapses to a
    # scalar (PyO3 maps Vec<u8> to `bytes`, which np.asarray turns into a 0-d
    # array). That silently produced a 1-unknown "solve" that converged to
    # nonsense, so it is checked rather than trusted.
    if kinds.shape != (n,):
        raise SystemExit(f"bus_spec returned kinds of shape {kinds.shape}, expected ({n},)")
    if not set(np.unique(kinds)) <= {SLACK, PV, PQ}:
        raise SystemExit(f"unexpected bus-type codes: {np.unique(kinds)}")

    vm0, va0 = model.initial_guess()
    return model, {
        "ybus": jnp.asarray(ybus),
        "kind": kinds,
        "p_spec": jnp.asarray(p_spec),
        "q_spec": jnp.asarray(q_spec),
        "vm0": jnp.asarray(vm0),
        "va0": jnp.asarray(va0),
    }


def power_injections(vm, va, ybus):
    """S = V * conj(Y V), split into P and Q. Mirrors network::power_injections."""
    v = vm * jnp.exp(1j * va)
    s = v * jnp.conj(ybus @ v)
    return jnp.real(s), jnp.imag(s)


def jacobian(vm, va, ybus, p_calc, q_calc, non_slack, pq):
    """Dense H/N/M/L Jacobian, transliterated from `build_jacobian_triplets`.

        J = [ H  N ]   H = dP/d_ang, N = dP/d_vmag
            [ M  L ]   M = dQ/d_ang, L = dQ/d_vmag
    """
    g, b = jnp.real(ybus), jnp.imag(ybus)
    ang = va[:, None] - va[None, :]
    sin, cos = jnp.sin(ang), jnp.cos(ang)
    vv = vm[:, None] * vm[None, :]

    # Off-diagonal forms, everywhere; diagonals overwritten below.
    h = vv * (g * sin - b * cos)
    n = vm[:, None] * (g * cos + b * sin)
    m = -vv * (g * cos + b * sin)
    ll = vm[:, None] * (g * sin - b * cos)

    d = jnp.arange(vm.shape[0])
    h = h.at[d, d].set(-q_calc - vm**2 * jnp.diag(b))
    n = n.at[d, d].set(p_calc / vm + vm * jnp.diag(g))
    m = m.at[d, d].set(p_calc - vm**2 * jnp.diag(g))
    ll = ll.at[d, d].set(q_calc / vm - vm * jnp.diag(b))

    return jnp.block(
        [
            [h[jnp.ix_(non_slack, non_slack)], n[jnp.ix_(non_slack, pq)]],
            [m[jnp.ix_(pq, non_slack)], ll[jnp.ix_(pq, pq)]],
        ]
    )


def mismatch(vm, va, ybus, p_spec, q_spec, non_slack, pq):
    p_calc, q_calc = power_injections(vm, va, ybus)
    return (
        jnp.concatenate([(p_spec - p_calc)[non_slack], (q_spec - q_calc)[pq]]),
        p_calc,
        q_calc,
    )


def newton(case, p_spec, q_spec, tol=1e-6, max_iter=20, solve_fn=None):
    """One scenario. `solve_fn` defaults to a dense LU of this scenario alone."""
    ybus, kind = case["ybus"], case["kind"]
    non_slack = jnp.asarray(np.flatnonzero(kind != SLACK))
    pq = jnp.asarray(np.flatnonzero(kind == PQ))
    n_ang = int(non_slack.shape[0])

    vm, va = case["vm0"], case["va0"]
    solve_fn = solve_fn or (lambda j, f: jnp.linalg.solve(j, f))

    for it in range(max_iter):
        f, p_calc, q_calc = mismatch(vm, va, ybus, p_spec, q_spec, non_slack, pq)
        if float(jnp.max(jnp.abs(f))) < tol:
            return vm, va, it + 1, True
        j = jacobian(vm, va, ybus, p_calc, q_calc, non_slack, pq)
        dx = solve_fn(j, f)
        va = va.at[non_slack].add(dx[:n_ang])
        vm = vm.at[pq].add(dx[n_ang:])
    return vm, va, max_iter, False


def solve_batch_vmap(case, scenarios, tol=1e-6, max_iter=20):
    """B independent solves. The reference the embedded version must match."""
    return [newton(case, p, q, tol, max_iter) for p, q in scenarios]


def solve_batch_bde(case, scenarios, tol=1e-6, max_iter=20):
    """Block-diagonal embedding: stack every scenario's Jacobian into ONE
    matrix and take a single LU per iteration.

        J = diag(J_1 .. J_B),  dx = J^-1 f

    This is the formulation `plans/GPU_PLAN.md` §3 adopts. Because the blocks
    share no rows or columns, LU of the stacked matrix is block-diagonal too:
    no fill crosses a block, and partial pivoting cannot select across blocks
    since a column in block i has nonzeros only in block i's rows. So this
    *should* be bit-comparable to independent solves — which is exactly what
    is being tested, not assumed.

    Scenarios are stepped in lockstep with a per-scenario active mask, the
    convergence masking §3 requires: a scenario that has converged stops being
    updated, and a diverging one never poisons its neighbours.
    """
    ybus, kind = case["ybus"], case["kind"]
    non_slack = jnp.asarray(np.flatnonzero(kind != SLACK))
    pq = jnp.asarray(np.flatnonzero(kind == PQ))
    n_ang = int(non_slack.shape[0])
    nb = len(scenarios)
    blk = n_ang + int(pq.shape[0])

    vms = [case["vm0"]] * nb
    vas = [case["va0"]] * nb
    active = [True] * nb
    iters = [max_iter] * nb
    converged = [False] * nb

    for it in range(max_iter):
        blocks, rhs, live = [], [], []
        for k in range(nb):
            if not active[k]:
                continue
            p_spec, q_spec = scenarios[k]
            f, p_calc, q_calc = mismatch(vms[k], vas[k], ybus, p_spec, q_spec, non_slack, pq)
            if float(jnp.max(jnp.abs(f))) < tol:
                active[k], converged[k], iters[k] = False, True, it + 1
                continue
            blocks.append(jacobian(vms[k], vas[k], ybus, p_calc, q_calc, non_slack, pq))
            rhs.append(f)
            live.append(k)

        if not live:
            break

        # The embedding itself: one block-diagonal matrix, one solve.
        big_j = jax.scipy.linalg.block_diag(*blocks)
        big_f = jnp.concatenate(rhs)
        big_dx = jnp.linalg.solve(big_j, big_f)

        for slot, k in enumerate(live):
            dx = big_dx[slot * blk : (slot + 1) * blk]
            vas[k] = vas[k].at[non_slack].add(dx[:n_ang])
            vms[k] = vms[k].at[pq].add(dx[n_ang:])

    return [(vms[k], vas[k], iters[k], converged[k]) for k in range(nb)]


def cap_scenarios(case, n_scen, budget_gb=1.0):
    """Block-diagonal embedding builds a dense (B*blk)^2 matrix here, so B is
    capped to keep it inside `budget_gb`. A sparse implementation would not
    need this — the dense solve is an oracle-simplicity choice, not a property
    of the formulation being validated."""
    kind = case["kind"]
    blk = int(np.sum(kind != SLACK) + np.sum(kind == PQ))
    per = (blk**2) * 8 / 1e9
    if per <= 0:
        return n_scen, blk
    max_b = max(1, int((budget_gb / per) ** 0.5))
    return min(n_scen, max_b), blk


def make_scenarios(case, n, seed=20260726):
    """+/-20% uniform load scalings — the same shape bench_batch.py drives."""
    rng = np.random.default_rng(seed)
    return [
        (case["p_spec"] * f, case["q_spec"] * f)
        for f in rng.uniform(0.8, 1.2, size=n)
    ]


def main():
    path = sys.argv[1]
    n_scen = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    model, case = load_case(path)
    n = model.n_nodes
    print(f"case: {path}")
    print(f"buses: {n}  scenarios: {n_scen}  dtype: {case['ybus'].dtype}")
    assert case["ybus"].dtype == jnp.complex128, "x64 not enabled — results would be meaningless"
    print()

    # --- 1. base case vs gridoxide's klu backend -------------------------
    model.solve()
    ref_vm = np.asarray(model.voltage_mag())
    ref_va = np.asarray(model.voltage_ang())

    t0 = time.perf_counter()
    vm, va, iters, ok = newton(case, case["p_spec"], case["q_spec"])
    t1 = time.perf_counter()
    if not ok:
        raise SystemExit("oracle did not converge on the base case")

    d_vm = float(np.max(np.abs(np.asarray(vm) - ref_vm)))
    # Compare angles modulo 2*pi to avoid a spurious 2*pi wrap difference.
    dva = np.asarray(va) - ref_va
    d_va = float(np.max(np.abs(np.arctan2(np.sin(dva), np.cos(dva)))))
    print("1. oracle vs gridoxide klu, base case")
    print(f"   oracle converged in {iters} iters ({(t1 - t0) * 1e3:.1f} ms)")
    print(f"   max |dVm| = {d_vm:.3e}")
    print(f"   max |dVa| = {d_va:.3e} rad")
    base_ok = d_vm < 1e-8 and d_va < 1e-8
    print(f"   {'PASS' if base_ok else 'FAIL'} (tolerance 1e-8)")
    print()

    # --- 2. block-diagonal embedding vs independent solves ---------------
    capped, blk = cap_scenarios(case, n_scen)
    if capped < n_scen:
        print(f"   (block size {blk}; capping {n_scen} -> {capped} scenarios to bound the dense solve)")
    n_scen = capped
    scenarios = make_scenarios(case, n_scen)
    t0 = time.perf_counter()
    indep = solve_batch_vmap(case, scenarios)
    t1 = time.perf_counter()
    bde = solve_batch_bde(case, scenarios)
    t2 = time.perf_counter()

    worst_vm = worst_va = 0.0
    iter_mismatch = 0
    for (a_vm, a_va, a_it, a_ok), (b_vm, b_va, b_it, b_ok) in zip(indep, bde):
        if not (a_ok and b_ok):
            raise SystemExit("a scenario failed to converge in one of the two paths")
        worst_vm = max(worst_vm, float(np.max(np.abs(np.asarray(a_vm) - np.asarray(b_vm)))))
        worst_va = max(worst_va, float(np.max(np.abs(np.asarray(a_va) - np.asarray(b_va)))))
        iter_mismatch += a_it != b_it

    print("2. block-diagonal embedding vs independent solves")
    print(f"   independent: {(t1 - t0) * 1e3:.1f} ms   embedded: {(t2 - t1) * 1e3:.1f} ms")
    print(f"   max |dVm| = {worst_vm:.3e}")
    print(f"   max |dVa| = {worst_va:.3e} rad")
    print(f"   iteration-count mismatches: {iter_mismatch}/{n_scen}")
    # This is the claim GPU_PLAN section 3 property 2 rests on. It should hold
    # to machine precision, not merely to solver tolerance.
    bde_ok = worst_vm < 1e-12 and worst_va < 1e-12 and iter_mismatch == 0
    print(f"   {'PASS' if bde_ok else 'FAIL'} (tolerance 1e-12)")
    print()

    # --- 3. batch vs gridoxide's own batch path --------------------------
    scales = list(np.random.default_rng(20260726).uniform(0.8, 1.2, size=n_scen))
    got = model.solve_batch_scaled(scales)
    worst = 0.0
    for r, (o_vm, _, _, _) in zip(got, bde):
        worst = max(worst, float(np.max(np.abs(np.asarray(r.voltage_mag) - np.asarray(o_vm)))))
    print("3. oracle batch vs gridoxide BatchSolver")
    print(f"   max |dVm| = {worst:.3e}")
    batch_ok = worst < 1e-8
    print(f"   {'PASS' if batch_ok else 'FAIL'} (tolerance 1e-8)")
    print()

    all_ok = base_ok and bde_ok and batch_ok
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

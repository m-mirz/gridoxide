#!/usr/bin/env python3
"""Absolute correctness check for a solved MATPOWER case — no tool used as the
reference.

`accuracy_case_suite.py` can only report how far tools drift from *each other*,
which says nothing about which one is right. This script instead rebuilds the
bus admittance matrix straight from the MATPOWER `.m` file, following
`makeYbus.m`'s own conventions, and evaluates a converged solution against the
original power-flow equations:

    dS = V .* conj(Ybus @ V) - S_specified

A solution that actually solves the stated problem drives that residual to
zero. Which components of it are meaningful depends on the bus type, and this
is where a naive check goes wrong:

- PQ bus: both P and Q are specified, so both residuals must vanish.
- PV bus: only P is specified; reactive output is a free variable of the
  solve, so its Q residual is meaningless and is skipped.
- Slack bus: neither P nor Q is specified; both are skipped.

The residual is invariant to a global rotation of all voltage angles, so tools
that pick different angle references need no alignment before comparing.

Usage:
    python3 check_matpower_residual.py <case> [<case> ...]
    python3 check_matpower_residual.py --all
    python3 check_matpower_residual.py case1888rte --zero-phase-shifts

`--zero-phase-shifts` rebuilds the reference Ybus with every branch's `angle`
column forced to 0. That is not a correctness option — it exists to isolate one
specific, known conversion loss: PGM's `transformer.clock` cannot represent a
continuous phase shift (see `gridoxide.matpower`'s module docstring), so every
phase shift in these 12 cases is rounded away. Against the as-published case
that shows up as a large residual concentrated on the shifting branches'
endpoints; against the zeroed-shift network it vanishes, which is what
identifies it as *that* loss rather than a solver error.

Only tools reading this repo's own vendored `.m` files are meaningfully checked
this way. gridoxide (via `matpower_to_pgm.py`'s converted JSON), PGM (the same
JSON), and VeraGrid (the `.m` directly) all qualify. pandapower and
lightsim2grid load `pandapower.networks.<case>()`, whose bundled copy of a case
is not always the same data as the vendored `.m` (confirmed: the bus counts
themselves differ on `case1354pegase`/`case2869pegase`), so a residual computed
for them against this `.m` file measures that data difference, not solver
accuracy — they are excluded rather than reported misleadingly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import CASE_NAMES, matpower_filename
from matpower_to_pgm import load_mpc

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MATPOWER_DIR = REPO_ROOT / "tests" / "data" / "benchmark-grids" / "matpower"
CACHE_DIR = Path(__file__).resolve().parent / ".case-cache"

# MATPOWER column indices (0-based).
BUS_I, BUS_TYPE, PD, QD, GS, BS = 0, 1, 2, 3, 4, 5
GEN_BUS, PG, QG, GEN_STATUS = 0, 1, 2, 7
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATIO, ANGLE, BR_STATUS = 0, 1, 2, 3, 4, 8, 9, 10
PQ, REF = 1, 3


def build_ybus(mpc, zero_phase_shifts=False):
    """Dense Ybus, following `makeYbus.m` exactly (pi-model, complex tap on the
    from side, bus shunts on the diagonal)."""
    bus = np.asarray(mpc["bus"], dtype=float)
    branch = np.asarray(mpc["branch"], dtype=float)
    base_mva = float(np.asarray(mpc["baseMVA"]).ravel()[0])
    ids = bus[:, BUS_I].astype(int)
    pos = {b: i for i, b in enumerate(ids)}
    y = np.zeros((len(ids), len(ids)), dtype=complex)

    for br in branch:
        if br.shape[0] > BR_STATUS and br[BR_STATUS] == 0:
            continue
        f, t = pos[int(br[F_BUS])], pos[int(br[T_BUS])]
        ys = 1.0 / (br[BR_R] + 1j * br[BR_X])
        # ratio == 0 means "no transformer", i.e. a unity tap.
        tap = br[RATIO] if br[RATIO] != 0 else 1.0
        shift = 0.0 if zero_phase_shifts else br[ANGLE]
        tap = tap * np.exp(1j * np.deg2rad(shift))
        ytt = ys + 1j * br[BR_B] / 2.0
        y[f, f] += ytt / (tap * np.conj(tap))
        y[f, t] += -ys / np.conj(tap)
        y[t, f] += -ys / tap
        y[t, t] += ytt

    for i in range(len(ids)):
        y[i, i] += (bus[i, GS] + 1j * bus[i, BS]) / base_mva
    return ids, y, base_mva


def specified_injections(mpc, ids):
    """Net specified complex injection per bus, in per-unit."""
    bus = np.asarray(mpc["bus"], dtype=float)
    gen = np.asarray(mpc["gen"], dtype=float)
    base_mva = float(np.asarray(mpc["baseMVA"]).ravel()[0])
    pos = {b: i for i, b in enumerate(ids)}
    s = -(bus[:, PD] + 1j * bus[:, QD]) / base_mva
    for g in gen:
        if g[GEN_STATUS] > 0:
            s[pos[int(g[GEN_BUS])]] += (g[PG] + 1j * g[QG]) / base_mva
    return s, bus[:, BUS_TYPE].astype(int)


def gridoxide_solution(json_path, backend="klu_native"):
    import json

    import gridoxide

    with open(json_path) as f:
        node_ids = [n["id"] for n in json.load(f)["data"]["node"]]
    model = gridoxide.PowerFlowModel.from_pgm_json(str(json_path), backend=backend)
    model.solve()
    # from_pgm_json appends one virtual slack bus per source; trim it back off.
    vm = model.voltage_mag()[: len(node_ids)]
    va = model.voltage_ang()[: len(node_ids)]
    return dict(zip(node_ids, vm)), dict(zip(node_ids, va))


def residual(ids, ybus, s_spec, bus_type, vm, va, base_mva):
    v = np.array([vm[b] for b in ids]) * np.exp(1j * np.array([va[b] for b in ids]))
    ds = (v * np.conj(ybus @ v) - s_spec) * base_mva
    p_checked = bus_type != REF          # P specified at PV and PQ
    q_checked = bus_type == PQ           # Q specified at PQ only
    dp, dq = np.abs(ds.real)[p_checked], np.abs(ds.imag)[q_checked]
    worst = int(ids[p_checked][int(np.argmax(dp))]) if dp.size else -1
    return dp.max(), dq.max(), worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="case names (default: --all)")
    ap.add_argument("--all", action="store_true", help="check all 12 benchmark cases")
    ap.add_argument("--zero-phase-shifts", action="store_true",
                    help="zero every branch phase shift in the reference network (see docstring)")
    ap.add_argument("--backend", default="klu_native")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |residual| in MVA still reported as OK (default 1e-3)")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()

    cases = CASE_NAMES if (args.all or not args.cases) else args.cases
    print(f"{'case':<18}{'buses':>7}{'max|dP| MVA':>14}{'max|dQ| MVA':>14}{'worst bus':>11}  status")
    worst_overall = 0.0
    for case in cases:
        m_path = MATPOWER_DIR / matpower_filename(case)
        json_path = args.cache_dir / f"{case}.json"
        if not m_path.exists() or not json_path.exists():
            print(f"{case:<18}{'':>7}  skipped (missing input — run run_case_suite.py first)")
            continue
        mpc = load_mpc(m_path)
        ids, ybus, base_mva = build_ybus(mpc, zero_phase_shifts=args.zero_phase_shifts)
        s_spec, bus_type = specified_injections(mpc, ids)
        try:
            vm, va = gridoxide_solution(json_path, args.backend)
        except Exception as e:  # noqa: BLE001
            print(f"{case:<18}{len(ids):>7}  FAILED ({e})")
            continue
        dp, dq, worst = residual(ids, ybus, s_spec, bus_type, vm, va, base_mva)
        worst_overall = max(worst_overall, dp, dq)
        ok = "OK" if max(dp, dq) <= args.tol else "MISMATCH"
        print(f"{case:<18}{len(ids):>7}{dp:>14.4f}{dq:>14.4f}{worst:>11}  {ok}")

    return 0 if worst_overall <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())

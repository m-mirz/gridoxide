#!/usr/bin/env python3
"""Absolute correctness check for gridoxide's DC power flow — no tool used as
the reference.

`tests/dc_powerflow_test.rs`'s oracle checks DC against gridoxide's *own* AC
branch-flow code. That pins the formulation's internal consistency — the phase
shift sign, the `1/k` versus `1/k²` — but it cannot catch a convention this
project got wrong in both places at once, and it says nothing about agreeing
with the rest of the field. This script closes that: it rebuilds the DC system
straight from the MATPOWER `.m` file, following `makeBdc.m` and `dcpf.m`'s own
conventions, and compares.

    b       = status / (x * tap)                     makeBdc.m
    Bf      = b at (l, f), -b at (l, t)
    Bbus    = Cft' * Bf
    Pfinj   = b * (-shift_rad)
    Pbusinj = Cft' * Pfinj
    Va[pvpq] = Bbus[pvpq,pvpq] \\ (Pbus[pvpq] - Bbus[pvpq,ref] * Va0[ref])   dcpf.m
    Pf      = (Bf * Va + Pfinj) * baseMVA

Both solutions are then pushed through the *same* `Pf` formula, so the
comparison is on branch flows in MW and needs no mapping between MATPOWER's
branch rows and gridoxide's own flat branch index. DC angles are determined
only up to a global constant, so both angle vectors are first shifted to put
the MATPOWER reference bus at zero; that also cancels the angle drop across the
near-ideal `source` impedance `matpower_to_pgm.py` inserts, which is a pure
offset on a radially-attached virtual slack.

Usage:
    python3 check_dc_matpower.py <case> [<case> ...]
    python3 check_dc_matpower.py --all
    python3 check_dc_matpower.py --all --zero-phase-shifts

Two flags isolate *known, documented* model differences rather than excusing
errors. Both leave a real discrepancy visible when omitted.

`--zero-phase-shifts` rebuilds the reference with every branch's `angle` column
forced to 0. This is not optional in practice: `gridoxide.matpower` rounds
`angle` to the nearest 60-degree clock position, which for every shift in these
12 cases means **dropping it**, so the as-published comparison measures that
conversion loss and not the solver. It is the same flag, for the same reason, as
`check_matpower_residual.py`'s — and it bites harder here, because a phase shift
enters the DC right-hand side directly rather than as a second-order term.

`--ignore-gs` drops the `Gs/baseMVA` constant-load term from the reference's
`Pbus`. MATPOWER and pandapower fold shunt conductance in that way; gridoxide's
DC deliberately does not (see `docs/src/powerflow/dc.md`'s Scope section), so
without this flag the two differ by exactly the total `Gs` of the case.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import CASE_NAMES, matpower_filename
from matpower_to_pgm import load_mpc

MATPOWER_DIR = (
    Path(__file__).resolve().parents[2] / "tests/data/benchmark-grids/matpower"
)
CACHE_DIR = Path(__file__).resolve().parent / ".case-cache"

# MATPOWER column indices (0-based).
BUS_I, BUS_TYPE, PD, GS, VA = 0, 1, 2, 4, 8
GEN_BUS, PG, GEN_STATUS = 0, 1, 7
F_BUS, T_BUS, BR_X, RATIO, ANGLE, BR_STATUS = 0, 1, 3, 8, 9, 10
REF = 3


def build_bdc(mpc, zero_phase_shifts=False):
    """`makeBdc.m`, densely. Returns (ids, Bbus, Bf, Pbusinj, Pfinj)."""
    bus = np.asarray(mpc["bus"], dtype=float)
    branch = np.asarray(mpc["branch"], dtype=float)
    ids = bus[:, BUS_I].astype(int)
    pos = {b: i for i, b in enumerate(ids)}
    nb, nl = len(ids), len(branch)

    bf = np.zeros((nl, nb))
    cft = np.zeros((nl, nb))
    pfinj = np.zeros(nl)

    for l, br in enumerate(branch):
        status = br[BR_STATUS] if br.shape[0] > BR_STATUS else 1.0
        f, t = pos[int(br[F_BUS])], pos[int(br[T_BUS])]
        x = br[BR_X]
        if status == 0 or x == 0.0:
            # `stat ./ x` is 0 for an open branch; a zero-reactance branch has
            # no DC susceptance under this approximation at all, and MATPOWER
            # would produce an Inf here. Neither contributes.
            continue
        tap = br[RATIO] if br[RATIO] != 0 else 1.0
        b = status / (x * tap)
        shift = 0.0 if zero_phase_shifts else np.deg2rad(br[ANGLE])

        bf[l, f], bf[l, t] = b, -b
        cft[l, f], cft[l, t] = 1.0, -1.0
        pfinj[l] = b * (-shift)

    return ids, cft.T @ bf, bf, cft.T @ pfinj, pfinj


def specified_p(mpc, ids, ignore_gs=False):
    """`real(makeSbus(...)) - bus Gs/baseMVA`, per-unit, plus the bus types."""
    bus = np.asarray(mpc["bus"], dtype=float)
    gen = np.asarray(mpc["gen"], dtype=float)
    base_mva = float(np.asarray(mpc["baseMVA"]).ravel()[0])
    pos = {b: i for i, b in enumerate(ids)}

    p = -bus[:, PD] / base_mva
    if not ignore_gs:
        p -= bus[:, GS] / base_mva
    for g in gen:
        if g[GEN_STATUS] > 0:
            p[pos[int(g[GEN_BUS])]] += g[PG] / base_mva
    return p, bus[:, BUS_TYPE].astype(int), base_mva, np.deg2rad(bus[:, VA])


def reference_angles(bbus, pbus, pbusinj, bus_type, va0):
    """`dcpf.m`. Returns angles in radians, or None if the system is singular."""
    ref = np.flatnonzero(bus_type == REF)
    pvpq = np.flatnonzero(bus_type != REF)
    va = va0.copy()
    rhs = (pbus - pbusinj)[pvpq] - bbus[np.ix_(pvpq, ref)] @ va0[ref]
    try:
        va[pvpq] = np.linalg.solve(bbus[np.ix_(pvpq, pvpq)], rhs)
    except np.linalg.LinAlgError:
        return None
    return va


def gridoxide_angles(json_path, approximation="ignore_r"):
    import json

    import gridoxide

    with open(json_path) as f:
        node_ids = [n["id"] for n in json.load(f)["data"]["node"]]
    model = gridoxide.PowerFlowModel.from_pgm_json(
        str(json_path), method="dc", dc_approximation=approximation
    )
    model.solve()
    # from_pgm_json appends one virtual slack bus per source; trim it back off.
    return dict(zip(node_ids, model.voltage_ang()[: len(node_ids)]))


def check(case, zero_phase_shifts, ignore_gs, approximation):
    from matpower_to_pgm import convert

    mpc = load_mpc(MATPOWER_DIR / matpower_filename(case))
    ids, bbus, bf, pbusinj, pfinj = build_bdc(mpc, zero_phase_shifts)
    pbus, bus_type, base_mva, va0 = specified_p(mpc, ids, ignore_gs)

    va_ref = reference_angles(bbus, pbus, pbusinj, bus_type, va0)
    if va_ref is None:
        return None

    CACHE_DIR.mkdir(exist_ok=True)
    json_path = CACHE_DIR / f"{case}.json"
    if not json_path.exists():
        convert(MATPOWER_DIR / matpower_filename(case), json_path)
    va_go_by_id = gridoxide_angles(json_path, approximation)
    va_go = np.array([va_go_by_id[int(b)] for b in ids])

    # DC angles are fixed only up to a global constant. Pinning both at the
    # MATPOWER reference bus also cancels the drop across the virtual slack's
    # source impedance, which is a pure offset on a radial attachment.
    ref = int(np.flatnonzero(bus_type == REF)[0])
    va_ref = va_ref - va_ref[ref]
    va_go = va_go - va_go[ref]

    # Push both angle vectors through the *same* flow formula, so no mapping
    # between MATPOWER branch rows and gridoxide's flat branch index is needed.
    pf_ref = (bf @ va_ref + pfinj) * base_mva
    pf_go = (bf @ va_go + pfinj) * base_mva

    d_ang = np.abs(va_go - va_ref)
    d_flow = np.abs(pf_go - pf_ref)
    return {
        "buses": len(ids),
        "branches": bf.shape[0],
        "max_angle_deg": np.rad2deg(d_ang.max()),
        "worst_bus": int(ids[int(np.argmax(d_ang))]),
        "max_flow_mw": d_flow.max() if d_flow.size else 0.0,
        "peak_flow_mw": np.abs(pf_ref).max() if pf_ref.size else 0.0,
        "shifters": int(np.count_nonzero(np.asarray(mpc["branch"], dtype=float)[:, ANGLE])),
        "gs_mw": float(np.asarray(mpc["bus"], dtype=float)[:, GS].sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--zero-phase-shifts", action="store_true")
    ap.add_argument("--ignore-gs", action="store_true")
    ap.add_argument("--approximation", default="ignore_r",
                    choices=["ignore_r", "ignore_g"])
    args = ap.parse_args()

    cases = CASE_NAMES if args.all else args.cases
    if not cases:
        ap.error("name at least one case, or pass --all")

    print(f"{'case':<20}{'buses':>7}{'branch':>8}{'max Δangle':>12}"
          f"{'max Δflow':>12}{'peak flow':>12}{'PSTs':>6}{'ΣGs':>9}")
    worst = 0.0
    for case in cases:
        r = check(case, args.zero_phase_shifts, args.ignore_gs, args.approximation)
        if r is None:
            print(f"{case:<20}{'singular reference system':>50}")
            continue
        worst = max(worst, r["max_flow_mw"])
        print(f"{case:<20}{r['buses']:>7}{r['branches']:>8}"
              f"{r['max_angle_deg']:>11.2e}°{r['max_flow_mw']:>11.2e} "
              f"{r['peak_flow_mw']:>11.1f}{r['shifters']:>6}{r['gs_mw']:>9.1f}")

    print(f"\nworst branch-flow disagreement across all cases: {worst:.3e} MW")


if __name__ == "__main__":
    main()

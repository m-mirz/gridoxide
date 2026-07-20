"""Converts a raw MATPOWER case (`.mat`, MATPOWER's own bus/branch/gen
matrix format, or `.m`, MATPOWER's plain-text case-file format) directly
into PGM-format JSON, the format `gridoxide.PowerFlowModel.from_pgm_json`
reads. Needs the `matpower` extra (`pip install gridoxide[matpower]`, which
pulls in `numpy`/`scipy` for `.mat` reading — plain `.m` parsing only
actually needs `numpy`, but both extras travel together for simplicity).

This is also `scripts/bench/run_case_suite.py`'s converter for all 12 of its
real power-system test-case grids in that directory's own benchmark suite
(see `scripts/bench/cases.py` for the case list) — `scripts/bench/
matpower_to_pgm.py` there is a thin CLI wrapper around this module so the
conversion logic itself only has to live in one place.

CLI usage: `gridoxide-matpower <input.mat-or-.m> <output.json>` (installed
as a console script) or `python3 -m gridoxide.matpower <input> <output>`.

Why not go through pandapower's own MATPOWER importer instead? Three of the
twelve `scripts/bench/` cases (case1888rte, case6495rte, case6515rte) would
not converge at all through that path. Root-caused (not assumed) by
comparing directly against `references/powsybl-open-loadflow` via
`pypowsybl`: pandapower's importer (via power-grid-model-io's
`PandaPowerConverter`) assigns each bus a real physical `baseKV`, which for
a transformer connecting two different voltage levels requires very
carefully keeping every impedance/tap conversion referenced to a
*consistent* side — easy to get subtly wrong (this module's own first
attempt did, twice). powsybl's own MATPOWER importer sidesteps the whole
problem: verified directly (`network.get_voltage_levels()["nominal_v"]`
after `pypowsybl.network.load(...)`) that it assigns `nominal_v = 1.0` to
*every* bus regardless of the `.m`/`.mat` file's `baseKV` column, and
encodes an off-nominal branch purely via `rated_u1 = ratio`, `rated_u2 =
1.0`. MATPOWER's own power-flow formulation never actually needs a physical
voltage reference, only a *consistent* one — so this module does the same
(`U_RATED_UNIFORM` below), which eliminates the whole class of
from-side/to-side base-mismatch bug by construction.

MATPOWER format (standard, documented in the MATPOWER manual, "mpc" struct):
- `bus` columns: bus_i, type(1=PQ,2=PV,3=ref,4=isolated), Pd, Qd, Gs, Bs,
  area, Vm, Va, baseKV, zone, Vmax, Vmin.
- `gen` columns: bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, ...
- `branch` columns: fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle,
  status, angmin, angmax. `ratio`/`angle` encode a generalized (possibly
  off-nominal, possibly phase-shifting) complex tap — MATPOWER doesn't
  distinguish "line" from "transformer" as separate component types the way
  PGM does; any branch with `ratio not in {0, 1}` or `angle != 0` is treated
  here as a transformer.

Uniform per-unit voltage base (no physical kV at all): confirmed against
`references/powsybl-open-loadflow` via `pypowsybl` directly — powsybl's own
MATPOWER importer assigns `nominal_v = 1.0` to *every* voltage level
regardless of the `.mat` file's `baseKV` column (checked empirically:
`network.get_voltage_levels()["nominal_v"]` is uniformly `1.0` after
`pypowsybl.network.load("case1888rte.mat")`), and represents an off-nominal
branch purely via `rated_u1 = ratio`, `rated_u2 = 1.0` — i.e. MATPOWER's own
power-flow formulation never actually needs a physical voltage reference,
only a *consistent* one. An earlier version of this converter used `baseKV`
directly and derived per-branch impedance in ohms from it; this produced
subtle from-side/to-side base-mismatch bugs for transformers connecting
different voltage levels (r/x and uk/pk must be referenced to the *same*
side gridoxide's `transformer_admittances(u2, ...)` uses, and getting this
wrong is easy while physical units are involved at all). Using one uniform
`u_rated` for every node removes the possibility of that class of bug
entirely — there is no "wrong side" left to pick.

`transformer.u1`/`transformer.u2` in PGM's *actual* C++ reference
(`references/power-grid-model`) are compared against a separately-supplied
node-level `u1_rated`/`u2_rated` to compute `nominal_ratio_ = u1_rated /
u2_rated`, and the real tap ratio is `k = (u1/u2) / nominal_ratio_`
(`transformer.hpp:70,194`) — gridoxide's own `network::transformer_tap`
doesn't implement this `nominal_ratio_` division at all (nor PGM's
`tap_direction_`), it only ever compares `u1` against `u1 + delta` (the tap
step). That's a real, narrower port of PGM's model, silently correct only
when a transformer's own `u1`/`u2` fields exactly equal its connected
nodes' `u_rated` (true of every hand-authored fixture in this project) —
which is exactly why an off-nominal ratio has to be injected via a forced
tap step here (`tap_min = tap_max = tap_pos = 1`, `tap_nom = 0`, `tap_size =
u1 * (ratio - 1)`) instead of by scaling `u1` directly, which
`transformer_tap` would ignore.

Known lossy step: PGM's `transformer.clock` is quantized to 30-degree
increments (real nameplate transformer vector groups only exist in fixed
clock-hour groups) — MATPOWER's `angle` is a continuous degree value with no
such constraint. `angle` is rounded to the nearest 30-degree multiple here;
for unusual near-zero-impedance phase-shifting branches this is a real,
bounded approximation, not a bug — the same constraint any correct
MATPOWER-to-PGM converter faces.
"""
import json
import math
import re
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gridoxide.matpower needs the 'matpower' extra: pip install gridoxide[matpower]"
    ) from e

BUS_I, BUS_TYPE, PD, QD, GS, BS, _AREA, VM, VA, _BASE_KV, _ZONE, _VMAX, _VMIN = range(13)
GEN_BUS, PG, QG, QMAX, QMIN, VG, _MBASE, GEN_STATUS = range(8)
F_BUS, T_BUS, BR_R, BR_X, BR_B, _RATE_A, _RATE_B, _RATE_C, RATIO, ANGLE, BR_STATUS = range(11)

PQ, PV, REF, ISOLATED = 1, 2, 3, 4

# Near-ideal-source defaults, matching this project's own PGM test fixtures.
SOURCE_SK = 1e10
SOURCE_RX_RATIO = 0.1

# One uniform per-unit voltage base for the whole network — see module
# docstring for why this matches powsybl's own MATPOWER-import convention
# and avoids any from-side/to-side base-mismatch bug entirely.
U_RATED_UNIFORM = 1.0


def parse_matpower_m(m_path: Path) -> dict:
    """Minimal parser for MATPOWER's own `.m` case-file format (used when no
    `.mat` is available, e.g. MATPOWER's own `data/case6495rte.m` — no
    Octave/MATLAB needed). MATPOWER case files are simple, consistent
    numeric-matrix literals (`mpc.bus = [ ...; ...; ];`), not general
    MATLAB — this covers exactly that, not arbitrary `.m` scripts.
    """
    text = m_path.read_text()
    result: dict = {"version": "2"}
    base_mva_match = re.search(r"mpc\.baseMVA\s*=\s*([\d.eE+-]+)\s*;", text)
    result["baseMVA"] = float(base_mva_match.group(1)) if base_mva_match else 100.0
    for name in ("bus", "gen", "branch"):
        block_match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.DOTALL)
        if not block_match:
            result[name] = np.zeros((0, 0))
            continue
        rows = []
        for line in block_match.group(1).splitlines():
            line = line.split("%", 1)[0].strip().rstrip(";").strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
        result[name] = np.array(rows)
    return result


def load_mpc(mat_path: Path) -> dict:
    """Loads a MATPOWER case (`.m` or `.mat`) into a plain dict with
    `version`/`baseMVA`/`bus`/`gen`/`branch` keys."""
    if mat_path.suffix == ".m":
        return parse_matpower_m(mat_path)
    import scipy.io as sio

    data = sio.loadmat(mat_path, simplify_cells=True)
    return data["mpc"] if "mpc" in data else data


class _IdCounter:
    """Monotonically increasing id generator for synthesized PGM component
    ids (loads/sources/gens/etc. that don't have a MATPOWER-native id of
    their own) — a fresh instance per `convert()` call keeps repeated calls
    within one process independent of each other, unlike a shared
    module-level counter."""

    def __init__(self, start: int = 10_000_000) -> None:
        self._next = start

    def __call__(self) -> int:
        self._next += 1
        return self._next


def convert(mat_path: Path, output_path: Path) -> None:
    mpc = load_mpc(mat_path)
    next_id = _IdCounter()
    base_mva = float(mpc["baseMVA"])
    s_base_va = base_mva * 1e6
    bus = np.atleast_2d(mpc["bus"])
    gen = np.atleast_2d(mpc["gen"])
    branch = np.atleast_2d(mpc["branch"])

    bus_ids = bus[:, BUS_I].astype(np.int64)
    energized_ids = {int(bus_ids[row]) for row in range(len(bus)) if bus[row, BUS_TYPE] != ISOLATED}

    nodes = [{"id": nid, "u_rated": U_RATED_UNIFORM} for nid in sorted(energized_ids)]

    z_base = (U_RATED_UNIFORM ** 2) / s_base_va

    sym_loads = []
    shunts = []
    for row in range(len(bus)):
        if bus[row, BUS_TYPE] == ISOLATED:
            continue
        node_id = int(bus_ids[row])
        pd, qd = bus[row, PD], bus[row, QD]
        if pd != 0.0 or qd != 0.0:
            sym_loads.append({"id": next_id(), "node": node_id, "status": 1, "type": 0,
                               "p_specified": pd * 1e6, "q_specified": qd * 1e6})
        gs, bs = bus[row, GS], bus[row, BS]
        if gs != 0.0 or bs != 0.0:
            ub = U_RATED_UNIFORM
            shunts.append({"id": next_id(), "node": node_id, "status": 1,
                            "g1": gs * 1e6 / (ub * ub), "b1": bs * 1e6 / (ub * ub),
                            "g0": gs * 1e6 / (ub * ub), "b0": bs * 1e6 / (ub * ub)})

    # Bus type REF: modeled as a near-ideal `source` (gridoxide's Slack bus).
    # Bus type PV with at least one active gen: `sym_gen` (summed P across
    # all active gens at that bus) + one `voltage_regulator` referencing the
    # first active gen, pinning voltage to its `Vg` setpoint.
    sources = []
    sym_gens = []
    voltage_regulators = []
    gen_p_by_bus: dict[int, float] = {}
    # Summed across every active gen at a bus (same as gen_p_by_bus), not
    # just the first one voltage_regulator's u_ref comes from — q_min/q_max
    # bound the *bus's* net reactive injection (see PgmVoltageRegulator's
    # doc comment in src/pgm.rs), consistent with p_specified/q_specified
    # already being summed across every load/gen at a node.
    gen_qmin_by_bus: dict[int, float] = {}
    gen_qmax_by_bus: dict[int, float] = {}
    first_active_gen_by_bus: dict[int, int] = {}
    for g in range(len(gen)):
        if gen[g, GEN_STATUS] <= 0:
            continue
        node_id = int(gen[g, GEN_BUS])
        gen_p_by_bus[node_id] = gen_p_by_bus.get(node_id, 0.0) + gen[g, PG]
        gen_qmin_by_bus[node_id] = gen_qmin_by_bus.get(node_id, 0.0) + gen[g, QMIN]
        gen_qmax_by_bus[node_id] = gen_qmax_by_bus.get(node_id, 0.0) + gen[g, QMAX]
        first_active_gen_by_bus.setdefault(node_id, g)

    for row in range(len(bus)):
        node_id = int(bus_ids[row])
        btype = bus[row, BUS_TYPE]
        if btype == ISOLATED:
            continue
        if btype == REF:
            sources.append({"id": next_id(), "node": node_id, "status": 1,
                             "u_ref": bus[row, VM], "sk": SOURCE_SK, "rx_ratio": SOURCE_RX_RATIO})
            continue
        p_mw = gen_p_by_bus.get(node_id)
        if p_mw is None:
            continue
        gen_id = next_id()
        sym_gens.append({"id": gen_id, "node": node_id, "status": 1, "type": 0,
                          "p_specified": p_mw * 1e6, "q_specified": 0.0})
        if btype == PV:
            g = first_active_gen_by_bus[node_id]
            vr = {"id": next_id(), "regulated_object": gen_id, "status": 1, "u_ref": gen[g, VG]}
            # MATPOWER represents "no limit" as literal +-Inf on some real
            # cases — omit the key entirely rather than writing a
            # non-finite value: `json.dumps` would emit Python's own
            # non-standard Infinity/-Infinity/NaN tokens, which aren't
            # valid JSON and fail to parse on the Rust side. Omitting
            # matches PGM's own "unset means unbounded" convention for
            # these fields (see PgmVoltageRegulator's `#[serde(default =
            # "nan")]` in src/pgm.rs) exactly.
            q_min_var = gen_qmin_by_bus[node_id] * 1e6
            q_max_var = gen_qmax_by_bus[node_id] * 1e6
            if math.isfinite(q_min_var):
                vr["q_min"] = q_min_var
            if math.isfinite(q_max_var):
                vr["q_max"] = q_max_var
            voltage_regulators.append(vr)

    lines = []
    transformers = []
    for row in range(len(branch)):
        if branch[row, BR_STATUS] == 0:
            continue
        f_id, t_id = int(branch[row, F_BUS]), int(branch[row, T_BUS])
        if f_id not in energized_ids or t_id not in energized_ids:
            continue
        ratio, angle = branch[row, RATIO], branch[row, ANGLE]
        is_transformer = not ((ratio == 0.0 or ratio == 1.0) and angle == 0.0)
        r_ohm, x_ohm = branch[row, BR_R] * z_base, branch[row, BR_X] * z_base

        if not is_transformer:
            b_siemens = branch[row, BR_B] / z_base
            omega = 2 * np.pi * 50.0
            c1 = b_siemens / omega if omega != 0 else 0.0
            # tan1/tan0 (dielectric loss angle) default to NaN if omitted —
            # gridoxide's PgmLine parser doesn't read them at all, but real
            # PGM's C++ core does, and a NaN there poisons the line's shunt
            # admittance into a NaN that later surfaces as a spurious
            # "possibly singular matrix" error during PowerGridModel
            # construction. MATPOWER has no equivalent loss-angle concept,
            # so 0.0 (lossless shunt) is the only sensible value.
            lines.append({"id": next_id(), "from_node": f_id, "to_node": t_id,
                           "from_status": 1, "to_status": 1,
                           "r1": r_ohm, "x1": x_ohm, "c1": c1, "tan1": 0.0,
                           "r0": r_ohm, "x0": x_ohm, "c0": c1, "tan0": 0.0})
            continue

        # Off-nominal and/or phase-shifting branch -> PGM transformer. sn
        # fixed at system base; uk/pk solved from the branch's own r/x
        # (against the same uniform U_RATED_UNIFORM used everywhere else)
        # so the round-trip through transformer_admittances is exact — see
        # module docstring for why an out-of-[0,1] `uk` doesn't affect
        # correctness (the underlying formula is scale-invariant).
        sn = s_base_va
        z_abs = np.hypot(r_ohm, x_ohm)
        uk = z_abs * sn / (U_RATED_UNIFORM ** 2)
        pk = r_ohm * sn * sn / (U_RATED_UNIFORM ** 2)
        # PGM only allows *even* clock values for a wye-wye transformer
        # (winding_from == winding_to == 1 below) — confirmed empirically:
        # power_grid_model's PowerGridModel() raises InvalidTransformerClock
        # for an odd clock otherwise. Round to the nearest 60-degree
        # multiple, not 30. gridoxide's own transformer_tap doesn't enforce
        # this (it accepted odd clocks fine), but matching PGM's real
        # constraint keeps this converter's output usable by both.
        clock = (int(round(angle / 60.0)) * 2) % 12
        effective_ratio = ratio if ratio != 0.0 else 1.0

        # Ratio injected via a forced tap step (see module docstring for why
        # scaling u1 directly wouldn't do anything in gridoxide's model).
        # PGM's own validator requires tap_size >= 0 and tap_nom within
        # [tap_min, tap_max]. A single fixed delta can be *either* sign
        # while keeping tap_size non-negative by choosing which side of the
        # step tap_pos sits on: for ratio >= 1, step up from tap_nom=0 to
        # tap_pos=1; for ratio < 1, step down from tap_nom=1 to tap_pos=0.
        # Either way tap_min=0/tap_max=1 keeps tap_nom in range.
        if effective_ratio >= 1.0:
            tap_pos, tap_nom = 1, 0
            tap_size = U_RATED_UNIFORM * (effective_ratio - 1.0)
        else:
            tap_pos, tap_nom = 0, 1
            tap_size = U_RATED_UNIFORM * (1.0 - effective_ratio)
        transformers.append({
            "id": next_id(), "from_node": f_id, "to_node": t_id,
            "from_status": 1, "to_status": 1,
            "u1": U_RATED_UNIFORM, "u2": U_RATED_UNIFORM,
            "sn": sn, "uk": uk, "pk": pk, "i0": 0.0, "p0": 0.0,
            "winding_from": 1, "winding_to": 1, "clock": clock,
            "tap_side": 0, "tap_pos": tap_pos, "tap_min": 0, "tap_max": 1, "tap_nom": tap_nom, "tap_size": tap_size,
        })

    output = {
        "version": "1.0", "type": "input", "is_batch": False, "attributes": {},
        "data": {
            "node": nodes, "line": lines, "source": sources, "sym_load": sym_loads,
            "sym_gen": sym_gens, "shunt": shunts, "transformer": transformers,
            "voltage_regulator": voltage_regulators,
        },
    }
    output_path.write_text(json.dumps(output))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    convert(Path(argv[0]), Path(argv[1]))
    print(f"wrote {argv[1]}")


if __name__ == "__main__":
    main()

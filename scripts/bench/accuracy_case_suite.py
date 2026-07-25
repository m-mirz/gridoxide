#!/usr/bin/env python3
"""Cross-tool accuracy comparison on the same 12 MATPOWER cases
`run_case_suite.py` benchmarks for timing — this script instead reports how
closely each tool's converged bus voltages agree with gridoxide's own,
per bus, matched by MATPOWER bus number (not array position — see below).

Unlike the CGMES benchmark (`bench_gridoxide_cgmes.py`/`bench_pypowsybl_cgmes.py`),
none of these MATPOWER cases ship a published/authoritative reference
solution to check against (a `.m` file's `bus`/`gen` matrices are power-flow
*input*, not a solved-case snapshot) — so this uses gridoxide's own solve as
the comparison anchor instead. That's a reasonable anchor specifically
because gridoxide's five backends (scalar/block/klu/klu_native/pardiso)
already independently converge to the same voltages on every one of these
12 cases (see this directory's README, section 4) — cross-backend agreement
within one implementation, though not proof of absolute correctness, at
least means the anchor isn't an arbitrary pick.

Usage: python3 accuracy_case_suite.py [--python PYTHON] [--cache-dir DIR] [--out FILE]

Needs the same environment as `run_case_suite.py` (see this directory's
README's step 4) — the `gridoxide` extension built with `--features
python,klu` (no `pardiso`/MKL requirement here), plus `power-grid-model`,
`pypowsybl`, `pandapower`, `lightsim2grid`, and `VeraGridEngine`. Unlike
`run_case_suite.py`, this runs every tool in-process in one interpreter
(no subprocess-per-tool) since it needs each tool's full per-bus array, not
just a `mean=Xms` line to regex out of stdout — run with the `--python`
interpreter's own environment already active, not via the `--python` flag
(which `run_case_suite.py` uses to shell out; this script doesn't).

Matching buses across tools by MATPOWER bus number, not array position, is
essential: `gridoxide.matpower.convert` builds gridoxide/PGM's shared PGM
JSON with nodes in *sorted*-bus-number order (`nodes = [... for nid in
sorted(energized_ids)]`), so JSON array position already equals sorted-id
rank for both of those two tools. The other four don't share that
convention at all, but each turns out to expose the original MATPOWER bus
number somewhere, confirmed empirically per tool (not assumed):

- pypowsybl's own MATPOWER importer names each bus `"VL-<matpower id>_<k>"`
  (`k` is always `0` for these cases — one bus per voltage level).
- VeraGrid's own MATPOWER importer sets each `Bus.code` to the MATPOWER bus
  number as a string.
- pandapower's built-in `pandapower.networks.<case_name>()` grids set
  `net.bus["name"]` to the MATPOWER bus number, aligned by `net.bus.index`
  with `net.res_bus`.
- lightsim2grid's `ac_pf()` return array is index-aligned with the
  `pandapower` net it was built from (`init_from_pandapower`) — confirmed
  directly against `net.res_bus["vm_pu"]` on case14 (matches to ~1e-8), so
  it reuses the same `net.bus["name"]` mapping pandapower does.

gridoxide's own bus index needs one extra step beyond the JSON's sorted
order: `PowerFlowModel.from_pgm_json` appends one virtual Slack bus (the
`source` component's own ideal-source-behind-impedance bus, see
`src/pgm.rs`'s "Virtual Slack bus" comment) after all `n` physical buses,
so `voltage_mag()` has `n + 1` entries — the last one is trimmed off before
matching, matching `n_nodes` reported elsewhere in this suite always being
one more than the case's own real bus count.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import CASE_NAMES, matpower_filename
from matpower_to_pgm import load_mpc

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MATPOWER_DIR = REPO_ROOT / "tests" / "data" / "benchmark-grids" / "matpower"


def gridoxide_voltages(json_path: Path) -> dict[int, float]:
    import json as jsonlib

    import gridoxide

    with open(json_path) as f:
        ids = [n["id"] for n in jsonlib.load(f)["data"]["node"]]
    model = gridoxide.PowerFlowModel.from_pgm_json(str(json_path), backend="klu_native")
    model.solve()
    vm = model.voltage_mag()[: len(ids)]  # drop the trailing virtual Slack bus
    return dict(zip(ids, vm))


def pgm_voltages(json_path: Path) -> dict[int, float]:
    from power_grid_model import CalculationMethod, PowerGridModel
    from power_grid_model.utils import json_deserialize

    with open(json_path) as f:
        dataset = json_deserialize(f.read())
    model = PowerGridModel(dataset)
    result = model._calculate_power_flow(  # noqa: SLF001
        calculation_method=CalculationMethod.newton_raphson, symmetric=True, experimental_features="enabled"
    )
    ids = dataset["node"]["id"]
    u_pu = result["node"]["u_pu"]
    return dict(zip(ids.tolist(), u_pu.tolist()))


def pypowsybl_voltages(m_path: Path) -> dict[int, float]:
    import scipy.io as sio
    import pypowsybl.loadflow as lf
    import pypowsybl.network as pn

    mpc = load_mpc(m_path)
    mpc.setdefault("version", "2")
    tmp_mat = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
    sio.savemat(tmp_mat.name, {"mpc": mpc})
    network = pn.load(tmp_mat.name)
    params = lf.Parameters(
        voltage_init_mode=lf.VoltageInitMode.UNIFORM_VALUES,
        distributed_slack=False,
        use_reactive_limits=False,
        phase_shifter_regulation_on=False,
        transformer_voltage_control_on=False,
        connected_component_mode=lf.ConnectedComponentMode.MAIN,
    )
    result = lf.run_ac(network, parameters=params)
    if result[0].status != lf.ComponentStatus.CONVERGED:
        raise RuntimeError(f"pypowsybl did not converge: {result[0].status_text}")
    buses = network.get_buses()
    bus_to_matpower_id = pypowsybl_bus_to_matpower_id(network)
    out = {}
    for bus_id, v_mag in buses["v_mag"].items():
        matpower_id = bus_to_matpower_id.get(bus_id)
        if matpower_id is not None:
            out[matpower_id] = v_mag
    return out


def pypowsybl_bus_to_matpower_id(network) -> dict[str, int]:
    """Maps each pypowsybl bus id to its true MATPOWER bus number, via every
    `Line`/`TwoWindingsTransformer`'s own `"LINE-<from>-<to>"`/
    `"TWT-<from>-<to>"` id string (confirmed empirically to always encode the
    two true MATPOWER endpoint numbers, checked directly on case14 through
    case9241pegase) — **not** via the `"VL-<n>_<k>"` bus-id naming scheme
    `<n>` might suggest.

    That naming scheme is actively misleading for any bus that is only ever
    a transformer's *secondary* side: pypowsybl's own MATPOWER importer
    names a transformer's secondary-side voltage level after its *primary*
    side's bus number with an incrementing suffix, not the secondary side's
    own true bus number — confirmed directly on case14, where
    `TWT-4-7`/`TWT-4-9`/`TWT-5-6` connect `VL-4_0` to `VL-4_1`, `VL-4_0` to
    `VL-4_2`, and `VL-5_0` to `VL-5_1` respectively — i.e. `"VL-4_1"` is
    really MATPOWER bus **7**, not bus 4, and `"VL-4_2"` is bus **9**;
    naively parsing the leading number out of the bus id string itself (an
    earlier version of this function did exactly that) silently
    mismatches/overwrites entries for every such bus, incorrectly
    attributing a *different* bus's voltage to bus 4 twice over and
    dropping buses 4/7/9's own true identities down to one merged (wrong)
    entry — inflating this script's own reported gridoxide-vs-pypowsybl
    deviation with a measurement artifact, not a real solver difference."""
    mapping: dict[str, int] = {}
    for df in (network.get_lines(), network.get_2_windings_transformers()):
        for elem_id, row in df.iterrows():
            # Parallel branches between the same bus pair get a "#<k>"
            # disambiguator appended to the id's own last number (confirmed
            # on case118, e.g. "LINE-42-49#0") — strip it before parsing.
            _, from_id, to_id = elem_id.split("-")
            mapping[row["bus1_id"]] = int(from_id)
            mapping[row["bus2_id"]] = int(to_id.split("#")[0])
    return mapping


def veragrid_voltages(m_path: Path) -> dict[int, float]:
    import VeraGridEngine as vg

    grid, _logger = vg.parse_matpower_file(str(m_path))
    options = vg.PowerFlowOptions(
        solver_type=vg.SolverType.NR,
        retry_with_other_methods=False,
        distributed_slack=False,
        control_q=False,
        control_taps_modules=False,
        control_taps_phase=False,
        control_remote_voltage=False,
        verbose=0,
    )
    driver = vg.PowerFlowDriver(grid, options)
    driver.run()
    if not driver.results.converged:
        raise RuntimeError("VeraGrid did not converge")
    vm = np.abs(driver.results.voltage)
    ids = [int(b.code) for b in grid.get_buses()]
    return dict(zip(ids, vm.tolist()))


def pandapower_voltages(case_name: str):
    """Returns (voltages dict, the solved pandapower net) — the net is
    reused by `lightsim2grid_voltages` below, matching
    `bench_lightsim2grid.py`'s own construction order (`pp.runpp` before
    `init_from_pandapower`)."""
    import pandapower as pp
    import pandapower.networks as pn

    net = getattr(pn, case_name)()
    pp.runpp(net)
    ids = net.bus["name"].astype(int)
    vm = net.res_bus["vm_pu"]
    return dict(zip(ids.tolist(), vm.tolist())), net


def lightsim2grid_voltages(net) -> dict[int, float]:
    from lightsim2grid import SolverType
    from lightsim2grid.gridmodel import init_from_pandapower

    grid = init_from_pandapower(net)
    grid.change_solver(SolverType.KLU)
    n_bus = len(grid.get_bus_vn_kv())
    v_init = np.ones(n_bus, dtype=complex) * grid.get_init_vm_pu()
    v = grid.ac_pf(v_init, 20, 1e-6)
    if v.shape[0] == 0:
        raise RuntimeError("lightsim2grid diverged")
    ids = net.bus["name"].astype(int)
    return dict(zip(ids.tolist(), np.abs(v).tolist()))


def deviation_stats(ref: dict[int, float], other: dict[int, float]) -> str:
    shared = sorted(set(ref) & set(other))
    if not shared:
        return "N/A (no shared buses)"
    errs = np.array([abs(other[i] - ref[i]) / ref[i] for i in shared])
    errs.sort()
    n = len(errs)
    return f"n={n} median={errs[n // 2]:.4%} p90={errs[n * 9 // 10]:.4%} max={errs[-1]:.4%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / ".case-cache")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for case_name in CASE_NAMES:
        print(f"=== {case_name} ===", file=sys.stderr)
        m_path = MATPOWER_DIR / matpower_filename(case_name)
        json_path = args.cache_dir / f"{case_name}.json"
        if not m_path.exists() or not json_path.exists():
            print(f"  skipping (missing input — run run_case_suite.py first to populate the cache)", file=sys.stderr)
            rows.append((case_name, "N/A", "N/A", "N/A", "N/A", "N/A"))
            continue

        try:
            ref = gridoxide_voltages(json_path)
        except Exception as e:  # noqa: BLE001
            print(f"  gridoxide FAILED: {e}", file=sys.stderr)
            rows.append((case_name, f"FAILED ({e})", "-", "-", "-", "-"))
            continue

        cells = []
        for label, fn in [
            ("PGM", lambda: pgm_voltages(json_path)),
            ("pypowsybl", lambda: pypowsybl_voltages(m_path)),
            ("VeraGrid", lambda: veragrid_voltages(m_path)),
        ]:
            try:
                cells.append(deviation_stats(ref, fn()))
            except Exception as e:  # noqa: BLE001
                cells.append(f"FAILED ({e})")
            print(f"  {label}: {cells[-1]}", file=sys.stderr)

        try:
            pp_voltages, net = pandapower_voltages(case_name)
            cells.append(deviation_stats(ref, pp_voltages))
        except Exception as e:  # noqa: BLE001
            cells.append(f"FAILED ({e})")
            net = None
        print(f"  pandapower: {cells[-1]}", file=sys.stderr)

        if net is not None:
            try:
                cells.append(deviation_stats(ref, lightsim2grid_voltages(net)))
            except Exception as e:  # noqa: BLE001
                cells.append(f"FAILED ({e})")
        else:
            cells.append("N/A (pandapower net unavailable)")
        print(f"  lightsim2grid: {cells[-1]}", file=sys.stderr)

        rows.append((case_name, *cells))

    header = ["case", "PGM", "pypowsybl", "VeraGrid", "pandapower", "lightsim2grid"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)

    print()
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

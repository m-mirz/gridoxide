#!/usr/bin/env python3
"""Converts one of pandapower's bundled power-system test-case grids (see
cases.py) into PGM-format JSON, so it can be fed straight into gridoxide's
existing `examples/bench_network.rs` / `bench_pgm.py` unchanged.

Usage: python3 convert_pandapower_case.py <case_name> <output.json>

`case_name` is a function in `pandapower.networks` (e.g. "case14",
"case118" — see cases.py for the full list this project benchmarks against).

Requires `pandapower` and `power-grid-model-io` (which pulls in
`power-grid-model` itself): pip install pandapower power-grid-model-io

Two gaps are patched in after conversion, both harmless for a *symmetric*
power flow benchmark (the affected fields are only read by asymmetric/
zero-sequence calculations, which this project doesn't run against these
grids):

- `power_grid_model_io`'s `PandaPowerConverter` only emits positive-sequence
  line impedance (`r1`/`x1`) — gridoxide's PGM parser (`src/pgm.rs`) requires
  `r0`/`x0` on every line unconditionally (mirroring the hand-written PGM
  test fixtures under `tests/data/pgm/`, which always specify both). Backfill
  `r0`/`x0` from `r1`/`x1` when absent.
- pandapower's `ext_grid` doesn't carry short-circuit power (`sk`) by
  default, so the converter can't populate `source.sk`/`source.rx_ratio` —
  again unconditionally required by gridoxide's parser. Backfill with the
  same near-ideal-source values already used throughout this project's own
  PGM test fixtures (e.g. `tests/data/pgm/powerflow/symmetric/
  transmission-case/input.json`): `sk=1e10`, `rx_ratio=0.1`.

pandapower's `gen` table (as opposed to `sgen`) is PV-controlled (fixed
voltage magnitude, floating Q) — the converter represents this as a
`sym_gen` (fixed P) plus an attached `voltage_regulator` record
(`regulated_object` = the `sym_gen`'s id, `u_ref` = the voltage setpoint).
This is PGM's actual PV-bus mechanism (see PGM's own
`newton_raphson_pf_solver.hpp::set_u_ref_and_bus_types`, which assigns
`BusType::pv` to any bus with an active regulator) — not an approximation.
gridoxide's `pgm_to_buses_and_branches` (`src/pgm.rs`) now parses
`voltage_regulator` the same way, assigning `BusType::PV` and pinning
`voltage_mag` to `u_ref` for the regulated bus, so it solves these cases as
true PV buses rather than fixed-PQ approximations. Reactive power limits
(PGM's `voltage_regulator.q_min`/`q_max`) aren't enforced (no PV→PQ
switching) — out of scope for now, see `PgmVoltageRegulator`'s doc comment.

Known data-quality caveat, present in every one of the 12 lightsim2grid
cases (checked directly, not assumed): pandapower's `case14`/`case118`/etc.
loaders derive their `trafo` table from raw MATPOWER branch data by
inflating `sn_mva` (e.g. 9900 MVA for a normal-sized transformer) so that
`vk_percent` still reproduces the correct physical impedance in ohms on
the *real* system base — self-consistent for the underlying physics, but
it pushes the converted PGM `uk` field (a fraction PGM's own
`validate_input_data` requires to be in [0, 1]) up to ~20 (2000%), and
several of these transformers carry a `tap_pos` outside `[tap_min,
tap_max]` too. `power_grid_model.PowerGridModel` still happily solves this
out-of-spec data (its arithmetic is scale-invariant), and gridoxide's own
`transformer_admittances`/`transformer_tap` formulas are the same
scale-invariant arithmetic mirrored from PGM's C++ reference — but because
the input is outside PGM's own documented contract, exact voltage parity
with pandapower's/lightsim2grid's PV-bus solve on these specific real-world
cases isn't guaranteed the way it is for this project's synthetic PGM
benchmark grid (`generate_grid.py`) or its hand-authored PGM test fixtures.
Runs `power_grid_model.validation.validate_input_data` after conversion and
prints (not blocks on) whatever it finds, so this is visible per case
rather than silently assumed away. (Corroborated independently:
lightsim2grid's own `init_from_pandapower` warns "There were some Nan in
the pp_net.trafo['tap_pos'/'tap_neutral'/...], they have been replaced by
0" for these same cases — this is a real property of how pandapower's
MATPOWER-derived loaders populate `net.trafo`, not an artifact specific to
one converter.)
"""
import sys
from pathlib import Path

import pandapower as pp
import pandapower.networks as pn
from power_grid_model.utils import export_json_data
from power_grid_model.validation import validate_input_data
from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter

# Near-ideal-source defaults matching this project's own PGM test fixtures
# (see tests/data/pgm/powerflow/symmetric/transmission-case/input.json).
DEFAULT_SOURCE_SK = 1e10
DEFAULT_SOURCE_RX_RATIO = 0.1


def convert(case_name: str, output_path: Path) -> None:
    net = getattr(pn, case_name)()
    pp.runpp(net)

    converter = PandaPowerConverter()
    input_data, extra_info = converter.load_input_data(net)

    _backfill_line_zero_sequence(input_data)
    _backfill_source_sk(input_data)

    export_json_data(output_path, input_data, use_deprecated_format=False)

    errors = validate_input_data(input_data, symmetric=True)
    if errors:
        print(f"{case_name}: power_grid_model flags {len(errors)} input-validity issue(s) "
              f"(see this script's docstring for why these MATPOWER-derived cases trip PGM's "
              f"own validator):", file=sys.stderr)
        for err in errors[:10]:
            print(f"  {err}", file=sys.stderr)


def _backfill_line_zero_sequence(input_data: dict) -> None:
    lines = input_data.get("line")
    if lines is None:
        return
    for field, fallback in (("r0", "r1"), ("x0", "x1")):
        if field not in lines.dtype.names:
            continue
        missing = lines[field] != lines[field]  # NaN
        lines[field][missing] = lines[fallback][missing]


def _backfill_source_sk(input_data: dict) -> None:
    sources = input_data.get("source")
    if sources is None:
        return
    for field, default in (("sk", DEFAULT_SOURCE_SK), ("rx_ratio", DEFAULT_SOURCE_RX_RATIO)):
        if field not in sources.dtype.names:
            continue
        missing = sources[field] != sources[field]  # NaN
        sources[field][missing] = default


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], Path(sys.argv[2]))
    print(f"wrote {sys.argv[2]}")

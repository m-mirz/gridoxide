"""Shared list of real power-system test-case grids used by the lightsim2grid
comparison (matpower_to_pgm.py, bench_lightsim2grid.py, run_case_suite.py).

Mirrors the case list in lightsim2grid's own benchmark:
https://github.com/m-mirz/lightsim2grid/blob/master/benchmarks/benchmark_grid_size.py

`GBnetwork` (2,224 buses) is commented out in lightsim2grid's own list too
and is excluded here for the same reason.

Each name is also a function in `pandapower.networks` (used by
bench_lightsim2grid.py, since lightsim2grid needs a pandapower net directly)
and, via MATPOWER_FILENAMES, a case file vendored in the `benchmark-grids`
git submodule (`tests/data/benchmark-grids/matpower/`, originally from
https://github.com/m-mirz/matpower/tree/master/data — see that submodule's
own PROVENANCE.md) used by matpower_to_pgm.py — see that script's docstring
for why gridoxide's PGM JSON is derived directly from MATPOWER's own case
files rather than through pandapower's MATPOWER importer.
"""

CASE_NAMES = [
    "case14",
    "case118",
    "case_illinois200",
    "case300",
    "case1354pegase",
    "case1888rte",
    "case2848rte",
    "case2869pegase",
    "case3120sp",
    "case6495rte",
    "case6515rte",
    "case9241pegase",
]

# MATPOWER's own filename, where it differs from the case name above.
# case_illinois200 is Texas A&M's ACTIVSg200 synthetic grid, bundled in
# MATPOWER under that name instead.
MATPOWER_FILENAMES = {
    "case_illinois200": "case_ACTIVSg200",
}


def matpower_filename(case_name: str) -> str:
    return MATPOWER_FILENAMES.get(case_name, case_name) + ".m"

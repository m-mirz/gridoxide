"""Shared list of real power-system test-case grids used by the lightsim2grid
comparison (convert_pandapower_case.py, bench_lightsim2grid.py, run_case_suite.py).

Mirrors the case list in lightsim2grid's own benchmark:
https://github.com/m-mirz/lightsim2grid/blob/master/benchmarks/benchmark_grid_size.py

Each name is a function in `pandapower.networks` (`pandapower.networks.case14()`,
etc.) — these are pandapower's own bundled MATPOWER/PYPOWER-derived test cases,
not anything specific to a fork. `GBnetwork` (2,224 buses) is commented out in
lightsim2grid's own list too and is excluded here for the same reason.
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

#!/usr/bin/env python3
"""Thin CLI wrapper around `gridoxide.matpower` (see that module's own
docstring for the full conversion writeup — MATPOWER `.mat`/`.m` case ->
PGM-format JSON). The conversion logic now lives in the `gridoxide` pip
package itself (needs the `matpower` extra: `pip install
gridoxide[matpower]`) so it's usable without this benchmark suite; this
script stays around as `run_case_suite.py`'s subprocess entry point and for
`bench_pypowsybl.py`'s `load_mpc` import, both unchanged.

Usage: python3 matpower_to_pgm.py <input.mat-or-.m> <output.json>
"""
from gridoxide.matpower import convert, load_mpc, main, parse_matpower_m  # noqa: F401

if __name__ == "__main__":
    main()

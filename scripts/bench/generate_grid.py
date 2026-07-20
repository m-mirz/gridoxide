#!/usr/bin/env python3
"""Thin CLI wrapper around `gridoxide.generate_grid` (see that module's own
docstring for the full writeup — synthetic PGM-JSON grid generation, ported
from power-grid-model's C++ benchmark generator). The generator now lives
in the `gridoxide` pip package itself (no extra needed — pure stdlib) so
it's usable without this benchmark suite; this script stays around as a
familiar entry point in this directory.

Usage:
    python3 generate_grid.py <output.json> [--target-nodes N] [--seed N]
"""
from gridoxide.generate_grid import generate, main  # noqa: F401

if __name__ == "__main__":
    main()

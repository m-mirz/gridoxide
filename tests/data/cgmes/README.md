# CGMES test fixtures

`tests/cgmes_microgrid_be_test.rs` uses ENTSO-E's MicroGrid conformance test
configuration (Belgian area, "Model As Supplied"), referenced via the
`CGMES-Test-Configurations` git submodule at
`tests/data/CGMES-Test-Configurations` rather than copied into this
directory.

- **Why a submodule, not a copy**: `CGMES-Test-Configurations` is licensed
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  (NonCommercial) — a mismatch against gridoxide's own Apache-2.0 license, so
  its files are referenced rather than redistributed inside this repo's own
  tree. This mirrors how the MATPOWER benchmark cases in `scripts/bench/` are
  fetched at run time rather than committed.
- **Initializing**: `git submodule update --init tests/data/CGMES-Test-Configurations`
  (only needed for `--features cgmes` test runs — not part of a normal clone,
  and not run in default CI, same "local/manual verification" posture as the
  `klu`/`pardiso` backends).
- **Case used**: `v3.0/MicroGrid/MicroGid-BaseCase/MicroGrid-BE-MAS/` (EQ, SSH,
  TP, SV) plus `v3.0/MicroGrid/MicroGid-BaseCase/MicroGrid-BD-MAS/`'s boundary
  EQ file (needed for a few `BaseVoltage` references — e.g. the 380 kV
  boundary bus — that only resolve there, not in BE-MAS's own EQ).

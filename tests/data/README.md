# Test data

Only small fixtures live in this directory. The larger test data sets are
outsourced to git submodules, so a normal clone stays small and the tests that
need them opt in explicitly.

| Path | Kind | Size |
|---|---|---|
| `CGMES-Test-Configurations/` | submodule | ~158 MB |
| `benchmark-grids/` | submodule | ~5 MB |
| `relicapgrid/` | submodule | ~30 MB |
| `pgm/` | committed | ~3.5 MB |

The OPF tests use the pglib-opf cases in `benchmark-grids/pglib/`. The Rust
tests read JSON rather than MATLAB, so `pglib_opf.py` converts them with
`python/gridoxide/matpower.py` on first use, into `target/tmp/`. That needs a
Python with numpy: `python3`, or whatever `PYTHON` names. The script's
docstring explains why these cases and not `benchmark-grids/matpower/`.


## Submodules

- **`CGMES-Test-Configurations/`** — ENTSO-E conformance models (MicroGrid and
  friends) used by the `--features cgmes` tests. See below.
- **`benchmark-grids/`** — MATPOWER cases up to `case9241pegase`, used by the
  benchmark suite in `scripts/bench/`; the pglib-opf cases the OPF tests
  and `injection_hessian_test.rs` need; and powsybl-open-rao's networks,
  CRACs and Cucumber suite, which the UCTE, IIDM and RAO tests read through
  `tests/openrao/mod.rs` and `openrao-fixtures.txt`. See that directory's
  `README.md`.
- **`relicapgrid/`** — ENTSO-E's [ReliCapGrid](https://github.com/entsoe/relicapgrid),
  the dataset of the 2026 SV-IOP (CC BY-SA 4.0, see its `LICENSE.md`): nine IGMs, their
  boundary and common data and an assembled CGM, used by `cgmes_relicapgrid_test.rs`.
  Pinned to the commit cimoxide's `scripts/relicapgrid` results are for.

None is initialized by a plain `git clone`. CI fetches `benchmark-grids/`
but neither of the others. Pull one in when you need it:

```bash
git submodule update --init tests/data/CGMES-Test-Configurations
git submodule update --init tests/data/benchmark-grids
git submodule update --init tests/data/relicapgrid
# or all, plus anything added later:
git submodule update --init --recursive
```

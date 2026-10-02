# Test data

Only small fixtures live in this directory. The larger test data sets are
outsourced to git submodules, so a normal clone stays small and the tests that
need them opt in explicitly.

| Path | Kind | Size |
|---|---|---|
| `CGMES-Test-Configurations/` | submodule | ~158 MB |
| `benchmark-grids/` | submodule | ~5 MB |
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
  benchmark suite in `scripts/bench/`, and the pglib-opf cases the OPF tests
  and `injection_hessian_test.rs` need. See that directory's `README.md`.

Neither is initialized by a plain `git clone`. CI fetches `benchmark-grids/`
but not `CGMES-Test-Configurations/`. Pull one in when you need it:

```bash
git submodule update --init tests/data/CGMES-Test-Configurations
git submodule update --init tests/data/benchmark-grids
# or both, plus anything added later:
git submodule update --init --recursive
```

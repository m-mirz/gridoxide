# Test data

Only small fixtures live in this directory. The larger test data sets are
outsourced to git submodules, so a normal clone stays small and the tests that
need them opt in explicitly.

| Path | Kind | Size |
|---|---|---|
| `CGMES-Test-Configurations/` | submodule | ~158 MB |
| `benchmark-grids/` | submodule | ~5 MB |
| `pgm/` | committed | ~3.5 MB |


## Submodules

- **`CGMES-Test-Configurations/`** — ENTSO-E conformance models (MicroGrid and
  friends) used by the `--features cgmes` tests. See below.
- **`benchmark-grids/`** — MATPOWER cases up to `case9241pegase`, used by the
  benchmark suite in `scripts/bench/`. See that directory's `README.md`.

Neither is initialized by a plain `git clone`, and neither is fetched in default
CI. Pull one in when you need it:

```bash
git submodule update --init tests/data/CGMES-Test-Configurations
git submodule update --init tests/data/benchmark-grids
# or both, plus anything added later:
git submodule update --init --recursive
```

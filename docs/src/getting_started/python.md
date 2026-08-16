# Python Bindings

`gridoxide` ships as a pip-installable package exposing the solver to Python:

```bash
pip install gridoxide
```

Prebuilt wheels are published for Linux (x86_64), Windows, and macOS (arm64).

```python
import gridoxide

model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="klu_native")
model.solve()
print(model.voltage_mag())  # per-unit magnitude, one entry per node
print(model.voltage_ang())  # angle in radians, one entry per node
```

Grids are loaded from [power-grid-model (PGM)](https://github.com/PowerGridModel/power-grid-model)
JSON input files. `python/README.md` in the repository is the package's own PyPI landing page and
carries a full worked example of that input format.

## API

- `PowerFlowModel.from_pgm_json(path, backend="scalar", tol=1e-6, max_iter=20, s_base_va=1e6,
  freq_hz=50.0, method="newton_raphson", dc_approximation="ignore_r")` — loads a PGM JSON file and
  builds the Y-bus admittance matrix.
- `model.n_nodes` / `model.n_branches` — bus count (including one virtual slack bus per active
  `source`) and branch count (lines first, then transformers — the index every branch-keyed vector
  below uses).
- `model.solve()` — runs whichever `method` the model was built with; raises `RuntimeError` if it
  doesn't converge within `max_iter` iterations, or if a direct method's matrix is singular.
- `model.reset()` — discards the cached symbolic factorization; call before the next `solve()` if
  the topology has changed.
- `model.voltage_mag()` / `model.voltage_ang()` — per-bus results in node order.

### Linear methods

`method` selects between three solvers. `"newton_raphson"` is the default and the full nonlinear
solve; `"dc"` is real Bθ; `"linear_impedance"` is the complex constant-admittance linearization
(power-grid-model's `CalculationMethod.linear`). The two are different algorithms rather than two
spellings of one — see [DC (Bθ) Power Flow](../powerflow/dc.md) and
[The Constant-Admittance Linearization](../powerflow/linear_impedance.md).

```python
model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", method="dc")
model.solve()
flows = model.branch_flow_p()          # per-unit, at each branch's from terminal
pickup = model.dc_slack_pickup()       # [(bus_indices, p_pu)] per island
```

- `model.solve_dc()` — runs a DC solve on a model built for any method, so a Newton model can take
  a DC reading without being rebuilt.
- `model.branch_flow_p()` — per-branch active flow, per-unit. DC only; the other methods produce
  complex flows, computed from the solved voltages.
- `model.dc_slack_pickup()` / `model.dc_max_residual()` — per-island reference supply, and the
  solve's residual (round-off on a healthy network; large means `B` was ill-conditioned).
- `dc_approximation` is `"ignore_r"` (default, `b = 1/x`, what every other tool computes) or
  `"ignore_g"` (`b = x/(r²+x²)`, the exact coefficient). They differ only where `r/x` varies across
  the network — a uniform ratio makes the difference a pure rescaling of the angles.

### DC sensitivity factors (PTDF and LODF)

PTDF and LODF are properties of the topology, not of an operating point, so these need no `solve()`
first. The factorization behind them is built on first use and reused until `reset()`.

```python
model = gridoxide.PowerFlowModel.from_pgm_json("grid.json")
overloaded = model.ptdf_row(branch=12)         # sensitivity to every bus, one solve
if not model.is_radial(7):
    redistribution = model.lodf_column(7)      # who picks up branch 7's flow if it trips
```

- `model.ptdf_column(bus)` — `∂P_branch/∂P_bus` over every branch. `None` if the bus is in an
  island with no reference.
- `model.ptdf_row(branch)` — the same factors over every bus, in **one** solve rather than one per
  bus (the reduced susceptance matrix is symmetric).
- `model.lodf_column(branch)` — the fraction of that branch's flow each other branch picks up when
  it trips. `None` if the branch is radial.
- `model.is_radial(branch)` — whether removing it would disconnect the network, in which case no
  redistribution factors exist.
- `model.transfer_factors(injections)` — branch-flow response to an arbitrary per-bus injection
  pattern; the primitive the others are special cases of.
- `model.outage_flows(branch, base_flows=None)` — the N-1 primitive: flows *after* that branch
  trips, one solve and no re-solve of the network. Defaults to the last DC solve's own flows, so
  the usual call is `model.solve(); model.outage_flows(7)`.
- `model.multi_outage_flows(branches, base_flows=None)` — N-2 and N-k, for a whole set tripping at
  once. **Not** the same as calling `outage_flows` repeatedly: each single-branch factor was
  computed on the intact network, so chaining them ignores how the outages interact.
- `model.is_breaking_set(branches)` — whether removing the whole set would disconnect the network.
  Two individually non-radial lines can be jointly breaking, which is exactly what single-branch
  screening misses.

### AC sensitivity

The DC factors above are exact but blind to voltage and reactive power. `AcSensitivityModel`
differentiates a converged *AC* operating point instead — so it has a solve behind it, and the
answers are local derivatives rather than global constants. See
[AC Sensitivity Analysis](../sensitivity/ac.md) for the formulation and the trade-off.

It is a class rather than a function because there is something worth keeping between calls: one
Jacobian factorization, arbitrarily many questions.

```python
s = gridoxide.AcSensitivityModel("grid.json")

col = s.column(active_injection=2)   # bus 2 ramps — what responds?
row = s.row(branch=8)                # branch 8 is loaded — what moves it?

# Which tap has the most leverage on branch 8?
best = max(range(s.n_branches), key=lambda b: abs(row.d_transformer_ratio[b]))
```

- `AcSensitivityModel(path, s_base_va=1e6, freq_hz=50.0, tol=1e-8, max_iter=50)` — solves the power
  flow and factorizes its Jacobian. **Raises if the solve does not converge**: a derivative taken at
  a non-converged point is meaningless rather than merely imprecise.
- `s.column(...)` — the *forward* direction, one solve per variable. Pass exactly one of
  `active_injection=bus`, `reactive_injection=bus`, `transformer_ratio=branch` or
  `phase_shift=branch`, plus `terminal="from"|"to"`. Returns `d_branch_active`,
  `d_branch_reactive` (per branch) and `d_voltage_magnitude`, `d_voltage_angle` (per bus).
- `s.row(...)` — the *adjoint* direction, one solve per monitored quantity. Pass either
  `branch=` with `quantity="active"|"reactive"`, or `bus=` with `quantity="magnitude"|"angle"`.
  Returns `d_active_injection`, `d_reactive_injection` (per bus) and `d_transformer_ratio`,
  `d_phase_shift` (per branch).
- `s.n_buses`, `s.n_branches` — branch indices are lines first, then transformers.

Pick the direction by which side is small: forward when few inputs move and many outputs are
watched, adjoint when one output is watched and many inputs could move. Both cost one triangular
solve against the same factorization.

Two contracts worth knowing. A variable the solve itself determines has an identically **zero**
column — a slack bus's injection, or reactive power where the voltage is held — and that is the
correct derivative, not a gap. And a line has no tap, so its `d_transformer_ratio` and
`d_phase_shift` entries are zero rather than an error.

### DC optimal power flow

`dc_opf` answers *what should each generator produce, so demand is met at least cost without
overloading anything*. A plain function rather than a class: each solve builds its own program,
so there is nothing worth keeping between calls.

**Only present when the extension was built with the `opf` feature.** That feature needs nothing
installed — the solver is gridoxide's own interior-point method, pure Rust — so enabling it costs
only build time. Importing `gridoxide` never fails for want of a solver; `dc_opf` is simply
absent, so `hasattr(gridoxide, "dc_opf")` is the check.

```python
import gridoxide

result = gridoxide.dc_opf("grid.json")

print(result.objective)                       # total cost, $/h
for index, mw in zip(result.generator_index, result.dispatch):
    print(index, mw)

# The price spread between buses *is* the congestion.
spread = max(result.lmp) - min(result.lmp)
for b in result.binding:
    print(f"branch {b.branch} at {b.flow:.1f}/{b.rate:.1f} MW, "
          f"worth {abs(b.price):.2f} $/MWh to relieve")
```

- `dc_opf(path, data_path=None, shed_price=10000.0, allow_shedding=True,
  dc_approximation="ignore_g", solver="ipm", freq_hz=50.0)` — costs and limits come from a companion OPF
  document, defaulting to `path` with its extension replaced by `.opf.json`, which is the pair
  `gridoxide-matpower` writes. Raises if no optimal dispatch exists.
- `solver` is `"ipm"` (default), the built-in interior-point method, or `"highs"`, available only
  when the extension was also built with `opf-highs` (which links a local HiGHS). The two are
  cross-checked against each other, so this picks a dependency rather than an answer — see
  [Optimal Power Flow](../opf/index.md#why-keep-both).
- `dc_approximation` defaults to `"ignore_g"` (`b = x/(r²+x²)`), the *opposite* of
  `PowerFlowModel.from_pgm_json`'s default and deliberately so — each matches what its own
  field's reference tools compute. The choice moves which branch binds, so it is not cosmetic;
  see [Optimal Power Flow](../opf/index.md#which-susceptance--and-why-it-is-not-a-detail).
- `result.objective` — total cost, $/h.
- `result.dispatch` / `result.generator_index` — MW per generator, and the source case's own
  generator index for each, so results match back to the case file.
- `result.lmp` — **locational marginal price** per bus, $/MWh: the cost of serving one more MW
  there. Uniform when nothing is congested.
- `result.flows`, `result.angles` — per branch and per bus.
- `result.shed` — MW of unserved demand per load. All zero on a case that can be served; with
  `allow_shedding=False` such a case raises instead, which is sometimes the answer wanted.
- `result.binding` — the branches at their limit, each with `branch`, `flow`, `rate` and
  `price`.

Costs, limits and ratings are not in the PGM network document — it has nowhere to put them — so
`gridoxide-matpower` writes both files from a MATPOWER case:

```bash
python -m gridoxide.matpower case14.m case14.json    # also writes case14.opf.json
```

### Short-circuit calculation

`short_circuit` is a plain function, not a class: it is a single direct solve with nothing worth
caching between calls, because the fault's boundary conditions change the matrix itself. The fault
is declared in the document as power-grid-model's `fault` component. See
[The Short-Circuit Problem](../short_circuit/index.md).

```python
result = gridoxide.short_circuit("grid.json", scaling="max")

for fault in result.faults:
    print(fault.id, fault.i_f)          # per-phase current, amperes

# A two-phase fault clear of ground has no zero-sequence component at all —
# the sharpest available check that a result is sane.
for node in result.nodes:
    zero, positive, negative = node.sequence
```

- `short_circuit(path, scaling="max", s_base_va=1e6, freq_hz=50.0)` — `scaling` picks the IEC 60909
  voltage factor `c`: `"max"` for the largest current (what equipment ratings are sized against),
  `"min"` for the smallest (the protection-sensitivity study).
- `result.nodes` — per node: `u_pu`, `u_angle`, `u` (line-to-neutral volts), `energized`, and
  `sequence` as `(zero, positive, negative)` magnitudes.
- `result.faults` — per fault: `i_f`, `i_f_angle`, per phase.
- `result.sources` — each source's contribution, `i` and `i_angle`.

### Node-breaker topology and switches

CGMES models carry their substation switching arrangement; `topology="node_breaker"` keeps it
instead of merging it away at import, so a breaker becomes an element with an identity, a flow and a
position you can change.

```python
model = gridoxide.PowerFlowModel.from_cgmes(
    paths, topology="node_breaker", retain="busbar_adjacent"
)
model.solve()
for switch_id, mrid, kind, bus_from, bus_to, is_open, branch in model.switches():
    print(f"{kind} {mrid}: buses {bus_from}-{bus_to}, {'open' if is_open else 'closed'}")

model.set_switch(switch_id, open=True)   # no re-import
model.solve()
```

- `topology` is `"bus_branch"` (default, every switch merged — gridoxide's historical behaviour) or
  `"node_breaker"`.
- `retain` selects which switches survive as elements: `"none"` (merge them all, but still derive
  buses from `ConnectivityNode`), `"busbar_adjacent"` (the bus-breaker view) or `"all"` (the full
  node-breaker view). It is rejected with `topology="bus_branch"`, where it would mean nothing.
- `model.switches()` — one tuple per retained switch, identified by its **CGMES mRID**. `is_open` is
  the current position, so it follows `set_switch`.
- `model.set_switch(switch_id, open)` — flips a breaker and rebuilds the admittance matrix.
- `model.switch_flow_p()` — active power through each switch after a `solve()`, in `switches()` order.

Retention is what keeps this affordable. On SmallGrid, `"none"` gives 167 buses and `"all"` gives
1,369 — roughly a 16x larger system answering the same question — so keep only the switches you
intend to operate. See [DC (Bθ) Power Flow](../powerflow/dc.md) for what a switch costs once
retained.

**A switch is an ordinary branch to everything else**, which is the point of representing it this
way. Its `branch` index works with `lodf_column`, `outage_flows` and `solve_contingencies` with no
new machinery — a bus-split contingency is just a branch outage:

```python
for *_, branch in model.switches():
    if not model.is_radial(branch):
        redistribution = model.lodf_column(branch)   # who picks up its flow
```

### AC contingency screening

For the full nonlinear answer rather than the DC approximation:

```python
model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="klu_native")
results = model.solve_contingencies([[b] for b in range(model.n_branches)])
for branch, (status, vm, va) in enumerate(results):
    if status != "converged" or min(vm) < 0.9:
        print(f"outage of {branch}: {status}, min |V| = {min(vm):.3f}")
```

- `model.solve_contingencies(contingencies, threads=None)` — one entry per scenario, each a list of
  flat branch indices to take out. Returns `(status, voltage_mag, voltage_ang)` per scenario, in
  order. `status` is `"converged"`, `"max_iterations"` or `"singular"`; a contingency that leaves an
  unsolvable network is a screening *result*, so it does not raise.

The symbolic factorization is shared across every contingency that leaves the network connected —
2.0–2.7x against independent solves before any parallelism. Contingencies that sever the network
fall back to a full rebuild, and their orphaned buses come back pinned to zero rather than as a
spurious singular solve.

```python
model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", method="dc")
model.solve()
for branch in range(model.n_branches):
    after = model.outage_flows(branch)        # None if the branch is radial
    if after and max(map(abs, after)) > limit:
        print(f"outage of {branch} overloads the network")
```

There is deliberately no dense-matrix accessor here: a full PTDF on `case9241pegase` is 1.19 GB and
a full LODF 2.06 GB. The Rust API offers them (`ptdf_dense`/`lodf_dense`) with those numbers in
their doc comments.

## Reusing factorization across repeated solves

`PowerFlowModel` wraps the Rust `solver::PersistentSolver` — it *is* that API, not a reimplementation
of it — so repeated `.solve()` calls on one model reuse the cached symbolic factorization exactly as
described in [Backends and Factorization Reuse](../solvers/backends.md). Construct one model per
topology, then solve as many times as needed:

```python
model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="scalar")
for scenario in scenarios:
    apply_scenario(model, scenario)  # changes p/q values only
    model.solve()
    results.append(model.voltage_mag())
```

Call `model.reset()` if the topology itself changes between solves, not just bus values.

This is what lets the benchmark suite run its whole comparison in pure Python
(`scripts/bench/bench_gridoxide_native.py`), timing gridoxide with the same
`time.perf_counter()`-around-a-persistent-solve-object methodology every other tool there already
uses (PGM's `PowerGridModel`, lightsim2grid's `GridModel`, pandapower's `net`), rather than shelling
out to a compiled Rust binary and parsing its stdout.

## Backends available from Python

| Backend | Notes |
|---|---|
| `"scalar"` (default) | Sparse LU via `faer`, no special build requirements. |
| `"block"` | Block-structured variant (one 2×2 block per bus); faster on some topologies. |
| `"klu_native"` | From-scratch Rust translation of SuiteSparse KLU, always available in the wheel. |

Two further backends exist in the source tree but are **not** in the published wheel, since they
need extra system dependencies at build time — build from source with the matching Cargo feature:

- `"klu"` — links vendored SuiteSparse C directly (`--features python,klu`).
- `"pardiso"` — Intel oneMKL's PARDISO solver (`--features python,pardiso`, needs `MKLROOT` set).

See [Backends and Factorization Reuse](../solvers/backends.md) for what each one actually does and
how they compare.

## Building input grids

Two helpers produce or convert PGM JSON, so a working grid doesn't require hand-writing one. Both
ship in the pip package itself (they used to live only under `scripts/bench/`) and are installed as
console scripts:

- `gridoxide.generate_grid` — synthetic radial MV/LV distribution grid generator at any scale, pure
  stdlib, no extra dependencies:

  ```python
  from gridoxide.generate_grid import generate
  generate(target_nodes=2200, seed=42, out_path="grid.json")  # ~2,600 nodes
  ```

  or `gridoxide-generate-grid grid.json --target-nodes 2200 --seed 42`.

- `gridoxide.matpower` (needs `pip install gridoxide[matpower]`) — converts a raw MATPOWER `.m`/`.mat`
  case into PGM JSON:

  ```python
  from gridoxide.matpower import convert
  convert("case14.m", "case14.json")
  ```

  or `gridoxide-matpower case14.m case14.json`.

`scripts/bench/generate_grid.py` and `matpower_to_pgm.py` are thin CLI wrappers delegating to these
same package modules, so the conversion logic only lives in one place.

There is deliberately no pandapower-based converter: it would pull in the full pandapower +
power-grid-model-io dependency chain, and `gridoxide.matpower` already covers the same real-world
test-case grids straight from their original MATPOWER sources. If you already have a
`pandapower.pandapowerNet` and pandapower installed, `scripts/bench/convert_pandapower_case.py` in
the main repo is a standalone (not packaged) converter.

## How the extension is built

`src/python.rs` exposes `PersistentSolver` and PGM-JSON loading as `gridoxide._gridoxide`, a private
compiled extension module built with [maturin](https://www.maturin.rs/):

```bash
maturin develop --release --features python,klu
```

It is gated entirely behind the opt-in `python` Cargo feature and compiled by nothing else, so a
plain `cargo build`/`cargo test` never touches it — the feature must never be combined with a plain
`cargo` invocation.

This is a mixed Rust/Python maturin project (`pyproject.toml`'s `python-source = "python"` plus
`module-name = "gridoxide._gridoxide"`): pure-Python code lives in `python/gridoxide/` and ships in
the same wheel as the compiled extension, re-exported through `python/gridoxide/__init__.py` so
callers only ever write `import gridoxide`.

`python/tests/` holds a pytest suite (`scalar`/`block` only — no `klu`, matching what's published)
checked against this project's own committed PGM reference fixtures, run by
`.github/workflows/python.yml` on every push/PR. `.github/workflows/pypi.yml` builds wheels
(Linux/Windows/macOS) plus an sdist and publishes to PyPI via
[trusted publishing](https://docs.pypi.org/trusted-publishers/) on `v*` tags. The published wheel
**deliberately omits the `klu` backend** — LGPL-2.1-or-later vendored SuiteSparse source, plus a C
compiler and libclang needed on every target platform. See
[Provenance and Licensing](../reference/provenance.md).

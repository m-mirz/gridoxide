"""`GpuBatchSolver`, the batched power flow on an NVIDIA GPU, against the CPU
`solve_batch` and `solve_contingencies` from the same start.

Skipped unless the extension was built with the `gpu` feature
(`maturin develop --release --features python,gpu`, with `CUDSS_DIR` set);
it then needs a GPU. Phase 5 of plans/GPU_BATCH_PLAN.md: every scenario agrees
with the CPU to 1e-8.
"""
import math
import random
from pathlib import Path

import pytest

import gridoxide

pytestmark = pytest.mark.skipif(
    not hasattr(gridoxide, "GpuBatchSolver"), reason="built without the `gpu` feature"
)

MATPOWER = Path(__file__).resolve().parents[2] / "tests/data/benchmark-grids/matpower"
CASE1354 = MATPOWER / "case1354pegase.m"
needs_case1354 = pytest.mark.skipif(
    not CASE1354.exists(), reason="benchmark-grids submodule not checked out"
)
TOL = 1e-8


def solved(path):
    """The model of `path` with its base case solved: the start of every
    scenario on both sides."""
    model = gridoxide.PowerFlowModel.from_matpower(str(path), tol=TOL)
    model.solve()
    return model


def sweep(model, count, seed=1):
    """grid-bench's kind of sweep: a daily load curve between 0.6 and 1.0,
    5% noise per bus."""
    rng = random.Random(seed)
    _, p, q = model.bus_spec()
    loaded = [i for i in range(len(p)) if p[i] != 0.0 or q[i] != 0.0]
    scenarios = []
    for k in range(count):
        scale = 0.6 + 0.4 * (0.5 - 0.5 * math.cos(2 * math.pi * k / count))
        scenario = []
        for i in loaded:
            f = scale * (1 + 0.05 * rng.gauss(0, 1))
            scenario.append((i, p[i] * f, q[i] * f))
        scenarios.append(scenario)
    return scenarios


def assert_same_batch(gpu, cpu):
    assert len(gpu) == len(cpu)
    for s, (g, c) in enumerate(zip(gpu, cpu)):
        assert g.converged == c.converged, f"scenario {s}"
        if not c.converged:
            continue
        assert g.iterations == c.iterations, f"scenario {s}"
        assert g.max_mismatch <= TOL
        assert g.voltage_mag == pytest.approx(c.voltage_mag, abs=1e-8), f"scenario {s}"
        assert g.voltage_ang == pytest.approx(c.voltage_ang, abs=1e-8), f"scenario {s}"


@needs_case1354
def test_a_sweep_matches_the_cpu_scenario_for_scenario():
    model = solved(CASE1354)
    scenarios = sweep(model, 300)
    cpu = model.solve_batch(scenarios, start="last")
    gpu = gridoxide.GpuBatchSolver(model, capacity=300).solve_batch(scenarios)
    assert all(c.converged for c in cpu)
    assert_same_batch(gpu, cpu)


@needs_case1354
def test_voltages_come_back_whole():
    """`voltages()` is every result's voltages, scenario after scenario."""
    np = pytest.importorskip("numpy")
    model = solved(CASE1354)
    session = gridoxide.GpuBatchSolver(model, capacity=8)
    for solve in (lambda: session.solve_batch(sweep(model, 20)),
                  lambda: session.solve_contingencies([[3], [10], [40]])):
        results = solve()
        vm, va = (np.frombuffer(b).reshape(len(results), -1) for b in session.voltages())
        assert vm.tolist() == [r.voltage_mag for r in results]
        assert va.tolist() == [r.voltage_ang for r in results]


@needs_case1354
def test_a_session_solves_in_chunks_and_again():
    """More scenarios than the capacity are solved in chunks of it, the last
    one short; a second call on the same session reuses its analysis and
    gives the same answers, stored scenarios as given ones, though not to
    the bit: cuDSS is not deterministic by default, and the same call twice
    differs by up to 1e-13."""
    model = solved(CASE1354)
    scenarios = sweep(model, 250, seed=2)
    cpu = model.solve_batch(scenarios, start="last")
    session = gridoxide.GpuBatchSolver(model, capacity=64)
    assert session.capacity == 64
    first = session.solve_batch(scenarios)
    assert_same_batch(first, cpu)
    session.set_scenarios(scenarios[:10])
    again = session.solve_batch()
    for a, b in zip(again, first):
        assert a.voltage_mag == pytest.approx(b.voltage_mag, abs=1e-11)
        assert a.iterations == b.iterations


@needs_case1354
def test_every_outage_of_case1354pegase_matches_the_cpu():
    """N-1 over every branch: the connected outages on the device, those that
    split the network on the CPU, all in the order asked for."""
    model = solved(CASE1354)
    contingencies = [[b] for b in range(model.n_branches)]
    cpu = model.solve_contingencies(contingencies, start="last")
    gpu = gridoxide.GpuBatchSolver(model, capacity=1024).solve_contingencies(contingencies)
    assert len(gpu) == len(cpu)
    agree = 0
    for b, (g, c) in enumerate(zip(gpu, cpu)):
        if c.status != "converged":
            # The hopeless outages (branches 75 and 1754 run out of steps
            # on both): the device may call one diverged, never converged.
            assert g.status in ("max_iterations", "diverged", c.status), f"branch {b}"
            continue
        assert g.status == "converged", f"branch {b}"
        assert g.iterations == c.iterations, f"branch {b}"
        assert g.voltage_mag == pytest.approx(c.voltage_mag, abs=1e-8), f"branch {b}"
        assert g.voltage_ang == pytest.approx(c.voltage_ang, abs=1e-8), f"branch {b}"
        agree += 1
    assert agree >= len(cpu) - 5


@needs_case1354
def test_the_cpu_handoff_finishes_the_stragglers_alike():
    model = solved(CASE1354)
    contingencies = [[b] for b in range(0, model.n_branches, 5)]
    cpu = model.solve_contingencies(contingencies, start="last")
    gpu = gridoxide.GpuBatchSolver(model, cpu_handoff=20).solve_contingencies(contingencies)
    for b, (g, c) in enumerate(zip(gpu, cpu)):
        if c.status == "converged":
            assert g.status == "converged", f"outage {b}"
            assert g.voltage_mag == pytest.approx(c.voltage_mag, abs=1e-8), f"outage {b}"


@needs_case1354
def test_factor_reuse_converges_within_the_tolerance():
    """With `reuse_below` a scenario ends at another iterate than the CPU's,
    but within the tolerance and close to the CPU's voltages."""
    model = solved(CASE1354)
    scenarios = sweep(model, 100, seed=3)
    cpu = model.solve_batch(scenarios, start="last")
    gpu = gridoxide.GpuBatchSolver(model, reuse_below=0.1).solve_batch(scenarios)
    for s, (g, c) in enumerate(zip(gpu, cpu)):
        assert g.converged, f"scenario {s}"
        assert g.max_mismatch <= TOL
        assert g.voltage_mag == pytest.approx(c.voltage_mag, abs=1e-6), f"scenario {s}"


@needs_case1354
def test_bad_input_raises():
    model = solved(CASE1354)
    session = gridoxide.GpuBatchSolver(model, capacity=4)
    with pytest.raises(ValueError):
        session.solve_batch([[(10**6, 0.0, 0.0)]])
    with pytest.raises(ValueError):
        session.solve_contingencies([[model.n_branches]])

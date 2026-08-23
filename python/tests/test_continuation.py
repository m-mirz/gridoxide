"""The `continuation` binding.

The physics is checked in `tests/continuation_test.rs` — against a closed-form
two-bus nose — and in `tests/continuation_events_test.rs`, against a brute-force
bisection. What is checked here is that the binding reaches it, and that the
arguments mean what the docstring says.
"""

from pathlib import Path

import pytest

import gridoxide

CASE14 = Path(__file__).resolve().parent.parent.parent / "tests/data/pglib-opf/pglib_opf_case14_ieee.json"


pytestmark = pytest.mark.skipif(
    not hasattr(gridoxide, "continuation"), reason="built without the continuation binding"
)


@pytest.fixture(scope="module")
def case14():
    if not CASE14.exists():
        pytest.skip(f"missing fixture: {CASE14}")
    return str(CASE14)


def test_it_traces_a_curve_and_finds_the_limit(case14):
    curve = gridoxide.continuation(case14)

    assert curve.status == "NoseReached"
    assert curve.lambda_max is not None and curve.lambda_max > 0
    assert curve.critical_kind == "saddle_node"
    assert curve.critical_bus is not None
    assert curve.margin_mw is not None and curve.margin_mw > 0

    # The curve itself: one lambda and one voltage vector per point, starting
    # at the base case.
    assert len(curve.lambdas) == len(curve.voltage_mag) == len(curve.arclengths)
    assert curve.lambdas[0] == pytest.approx(0.0, abs=1e-9)
    assert curve.lambdas[-1] <= curve.lambda_max + 1e-6
    assert all(v > 0 for point in curve.voltage_mag for v in point)

    # The weakest-bus ranking is normalized and sorted.
    assert curve.weakest[0][0] == curve.critical_bus
    assert curve.weakest[0][1] == pytest.approx(1.0)
    assert curve.weakest == sorted(curve.weakest, key=lambda w: -w[1])


def test_reactive_limits_are_reported_and_lower_the_limit(case14):
    free = gridoxide.continuation(case14)
    limited = gridoxide.continuation(case14, enforce_q_limits=True)

    assert not free.events
    assert limited.events, "case14's machines should saturate under load"
    for event in limited.events:
        assert event.limit in ("max", "min")
        assert 0 < event.lambda_ < limited.lambda_max

    # Events come in the order they happen along the curve.
    assert [e.lambda_ for e in limited.events] == sorted(e.lambda_ for e in limited.events)
    assert limited.lambda_max < free.lambda_max


def test_a_target_lambda_stops_short(case14):
    curve = gridoxide.continuation(case14, target_lambda=0.3)
    assert curve.status == "TargetReached"
    assert curve.lambdas[-1] == pytest.approx(0.3, abs=1e-6)
    # Stopping early is not finding a limit, and the binding says so rather
    # than passing the last point off as one.
    assert curve.lambda_max is None
    assert curve.critical_bus is None


def test_the_lower_branch_is_the_low_voltage_solution(case14):
    curve = gridoxide.continuation(case14, lower_branch=True, max_steps=400)
    assert "upper" in curve.branch
    assert "lower" in curve.branch

    bus = curve.critical_bus
    upper = [v[bus] for v, b in zip(curve.voltage_mag, curve.branch) if b == "upper"]
    lower = [v[bus] for v, b in zip(curve.voltage_mag, curve.branch) if b == "lower"]
    assert min(lower) < min(upper)


def test_the_direction_changes_the_answer(case14):
    """lambda_max is a property of the loading direction, not of the network."""
    default = gridoxide.continuation(case14)
    one_bus = gridoxide.continuation(case14, buses=[13])
    assert one_bus.lambda_max != pytest.approx(default.lambda_max)


def test_bad_arguments_are_refused(case14):
    with pytest.raises(ValueError):
        gridoxide.continuation(case14, parametrization="wibble")
    with pytest.raises(ValueError):
        gridoxide.continuation(case14, target_lambda=0.3, lower_branch=True)
    with pytest.raises(ValueError):
        gridoxide.continuation(case14, buses=[9999])
    with pytest.raises(RuntimeError):
        gridoxide.continuation("/nonexistent/network.json")

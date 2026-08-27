"""Q-V curves through the binding.

The physics is gated in `tests/qv_test.rs`, against a closed-form two-bus
curve. What is checked here is that the binding reaches it and the shapes are
what the docstring promises.
"""

from pathlib import Path

import pytest

import gridoxide

ROOT = Path(__file__).resolve().parent.parent.parent
CASE14 = ROOT / "tests/data/pglib-opf/pglib_opf_case14_ieee.json"


pytestmark = pytest.mark.skipif(
    not hasattr(gridoxide.PowerFlowModel, "qv_curve"), reason="built without the qv binding"
)


@pytest.fixture(scope="module")
def model():
    if not CASE14.exists():
        pytest.skip(f"missing fixture: {CASE14}")
    return gridoxide.PowerFlowModel.from_pgm_json(str(CASE14), s_base_va=100e6)


def test_it_traces_a_curve_and_finds_a_margin(model):
    curve = model.qv_curve(13, v_min=0.20)

    assert curve["status"] == "nose_found"
    assert curve["bus"] == 13
    assert curve["nose"] is not None
    assert curve["nose"]["margin_pu"] > 0
    assert 0.2 < curve["nose"]["voltage"] < 1.1
    assert curve["nose"]["refined"] is True

    pts = curve["points"]
    assert len(pts) > 50
    # Swept downward, and every setpoint distinct.
    volts = [p["voltage"] for p in pts]
    assert volts == sorted(volts, reverse=True)
    # The nose is the minimum of what was sampled, give or take interpolation.
    assert curve["nose"]["q"] <= min(p["q"] for p in pts) + 1e-9


def test_the_curve_crosses_zero_at_the_buses_own_voltage(model):
    curve = model.qv_curve(13, v_min=0.20, step=0.005)
    base = curve["base_voltage"]
    above = [p["q"] for p in curve["points"] if p["voltage"] > base + 0.02]
    below = [p["q"] for p in curve["points"] if base - 0.20 < p["voltage"] < base - 0.02]
    assert all(q > 0 for q in above), "holding a bus above where it sits needs support"
    assert all(q < 0 for q in below), "holding it below needs absorption"


def test_a_truncated_sweep_reports_a_bound_not_a_margin(model):
    truncated = model.qv_curve(13, v_min=0.85)
    full = model.qv_curve(13, v_min=0.20)
    assert truncated["status"] == "nose_not_reached"
    assert full["status"] == "nose_found"
    assert truncated["nose"]["margin_pu"] < full["nose"]["margin_pu"]
    assert truncated["nose"]["refined"] is False


def test_the_weakest_bus_can_be_found_by_ranking(model):
    margins = {}
    for bus in range(15):
        try:
            c = model.qv_curve(bus, v_min=0.20)
        except ValueError:
            continue  # the slack bus is refused
        if c["status"] == "nose_found":
            margins[bus] = c["nose"]["margin_pu"]
    assert len(margins) >= 10
    weakest = min(margins, key=margins.get)
    assert margins[weakest] > 0
    # Bus 7 is case14's weakest by this measure — recorded, not derived.
    assert weakest == 7


def test_a_slack_bus_is_refused(model):
    """The slack already fixes its magnitude and already has a free reactive
    injection, so there is no condenser to add. Bus 14 is case14's slack."""
    with pytest.raises(ValueError):
        model.qv_curve(14)
    with pytest.raises(ValueError):
        model.qv_curve(999)

"""Transformer tap control, and the outer loops generally, through Python.

Needs the `cgmes` feature as well as `python`:
`maturin develop --features python,cgmes`. Skipped otherwise, and skipped when
the CGMES conformance submodule is not checked out — the conformity fixtures
are the only material anywhere that declares tap regulating controls.

The physics is checked in `tests/tap_control_test.rs`; what is checked here is
that the binding reaches it — including the composition that no single entry
point could express before the outer-loop layer existed.
"""
from pathlib import Path

import pytest

import gridoxide

CONFIGS = Path(__file__).resolve().parent.parent.parent / "tests/data/CGMES-Test-Configurations/v3.0"


def profiles(name):
    # `tap_positions` exists in every build; `from_cgmes` is the
    # feature-gated one, and the fixtures are the only source of tap controls.
    if not hasattr(gridoxide.PowerFlowModel, "from_cgmes"):
        pytest.skip("built without the cgmes feature")
    directory = CONFIGS / name
    if not directory.exists():
        pytest.skip(f"{directory} not found; CGMES submodule not checked out")
    return sorted(str(p) for p in directory.glob("*.xml"))


def svedala(**kwargs):
    paths = profiles("Svedala/Svedala-Merged")
    return gridoxide.PowerFlowModel.from_cgmes(paths, tol=1e-8, max_iter=30, **kwargs)


def test_a_model_reports_the_controls_it_read():
    model = svedala()
    assert model.tap_control_count == 11
    positions = model.tap_positions()
    assert len(positions) == 53, "one entry per transformer, None where there is no changer"
    assert sum(p is not None for p in positions) == 11


def test_a_plain_solve_reports_no_loop():
    model = svedala()
    model.solve()
    assert model.outer_loops() == []
    assert model.tap_controllers() == []
    assert model.q_limit_switches() == []
    assert model.slack_shift() == []


def test_tap_control_brings_every_controller_into_its_deadband():
    model = svedala()
    model.solve(control_taps=True)

    controllers = model.tap_controllers()
    assert len(controllers) == 11
    assert all(c["outcome"] == "in_deadband" for c in controllers), controllers
    # Svedala's own published solution sits 2-5% outside these deadbands, so a
    # loop that changed nothing would be the defective one — see
    # `svedalas_published_solution_violates_its_own_deadbands` in
    # `tests/tap_control_test.rs`.
    assert any(c["final_position"] != c["initial_position"] for c in controllers)

    for c in controllers:
        assert c["id"], "every controller names the element it came from"
        assert 0 <= c["transformer"] < 53
        assert c["direction_changes"] < 3, "a hunting controller should have said so"

    names = [name for name, _, _ in model.outer_loops()]
    assert names == ["PhaseControl", "TransformerVoltageControl"]


def test_all_three_controls_compose():
    """The capability the outer-loop layer exists for, from Python."""
    model = svedala()
    model.solve(control_taps=True, enforce_q_limits=True, distribute_slack=True)

    names = [name for name, _, _ in model.outer_loops()]
    assert names == [
        "DistributedSlack",
        "ReactiveLimits",
        "PhaseControl",
        "TransformerVoltageControl",
    ], "innermost first: slack, then reactive limits, then the tap controls"
    assert all(converged for _, _, converged in model.outer_loops())

    assert model.q_limit_switches(), "this network does saturate a reactive limit"
    shift = model.slack_shift()
    assert len(shift) == model.n_nodes
    assert abs(sum(shift)) > 0, "something should have been distributed"
    assert all(c["outcome"] == "in_deadband" for c in model.tap_controllers())


def test_tap_positions_reflect_what_the_loop_did():
    before = svedala()
    after = svedala()
    after.solve(control_taps=True)
    assert before.tap_positions() != after.tap_positions()

    # And the reported controller positions agree with the position list.
    positions = after.tap_positions()
    for c in after.tap_controllers():
        assert positions[c["transformer"]] == c["final_position"]


def test_control_taps_is_a_no_op_without_controls():
    paths = profiles("SmallGrid/SmallGrid-Merged")
    model = gridoxide.PowerFlowModel.from_cgmes(paths, tol=1e-8, max_iter=30)
    assert model.tap_control_count == 0
    before = model.tap_positions()
    model.solve(control_taps=True)
    assert model.tap_controllers() == []
    assert model.tap_positions() == before, "no control, so no movement"


def test_a_solve_without_loops_clears_a_previous_report():
    model = svedala()
    model.solve(control_taps=True)
    assert model.tap_controllers()
    model.solve()
    assert model.tap_controllers() == [], "a stale report must not be served"

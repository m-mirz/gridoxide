"""Node-breaker topology through the Python bindings.

Needs the `cgmes` feature as well as `python`:
`maturin develop --release --features python,cgmes`. Skipped otherwise, and
skipped when the CGMES conformance submodule is not checked out.

The point of these is that a switch is *addressable*: identified by its own
mRID, reporting its own flow, and openable without rebuilding the model. Every
calculation underneath already treats it as an ordinary branch — see
`tests/cgmes_node_breaker_solve_test.rs` — so what is checked here is the
binding, not the physics.
"""
from pathlib import Path

import pytest

import gridoxide

MINIGRID = (
    Path(__file__).resolve().parent.parent.parent
    / "tests/data/CGMES-Test-Configurations/v3.0/MiniGrid/MiniGrid-Merged"
)


def profiles():
    """The MiniGrid profile set, skipping unless both the feature and the
    fixtures are present."""
    if not hasattr(gridoxide.PowerFlowModel, "switches"):
        pytest.skip("built without the cgmes feature")
    if not MINIGRID.exists():
        pytest.skip(f"{MINIGRID} not found; CGMES submodule not checked out")
    paths = [str(MINIGRID / f"MiniGrid_{p}.xml") for p in ("EQBD", "EQ", "SSH", "TP", "SV")]
    return [p for p in paths if Path(p).exists()]


def node_breaker(retain="busbar_adjacent", **kwargs):
    # `profiles()` first, and into a variable: Python resolves the attribute
    # before evaluating the arguments, so an inline `from_cgmes(profiles())`
    # raises AttributeError without the feature instead of skipping.
    paths = profiles()
    return gridoxide.PowerFlowModel.from_cgmes(
        paths, topology="node_breaker", retain=retain, **kwargs
    )


def test_switches_are_reported_with_identity_and_state():
    model = node_breaker()
    switches = model.switches()
    assert len(switches) >= 20, f"only {len(switches)} switches retained"

    ids = set()
    for switch_id, label, kind, bus_from, bus_to, is_open, branch in switches:
        # A switch is identified by its source mRID, not by our index.
        assert label.startswith("_"), f"expected a CGMES mRID, got {label!r}"
        assert kind in {
            "Generic", "Breaker", "Disconnector", "LoadBreakSwitch",
            "DisconnectingCircuitBreaker", "GroundDisconnector", "Jumper",
            "Cut", "Fuse",
        }, kind
        assert bus_from != bus_to, "a degenerate switch should not be listed"
        assert isinstance(is_open, bool)
        assert 0 <= branch < model.n_branches
        ids.add(switch_id)

    assert len(ids) == len(switches), "switch ids are not distinct"
    # MiniGrid is all breakers and disconnectors, and every switch is closed.
    assert {s[2] for s in switches} <= {"Breaker", "Disconnector"}
    assert not any(s[5] for s in switches), "MiniGrid has no open switches"


def test_a_bus_branch_model_reports_no_switches():
    paths = profiles()
    model = gridoxide.PowerFlowModel.from_cgmes(paths)
    assert model.switches() == []
    with pytest.raises(RuntimeError, match="not node-breaker"):
        model.set_switch(0, True)
    with pytest.raises(RuntimeError, match="not node-breaker"):
        model.switch_flow_p()


def test_retention_policy_changes_how_much_survives():
    counts = {}
    for retain in ("none", "busbar_adjacent", "all"):
        model = node_breaker(retain=retain)
        counts[retain] = (len(model.switches()), model.n_nodes)

    assert counts["none"][0] == 0, "MergeAll must retain nothing"
    assert counts["none"][1] < counts["busbar_adjacent"][1] < counts["all"][1]
    assert counts["busbar_adjacent"][0] < counts["all"][0]

    with pytest.raises(ValueError, match="unknown retain"):
        node_breaker(retain="nonsense")
    paths = profiles()
    with pytest.raises(ValueError, match="unknown topology"):
        gridoxide.PowerFlowModel.from_cgmes(paths, topology="nonsense")
    # `retain` is meaningless without a node-breaker view, and says so rather
    # than being silently ignored.
    with pytest.raises(ValueError, match="only applies"):
        gridoxide.PowerFlowModel.from_cgmes(paths, retain="all")


def test_switch_flows_are_reported_after_a_solve():
    model = node_breaker()
    model.solve()

    flows = model.switch_flow_p()
    assert len(flows) == len(model.switches())
    assert all(f == f for f in flows), "a switch flow came back NaN"
    assert any(abs(f) > 1e-9 for f in flows), "every switch reported zero flow"


def test_opening_a_switch_changes_the_answer_and_can_be_undone():
    model = node_breaker()
    model.solve()
    before = model.voltage_ang()

    # Pick the switch carrying the most power, so opening it must matter.
    flows = model.switch_flow_p()
    worst = max(range(len(flows)), key=lambda i: abs(flows[i]))
    switch_id = model.switches()[worst][0]
    assert abs(flows[worst]) > 1e-6

    model.set_switch(switch_id, True)
    assert model.switches()[worst][5] is True
    model.solve()
    after = model.voltage_ang()
    assert max(abs(a - b) for a, b in zip(before, after)) > 1e-9, (
        "opening a load-carrying switch changed nothing"
    )

    model.set_switch(switch_id, False)
    model.solve()
    restored = model.voltage_ang()
    assert max(abs(a - b) for a, b in zip(before, restored)) < 1e-9


def test_a_switch_is_an_ordinary_branch_to_the_sensitivities():
    """The payoff of representing a switch as a branch: LODF at its branch
    index *is* its bus-split distribution factor, with no new machinery."""
    model = node_breaker()
    model.solve()

    with_factors = 0
    for *_, branch in model.switches():
        column = model.lodf_column(branch)
        # A radial switch has no redistribution factors — opening it islands
        # whatever sits behind it, which is a result, not a failure.
        assert (column is None) == model.is_radial(branch)
        if column is not None:
            assert column[branch] == -1.0
            with_factors += 1

    assert with_factors > 0, "no switch has redistribution factors"


def test_set_switch_rejects_an_unknown_switch():
    model = node_breaker()
    with pytest.raises(ValueError, match="not retained"):
        model.set_switch(10_000, True)

"""Tests for the `StateEstimationModel` bindings (src/python.rs).

Build first: `maturin develop --release --features python`.

Uses the committed power-grid-model state-estimation fixtures under
`tests/data/pgm/state_estimation/`, so these check the same answers the Rust
integration tests do, reached through the Python surface. `1os2msr` is
power-grid-model's canonical worked example; `node-injection-sensor-and-zero-injection`
is its conflicting-sensor case, which exists to be rejected.
"""
import json
from pathlib import Path

import pytest

import gridoxide

FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests/data/pgm/state_estimation"


def expected_nodes(name):
    """`{node id: (u_pu, u_angle)}` from a fixture's published output."""
    data = json.loads((FIXTURES / name / "sym_output.json").read_text())["data"]
    return {n["id"]: (n["u_pu"], n["u_angle"]) for n in data["node"]}


def model(name, **kwargs):
    return gridoxide.StateEstimationModel.from_pgm_json(
        str(FIXTURES / name / "input.json"), **kwargs
    )


def test_estimate_matches_pgm():
    """The canonical fixture, end to end through the bindings.

    Its voltage sensors carry angles, so the phase is determined by the data
    and absolute angles are comparable — see the Rust test's module comment for
    why that is not true of every fixture.
    """
    m = model("1os2msr")
    assert m.n_measurements == 20
    m.solve()

    vm, va = m.voltage_mag(), m.voltage_ang()
    # Node ids sort to gridoxide's 0-based indices in order, and the physical
    # nodes come before the virtual slack bus gridoxide adds per source.
    for idx, (u_pu, u_angle) in enumerate(expected_nodes("1os2msr").values()):
        assert vm[idx] == pytest.approx(u_pu, abs=1e-6)
        assert va[idx] == pytest.approx(u_angle, abs=1e-6)


def test_residuals_and_objective_are_available_after_solve():
    m = model("1os2msr")
    m.solve()
    assert len(m.residuals()) == m.n_measurements
    # The fixture's readings were generated from the true state without noise,
    # so a correct estimate reproduces them and the objective collapses.
    assert m.objective == pytest.approx(0.0, abs=1e-12)


def test_backends_agree():
    """The gain matrix is an ordinary sparse system, so the backend choice must
    not change the answer."""
    answers = []
    for backend in ("scalar", "block", "klu_native"):
        m = model("1os2msr", backend=backend)
        m.solve()
        answers.append(m.voltage_mag())
    for other in answers[1:]:
        assert other == pytest.approx(answers[0], abs=1e-9)


def test_iterative_linear_method_agrees():
    """power-grid-model's default method, reached through the bindings.

    It converges linearly rather than quadratically, hence the larger iteration
    budget — trading more iterations for much cheaper ones is the point of it.
    """
    newton = model("transmission-case")
    newton.solve()

    linear = model("transmission-case", method="iterative_linear", max_iter=100)
    linear.solve()

    assert linear.voltage_mag() == pytest.approx(newton.voltage_mag(), abs=1e-6)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        model("1os2msr", method="bogus")


def test_bad_data_rejects_the_conflicting_sensor():
    """A 0.1 p.u. injection sensor on a node with nothing attached to it.

    The zero-injection constraint holds the state at the truth, so the whole
    conflict lands in that measurement's residual — and the chi-squared test
    should say so rather than the estimate quietly absorbing it.
    """
    m = model("node-injection-sensor-and-zero-injection")
    m.solve()

    chi_squared, dof, p_value, suspects = m.bad_data()
    assert chi_squared > 1e3, chi_squared
    assert dof > 0
    assert p_value < 0.05, "a 100-sigma conflict must be rejected"
    assert suspects, "the culprit should be identified"
    _, normalized = suspects[0]
    assert normalized > 3.0


def test_bad_data_is_quiet_on_consistent_data():
    m = model("1os2msr")
    m.solve()
    _, _, p_value, _ = m.bad_data()
    assert p_value > 0.05, "consistent data must not be flagged"


def test_bad_data_requires_solve_first():
    m = model("1os2msr")
    with pytest.raises(RuntimeError):
        m.bad_data()


def test_observability_reports_only_synthesized_buses():
    """Anything undetermined must be one of gridoxide's own synthesized buses.

    A source's virtual slack bus is unobservable whenever the source's own
    power is unmeasured — that is a property of gridoxide's network model, not
    of the fixture. A *physical* node appearing here would mean the estimate
    cannot determine something power-grid-model evidently does.
    """
    m = model("transmission-case")
    m.solve()
    n_physical = len(expected_nodes("transmission-case"))
    for bus, quantity in m.observability():
        assert bus >= n_physical, f"physical bus {bus} ({quantity}) is unobservable"


def test_document_without_sensors_is_rejected():
    """A power-flow document has nothing to estimate, and saying so beats
    returning an empty answer."""
    power_flow_input = (
        Path(__file__).resolve().parent.parent.parent
        / "tests/data/pgm/powerflow/symmetric/line/input.json"
    )
    with pytest.raises(ValueError, match="no usable sensors"):
        gridoxide.StateEstimationModel.from_pgm_json(str(power_flow_input))

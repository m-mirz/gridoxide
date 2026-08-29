"""The `dynamics` binding.

The physics is checked in the Rust suite — against closed forms in
`tests/dynamics_test.rs` and `tests/dynamics_events_test.rs`, against a
finite-difference oracle in `tests/dynamics_models_test.rs`, and against
Dynawo's own published answer in `tests/dynamics_reference_test.rs`. What is
checked here is that the binding reaches it, and that the arguments mean what
the docstring says.
"""

from pathlib import Path

import pytest

import gridoxide

CASE = Path(__file__).resolve().parent.parent.parent / "tests/data/dynamics/smib.json"

pytestmark = pytest.mark.skipif(
    not hasattr(gridoxide, "dynamics"), reason="built without the dynamics binding"
)


@pytest.fixture(scope="module")
def result():
    return gridoxide.dynamics(str(CASE), stop=6.0, step=0.005)


def test_it_runs_the_documents_own_schedule(result):
    assert result.completed()
    assert result.status == "completed"
    assert result.events_applied == 3
    assert result.warnings == []
    assert result.steps > 1000
    # Every device state, plus a magnitude and an angle per bus.
    assert len(result.names) == 13 + 2 * 3
    assert len(result.time) == len(result.values())


def test_observables_are_named_by_machine_and_bus(result):
    assert "G1.delta" in result.names
    assert "G1.omega" in result.names
    assert "G1.efd" in result.names, "the exciter's states are part of the unit"
    assert "bus0.vmag" in result.names
    assert "bus1.vang" in result.names

    with pytest.raises(ValueError, match="no observable named"):
        result.series("G1.nonsense")


def test_the_machine_survives_the_documents_fault(result):
    omega = result.series("G1.omega")
    assert len(omega) == len(result.time)
    assert omega[0] == pytest.approx(1.0, abs=1e-9)
    # A hundred-millisecond fault on this case is survivable, so the speed
    # stays within a per cent of synchronous throughout.
    assert max(abs(w - 1.0) for w in omega) < 0.02

    # And the fault is visible: the faulted bus collapses.
    assert min(result.series("bus1.vmag")) < 1e-3


def test_events_override_the_documents_schedule():
    """A caller sweeping clearing times replaces the schedule, in the same
    vocabulary the document uses rather than a second one."""
    brief = gridoxide.dynamics(
        str(CASE),
        stop=4.0,
        events=[
            {"kind": "bus_fault", "t": 1.0, "bus": 1},
            {"kind": "clear_fault", "t": 1.05, "bus": 1},
        ],
    )
    long = gridoxide.dynamics(
        str(CASE),
        stop=4.0,
        events=[
            {"kind": "bus_fault", "t": 1.0, "bus": 1},
            {"kind": "clear_fault", "t": 1.25, "bus": 1},
        ],
    )
    assert brief.events_applied == 2
    assert long.events_applied == 2

    swing = lambda r: max(r.series("G1.delta")) - min(r.series("G1.delta"))  # noqa: E731
    assert swing(long) > swing(brief), "a longer fault must swing the rotor further"


def test_an_empty_schedule_leaves_the_case_at_rest():
    """The equilibrium invariant, through the binding: with nothing scheduled,
    nothing moves. This is the check that would catch an initialization
    mistake, and it is worth having on this side of the boundary too."""
    still = gridoxide.dynamics(str(CASE), stop=5.0, step=0.01, events=[])
    assert still.events_applied == 0
    for name in still.names:
        # An islanded rotor angle may drift with the system frequency; this
        # case has a fixed bus, so nothing may.
        series = still.series(name)
        assert max(abs(v - series[0]) for v in series) < 1e-9, name


def test_the_backend_is_a_performance_choice_not_an_answer():
    scalar = gridoxide.dynamics(str(CASE), stop=3.0, backend="scalar")
    klu = gridoxide.dynamics(str(CASE), stop=3.0, backend="klu_native")
    a, b = scalar.series("G1.delta"), klu.series("G1.delta")
    assert len(a) == len(b)
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9

    with pytest.raises(ValueError, match="backend must be"):
        gridoxide.dynamics(str(CASE), backend="quantum")


def test_a_bad_document_is_a_value_error():
    with pytest.raises(ValueError):
        gridoxide.dynamics("/nonexistent/case.json")


def test_the_machine_formulation_is_selectable():
    """`speed_voltages` chooses between the classical RMS approximation and
    Dynawo's fuller form. It agrees exactly at synchronous speed and parts in
    proportion to the speed deviation, so an undisturbed run must be identical
    and a disturbed one must not."""
    still_a = gridoxide.dynamics(str(CASE), stop=2.0, events=[], speed_voltages=False)
    still_b = gridoxide.dynamics(str(CASE), stop=2.0, events=[], speed_voltages=True)
    assert still_a.series("G1.delta") == still_b.series("G1.delta")

    fault = [
        {"kind": "bus_fault", "t": 1.0, "bus": 1},
        {"kind": "clear_fault", "t": 1.1, "bus": 1},
    ]
    approx = gridoxide.dynamics(str(CASE), stop=4.0, events=fault, speed_voltages=False)
    full = gridoxide.dynamics(str(CASE), stop=4.0, events=fault, speed_voltages=True)
    gap = max(abs(a - b) for a, b in zip(approx.series("G1.delta"), full.series("G1.delta")))
    assert gap > 1e-5, "the argument must reach the machines"
    assert gap < 0.1, "but it is a per-cent effect on the swing, not a different model"

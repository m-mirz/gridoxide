"""The `gridoxide.AcSensitivityModel` binding.

The numerics are validated against a finite-difference re-solve in
`tests/ac_sensitivity_test.rs`. What is checked here is that the binding
exposes them faithfully — right shapes, right index spaces, and the two
directions agreeing — plus the physical identities a caller would sanity-check
against.
"""

import os

import pytest

import gridoxide

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests",
    "data",
    "pgm",
    "powerflow",
    "symmetric",
)


def fixture(name):
    return os.path.join(FIXTURES, name, "input.json")


@pytest.fixture(scope="module")
def model():
    return gridoxide.AcSensitivityModel(fixture("distribution-case"))


def test_shapes_follow_the_network(model):
    assert model.n_buses == 9
    assert model.n_branches == 10

    col = model.column(active_injection=2)
    assert len(col.d_branch_active) == model.n_branches
    assert len(col.d_branch_reactive) == model.n_branches
    assert len(col.d_voltage_magnitude) == model.n_buses
    assert len(col.d_voltage_angle) == model.n_buses

    row = model.row(branch=8)
    assert len(row.d_active_injection) == model.n_buses
    assert len(row.d_transformer_ratio) == model.n_branches


def test_forward_and_adjoint_agree(model):
    """The same derivative, bracketed the other way — so they must match to
    round-off, not merely to plotting accuracy."""
    for branch in range(model.n_branches):
        row = model.row(branch=branch)
        for bus in (0, 2, 5, 8):
            forward = model.column(active_injection=bus).d_branch_active[branch]
            assert row.d_active_injection[bus] == pytest.approx(forward, abs=1e-9)


def test_tap_sensitivities_are_zero_for_lines_and_nonzero_for_transformers(model):
    row = model.row(branch=8)
    # This fixture is 8 lines then 2 transformers.
    for line in range(8):
        assert row.d_transformer_ratio[line] == 0.0
        assert row.d_phase_shift[line] == 0.0
    assert any(abs(row.d_transformer_ratio[t]) > 1e-6 for t in (8, 9))
    assert any(abs(row.d_phase_shift[t]) > 1e-6 for t in (8, 9))


def test_a_slack_bus_injection_moves_nothing(model):
    """A slack bus's injection is the solve's own output, so its column is
    identically zero — the same contract a DC PTDF column has at its reference.

    The slack here is not a bus from the document: converting PGM input appends
    one *virtual* slack bus per source after the physical nodes, so it is the
    last index. Asserting that rather than assuming bus 0 is what makes this a
    check on the contract instead of on the fixture's ordering.
    """
    reference = model.n_buses - 1
    col = model.column(active_injection=reference)
    assert all(v == 0.0 for v in col.d_voltage_angle)
    assert all(v == 0.0 for v in col.d_voltage_magnitude)
    assert all(v == 0.0 for v in col.d_branch_active)

    # And it is the only such bus — every other injection moves something.
    for bus in range(reference):
        other = model.column(active_injection=bus)
        assert any(abs(v) > 1e-9 for v in other.d_voltage_angle), bus


def test_the_two_terminals_differ(model):
    """A branch's two ends do not carry the same flow — losses sit between them
    — so a sensitivity read at each end should differ."""
    a = model.column(active_injection=2, terminal="from").d_branch_active
    b = model.column(active_injection=2, terminal="to").d_branch_active
    assert a != b


def test_voltage_functions_are_available(model):
    row = model.row(bus=5, quantity="magnitude")
    forward = model.column(active_injection=2).d_voltage_magnitude[5]
    assert row.d_active_injection[2] == pytest.approx(forward, abs=1e-9)

    row = model.row(bus=5, quantity="angle")
    forward = model.column(active_injection=2).d_voltage_angle[5]
    assert row.d_active_injection[2] == pytest.approx(forward, abs=1e-9)


def test_exactly_one_variable_is_required(model):
    with pytest.raises(ValueError, match="exactly one"):
        model.column()
    with pytest.raises(ValueError, match="exactly one"):
        model.column(active_injection=1, reactive_injection=2)
    with pytest.raises(ValueError, match="exactly one"):
        model.row()
    with pytest.raises(ValueError, match="exactly one"):
        model.row(branch=1, bus=2)


def test_bad_arguments_are_rejected(model):
    with pytest.raises(ValueError, match="terminal"):
        model.column(active_injection=1, terminal="sideways")
    with pytest.raises(ValueError, match="quantity"):
        model.row(branch=1, quantity="sideways")
    with pytest.raises(ValueError):
        model.column(active_injection=999)


def test_a_missing_file_is_reported_clearly():
    with pytest.raises(RuntimeError, match="reading"):
        gridoxide.AcSensitivityModel(fixture("no-such-case"))

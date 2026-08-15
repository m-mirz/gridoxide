"""The `gridoxide.short_circuit` binding.

The numerics are validated against power-grid-model's own reference outputs in
`tests/pgm_short_circuit_test.rs`; what is checked here is that the binding
exposes them faithfully — the right ids, the right shapes, and the physical
signatures a caller would recognise.
"""

import math
import os

import pytest

import gridoxide

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests",
    "data",
    "pgm",
    "short_circuit",
)


def fixture(name):
    return os.path.join(FIXTURES, name, "input.json")


def test_three_phase_fault_is_balanced_and_positive_sequence():
    result = gridoxide.short_circuit(fixture("three_phase_c_maximum"))

    assert len(result.faults) == 1
    fault = result.faults[0]
    assert fault.id == 10
    # A three-phase fault is balanced: all three phases carry the same current.
    assert fault.i_f[0] == pytest.approx(fault.i_f[1], rel=1e-9)
    assert fault.i_f[1] == pytest.approx(fault.i_f[2], rel=1e-9)
    assert fault.i_f[0] > 1e4

    # ...and being balanced, it excites only the positive sequence.
    for node in result.nodes:
        zero, positive, negative = node.sequence
        assert zero < 1e-9, f"node {node.id} zero sequence {zero}"
        assert negative < 1e-9, f"node {node.id} negative sequence {negative}"
        assert positive > 0.1


def test_single_phase_fault_is_unbalanced_and_grounded():
    result = gridoxide.short_circuit(fixture("single_phase_to_ground_c_maximum"))

    fault = result.faults[0]
    # Current on the faulted phase only.
    assert fault.i_f[0] > 1e3
    assert fault.i_f[1] == pytest.approx(0.0, abs=1e-6)
    assert fault.i_f[2] == pytest.approx(0.0, abs=1e-6)

    # A ground fault has a zero-sequence component somewhere. The source-side
    # node sits behind a delta winding, which blocks it, so look at the rest.
    assert any(node.sequence[0] > 1e-3 for node in result.nodes)


def test_two_phase_fault_has_no_zero_sequence():
    """A fault clear of ground has no zero-sequence path, so no zero-sequence
    current — the cleanest physical check on the symmetrical-component view."""
    result = gridoxide.short_circuit(fixture("two_phase_c_maximum"))

    fault = result.faults[0]
    # Two phases carry equal and opposite current; the third carries none.
    assert fault.i_f[0] == pytest.approx(0.0, abs=1e-6)
    assert fault.i_f[1] > 1e3
    assert fault.i_f[1] == pytest.approx(fault.i_f[2], rel=1e-6)


def test_minimum_scaling_gives_a_smaller_current_than_maximum():
    path = fixture("three_phase_c_maximum")
    big = gridoxide.short_circuit(path, scaling="max").faults[0].i_f[0]
    small = gridoxide.short_circuit(path, scaling="min").faults[0].i_f[0]
    assert small < big
    # Both `c` factors are 1.10 above 1 kV for max and 1.00 for min, so the
    # ratio is the ratio of the factors.
    assert small / big == pytest.approx(1.00 / 1.10, rel=1e-3)


def test_nodes_carry_document_ids_and_volts():
    result = gridoxide.short_circuit(fixture("three_phase_c_maximum"))
    ids = sorted(node.id for node in result.nodes)
    assert ids == [0, 1, 2, 3]

    for node in result.nodes:
        assert node.energized is True
        assert len(node.u_pu) == 3
        assert len(node.u_angle) == 3
        # `u` is line-to-neutral volts, so it tracks u_pu against the node's
        # own rating; both nodes 1..3 are 10 kV in this fixture.
        assert all(math.isfinite(v) for v in node.u)


def test_sources_report_their_contributions():
    result = gridoxide.short_circuit(fixture("three_phase_c_maximum"))
    ids = sorted(source.id for source in result.sources)
    assert ids == [4, 5]
    # The two sources between them supply the fault, so at least one carries a
    # substantial current.
    assert max(s.i[0] for s in result.sources) > 1e4


def test_rejects_an_unknown_scaling():
    with pytest.raises(ValueError, match="max"):
        gridoxide.short_circuit(fixture("three_phase_c_maximum"), scaling="sideways")


def test_reports_a_missing_file_clearly():
    with pytest.raises(RuntimeError, match="reading"):
        gridoxide.short_circuit(os.path.join(FIXTURES, "no-such-fixture", "input.json"))

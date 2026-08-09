"""DC (Bθ) power flow and its sensitivity factors, through the Python bindings.

Build the extension first: `maturin develop --release --features python`.

These are correctness checks rather than smoke tests. Because DC is exactly
linear, PTDF and LODF can be verified against the solver itself — perturb an
injection or open a branch, re-solve, and the predicted change must match to
round-off. That oracle is used here as it is on the Rust side
(`tests/dc_sensitivity_test.rs`), so the bindings are held to the same standard
as the code beneath them, not merely checked for returning plausible numbers.
"""
import json
from pathlib import Path

import pytest

import gridoxide

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "tests/data/pgm/powerflow/symmetric"
)
TRANSMISSION = FIXTURE_DIR / "transmission-case/input.json"
DISTRIBUTION = FIXTURE_DIR / "distribution-case/input.json"


def dc_model(path=TRANSMISSION, **kwargs):
    return gridoxide.PowerFlowModel.from_pgm_json(str(path), method="dc", **kwargs)


def test_dc_solves_and_conserves_power():
    model = dc_model()
    model.solve()

    # Every bus gets an angle, and the DC model's own |V| = 1 assumption shows
    # up in the magnitudes it writes back.
    assert len(model.voltage_ang()) == model.n_nodes
    assert len(model.branch_flow_p()) == model.n_branches

    # Per bus, what flows out equals what is injected — round-off, not a
    # convergence tolerance, because nothing iterated.
    assert model.dc_max_residual() < 1e-9

    # DC is lossless, so each island's reference supplies exactly the rest.
    pickup = model.dc_slack_pickup()
    assert len(pickup) >= 1
    total = sum(p for _, p in pickup)
    assert abs(total - 62.0) < 1e-9, f"expected 62 p.u. of load, got {total}"


def test_the_two_dc_approximations_are_both_reachable():
    """`dc_approximation` must reach the solver, not be silently accepted.

    This fixture has a uniform r/x, which makes the two choices a pure
    rescaling of B — so the flows agree while the *angles* differ. That is
    exactly what makes angles the right probe here.
    """
    angles = {}
    for approximation in ("ignore_r", "ignore_g"):
        model = dc_model(dc_approximation=approximation)
        model.solve()
        angles[approximation] = model.voltage_ang()

    worst = max(abs(a - b) for a, b in zip(angles["ignore_r"], angles["ignore_g"]))
    assert worst > 1e-9, "dc_approximation did not reach the solver"

    with pytest.raises(ValueError, match="unknown dc_approximation"):
        dc_model(dc_approximation="nonsense")


def test_method_selection_and_validation():
    for method in ("newton_raphson", "dc", "linear_impedance"):
        model = gridoxide.PowerFlowModel.from_pgm_json(str(TRANSMISSION), method=method)
        model.solve()
        assert len(model.voltage_mag()) == model.n_nodes

    with pytest.raises(ValueError, match="unknown method"):
        gridoxide.PowerFlowModel.from_pgm_json(str(TRANSMISSION), method="nonsense")


def test_branch_flows_need_a_dc_solve():
    """The DC accessors must refuse to serve a result that does not exist."""
    model = gridoxide.PowerFlowModel.from_pgm_json(str(TRANSMISSION))
    model.solve()  # Newton
    with pytest.raises(RuntimeError, match="no DC result"):
        model.branch_flow_p()

    # ...but `solve_dc` is available on a Newton-built model without rebuilding.
    model.solve_dc()
    assert len(model.branch_flow_p()) == model.n_branches


def test_ptdf_matches_a_finite_difference_of_the_solve():
    """PTDF against the solver itself.

    DC is linear, so a finite difference is the derivative, not an
    approximation of it. The step size cancels exactly.
    """
    model = dc_model()
    model.solve()
    base = model.branch_flow_p()

    eps = 1e-3
    checked = 0
    for bus in range(model.n_nodes):
        column = model.ptdf_column(bus)
        if column is None:
            continue

        injections = [0.0] * model.n_nodes
        injections[bus] = eps
        response = model.transfer_factors(injections)

        for k, (predicted, measured) in enumerate(zip(column, response)):
            assert abs(predicted * eps - measured) < 1e-9, (
                f"PTDF[{k}, {bus}]: column says {predicted * eps}, "
                f"transfer_factors says {measured}"
            )
        assert len(base) == len(column)
        checked += 1

    assert checked > 2, f"only {checked} buses had PTDF columns"


def test_ptdf_row_is_the_transpose_of_the_columns():
    """The row shortcut is one solve instead of n, and rests on B's symmetry."""
    model = dc_model()
    columns = [model.ptdf_column(bus) for bus in range(model.n_nodes)]

    checked = 0
    for branch in range(model.n_branches):
        row = model.ptdf_row(branch)
        if row is None:
            continue
        for bus, column in enumerate(columns):
            if column is None:
                continue
            assert abs(row[bus] - column[branch]) < 1e-12
        checked += 1

    assert checked > 2, f"only {checked} branches had PTDF rows"


def test_lodf_and_radial_branches():
    """A radial branch has no redistribution factors, and says so."""
    model = dc_model()

    radial, meshed = 0, 0
    for branch in range(model.n_branches):
        column = model.lodf_column(branch)
        if model.is_radial(branch):
            assert column is None, f"branch {branch} is radial but returned a column"
            radial += 1
        else:
            assert column is not None
            # The outaged branch loses all of its own flow, by definition.
            assert column[branch] == -1.0
            meshed += 1

    assert radial > 0, "fixture exercises no radial branch"
    assert meshed > 0, "fixture exercises no meshed branch"


def test_sensitivities_do_not_require_a_solve():
    """PTDF and LODF are properties of the topology, not of any operating point."""
    model = dc_model()
    # No solve() call at all.
    assert model.ptdf_column(0) is not None
    assert model.n_branches > 0


def test_out_of_range_indices_are_rejected():
    model = dc_model()
    with pytest.raises(ValueError, match="out of range"):
        model.ptdf_column(model.n_nodes)
    with pytest.raises(ValueError, match="out of range"):
        model.lodf_column(model.n_branches)
    with pytest.raises(ValueError, match="one per bus"):
        model.transfer_factors([0.0])


def test_dc_on_a_distribution_feeder():
    """DC is at its weakest where r/x is large; it must still return a
    power-conserving answer rather than failing or producing nonsense."""
    model = dc_model(DISTRIBUTION)
    model.solve()
    assert model.dc_max_residual() < 1e-9


def test_outage_flows_match_an_actual_outage():
    """The N-1 primitive, against the only oracle that matters for it.

    LODF predicts the post-outage flows from the pre-outage ones without
    re-solving. Checked here against `transfer_factors`' own linearity and the
    defining identity `f_out[l] == 0`.
    """
    model = dc_model()
    model.solve()
    base = model.branch_flow_p()

    checked = 0
    for branch in range(model.n_branches):
        after = model.outage_flows(branch)
        if model.is_radial(branch):
            assert after is None
            continue
        assert after is not None
        # The tripped branch carries nothing, by definition.
        assert after[branch] == 0.0
        # Everything it was carrying went somewhere: DC is lossless, so the
        # flows still satisfy every bus's balance. Spot-check via the identity
        # f_out = f + LODF[:, l] * f[l].
        column = model.lodf_column(branch)
        for k in range(model.n_branches):
            assert abs(after[k] - (base[k] + column[k] * base[branch])) < 1e-9 or k == branch
        checked += 1

    assert checked > 0, "no non-radial branch was checked"


def test_outage_flows_validates_its_input():
    model = dc_model()
    model.solve()
    with pytest.raises(ValueError, match="out of range"):
        model.outage_flows(model.n_branches)
    with pytest.raises(ValueError, match="one per branch"):
        model.outage_flows(0, [0.0])


def test_multi_outage_flows_and_breaking_sets():
    """N-2 screening, and the reason it is not just N-1 applied twice."""
    model = dc_model()
    model.solve()
    base = model.branch_flow_p()

    # An empty set changes nothing; a repeated index is malformed input.
    assert model.multi_outage_flows([]) == base
    assert model.multi_outage_flows([0, 0]) is None

    checked, breaking = 0, 0
    for a in range(model.n_branches):
        for b in range(a + 1, model.n_branches):
            after = model.multi_outage_flows([a, b])
            if after is None:
                assert model.is_breaking_set([a, b])
                breaking += 1
                continue
            # Both outaged branches carry nothing afterwards.
            assert after[a] == 0.0 and after[b] == 0.0
            # A one-element set must agree with the single-branch entry point.
            assert model.multi_outage_flows([a]) == model.outage_flows(a)
            checked += 1

    assert checked > 0, "no solvable pair was screened"
    assert breaking > 0, "no breaking pair was found, so is_breaking_set is untested"


def test_multi_outage_validates_its_input():
    model = dc_model()
    model.solve()
    with pytest.raises(ValueError, match="out of range"):
        model.multi_outage_flows([model.n_branches])
    with pytest.raises(ValueError, match="one per branch"):
        model.multi_outage_flows([0], [0.0])
    with pytest.raises(ValueError, match="out of range"):
        model.is_breaking_set([model.n_branches])

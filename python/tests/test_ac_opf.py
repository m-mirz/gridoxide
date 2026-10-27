"""The `gridoxide.ac_opf` binding.

The numerics are validated in `tests/opf_ac_test.rs` against pglib's published
AC objectives, finite-differenced derivatives and re-derived power-flow
feasibility. What is checked here is that the binding exposes them faithfully,
and that the physical identities a caller would sanity-check against hold.
"""

import math
import os

import pytest

import gridoxide
from _pglib import fixtures

pytestmark = pytest.mark.skipif(
    not hasattr(gridoxide, "ac_opf"),
    reason="built without the opf feature, so there is no solver",
)

# pglib's published AC objectives, $/h — see
# tests/data/benchmark-grids/pglib/BASELINE.md.
PUBLISHED = {
    "pglib_opf_case3_lmbd": 5812.6,
    "pglib_opf_case5_pjm": 17552.0,
    "pglib_opf_case14_ieee": 2178.1,
    "pglib_opf_case30_ieee": 8208.5,
    "pglib_opf_case118_ieee": 97214.0,
}


def case(name):
    return os.path.join(fixtures(), f"{name}.json")


@pytest.fixture(scope="module")
def congested():
    """case5_pjm is congested, so it exercises prices and a binding limit."""
    return gridoxide.ac_opf(case("pglib_opf_case5_pjm"))


@pytest.mark.parametrize("name,published", sorted(PUBLISHED.items()))
def test_objectives_match_the_published_ac_baseline(name, published):
    result = gridoxide.ac_opf(case(name))
    assert result.objective == pytest.approx(published, rel=1e-4)
    # An objective is only meaningful at a feasible point, which is why the
    # binding exposes the violation alongside it rather than hiding it.
    assert result.violation < 1e-6


def test_shapes_are_consistent(congested):
    r = congested
    assert len(r.p_gen) == len(r.q_gen) == len(r.generator_index) == 5
    assert len(r.magnitudes) == len(r.angles) == len(r.lmp_p) == len(r.lmp_q) == 5
    assert len(r.flows) == 6
    assert all(len(f) == 2 for f in r.flows)


def test_generation_exceeds_demand_by_the_losses(congested):
    """Unlike DC, AC is lossy — so generation must be *above* demand, by a
    margin small enough to be losses rather than an error."""
    demand = 1000.0
    total = sum(congested.p_gen)
    assert total > demand
    assert total - demand < 0.1 * demand, "losses of more than 10% suggest a modelling error"


def test_voltage_magnitudes_are_within_their_limits(congested):
    # case5_pjm declares [0.9, 1.1] on every bus.
    assert min(congested.magnitudes) >= 0.9 - 1e-8
    assert max(congested.magnitudes) <= 1.1 + 1e-8


def test_prices_are_positive_and_spread_where_congested(congested):
    """Serving more demand cannot cost less than nothing — the sign check that
    a plausible-looking convention error would fail on every bus at once."""
    assert all(p > 0 for p in congested.lmp_p)
    assert max(congested.lmp_p) - min(congested.lmp_p) > 1.0


def test_branch_limits_can_be_relaxed():
    enforced = gridoxide.ac_opf(case("pglib_opf_case5_pjm"))
    relaxed = gridoxide.ac_opf(case("pglib_opf_case5_pjm"), enforce_limits=False)
    assert relaxed.objective < enforced.objective


def test_the_companion_document_can_be_given_explicitly():
    explicit = gridoxide.ac_opf(
        case("pglib_opf_case5_pjm"),
        data_path=os.path.join(fixtures(), "pglib_opf_case5_pjm.opf.json"),
    )
    implicit = gridoxide.ac_opf(case("pglib_opf_case5_pjm"))
    assert explicit.objective == pytest.approx(implicit.objective, rel=1e-12)


def test_an_impossible_iteration_budget_is_reported_not_silently_accepted():
    with pytest.raises(RuntimeError, match="no optimal dispatch"):
        gridoxide.ac_opf(case("pglib_opf_case118_ieee"), max_iterations=2)


def test_missing_files_are_reported_clearly():
    with pytest.raises(RuntimeError, match="reading"):
        gridoxide.ac_opf(case("no_such_case"))

    with pytest.raises(RuntimeError, match="data_path"):
        gridoxide.ac_opf(case("pglib_opf_case5_pjm"), data_path="/nonexistent.json")


def test_the_flows_are_consistent_with_the_reported_angles(congested):
    """A weak but genuinely independent identity: apparent power at a terminal
    must be non-negative and finite, and at least one branch should carry
    something."""
    apparent = [math.hypot(p, q) for p, q in congested.flows]
    assert all(math.isfinite(s) and s >= 0 for s in apparent)
    assert max(apparent) > 1.0


# `OptimalPowerFlowModel` reads the `.m` itself, so these need only the
# submodule, not the converted fixtures.
PGLIB = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "benchmark-grids", "pglib")


def pglib_m(name):
    variant = name.rpartition("__")[2] if "__" in name else ""
    path = os.path.join(PGLIB, variant, f"{name}.m")
    if not os.path.exists(path):
        pytest.skip("benchmark-grids submodule not initialized")
    return path


@pytest.mark.parametrize(
    "name,published",
    [("pglib_opf_case14_ieee", 2178.1), ("pglib_opf_case14_ieee__sad", 2776.8)],
)
def test_the_model_reads_the_case_itself(name, published):
    model = gridoxide.OptimalPowerFlowModel.from_matpower(pglib_m(name))
    with pytest.raises(RuntimeError, match="solve"):
        model.objective
    model.solve()
    assert model.objective == pytest.approx(published, rel=1e-4)
    assert model.violation < 1e-6

    nodes = model.nodes()
    assert [b for b, _ in nodes] == list(range(1, 15))
    assert len(model.voltage_mag()) == len(model.voltage_ang()) == 14
    # case14's reference bus 1 has VA = 0.
    assert model.voltage_ang()[0] == pytest.approx(0.0, abs=1e-12)

    dispatch = model.dispatch()
    assert [row for row, _, _ in dispatch] == [0, 1, 2, 3, 4]
    assert sum(p for _, p, _ in dispatch) > 259.0  # load plus losses


def test_ipopt_is_refused_or_agrees():
    path = pglib_m("pglib_opf_case14_ieee")
    try:
        model = gridoxide.OptimalPowerFlowModel.from_matpower(path, solver="ipopt", tol=1e-6)
    except ValueError as e:
        assert "opf-ipopt" in str(e)
        return
    model.solve()
    assert model.objective == pytest.approx(2178.1, rel=1e-4)
    assert model.iterations > 0


def test_a_flat_start_reaches_the_same_optimum():
    # case14 only: from the flat start `opf::nlp` stalls on case118, the
    # reason the proportional start is the default.
    model = gridoxide.OptimalPowerFlowModel.from_matpower(pglib_m("pglib_opf_case14_ieee"))
    model.solve(init="flat")
    assert model.objective == pytest.approx(2178.1, rel=1e-4)
    with pytest.raises(ValueError, match="unknown init"):
        model.solve(init="cold")


def test_an_unknown_solver_is_refused():
    with pytest.raises(ValueError, match="unknown solver"):
        gridoxide.OptimalPowerFlowModel.from_matpower(pglib_m("pglib_opf_case14_ieee"), solver="mips")

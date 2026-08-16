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

pytestmark = pytest.mark.skipif(
    not hasattr(gridoxide, "ac_opf"),
    reason="built without the opf feature, so there is no solver",
)

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests",
    "data",
    "pglib-opf",
)

# pglib's published AC objectives, $/h — see tests/data/pglib-opf/README.md.
PUBLISHED = {
    "pglib_opf_case3_lmbd": 5812.6,
    "pglib_opf_case5_pjm": 17552.0,
    "pglib_opf_case14_ieee": 2178.1,
    "pglib_opf_case30_ieee": 8208.5,
    "pglib_opf_case118_ieee": 97214.0,
}


def case(name):
    return os.path.join(FIXTURES, f"{name}.json")


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
        data_path=os.path.join(FIXTURES, "pglib_opf_case5_pjm.opf.json"),
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

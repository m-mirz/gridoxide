"""The `gridoxide.dc_opf` binding.

The numerics are validated in `tests/opf_dc_test.rs` against analytic cases,
KKT certificates and pglib's published objectives. What is checked here is that
the binding exposes them faithfully, and that the physical identities a caller
would sanity-check against actually hold.

`dc_opf` needs a solver backend compiled in, so the whole module skips when the
extension was built without `opf-highs`.
"""

import os

import pytest

import gridoxide

pytestmark = pytest.mark.skipif(
    not hasattr(gridoxide, "dc_opf"),
    reason="built without the opf-highs feature, so there is no solver",
)

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests",
    "data",
    "pglib-opf",
)

# pglib's published DC objectives, $/h — see tests/data/pglib-opf/README.md.
PUBLISHED = {
    "pglib_opf_case3_lmbd": 5695.9,
    "pglib_opf_case5_pjm": 17480.0,
    "pglib_opf_case14_ieee": 2051.5,
    "pglib_opf_case30_ieee": 7472.8,
    "pglib_opf_case118_ieee": 93101.0,
}


def case(name):
    return os.path.join(FIXTURES, f"{name}.json")


@pytest.fixture(scope="module")
def congested():
    """case5_pjm is congested, so it exercises prices and binding limits."""
    return gridoxide.dc_opf(case("pglib_opf_case5_pjm"))


def test_shapes_are_consistent(congested):
    r = congested
    assert len(r.dispatch) == len(r.generator_index) == 5
    assert len(r.lmp) == len(r.angles) == 5
    assert len(r.flows) == 6
    assert len(r.shed) >= 0


def test_generation_meets_demand_exactly(congested):
    """DC is lossless, so generation plus shedding must equal demand. The
    cheapest check that the balance rows carry the right signs."""
    assert sum(congested.dispatch) + sum(congested.shed) == pytest.approx(1000.0, abs=1e-6)
    assert sum(congested.shed) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name,published", sorted(PUBLISHED.items()))
def test_objectives_track_the_published_dc_baseline(name, published):
    """All five agree to better than 0.03%; the Rust suite pins the gaps
    case by case."""
    result = gridoxide.dc_opf(case(name))
    assert result.objective == pytest.approx(published, rel=5e-4)


def test_the_susceptance_choice_is_exposed_and_changes_the_answer():
    """`dc_approximation` defaults to the series susceptance PowerModels uses.
    The textbook `1/x` is reachable, and on case30 it is measurably worse —
    see `DcOpfOptions` for why."""
    name = "pglib_opf_case30_ieee"
    published = PUBLISHED[name]
    default = gridoxide.dc_opf(case(name)).objective
    textbook = gridoxide.dc_opf(case(name), dc_approximation="ignore_r").objective

    assert abs(default - published) < abs(textbook - published)
    assert textbook == pytest.approx(published * 1.00423, rel=5e-4)

    with pytest.raises(ValueError, match="dc_approximation"):
        gridoxide.dc_opf(case(name), dc_approximation="nonsense")


def test_prices_spread_only_where_something_binds(congested):
    spread = max(congested.lmp) - min(congested.lmp)
    assert congested.binding, "case5_pjm should be congested"
    assert spread > 1.0, f"a congested case should show a price spread, got {spread}"

    # And an uncongested one should not.
    uncongested = gridoxide.dc_opf(case("pglib_opf_case14_ieee"))
    assert not uncongested.binding
    assert max(uncongested.lmp) - min(uncongested.lmp) == pytest.approx(0.0, abs=1e-6)


def test_binding_limits_report_flow_rate_and_price(congested):
    for b in congested.binding:
        assert 0 <= b.branch < len(congested.flows)
        assert b.rate > 0
        # The flow is at one of the two limits, and signed so it says which.
        assert abs(abs(b.flow) - b.rate) == pytest.approx(0.0, abs=1e-6)
        # The reported flow agrees with the flows array it indexes into.
        assert congested.flows[b.branch] == pytest.approx(b.flow, abs=1e-6)
        assert b.price != 0.0


def test_dispatch_respects_the_generator_limits(congested):
    """Every unit inside its own box — read from the OPF document rather than
    assumed, so this checks the limits reached the solver."""
    import json

    with open(os.path.join(FIXTURES, "pglib_opf_case5_pjm.opf.json")) as f:
        data = json.load(f)
    limits = {g["index"]: (g["p_min"], g["p_max"]) for g in data["generator"]}

    for index, p in zip(congested.generator_index, congested.dispatch):
        low, high = limits[index]
        assert low - 1e-6 <= p <= high + 1e-6, f"generator {index}: {p} outside [{low}, {high}]"


def test_shedding_can_be_disabled():
    """These cases can serve their demand, so switching shedding off must not
    change the answer."""
    with_shed = gridoxide.dc_opf(case("pglib_opf_case5_pjm"))
    without = gridoxide.dc_opf(case("pglib_opf_case5_pjm"), allow_shedding=False)
    assert without.objective == pytest.approx(with_shed.objective, rel=1e-9)


def test_the_companion_document_can_be_given_explicitly():
    explicit = gridoxide.dc_opf(
        case("pglib_opf_case5_pjm"),
        data_path=os.path.join(FIXTURES, "pglib_opf_case5_pjm.opf.json"),
    )
    implicit = gridoxide.dc_opf(case("pglib_opf_case5_pjm"))
    assert explicit.objective == pytest.approx(implicit.objective, rel=1e-12)


def test_missing_files_are_reported_clearly():
    with pytest.raises(RuntimeError, match="reading"):
        gridoxide.dc_opf(case("no_such_case"))

    with pytest.raises(RuntimeError, match="data_path"):
        gridoxide.dc_opf(case("pglib_opf_case5_pjm"), data_path="/nonexistent.json")

"""Per-machine reactive dispatch through the binding.

The physics is gated in `tests/cgmes_dispatch_test.rs`, against RealGrid's own
published per-terminal solution. What is checked here is that the binding
reaches it and that the shapes are what the docstring promises.
"""

from pathlib import Path

import pytest

import gridoxide

ROOT = Path(__file__).resolve().parent.parent.parent
REALGRID = ROOT / "tests/data/CGMES-Test-Configurations/v3.0/RealGrid/RealGrid-Merged"


pytestmark = pytest.mark.skipif(
    not hasattr(gridoxide.PowerFlowModel, "from_cgmes"), reason="built without the cgmes feature"
)


@pytest.fixture(scope="module")
def solved():
    if not REALGRID.exists():
        pytest.skip(f"missing fixture: {REALGRID}")
    model = gridoxide.PowerFlowModel.from_cgmes(
        [str(p) for p in sorted(REALGRID.glob("*.xml"))], s_base_va=100e6, max_iter=60
    )
    model.solve(enforce_q_limits=True, max_outer=60)
    return model


def test_it_attributes_each_bus_to_its_machines(solved):
    split = solved.machine_dispatch()
    assert split, "RealGrid has voltage-controlled buses"

    shared = [b for b in split if len(b["machines"]) > 1]
    assert len(shared) >= 62, "RealGrid's shared-control buses"

    for bus in split:
        assert bus["basis"] in ("explicit", "capability", "uniform")
        summed = sum(m["q"] for m in bus["machines"])
        assert summed == pytest.approx(bus["attributed"], abs=1e-12)
        assert bus["attributed"] + bus["unattributed"] == pytest.approx(bus["required"], abs=1e-12)
        for m in bus["machines"]:
            assert m["q_min"] - 1e-9 <= m["q"] <= m["q_max"] + 1e-9
            assert 0.0 <= m["share"] <= 1.0
        assert sum(m["share"] for m in bus["machines"]) == pytest.approx(1.0)


def test_shares_are_proportional_to_capability(solved):
    """RealGrid states a usable reactive range, so the split has a real basis."""
    shared = [b for b in solved.machine_dispatch() if len(b["machines"]) > 1]
    by_capability = [b for b in shared if b["basis"] == "capability"]
    assert len(by_capability) >= 60

    bus = by_capability[0]
    ranges = [m["q_max"] - m["q_min"] for m in bus["machines"]]
    total = sum(ranges)
    for m, r in zip(bus["machines"], ranges):
        assert m["share"] == pytest.approx(r / total)


def test_unattributed_power_is_surfaced_not_hidden(solved):
    """A bus whose own machines cannot produce what the solve put there.

    Not an allocation failure — it measures the bus model bounding the *net*
    injection with limits that describe the machines' own capability.
    """
    split = solved.machine_dispatch()
    short = [b for b in split if abs(b["unattributed"]) > 1e-6]
    assert short, "RealGrid has buses where a co-located load saturates the machine"
    for bus in short:
        assert all(m["at_limit"] for m in bus["machines"])


def test_a_model_without_machines_returns_nothing():
    """PGM states no per-machine capability, so there is nothing to attribute."""
    pgm = ROOT / "tests/data/pglib-opf/pglib_opf_case14_ieee.json"
    if not pgm.exists():
        pytest.skip(f"missing fixture: {pgm}")
    model = gridoxide.PowerFlowModel.from_pgm_json(str(pgm), s_base_va=100e6)
    model.solve()
    assert model.machine_dispatch() == []

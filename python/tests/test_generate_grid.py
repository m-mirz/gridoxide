"""Smoke test for `gridoxide.generate_grid` (pure stdlib, no extra needed):
generate a small synthetic grid and confirm `gridoxide.PowerFlowModel` can
load and solve it — proves the packaged generator produces valid PGM JSON,
not just that it runs without raising.
"""
import json

import pytest

import gridoxide
from gridoxide.generate_grid import generate


def test_generate_produces_solvable_grid(tmp_path):
    out_path = tmp_path / "grid.json"
    generate(target_nodes=50, seed=42, out_path=str(out_path))

    data = json.loads(out_path.read_text())["data"]
    assert len(data["node"]) > 0
    assert len(data["source"]) == 1

    model = gridoxide.PowerFlowModel.from_pgm_json(str(out_path))
    model.solve()
    vm = model.voltage_mag()
    assert len(vm) == model.n_nodes
    assert all(v == pytest.approx(v) for v in vm)  # all finite, no NaNs


def test_generate_is_deterministic_per_seed(tmp_path):
    out_1 = tmp_path / "grid1.json"
    out_2 = tmp_path / "grid2.json"
    generate(target_nodes=50, seed=7, out_path=str(out_1))
    generate(target_nodes=50, seed=7, out_path=str(out_2))
    assert out_1.read_text() == out_2.read_text()

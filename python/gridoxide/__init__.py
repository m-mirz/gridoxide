"""gridoxide: AC power flow analysis (Newton-Raphson) — Python bindings.

`PowerFlowModel` and `StateEstimationModel` are implemented in Rust
(`src/python.rs`) and built as the
private `_gridoxide` compiled extension alongside this package (see
`pyproject.toml`'s `python-source`/`module-name`), re-exported here so
callers only ever need `import gridoxide`.

`gridoxide.matpower` (needs the `matpower` extra: `pip install
gridoxide[matpower]`) converts raw MATPOWER case files into the PGM JSON
`PowerFlowModel.from_pgm_json` reads — imported lazily, not here, so the
core bindings never require numpy/scipy.
"""
from ._gridoxide import PowerFlowModel, StateEstimationModel

__all__ = ["PowerFlowModel", "StateEstimationModel"]

"""gridoxide: AC power flow analysis (Newton-Raphson) — Python bindings.

`PowerFlowModel`, `StateEstimationModel`, `AcSensitivityModel` and
`short_circuit` are implemented in Rust
(`src/python.rs`) and built as the
private `_gridoxide` compiled extension alongside this package (see
`pyproject.toml`'s `python-source`/`module-name`), re-exported here so
callers only ever need `import gridoxide`.

`gridoxide.matpower` (needs the `matpower` extra: `pip install
gridoxide[matpower]`) converts raw MATPOWER case files into the PGM JSON
`PowerFlowModel.from_pgm_json` reads — imported lazily, not here, so the
core bindings never require numpy/scipy.
"""
from ._gridoxide import (
    AcSensitivityModel,
    PowerFlowModel,
    StateEstimationModel,
    short_circuit,
)

__all__ = [
    "AcSensitivityModel",
    "PowerFlowModel",
    "StateEstimationModel",
    "short_circuit",
]

# `dc_opf` needs a solver backend compiled in (the `opf-highs` feature, which
# needs a local HiGHS install), so it is not present in every build. Exported
# when it is, absent when it is not — `hasattr(gridoxide, "dc_opf")` is the
# check, and importing gridoxide never fails for want of a solver.
try:
    from ._gridoxide import dc_opf  # noqa: F401
except ImportError:  # pragma: no cover - depends on build features
    pass
else:
    __all__.append("dc_opf")

"""gridoxide: AC power flow analysis (Newton-Raphson) — Python bindings.

`PowerFlowModel`, `StateEstimationModel`, `AcSensitivityModel`,
`short_circuit`, `continuation` and `dynamics` are implemented in Rust
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
    ContinuationCurve,
    ContinuationEvent,
    PowerFlowModel,
    StateEstimationModel,
    continuation,
    short_circuit,
)

__all__ = [
    "AcSensitivityModel",
    "ContinuationCurve",
    "ContinuationEvent",
    "PowerFlowModel",
    "StateEstimationModel",
    "continuation",
    "short_circuit",
]

# The optimal power flow entry points need the `opf` feature compiled in, so
# they are not present in every build. Exported when they are, absent when they
# are not — `hasattr(gridoxide, "dc_opf")` is the check, and importing
# gridoxide never fails for want of a solver.
#
# The feature itself needs nothing installed: the solvers behind both of these
# are gridoxide's own, pure Rust. Only the optional `opf-highs` reference
# backend needs a system library.
try:
    from ._gridoxide import ac_opf, dc_opf  # noqa: F401
except ImportError:  # pragma: no cover - depends on build features
    pass
else:
    __all__ += ["ac_opf", "dc_opf"]

# RMS dynamic simulation needs the `dynamics` feature compiled in, so like the
# optimal power flow it is present in some builds and not others.
# `hasattr(gridoxide, "dynamics")` is the check.
#
# It needs nothing installed either: the DAE, its integrator and the whole
# machine/exciter/governor model library are gridoxide's own, pure Rust on the
# same sparse backends the power flow uses.
try:
    from ._gridoxide import (  # noqa: F401
        DynamicsMode,
        DynamicsResult,
        ModeSensitivity,
        SmallSignalResult,
        dynamics,
        small_signal,
    )
except ImportError:  # pragma: no cover - depends on build features
    pass
else:
    __all__ += [
        "DynamicsMode",
        "DynamicsResult",
        "ModeSensitivity",
        "SmallSignalResult",
        "dynamics",
        "small_signal",
    ]

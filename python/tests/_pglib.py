"""The pglib-opf fixtures, converted once per session by
`tests/data/pglib_opf.py` from the `benchmark-grids` submodule."""

import atexit
import functools
import importlib.util
import shutil
import tempfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "tests" / "data" / "pglib_opf.py"
_spec = importlib.util.spec_from_file_location("pglib_opf", _SCRIPT)
pglib_opf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pglib_opf)

CASES = pglib_opf.CASES
PIECEWISE = pglib_opf.PIECEWISE


@functools.cache
def fixtures() -> Path:
    """Skips the calling module when the submodule or numpy is missing."""
    if not pglib_opf.SOURCE.is_dir():
        pytest.skip("benchmark-grids submodule not initialized", allow_module_level=True)
    pytest.importorskip("numpy")
    out = Path(tempfile.mkdtemp(prefix="pglib-opf-"))
    atexit.register(shutil.rmtree, out, ignore_errors=True)
    return pglib_opf.generate(out)

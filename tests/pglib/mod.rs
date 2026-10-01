//! The pglib-opf fixtures, converted from `tests/data/benchmark-grids/pglib/`
//! on first use by `tests/data/pglib_opf.py`.
//!
//! Needs the `benchmark-grids` submodule and a Python with numpy, `python3`
//! unless `PYTHON` names another.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Path of `<name><suffix>`, e.g. `("pglib_opf_case5_pjm", ".opf.json")`.
pub fn path(name: &str, suffix: &str) -> PathBuf {
    dir().join(format!("{name}{suffix}"))
}

pub fn read(name: &str, suffix: &str) -> String {
    let path = path(name, suffix);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// Converted once per test binary, into a directory of its own: binaries can
/// run concurrently, and a shared one would let one read another's half-written
/// file.
pub fn dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("pglib-opf")
            .join(env!("CARGO_CRATE_NAME"));
        let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pglib_opf.py");
        let out = Command::new(&python)
            .arg(&script)
            .arg(&dir)
            .output()
            .unwrap_or_else(|e| panic!("running {python}: {e}; set PYTHON to a Python with numpy"));
        assert!(
            out.status.success(),
            "{} failed (needs numpy and the benchmark-grids submodule):\n{}",
            script.display(),
            String::from_utf8_lossy(&out.stderr)
        );
        dir
    })
}

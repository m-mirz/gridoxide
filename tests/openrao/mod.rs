//! powsybl-open-rao's test resources, read from the benchmark-grids submodule
//! at `tests/data/benchmark-grids/powsybl-open-rao/`.
//!
//! `features/` and `files/` are its Cucumber suite, whole. Everything else the
//! tests use is listed by name in `tests/data/openrao-fixtures.txt`.
#![allow(dead_code)]

use std::path::PathBuf;

const MANIFEST: &str = include_str!("../data/openrao-fixtures.txt");

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/benchmark-grids/powsybl-open-rao")
}

/// A path relative to the root, which must exist: a missing file would
/// otherwise surface as a confusing parse error, or a gate testing nothing.
pub fn path(relative: &str) -> PathBuf {
    let path = root().join(relative);
    assert!(
        path.exists(),
        "{} does not exist; run `git submodule update --init tests/data/benchmark-grids`",
        path.display()
    );
    path
}

/// A file listed in `tests/data/openrao-fixtures.txt`, by its name there.
pub fn fixture(name: &str) -> PathBuf {
    let relative = entries()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("{name} is not listed in tests/data/openrao-fixtures.txt"))
        .1;
    path(relative)
}

/// The names of every listed file with this extension, e.g. `"xiidm"`, sorted.
pub fn names_with_extension(extension: &str) -> Vec<&'static str> {
    let mut names: Vec<&str> =
        entries().map(|(n, _)| n).filter(|n| n.rsplit('.').next() == Some(extension)).collect();
    names.sort();
    names
}

fn entries() -> impl Iterator<Item = (&'static str, &'static str)> {
    MANIFEST.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(|l| {
        let mut fields = l.split_whitespace();
        (fields.next().unwrap(), fields.next().expect("a name and a path"))
    })
}

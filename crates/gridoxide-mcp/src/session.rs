//! Networks, read once and kept until a file of theirs changes, with the
//! last power flow solved over each.
//!
//! A network is one file (`.m`, `.uct`, `.xiidm`) or a directory of CGMES
//! profiles; a single CGMES profile stands for its directory, as
//! `gridoxide solve` reads it.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use gridoxide::PowerFlowReport;
use gridoxide::load::{self, Format, Network, SolveSettings};

pub struct Loaded {
    /// The network as named, relative to the workspace.
    pub name: String,
    files: Vec<PathBuf>,
    stamps: Vec<Option<SystemTime>>,
    pub net: Network,
    /// The last power flow over it, and what it was asked to do.
    pub solved: Option<(SolveSettings, PowerFlowReport)>,
}

pub struct Session {
    /// Where relative paths resolve, and where `list_networks` looks.
    pub base: PathBuf,
    nets: HashMap<PathBuf, Loaded>,
}

impl Session {
    pub fn new(base: PathBuf) -> Self {
        let base = base.canonicalize().unwrap_or(base);
        Self {
            base,
            nets: HashMap::new(),
        }
    }

    /// `path` against the base directory.
    pub fn resolve(&self, path: Option<&str>) -> PathBuf {
        match path.filter(|p| !p.trim().is_empty()) {
            None => self.base.clone(),
            Some(p) => {
                let p = Path::new(p.trim());
                let full = if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    self.base.join(p)
                };
                full.canonicalize().unwrap_or(full)
            }
        }
    }

    /// `path` relative to the base directory, for display.
    pub fn relative(&self, path: &Path) -> String {
        match path.strip_prefix(&self.base) {
            Ok(p) if p.as_os_str().is_empty() => ".".into(),
            Ok(p) => p.display().to_string(),
            Err(_) => path.display().to_string(),
        }
    }

    /// The network `path` names, read if new or changed since.
    pub fn network(&mut self, path: &str) -> Result<&mut Loaded, String> {
        if path.trim().is_empty() {
            return Err("`path` names the network: a .m, .uct or .xiidm file, or a directory of CGMES profiles (list_networks finds them)".into());
        }
        let full = self.resolve(Some(path));
        if !full.exists() {
            return Err(format!(
                "{} does not exist (paths are relative to {})",
                full.display(),
                self.base.display()
            ));
        }
        // A CGMES profile is read as its directory, so it is cached as one.
        let key = if full.is_file() && is_xml(&full) {
            full.parent().map_or(full.clone(), Path::to_path_buf)
        } else {
            full.clone()
        };
        if load::detect(&key).is_none() {
            return Err(format!(
                "{} is not a network gridmcp reads: a MATPOWER .m, UCTE .uct, IIDM .xiidm, or a directory of CGMES profile .xml files",
                self.relative(&full)
            ));
        }
        let files = source_files(&key);
        let stamps = stamps(&files);
        let fresh = self
            .nets
            .get(&key)
            .is_some_and(|n| n.files == files && n.stamps == stamps);
        if !fresh {
            let name = self.relative(&key);
            let net =
                load::load(&key.to_string_lossy()).map_err(|e| format!("reading {name}: {e}"))?;
            self.nets.insert(
                key.clone(),
                Loaded {
                    name,
                    files,
                    stamps,
                    net,
                    solved: None,
                },
            );
        }
        Ok(self.nets.get_mut(&key).expect("inserted above"))
    }
}

fn is_xml(path: &Path) -> bool {
    path.extension()
        .is_some_and(|x| x.eq_ignore_ascii_case("xml"))
}

/// The files a network is read from: the file itself, or a CGMES directory's
/// `.xml` files in name order.
fn source_files(path: &Path) -> Vec<PathBuf> {
    if !path.is_dir() {
        return vec![path.to_path_buf()];
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_xml(p))
        .collect();
    files.sort();
    files
}

fn stamps(files: &[PathBuf]) -> Vec<Option<SystemTime>> {
    files
        .iter()
        .map(|f| f.metadata().and_then(|m| m.modified()).ok())
        .collect()
}

/// The first bytes of `path`, enough to tell what a file is.
fn head(path: &Path, len: usize) -> Option<String> {
    let mut buf = Vec::with_capacity(len);
    std::fs::File::open(path)
        .ok()?
        .take(len as u64)
        .read_to_end(&mut buf)
        .ok()?;
    // A cut through a multi-byte character costs that character only.
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// A network found on disk: where, in which format, and how many files.
pub struct Found {
    pub path: PathBuf,
    pub format: Format,
    pub files: usize,
    pub bytes: u64,
}

/// The networks at or below `root`, `depth` levels deep at most, in path
/// order. Hidden directories, `node_modules` and `target` are skipped.
///
/// Stricter than [`load::detect`], which goes by extension alone: a `.m`
/// must read like a MATPOWER case (`function mpc`, `mpc.`) rather than any MATLAB script,
/// and a directory's `.xml` files must include a CIM one.
pub fn find_networks(root: &Path, depth: usize) -> Vec<Found> {
    let mut out = Vec::new();
    walk(root, depth, &mut out);
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<Found>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subdirs = Vec::new();
    let (mut cim_files, mut cim_bytes) = (0, 0);
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            if name != "node_modules" && name != "target" {
                subdirs.push(path);
            }
            continue;
        }
        let bytes = entry.metadata().map_or(0, |m| m.len());
        if is_xml(&path) {
            if head(&path, 4096).is_some_and(|h| h.contains("http://iec.ch/TC57/")) {
                cim_files += 1;
                cim_bytes += bytes;
            }
            continue;
        }
        let format = match load::detect(&path) {
            Some(Format::Matpower) => head(&path, 1 << 16)
                .is_some_and(|h| h.contains("function mpc") || h.contains("mpc."))
                .then_some(Format::Matpower),
            other => other,
        };
        if let Some(format) = format {
            out.push(Found {
                path,
                format,
                files: 1,
                bytes,
            });
        }
    }
    if cim_files > 0 && load::detect(dir) == Some(Format::Cgmes) {
        out.push(Found {
            path: dir.to_path_buf(),
            format: Format::Cgmes,
            files: cim_files,
            bytes: cim_bytes,
        });
    }
    if depth > 0 {
        for sub in subdirs {
            walk(&sub, depth - 1, out);
        }
    }
}

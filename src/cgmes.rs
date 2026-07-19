//! CGMES (Common Grid Model Exchange Standard) dataset loading, built on
//! cimoxide's `cimdecoder`/`cimstructs` crates — see `CIMOXIDE_PROVENANCE.md`
//! for why this is a pinned git dependency rather than a crates.io one.

use std::path::Path;

pub use cimdecoder::CimDataset;

/// Loads and merges a set of CGMES profile files (e.g. EQ, SSH, TP, SV) into
/// one `CimDataset`, keyed by MRID across all of them.
pub fn load_profiles(paths: &[&Path]) -> Result<CimDataset, Box<dyn std::error::Error>> {
    CimDataset::decode_files(paths)
}

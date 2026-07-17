//! Sentinel-index encoding and fixed solver configuration shared across
//! every `klu_native` submodule — mirrors `vendor/suitesparse/KLU/Include/
//! klu_internal.h`'s `EMPTY`/`FLIP`/`UNFLIP` (identical, by inspection, to
//! `AMD/Include/amd_internal.h`'s and `BTF/Include/btf.h`'s own
//! `BTF_FLIP`/`BTF_UNFLIP`/`BTF_ISFLIPPED`) and `KLU/Source/klu_defaults.c`'s
//! `KLU_common` defaults.
//!
//! Ported deliberately literally, not "idiomatized" into an `Option`/`enum`
//! at this layer — `lpivot`'s diagonal-preference bookkeeping (`kernel.rs`,
//! landing in a later phase) reads `Pinv[diagrow] < 0` directly and recovers
//! `FLIP(Pinv[pivrow])` to reroute it, and BTF's `Q` uses the same encoding
//! for unmatched columns. This is exactly the kind of index bookkeeping
//! where `block_sparse.rs`'s `adj.row`/`adj.col` bug happened — silently
//! wrong only on inputs a casual test wouldn't cover. Conversion to
//! `Option<usize>`/checked `Vec<usize>` happens only at each phase's own
//! public API boundary, once the sentinel-form algorithm is done with it.

/// Sentinel for "not yet assigned" — matches every one of the three ported
/// packages' own `#define EMPTY (-1)` (confirmed identical across
/// `KLU/Include/klu_internal.h:77`, `AMD/Include/amd_internal.h:71,93`, and
/// implicitly in `BTF/Include/btf.h`'s `BTF_ISFLIPPED`/`BTF_UNFLIP`, which
/// treat `-1` the same way).
pub const EMPTY: i64 = -1;

/// "Negation about -1": marks an otherwise-non-negative index as "flipped"
/// (BTF uses this for unmatched columns; KLU's `lpivot` uses it to encode a
/// row's earlier diagonal preference once that row has been claimed by a
/// different column). `flip(flip(j)) == j` for all `j`, and
/// `flip(EMPTY) == EMPTY`. Matches `klu_internal.h:78`'s `FLIP` /
/// `amd_internal.h:72`'s `FLIP` / `btf.h`'s `BTF_FLIP` exactly (all three
/// upstream definitions are identical: `-(j)-2`).
pub const fn flip(i: i64) -> i64 {
    -i - 2
}

/// `true` iff `i` is currently in "flipped" form (`i < EMPTY`). Matches
/// `klu_internal.h`'s implicit `UNFLIP` condition / `btf.h`'s
/// `BTF_ISFLIPPED` exactly.
pub const fn is_flipped(i: i64) -> bool {
    i < EMPTY
}

/// Recovers the original non-negative index whether or not `i` is currently
/// flipped — a no-op if it wasn't. Matches `klu_internal.h:79`'s `UNFLIP` /
/// `amd_internal.h:73`'s `UNFLIP` / `btf.h`'s `BTF_UNFLIP` exactly.
pub const fn unflip(i: i64) -> i64 {
    if is_flipped(i) { flip(i) } else { i }
}

/// Fixed solver configuration, mirroring the specific `KLU_common` fields
/// `sparse_klu.rs::new_common()` (via `klu_defaults`) leaves at their
/// defaults — confirmed directly from `KLU/Source/klu_defaults.c` — and that
/// this port's algorithm actually consumes. Kept as a struct (not bare
/// constants) so differential tests (`#[cfg(all(test, feature = "klu"))]`,
/// comparing against the vendored C oracle) can vary `tol`/`scale` without
/// touching any production code path, which always uses `Options::default()`.
///
/// Two `KLU_common` fields are deliberately **absent** here, not just
/// defaulted, because gridoxide's call pattern makes the alternative
/// branches they'd select genuinely unreachable (confirmed dead, not merely
/// unused):
/// - `ordering`: real KLU supports AMD (0), COLAMD (1), user-supplied P/Q
///   (2), or a user callback (3). `klu_defaults.c` defaults to `0` (AMD) and
///   `sparse_klu.rs` never overrides it, so this port only ever implements
///   the AMD path — no `ordering` field exists to accidentally select
///   another one.
/// - `maxwork`: bounds the work `btf_maxtrans` may spend before giving up
///   with a possibly-incomplete matching. Defaults to `0` ("no limit") and
///   is never overridden, so this port's BTF implementation always runs
///   `btf_maxtrans` to completion — the work-limit bookkeeping threaded
///   through the upstream C is dropped outright, not merely defaulted off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Partial-pivoting threshold: a candidate pivot is accepted only if
    /// its magnitude is at least `tol` times the largest candidate
    /// available in that column (otherwise the largest is forced instead).
    /// `klu_defaults.c`: `Common->tol = 0.001`.
    pub tol: f64,
    /// Row-scaling strategy: `1` = sum of absolute values per row, `2` =
    /// max absolute value per row. `klu_defaults.c`: `Common->scale = 2`.
    pub scale: i32,
    /// Whether to BTF-preorder before per-block AMD ordering.
    /// `klu_defaults.c`: `Common->btf = TRUE`. gridoxide never disables
    /// this, but it's kept as a field (rather than assumed) so `analyze`
    /// phase tests can exercise the single-block (`btf: false`) path too.
    pub btf: bool,
    /// Abort immediately (and report singular) on a pivot that's still zero
    /// after the tolerance-based search, rather than substituting a tiny
    /// nonzero value and continuing. `klu_defaults.c`:
    /// `Common->halt_if_singular = TRUE`.
    pub halt_if_singular: bool,
}

impl Default for Options {
    fn default() -> Self {
        // Matches KLU_defaults exactly for every field this port consumes.
        Self { tol: 0.001, scale: 2, btf: true, halt_if_singular: true }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_is_its_own_inverse() {
        for j in [-1i64, 0, 1, 5, -5, 1000, -1000] {
            assert_eq!(flip(flip(j)), j, "flip(flip({j})) should equal {j}");
        }
    }

    #[test]
    fn flip_of_empty_is_empty() {
        assert_eq!(flip(EMPTY), EMPTY);
    }

    #[test]
    fn is_flipped_matches_upstream_condition() {
        assert!(!is_flipped(EMPTY)); // -1 is not "flipped"
        assert!(!is_flipped(0));
        assert!(!is_flipped(5));
        assert!(is_flipped(flip(0)));
        assert!(is_flipped(flip(5)));
    }

    #[test]
    fn unflip_recovers_original_regardless_of_flip_state() {
        for j in [0i64, 1, 5, 1000] {
            assert_eq!(unflip(j), j, "unflip of an already-plain index is a no-op");
            assert_eq!(unflip(flip(j)), j, "unflip should recover the original index");
        }
    }

    #[test]
    fn options_default_matches_klu_defaults_c() {
        let opts = Options::default();
        assert_eq!(opts.tol, 0.001);
        assert_eq!(opts.scale, 2);
        assert!(opts.btf);
        assert!(opts.halt_if_singular);
    }
}

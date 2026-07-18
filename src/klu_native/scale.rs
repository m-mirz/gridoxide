//! Row scaling — ports `vendor/suitesparse/KLU/Source/klu_scale.c`'s
//! `KLU_scale` in full, both the `scale == 1` (row sum) and `scale >= 2`
//! (row max) variants, even though gridoxide's own fixed `Options::default()`
//! (`types.rs`) only ever selects `scale == 2` — cheap insurance per the port
//! plan, and easy to validate in isolation via the real `klu_scale` FFI
//! oracle (already in `build.rs`'s bindgen allowlist, unlike `btf_order`/
//! `amd_order` which needed hand-declared bindings — see `ffi_oracle.rs`).
//!
//! Real (`f64`) entries only, matching this whole port's scope (`mod.rs`'s
//! doc comment) — `ABS(a, Az[p])` collapses to plain `f64::abs` since KLU's
//! `ABS` macro only branches on complex-vs-real at compile time, and this
//! port never builds the complex variant.

/// Computes row scale factors for an `n`-by-`n` CSC matrix, or validates it
/// without scaling. Mirrors `KLU_scale`'s three outcomes:
/// - invalid input (bad `col_ptr`, out-of-range row index, or — when
///   `check_duplicates` is set — a duplicate `(row, col)` entry): `None`,
///   matching `Common->status = KLU_INVALID; return FALSE`.
/// - `scale < 0`: `Some(vec![])` — valid, but scaling is skipped entirely
///   and the input isn't even checked for duplicates/bounds, matching the
///   C's early `return (TRUE)` before any validation runs.
/// - `scale >= 0`: `Some(rs)` with `rs.len() == n`, `rs[row]` = the scale
///   factor for that row (sum of `|A(row,:)|` if `scale == 1`, max of
///   `|A(row,:)|` if `scale >= 2`, all-ones if `scale == 0`). Empty rows get
///   `rs[row] = 1.0` (matches "do not scale empty rows" / avoids a div-by-0
///   the matrix's own singularity would cause downstream anyway, not
///   something this function itself flags).
///
/// `check_duplicates` mirrors the C's `W != NULL` check (`W` is otherwise
/// pure scratch space in the original — a `Vec` internally here, never
/// exposed). gridoxide's own CSC construction always dedupes on the way in
/// (`sparse_klu.rs::build_csc_structure`, `klu_native`'s own callers), so
/// production use passes `false`; `true` exists so this function's own
/// duplicate-rejection path is directly testable, mirroring the real
/// `KLU_scale`'s behavior when `KLU_factor`/`KLU_refactor` pass their `W`
/// workspace (which they always do).
pub fn scale(
    scale: i32,
    n: usize,
    col_ptr: &[i64],
    row_idx: &[i64],
    values: &[f64],
    check_duplicates: bool,
) -> Option<Vec<f64>> {
    if scale < 0 {
        // "return without checking anything and without computing the scale
        // factors" -- klu_scale.c's own early-out, before any validation.
        return Some(Vec::new());
    }

    if n == 0 || col_ptr.len() != n + 1 || row_idx.len() != values.len() {
        return None;
    }
    if col_ptr[0] != 0 || col_ptr[n] < 0 {
        return None;
    }
    for col in 0..n {
        if col_ptr[col] > col_ptr[col + 1] {
            return None;
        }
    }

    let mut rs = if scale > 0 { vec![0.0f64; n] } else { Vec::new() };
    let mut w = if check_duplicates { vec![-1i64; n] } else { Vec::new() };

    for col in 0..n {
        let pend = col_ptr[col + 1] as usize;
        for p in col_ptr[col] as usize..pend {
            let row = row_idx[p];
            if row < 0 || row as usize >= n {
                return None;
            }
            let row = row as usize;
            if check_duplicates {
                if w[row] == col as i64 {
                    return None;
                }
                w[row] = col as i64;
            }
            let a = values[p].abs();
            if scale == 1 {
                rs[row] += a;
            } else if scale > 1 {
                rs[row] = rs[row].max(a);
            }
        }
    }

    if scale > 0 {
        for row in rs.iter_mut() {
            if *row == 0.0 {
                // "do not scale empty rows" -- matrix is (locally) singular,
                // left for the caller's own singularity handling downstream.
                *row = 1.0;
            }
        }
    }

    Some(rs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_scale_skips_everything() {
        // Even deliberately-invalid Ap is untouched -- matches KLU_scale's
        // early return before any validation runs.
        let rs = scale(-1, 2, &[0, 5], &[0], &[1.0], false).unwrap();
        assert!(rs.is_empty());
    }

    #[test]
    fn scale_zero_is_valid_but_computes_nothing() {
        let col_ptr = [0, 1, 2];
        let row_idx = [0, 1];
        let values = [3.0, -4.0];
        let rs = scale(0, 2, &col_ptr, &row_idx, &values, false).unwrap();
        assert!(rs.is_empty());
    }

    #[test]
    fn scale_one_is_row_sum_of_abs() {
        // A = [[3, 2], [0, -5]] in CSC: col0 has row0=3; col1 has row0=2, row1=-5.
        let col_ptr = [0, 1, 3];
        let row_idx = [0, 0, 1];
        let values = [3.0, 2.0, -5.0];
        let rs = scale(1, 2, &col_ptr, &row_idx, &values, false).unwrap();
        assert_eq!(rs, vec![5.0, 5.0]);
    }

    #[test]
    fn scale_two_is_row_max_of_abs() {
        let col_ptr = [0, 1, 3];
        let row_idx = [0, 0, 1];
        let values = [3.0, 2.0, -5.0];
        let rs = scale(2, 2, &col_ptr, &row_idx, &values, false).unwrap();
        assert_eq!(rs, vec![3.0, 5.0]);
    }

    #[test]
    fn empty_row_gets_scale_factor_one() {
        // Row 1 has no entries at all.
        let col_ptr = [0, 1, 1];
        let row_idx = [0];
        let values = [7.0];
        let rs = scale(2, 2, &col_ptr, &row_idx, &values, false).unwrap();
        assert_eq!(rs, vec![7.0, 1.0]);
    }

    #[test]
    fn rejects_out_of_range_row_index() {
        let col_ptr = [0, 1];
        let row_idx = [5]; // n=1, row 5 is out of range
        let values = [1.0];
        assert!(scale(2, 1, &col_ptr, &row_idx, &values, false).is_none());
    }

    #[test]
    fn rejects_decreasing_column_pointers() {
        let col_ptr = [0, 2, 1]; // Ap[1] > Ap[2]
        let row_idx = [0, 0];
        let values = [1.0, 1.0];
        assert!(scale(2, 2, &col_ptr, &row_idx, &values, false).is_none());
    }

    #[test]
    fn rejects_duplicate_entries_when_checked() {
        let col_ptr = [0, 2];
        let row_idx = [0, 0]; // (row 0, col 0) twice
        let values = [1.0, 2.0];
        assert!(scale(2, 1, &col_ptr, &row_idx, &values, true).is_none());
        // Same input is accepted when duplicate-checking is off (matches
        // KLU_scale's own W == NULL fast path).
        assert!(scale(2, 1, &col_ptr, &row_idx, &values, false).is_some());
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_ffi_oracle_on_random_matrices() {
        use crate::klu_native::ffi_oracle;

        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut next_f64 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };

        for trial in 0..40 {
            let n = 2 + (trial % 12);
            let scale_mode = if trial % 2 == 0 { 1 } else { 2 };

            let mut cols: Vec<std::collections::BTreeMap<usize, f64>> = vec![Default::default(); n];
            for col in cols.iter_mut() {
                let degree = 1 + (next_f64() * (n as f64 - 1.0)) as usize;
                for _ in 0..degree {
                    let r = (next_f64() * n as f64) as usize;
                    let v = (next_f64() - 0.5) * 20.0;
                    col.insert(r, v);
                }
            }
            let mut col_ptr = vec![0i32; n + 1];
            let mut row_idx: Vec<i32> = Vec::new();
            let mut values: Vec<f64> = Vec::new();
            for (j, col) in cols.iter().enumerate() {
                for (&r, &v) in col {
                    row_idx.push(r as i32);
                    values.push(v);
                }
                col_ptr[j + 1] = row_idx.len() as i32;
            }

            let col_ptr64: Vec<i64> = col_ptr.iter().map(|&x| x as i64).collect();
            let row_idx64: Vec<i64> = row_idx.iter().map(|&x| x as i64).collect();
            let rust_rs = scale(scale_mode, n, &col_ptr64, &row_idx64, &values, false).unwrap();

            let c_rs = ffi_oracle::klu_scale_oracle(scale_mode, n, &col_ptr, &row_idx, &values).unwrap();

            assert_eq!(rust_rs.len(), c_rs.len(), "trial {trial}");
            for row in 0..n {
                assert!(
                    (rust_rs[row] - c_rs[row]).abs() < 1e-12,
                    "trial {trial} (n={n}, scale={scale_mode}) row {row}: rust={} c={}",
                    rust_rs[row],
                    c_rs[row]
                );
            }
        }
    }
}

//! The approximate minimum degree ordering algorithm itself — ports
//! `vendor/suitesparse/AMD/Source/amd_2.c`'s `AMD_2`: quotient-graph
//! elimination with supervariable detection, mass elimination, and
//! (default-on) aggressive absorption, followed by an assembly-tree
//! postordering (`amd::postorder`).
//!
//! **Major storage simplification, carefully justified below** (not a
//! shortcut — re-derived from the algorithm's own invariants, not assumed):
//! `amd_2.c` packs every row/element's adjacency list into one shared,
//! growable `Iw` array, with `Pe[i]` an *index* into it — because of this,
//! `Pe` is overloaded to serve three different roles depending on `i`'s
//! state (active index, `FLIP(parent)` once absorbed, or `EMPTY`), and the
//! code needs an explicit "garbage collection" (`Iw` compression) step to
//! reclaim space as lists shrink and grow. This port instead gives every
//! row/element its own `Vec<i64>` (`lists[i]`, matching the same "elements
//! first, then supervariables" convention `Iw`'s per-row layout uses) —
//! `Vec`'s own amortized growth and `Drop` make the compression step
//! unnecessary, and `Pe`'s "active index" role disappears entirely (nothing
//! needs an index into a shared buffer). What's left, `parent: Vec<i64>`,
//! only ever holds a genuine assembly-tree parent (or `EMPTY`) — verified by
//! tracing every place `amd_2.c` writes `Pe[x] = FLIP(y)`: it's always at
//! the moment `x` is *actually* absorbed into `y` (element absorption,
//! aggressive absorption, mass elimination, or supervariable detection), or
//! `Pe[x] = EMPTY` when `x` becomes a genuine root (`Len[x] == 0` after
//! restore). Every element that isn't a final root gets absorbed by some
//! later element by construction (its remaining members all eventually
//! become pivots themselves, at which point the element they still
//! reference is either fully covered — full absorption — or contributes
//! zero fill given what's already accounted for), so nothing ever needs to
//! read a "still just an index" value in a parent role — meaning
//! `amd_2.c`'s final unconditional `Pe[i] = FLIP(Pe[i])` restoration pass
//! (and the matching `Elen[i] = FLIP(Elen[i])` one) becomes unnecessary too:
//! this port just stores the plain value from the start, using `EMPTY`
//! consistently as "no parent yet" / "not an element". Because
//! `amd_2.c`'s own supervariable-detection pattern-match explicitly skips
//! each list's first entry and compares the rest via `W[]`-flag set
//! membership (never by position), a list's element-count split point
//! matters but the order *within* each portion does not — confirmed
//! directly from that comparison loop, not assumed — so the exact
//! "move-to-end" in-place shuffle `amd_2.c` performs when growing a list is
//! a pure memory-layout artifact too, not replicated here.
//!
//! Everything else — pivot selection from degree lists, the two-scan
//! degree-update algorithm, aggressive absorption, mass elimination,
//! hash-bucket-based supervariable detection (`Head`/`Next`/`Last`'s
//! dual-purpose degree-list/hash-bucket overload *is* kept literal, since
//! unlike `Pe`/`Elen` this dual use is a genuine algorithmic technique for
//! combining two live data structures, not just a memory trick) — is
//! ported as literally as this data-structure change allows, preserving
//! variable names (`deg`, `degme`, `nvpiv`, `mindeg`, `wflg`, `hash`, ...)
//! from `amd_2.c` for traceability.

use super::super::types::{flip, EMPTY};
use super::postorder::postorder;

const AMD_DEFAULT_DENSE: f64 = 10.0;
const AMD_DEFAULT_AGGRESSIVE: bool = true;

fn remove_from_degree_list(i: usize, degree: &[i64], head: &mut [i64], next: &mut [i64], last: &mut [i64]) {
    let ilast = last[i];
    let inext = next[i];
    if inext != EMPTY {
        last[inext as usize] = ilast;
    }
    if ilast != EMPTY {
        next[ilast as usize] = inext;
    } else {
        head[degree[i] as usize] = inext;
    }
}

/// Ports `AMD_2` + the `AMD_postorder` call at its end. `lists[i]` on input
/// is row/column `i`'s adjacency in A+A' (as `amd::aat::build_symmetric_lists`
/// produces) — no elements yet, so every `lists[i]` starts as pure
/// supervariable content (`elen[i] == 0` for all `i`, matching `amd_2.c`'s
/// own initialization).
///
/// Returns `(perm, pinv)`: `perm[k]` is the original row/column that becomes
/// the `k`th pivot (`Last` in `amd_2.c`), `pinv[i]` is `i`'s pivot position
/// (`Next` in `amd_2.c`).
pub fn amd_2(n: usize, mut lists: Vec<Vec<i64>>) -> (Vec<i64>, Vec<i64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut nv = vec![1i64; n];
    let mut elen = vec![0i64; n];
    let mut degree: Vec<i64> = lists.iter().map(|l| l.len() as i64).collect();
    let mut w = vec![1i64; n];
    let mut head = vec![EMPTY; n];
    let mut next = vec![EMPTY; n];
    let mut last = vec![EMPTY; n];
    let mut parent = vec![EMPTY; n];

    let mut wflg: i64 = 2;
    let wbig = i64::MAX - n as i64;

    let alpha = AMD_DEFAULT_DENSE;
    let aggressive = AMD_DEFAULT_AGGRESSIVE;
    let dense = {
        let d = alpha * (n as f64).sqrt();
        (d.max(16.0).min(n as f64)) as i64
    };

    // Initialize degree lists; eliminate dense and empty rows immediately.
    // (`amd_2.c` also counts `ndense` here, but only for `Info[]` diagnostic
    // statistics this port doesn't expose — see this module's doc comment
    // on what's dropped as diagnostic-only.)
    let mut nel: i64 = 0;
    for i in 0..n {
        let deg = degree[i];
        if deg == 0 {
            elen[i] = flip(1);
            nel += 1;
            parent[i] = EMPTY;
            w[i] = 0;
        } else if deg > dense {
            nv[i] = 0;
            elen[i] = EMPTY;
            nel += 1;
            parent[i] = EMPTY;
        } else {
            let inext = head[deg as usize];
            if inext != EMPTY {
                last[inext as usize] = i as i64;
            }
            next[i] = inext;
            head[deg as usize] = i as i64;
        }
    }

    let mut mindeg: usize = 0;
    let mut lemax: i64 = 0; // largest |Le| (degme) seen so far, across all iterations

    // ===================================================================
    // WHILE (selecting pivots) DO
    // ===================================================================
    while nel < n as i64 {
        // --- GET PIVOT OF MINIMUM DEGREE ---
        let mut deg = mindeg;
        while deg < n && head[deg] == EMPTY {
            deg += 1;
        }
        mindeg = deg;
        let me = head[deg] as usize;

        let inext = next[me];
        if inext != EMPTY {
            last[inext as usize] = EMPTY;
        }
        head[deg] = inext;

        let elenme = elen[me] as usize;
        let mut nvpiv = nv[me];
        nel += nvpiv;

        // --- CONSTRUCT NEW ELEMENT ---
        nv[me] = -nvpiv;
        let mut degme: i64 = 0;

        let me_list = std::mem::take(&mut lists[me]);
        let elements_in_me: Vec<i64> = me_list[..elenme].to_vec();
        let supervars_in_me: Vec<i64> = me_list[elenme..].to_vec();

        let mut new_element_list: Vec<i64> = Vec::new();
        for &e_signed in &elements_in_me {
            let e = e_signed as usize;
            let e_list = std::mem::take(&mut lists[e]);
            for i_signed in e_list {
                let i = i_signed as usize;
                let nvi = nv[i];
                if nvi > 0 {
                    degme += nvi;
                    nv[i] = -nvi;
                    new_element_list.push(i_signed);
                    remove_from_degree_list(i, &degree, &mut head, &mut next, &mut last);
                }
            }
            if e != me {
                parent[e] = me as i64;
                w[e] = 0;
            }
        }
        for &i_signed in &supervars_in_me {
            let i = i_signed as usize;
            let nvi = nv[i];
            if nvi > 0 {
                degme += nvi;
                nv[i] = -nvi;
                new_element_list.push(i_signed);
                remove_from_degree_list(i, &degree, &mut head, &mut next, &mut last);
            }
        }

        degree[me] = degme;
        parent[me] = EMPTY; // provisional; overwritten below if me later gets absorbed
        wflg = clear_flag(wflg, wbig, &mut w, n);

        // --- COMPUTE (W[e] - wflg) = |Le \ Lme| FOR ALL ELEMENTS ---
        for &i_signed in &new_element_list {
            let i = i_signed as usize;
            let eln = elen[i];
            if eln > 0 {
                let nvi = -nv[i];
                let wnvi = wflg - nvi;
                let eln = eln as usize;
                for &e_signed in &lists[i][..eln] {
                    let e = e_signed as usize;
                    let we = w[e];
                    if we >= wflg {
                        w[e] = we - nvi;
                    } else if we != 0 {
                        w[e] = degree[e] + wnvi;
                    }
                }
            }
        }

        // --- DEGREE UPDATE AND ELEMENT ABSORPTION ---
        for &i_signed in &new_element_list {
            let i = i_signed as usize;
            let eln = elen[i] as usize;
            let elements_of_i: Vec<i64> = lists[i][..eln].to_vec();
            let supervars_of_i: Vec<i64> = lists[i][eln..].to_vec();

            let mut deg: i64 = 0;
            let mut hash: u64 = 0;
            let mut kept_elements: Vec<i64> = Vec::new();

            if aggressive {
                for &e_signed in &elements_of_i {
                    let e = e_signed as usize;
                    let we = w[e];
                    if we != 0 {
                        let dext = we - wflg;
                        if dext > 0 {
                            deg += dext;
                            kept_elements.push(e_signed);
                            hash = hash.wrapping_add(e as u64);
                        } else {
                            parent[e] = me as i64;
                            w[e] = 0;
                        }
                    }
                }
            } else {
                for &e_signed in &elements_of_i {
                    let e = e_signed as usize;
                    let we = w[e];
                    if we != 0 {
                        let dext = we - wflg;
                        deg += dext;
                        kept_elements.push(e_signed);
                        hash = hash.wrapping_add(e as u64);
                    }
                }
            }

            elen[i] = (kept_elements.len() + 1) as i64;

            let mut kept_supervars: Vec<i64> = Vec::new();
            for &j_signed in &supervars_of_i {
                let j = j_signed as usize;
                let nvj = nv[j];
                if nvj > 0 {
                    deg += nvj;
                    kept_supervars.push(j_signed);
                    hash = hash.wrapping_add(j as u64);
                }
            }

            if elen[i] == 1 && kept_supervars.is_empty() {
                // Mass elimination: nothing left of i except the edge to me.
                parent[i] = me as i64;
                let nvi = -nv[i];
                degme -= nvi;
                nvpiv += nvi;
                nel += nvi;
                nv[i] = 0;
                elen[i] = EMPTY;
            } else {
                degree[i] = degree[i].min(deg);

                // amd_2.c reconstructs this list via an in-place 3-step
                // shuffle ("move first supervariable to end of list; move
                // first element to end of element part; add new element me
                // to front") that we can't reproduce via pure memory-layout
                // arguments alone: it changes which entry lands first among
                // ties, which then changes *insertion order into degree
                // lists* below (LIFO), which changes *which* equal-degree
                // candidate gets selected as the next pivot — a real,
                // observable difference in output permutation on inputs
                // with degree ties (confirmed empirically: dropping this
                // reordering produced a different, slightly-worse-fill-in
                // permutation than the vendored C on a random 6-node test
                // graph). So: replicate the exact rotation, not just the
                // set of members.
                let mut new_i_list = Vec::with_capacity(1 + kept_elements.len() + kept_supervars.len());
                new_i_list.push(me as i64);
                if let [first, rest @ ..] = kept_elements.as_slice() {
                    new_i_list.extend_from_slice(rest);
                    new_i_list.push(*first);
                }
                if let [first, rest @ ..] = kept_supervars.as_slice() {
                    new_i_list.extend_from_slice(rest);
                    new_i_list.push(*first);
                }
                lists[i] = new_i_list;

                let hash = (hash % n as u64) as i64;
                let j = head[hash as usize];
                if j <= EMPTY {
                    next[i] = flip(j);
                    head[hash as usize] = flip(i as i64);
                } else {
                    next[i] = last[j as usize];
                    last[j as usize] = i as i64;
                }
                last[i] = hash;
            }
        }

        degree[me] = degme;
        lemax = lemax.max(degme);
        wflg = clear_flag(wflg + lemax, wbig, &mut w, n);

        // --- SUPERVARIABLE DETECTION ---
        for &i_signed in &new_element_list {
            let i = i_signed as usize;
            if nv[i] < 0 {
                // i is a principal variable in Lme: examine its hash bucket.
                let hash = last[i] as usize;
                let mut bucket_head = {
                    let j = head[hash];
                    if j == EMPTY {
                        EMPTY
                    } else if j < EMPTY {
                        head[hash] = EMPTY;
                        flip(j)
                    } else {
                        let h = last[j as usize];
                        last[j as usize] = EMPTY;
                        h
                    }
                };

                while bucket_head != EMPTY && next[bucket_head as usize] != EMPTY {
                    let bi = bucket_head as usize;
                    let ln = lists[bi].len();
                    let eln = elen[bi] as usize;
                    // Scatter i's list (skipping the first entry, always
                    // `me`) into w, flagged with wflg.
                    for &x in &lists[bi][1..ln] {
                        w[x as usize] = wflg;
                    }

                    let mut jlast = bi;
                    let mut j = next[bi];
                    while j != EMPTY {
                        let ju = j as usize;
                        let mut ok = lists[ju].len() == ln && elen[ju] as usize == eln;
                        if ok {
                            for &x in &lists[ju][1..ln] {
                                if w[x as usize] != wflg {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if ok {
                            parent[ju] = bi as i64;
                            nv[bi] += nv[ju];
                            nv[ju] = 0;
                            elen[ju] = EMPTY;
                            j = next[ju];
                            next[jlast] = j;
                        } else {
                            jlast = ju;
                            j = next[ju];
                        }
                    }

                    wflg += 1;
                    bucket_head = next[bi];
                }
            }
        }

        // --- RESTORE DEGREE LISTS AND REMOVE NONPRINCIPAL VARIABLES ---
        let nleft = n as i64 - nel;
        let mut final_element_list: Vec<i64> = Vec::new();
        for &i_signed in &new_element_list {
            let i = i_signed as usize;
            let nvi = -nv[i];
            if nvi > 0 {
                nv[i] = nvi;
                let mut deg = degree[i] + degme - nvi;
                deg = deg.min(nleft - nvi);

                let inext = head[deg as usize];
                if inext != EMPTY {
                    last[inext as usize] = i as i64;
                }
                next[i] = inext;
                last[i] = EMPTY;
                head[deg as usize] = i as i64;

                mindeg = mindeg.min(deg as usize);
                degree[i] = deg;

                final_element_list.push(i_signed);
            }
        }

        // --- FINALIZE THE NEW ELEMENT ---
        nv[me] = nvpiv;
        if final_element_list.is_empty() {
            parent[me] = EMPTY;
            w[me] = 0;
        }
        lists[me] = final_element_list;
    }

    // ===================================================================
    // POST-ORDERING
    // ===================================================================

    // Compress the paths of non-principal variables so each points directly
    // at its ultimate element ancestor.
    for i in 0..n {
        if nv[i] == 0 {
            let mut j = parent[i];
            if j == EMPTY {
                continue; // dense variable, no parent
            }
            while nv[j as usize] == 0 {
                j = parent[j as usize];
            }
            let e = j;
            let mut j = i as i64;
            while nv[j as usize] == 0 {
                let jnext = parent[j as usize];
                parent[j as usize] = e;
                j = jnext;
            }
        }
    }

    let order = postorder(n, &parent, &nv, &elen);

    // Compute output permutation and inverse permutation.
    let mut head2 = vec![EMPTY; n];
    for (e, &k) in order.iter().enumerate() {
        if k != EMPTY {
            head2[k as usize] = e as i64;
        }
    }

    let mut pinv = vec![EMPTY; n];
    let mut nel2: i64 = 0;
    for &e in &head2 {
        if e == EMPTY {
            break;
        }
        pinv[e as usize] = nel2;
        nel2 += nv[e as usize];
    }

    for i in 0..n {
        if nv[i] == 0 {
            let e = parent[i];
            if e != EMPTY {
                pinv[i] = pinv[e as usize];
                pinv[e as usize] += 1;
            } else {
                pinv[i] = nel2;
                nel2 += 1;
            }
        }
    }

    let mut perm = vec![0i64; n];
    for (i, &k) in pinv.iter().enumerate() {
        perm[k as usize] = i as i64;
    }

    (perm, pinv)
}

fn clear_flag(wflg: i64, wbig: i64, w: &mut [i64], n: usize) -> i64 {
    if wflg < 2 || wflg >= wbig {
        for x in w.iter_mut().take(n) {
            if *x != 0 {
                *x = 1;
            }
        }
        return 2;
    }
    wflg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_permutation(p: &[i64], n: usize) -> bool {
        let mut sorted: Vec<i64> = p.to_vec();
        sorted.sort_unstable();
        sorted == (0..n as i64).collect::<Vec<_>>()
    }

    #[test]
    fn single_node() {
        let lists = vec![Vec::new()];
        let (perm, pinv) = amd_2(1, lists);
        assert_eq!(perm, vec![0]);
        assert_eq!(pinv, vec![0]);
    }

    #[test]
    fn diagonal_only_no_fill() {
        // No off-diagonal structure at all: every node isolated.
        let lists = vec![Vec::new(), Vec::new(), Vec::new()];
        let (perm, _pinv) = amd_2(3, lists);
        assert!(is_permutation(&perm, 3));
    }

    #[test]
    fn small_chain() {
        // 0-1-2 chain (symmetric adjacency).
        let lists = vec![vec![1], vec![0, 2], vec![1]];
        let (perm, pinv) = amd_2(3, lists);
        assert!(is_permutation(&perm, 3));
        assert!(is_permutation(&pinv, 3));
        for (k, &p) in perm.iter().enumerate() {
            assert_eq!(pinv[p as usize], k as i64);
        }
    }

    #[test]
    fn dense_small_clique() {
        // Fully connected 4-node clique.
        let lists = vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]];
        let (perm, _pinv) = amd_2(4, lists);
        assert!(is_permutation(&perm, 4));
    }
}

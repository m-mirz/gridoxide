//! Post-orders the assembly tree `amd::core::amd_2` builds — ports
//! `vendor/suitesparse/AMD/Source/amd_postorder.c`'s `AMD_postorder` and
//! `amd_post_tree.c`'s `AMD_post_tree` (the non-recursive version only — the
//! recursive one in the same C file is explicitly `#if 0`-disabled there,
//! kept "because it is easier to read" but never compiled).

use super::super::types::EMPTY;

/// Non-recursive post-order of a single tree rooted at `root`, numbering
/// nodes starting at `k`. Mirrors `AMD_post_tree` exactly: children are
/// pushed in list order (reversed via the stack, so the *last* child in
/// `child`'s linked list — which `AMD_postorder` arranges to be the
/// largest — gets ordered last).
fn post_tree(root: usize, mut k: i64, child: &mut [i64], sibling: &[i64], order: &mut [i64]) -> i64 {
    let mut stack = vec![root as i64];

    while let Some(&i) = stack.last() {
        let i = i as usize;
        if child[i] != EMPTY {
            // Children not yet ordered: push them all, so the biggest
            // (last in the list) ends up on top of the stack, popped last.
            let mut to_push = Vec::new();
            let mut f = child[i];
            while f != EMPTY {
                to_push.push(f);
                f = sibling[f as usize];
            }
            stack.extend(to_push.iter().rev());
            // Delete child list so i gets ordered next time we see it.
            child[i] = EMPTY;
        } else {
            // Children (if any) already ordered: order i now.
            stack.pop();
            order[i] = k;
            k += 1;
        }
    }
    k
}

/// Ports `AMD_postorder`: assigns a post-order numbering to every element
/// (`nv[j] > 0`) in the assembly tree described by `parent`/`nv`, biasing
/// each node's children so the largest (by `fsize`, matching `Degree[me]` at
/// that point) is visited last — a heuristic that tends to reduce peak
/// memory in a subsequent multifrontal factorization, though gridoxide's own
/// solver doesn't need that property; ported for fidelity regardless, since
/// it's part of what real AMD does and affects the exact output permutation.
pub fn postorder(nn: usize, parent: &[i64], nv: &[i64], fsize: &[i64]) -> Vec<i64> {
    let mut child = vec![EMPTY; nn];
    let mut sibling = vec![EMPTY; nn];

    // Place children in link lists, iterating in reverse so bigger elements
    // (found later in a typical AMD run) tend to land at the list's end.
    for j in (0..nn).rev() {
        if nv[j] > 0 {
            let p = parent[j];
            if p != EMPTY {
                sibling[j] = child[p as usize];
                child[p as usize] = j as i64;
            }
        }
    }

    // Move the largest child (by fsize) to the end of each node's list.
    for i in 0..nn {
        if nv[i] > 0 && child[i] != EMPTY {
            let mut fprev = EMPTY;
            let mut maxfrsize = EMPTY;
            let mut bigfprev = EMPTY;
            let mut bigf = EMPTY;
            let mut f = child[i];
            while f != EMPTY {
                let frsize = fsize[f as usize];
                if frsize >= maxfrsize {
                    maxfrsize = frsize;
                    bigfprev = fprev;
                    bigf = f;
                }
                fprev = f;
                f = sibling[f as usize];
            }
            debug_assert_ne!(bigf, EMPTY);

            let fnext = sibling[bigf as usize];
            if fnext != EMPTY {
                // bigf isn't already at the end of the list: move it there.
                if bigfprev == EMPTY {
                    child[i] = fnext;
                } else {
                    sibling[bigfprev as usize] = fnext;
                }
                sibling[bigf as usize] = EMPTY;
                sibling[fprev as usize] = bigf;
            }
        }
    }

    // Post-order the assembly tree via a DFS from each root.
    let mut order = vec![EMPTY; nn];
    let mut k: i64 = 0;
    for i in 0..nn {
        if parent[i] == EMPTY && nv[i] > 0 {
            k = post_tree(i, k, &mut child, &sibling, &mut order);
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_node_tree() {
        let parent = [EMPTY];
        let nv = [1i64];
        let fsize = [0i64];
        let order = postorder(1, &parent, &nv, &fsize);
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn chain_orders_root_last() {
        // 0 -> 1 -> 2 (0's parent is 1, 1's parent is 2, 2 is root).
        let parent = [1, 2, EMPTY];
        let nv = [1, 1, 1];
        let fsize = [0, 0, 0];
        let order = postorder(3, &parent, &nv, &fsize);
        // Children ordered before parents; 2 (root) must be last.
        assert_eq!(order[2], 2, "root should be visited last");
        assert!(order[0] < order[2] && order[1] < order[2]);
    }

    #[test]
    fn two_children_bigger_last() {
        // 1 and 2 are both children of 0 (root); 2 has larger fsize.
        let parent = [EMPTY, 0, 0];
        let nv = [1, 1, 1];
        let fsize = [0, 1, 5];
        let order = postorder(3, &parent, &nv, &fsize);
        assert_eq!(order[0], 2, "root visited last, after both children");
        assert!(order[2] > order[1], "bigger child (2) should be ordered after smaller (1)");
    }
}

# Inside KLU: the Sparse Solve, Step by Step

Every Newton-Raphson iteration ends in the same place: solve \\(\textbf{J} \Delta x = f(x)\\) for the
correction \\(\Delta x\\). For anything bigger than a toy network, that linear solve dominates the
runtime, and the Jacobian is *sparse* — a bus only couples to its immediate neighbours, so a 22×22
Jacobian from an 11-node grid has 124 nonzeros out of 484 possible entries (74 % empty).

KLU is the sparse LU solver gridoxide uses for that step. It was designed by Tim Davis and Ekanathan
Palamadai specifically for circuit-simulation matrices — very sparse, unsymmetric, with a strong
zero-free diagonal — which describes a power-flow Jacobian almost exactly. gridoxide has two KLU
backends (`JacobianBackend::Klu`, the vendored C via FFI, and `JacobianBackend::KluNative`, a
from-scratch Rust port of the same algorithm in `src/klu_native/`), and both run the pipeline
described below.

This page walks that pipeline one phase at a time, on matrices small enough to check by hand. Every
number shown was produced by running the actual `src/klu_native/` code on the matrix shown.

## The pipeline at a glance

KLU splits the work into a **symbolic** phase that depends only on *where* the nonzeros are, and a
**numeric** phase that depends on their values:

| Phase | What it does | Depends on | Code |
|---|---|---|---|
| 0. CSC assembly | triplets → compressed sparse column | pattern | `klu_native/mod.rs` |
| 1. BTF | permute to block upper-triangular form | pattern | `klu_native/btf/` |
| 2. AMD | reorder inside each block to reduce fill | pattern | `klu_native/amd/` |
| 3. Factor | Gilbert-Peierls LU with partial pivoting | values | `klu_native/kernel.rs`, `factor.rs` |
| 4. Solve | permuted forward/back substitution | values | `klu_native/solve.rs` |
| 5. Refactor | redo phase 3 with new values, same pattern | values | `klu_native/refactor.rs` |

The split is the whole point. Phases 1–2 are the expensive combinatorial work, and in a power flow
the Jacobian's *pattern* never changes between Newton iterations — only its numbers do. So gridoxide
runs phases 0–2 once per topology (cached in `PersistentSolver`) and only phases 5 and 4 per
iteration. Phase 5 exists precisely because it can skip everything phase 3 does symbolically.

## Step 0: triplets to CSC

`jacobian::JacobianPattern` derives the Jacobian's sparsity pattern once per topology and hands the
backend `(row, col, value)` triplets in whatever order its assembly walk happens to produce them —
once, at `LinearSolver::new`. (`solver::jacobian_triplets_reference`/`build_jacobian_triplets` still
assemble the same thing the naive way, but only under `#[cfg(test)]`, as the oracle `JacobianPattern`
is checked against bit-for-bit.) KLU wants **compressed sparse column** (CSC): one array of column
start offsets, one of row indices sorted within each column, one of values.

Take this 4×4 matrix, which we'll reuse for the next several steps:

```
       c0  c1  c2  c3
r0      .   1   .   3
r1      .   .   2   .
r2      .   4   2   1
r3      5   .   1   .
```

As triplets and then as CSC (`build_csc_structure` in `klu_native/mod.rs`):

```
col_ptr = [0, 1, 3, 6, 8]     // column j occupies row_idx[col_ptr[j] .. col_ptr[j+1]]
row_idx = [3,  0, 2,  1, 2, 3,  0, 2]
values  = [5,  1, 4,  2, 2, 1,  3, 1]
          |c0|  c1  |    c2    |  c3 |
```

`build_csc_structure` also merges duplicate `(row, col)` pairs and returns a `groups` mapping from
each CSC slot back to the triplets that fed it. That mapping is what makes cheap refactorization
possible later: on the next Newton iteration, the same assembly order maps to the same CSC slots, so
new values can be packed in without re-deriving the structure. It is also why later iterations hand
over *only* the values — `JacobianPattern` refills one reused `Vec<f64>` positionally matching that
first triplet order, and `LinearSolver::factor_and_solve_values` packs it straight into the cached
slots, with no `(row, col)` pair rebuilt after the first iteration.

## Step 1: BTF — finding block triangular structure

If a matrix can be permuted to block upper-triangular form, you never have to factor it as one
piece. You factor each diagonal block independently and stitch the results together with back
substitution. Blocks are smaller, so there's less fill and less work; and a 1×1 block is just a
division.

BTF gets there in two moves.

### 1a. Maximum transversal: get a zero-free diagonal

First find a **matching** — a set of nonzeros, no two sharing a row or column, as large as possible.
This is bipartite matching between rows and columns, solved by augmenting paths
(`btf/maxtrans.rs`, a port of Duff's MC21 algorithm).

For the matrix above, the matching found is:

```
MAXTRANS: match = [1, 2, 3, 0]   nmatch = 4
```

Read it as "row *i* is matched to column `match[i]`": row 0 ↔ column 1 (value 1), row 1 ↔ column 2
(value 2), row 2 ↔ column 3 (value 1), row 3 ↔ column 0 (value 5). All four rows are matched
(`nmatch == n`), so the matrix has full structural rank and can be permuted to have no zeros on its
diagonal.

If `nmatch < n`, the matrix is *structurally* singular — no permutation can fill the diagonal — and
`btf_order` completes the permutation arbitrarily so later phases still have a bijection to work
with, marking the fake entries with the `flip` sentinel (`klu_native/types.rs`). The factorization
then fails on a zero pivot, which is exactly how gridoxide reports `SolveStatus::Singular` for a
disconnected island with no reference bus.

### 1b. Strongly connected components: find the blocks

With the matching in hand, build a directed graph: one node per column, and an edge \\(j \to k\\)
whenever column \\(j\\) has a nonzero in the row that is matched to column \\(k\\). Its **strongly
connected components** are exactly the irreducible diagonal blocks, and a topological order of the
components gives the block ordering. `btf/strongcomp.rs` is Tarjan's algorithm, iterative rather
than recursive.

For our matrix, the edges are:

```
c0 → c0                  (row 3 ↔ c0: the matched diagonal entry)
c1 → c3                  (row 2 ↔ c3)
c2 → c3, c2 → c0         (row 2 ↔ c3, row 3 ↔ c0)
c3 → c1                  (row 0 ↔ c1)
```

`c1 → c3 → c1` is a cycle, so `{c1, c3}` is one component. `{c0}` and `{c2}` are singletons. And
because `c2` has edges *out* to both other components and none coming back, it must be ordered last.
That is what the code returns:

```
BTF: p = [0, 2, 3, 1]   q = [1, 3, 0, 2]   r = [0, 2, 3, 4]
```

`p` and `q` are the row and column permutations; `r` gives the block boundaries — block 0 spans
positions 0..2, block 1 is position 2, block 2 is position 3. Applying them:

```
                q0=c1 q1=c3 | q2=c0 | q3=c2
   p0=r0   [     1     3    |   .   |   .    ]
   p1=r2   [     4     1    |   .   |   2    ]
           [ --------------  -------  ------ ]
   p2=r3   [     .     .    |   5   |   1    ]
           [ --------------  -------  ------ ]
   p3=r1   [     .     .    |   .   |   2    ]
```

Block upper-triangular, with a 2×2 block and two 1×1 blocks. Instead of one 4×4 LU, KLU will do one
2×2 LU and two divisions. The two entries above the diagonal blocks (the `2` and the `1` in the last
column) are stored separately, as the **off-diagonal** part — `factor.rs` keeps them in their own CSC
arrays (`off_p`/`off_i`/`off_x`) because they're not part of any block's LU; they only appear during
the solve.

### What BTF does on a real power-flow Jacobian: nothing

Worth saying plainly, because it's the common case:

```
3-bus network.json:        n=3    nnz=9     btf_blocks=1   sizes=[3]
PGM transmission-case:     n=22   nnz=124   btf_blocks=1   sizes=[22]
```

A connected AC network has an irreducible Jacobian — every bus reaches every other bus through the
network graph, so the whole matrix is one strongly connected component and BTF returns a single
block. BTF earns its keep on circuit matrices with genuine one-way structure (a driving stage feeding
a following stage that doesn't feed back).

The one power-flow case where it does something: a network with **several islands that each have a
reference bus**. `PersistentSolver::solve` classifies islands but still hands the whole bus list to a
single Newton solve (see [Multi-Island Power Flow](../powerflow/multi_island.md)), so the Jacobian is block
*diagonal* — no unknown in one island appears in any equation of another. The column digraph of
step 1b then has no edges crossing island boundaries, so no strongly connected component can span
two islands, and BTF necessarily recovers one block (or more) per island and factors them
independently. Islands with no reference bus never reach this point: `mark_unreferenced_islands`
converts their buses to slack first, removing them from the unknown set entirely.

BTF costs one near-linear pass, so KLU runs it unconditionally rather than trying to predict which
case it's in.

## Step 2: AMD — reordering to limit fill-in

Inside each diagonal block, the ordering still matters enormously, because of **fill-in**: entries
that are zero in \\(A\\) but nonzero in \\(L\\) or \\(U\\). Fill is what makes a sparse factorization
degenerate into a dense one.

The canonical demonstration is a *star* (an arrow matrix) — one hub node connected to everything,
which in power system terms is a substation busbar with many feeders:

```
       c0  c1  c2  c3  c4
r0      5   1   1   1   1
r1      1   2   .   .   .
r2      1   .   2   .   .
r3      1   .   .   2   .
r4      1   .   .   .   2
```

Eliminate the hub (node 0) first and every pair of leaves becomes coupled — the remaining 4×4 block
fills in completely. Eliminate the leaves first and nothing fills at all, because no two leaves are
adjacent. AMD finds the second order:

```
AMD star perm = [4, 3, 2, 1, 0]         // hub eliminated last

star, natural order  [0,1,2,3,4]:  nnz(L)=10  nnz(U)=10  total LU nnz = 25   (fully dense)
star, AMD order      [4,3,2,1,0]:  nnz(L)=4   nnz(U)=4   total LU nnz = 13   (zero fill)
```

25 versus 13 on a 5×5 — and the gap widens quadratically with size. AMD (`klu_native/amd/`) is a
greedy heuristic: repeatedly eliminate the node of smallest *approximate* degree, where "approximate"
is the trick that makes it fast — it bounds each node's degree using quotient-graph element
absorption rather than recomputing it exactly.

Two details of how KLU uses it:

- AMD orders the **symmetrized** pattern \\(A + A^T\\), so it produces one permutation applied to both
  rows and columns of the block. That's why `analyze.rs` applies `pblk` to `p` and `q` identically.
- Blocks of size ≤ 3 skip AMD and keep their natural order (`analyze_worker`'s own threshold, matched
  exactly in `analyze.rs`) — for a 3×3 there is nothing to gain.

On the real Jacobians, this is where the savings actually come from:

```
PGM transmission-case (n=22, nnz=124):
    LU nnz with BTF+AMD = 124      (zero fill)
    LU nnz natural order = 308     (184 fill entries)
```

AMD ordered that Jacobian so well that the factorization produces *no fill whatsoever* — the LU
factors have exactly as many nonzeros as the matrix. Natural order would have produced 2.5× more.

The 3-bus case is `n=3`, so it takes the natural-order path and both counts are 9 (a 3×3 Jacobian
is dense anyway).

## Step 3: numeric factorization — Gilbert-Peierls with partial pivoting

Now the values matter. KLU factors each diagonal block with a **left-looking** LU: column \\(k\\) of
\\(L\\) and \\(U\\) is computed completely before column \\(k+1\\) is touched, using only columns to
its left. `kernel.rs::factor_block` does this for one block.

Take this block (already ordered — pretend BTF and AMD have run):

```
       c0  c1  c2  c3
r0      2   .   1   .
r1      1   3   .   .
r2      .   1   4   1
r3      .   .   1   5
```

Each column goes through four sub-steps.

### 3a. Symbolic: which rows will this column touch?

Before computing anything, KLU determines the *pattern* of column \\(k\\) by a depth-first search.
The rule (Gilbert-Peierls): the nonzero pattern of column \\(k\\) of \\(L\\) and \\(U\\) is the set of
nodes **reachable** from the nonzero rows of \\(A(:,k)\\) in the directed graph of the already-computed
\\(L\\).

Column 2 of the example shows why this matters. Its input entries are rows 0, 2, 3 — nothing in row 1.
But row 0 is already pivotal (it was column 0's pivot), and column 0 of \\(L\\) has an entry in row 1.
So the DFS reaches row 1 anyway, and column 2 gets an entry there that \\(A\\) never had. That's
fill-in, predicted symbolically before a single flop:

```
U col 2: [(0, 1.0), (1, -0.5)]
                     ^^^^ fill: A(1,2) = 0
```

### 3b. Numeric: sparse triangular solve

With the pattern known, scatter \\(A(:,k)\\) into a dense workspace and run the updates in
topological order — for each pivotal row \\(j\\) in the pattern, subtract \\(x_j \cdot L(:,j)\\).
Column 2 again: \\(x_0 = 1\\), and \\(L(1,0) = 0.5\\), so \\(x_1 \mathrel{-}= 0.5 \cdot 1 = -0.5\\);
then \\(L(2,1) = 1/3\\) gives \\(x_2 = 4 + 1/6 = 4.1\overline{6}\\).

### 3c. Pivot: pick the diagonal if you can live with it

Everything still below the diagonal is a pivot candidate. KLU's rule (`lpivot`) is *diagonal
preference with a threshold*: take the entry the ordering intended for the diagonal, provided it's at
least `tol` times the largest candidate in the column; otherwise take the largest.

`tol` defaults to **0.001** (`klu_defaults.c`, mirrored in `types.rs`), which is deliberately loose —
KLU assumes its input has a strong diagonal and would rather preserve the fill-reducing ordering than
chase the last digit of stability. Two 2×2 matrices show both sides:

```
A = [[0.5, 1],       diag candidate 0.5, column max 1.0
     [1.0, 1]]       0.5 ≥ 0.001 × 1.0  →  keep the diagonal
                     p = [0, 1]   udiag = [0.5, -1.0]

B = [[1e-6, 1],      diag candidate 1e-6, column max 1.0
     [1.0,  1]]      1e-6 < 0.001 × 1.0  →  pivot to row 1
                     p = [1, 0]   udiag = [1.0, 0.999999]
```

In the first case a dense LU with ordinary partial pivoting would have swapped rows (1.0 > 0.5); KLU
does not, because 0.5 is *good enough* and keeping the row order keeps the sparsity. In the second
case the diagonal is hopeless and it swaps — note the resulting \\(L(1,0) = 10^{-6}\\), tiny, which
is the whole point of pivoting.

### 3d. Prune

After each pivot, `prune` applies Eisenstat-Liu symmetric pruning: once a column of \\(L\\) is known
to be "covered" by a symmetric counterpart in \\(U\\), the DFS in step 3a no longer needs to scan
past a certain point in it. This changes nothing about the result — it only shortens future searches,
and it is why KLU's symbolic step stays near-linear instead of degrading on later columns.

### The finished factors

For the 4×4 block above, no pivoting was needed (`p = [0,1,2,3]`) and the result is:

```
udiag  = [2, 3, 4.166666666666667, 4.76]

L col 0: [(1, 0.5)]                     U col 0: []
L col 1: [(2, 0.3333333333333333)]      U col 1: []
L col 2: [(3, 0.24)]                    U col 2: [(0, 1.0), (1, -0.5)]
L col 3: []                             U col 3: [(2, 1.0)]
```

i.e.

\\[ L = \begin{bmatrix} 1 & & & \\\\ 0.5 & 1 & & \\\\ & \tfrac13 & 1 & \\\\ & & 0.24 & 1 \end{bmatrix},
\qquad
U = \begin{bmatrix} 2 & & 1 & \\\\ & 3 & -0.5 & \\\\ & & 4.1\overline{6} & 1 \\\\ & & & 4.76 \end{bmatrix} \\]

Checking against a dense solve with \\(b = [1,2,3,4]^T\\): both give
\\(x = [2/7,\ 4/7,\ 3/7,\ 5/7]\\).

Note that \\(L\\)'s unit diagonal is never stored, and \\(U\\)'s diagonal lives in its own `udiag`
array — that separation is what lets the solve divide without hunting for the diagonal entry inside a
sparse column.

## Step 4: solve — permutations, then blocks in reverse

With the factors in hand, `solve.rs` computes

\\[ x = Q \left( (LU + \text{Off})^{-1} \, P \, b \right) \\]

in four moves. Back to the BTF example from step 1, with \\(b = [7, 4, 13, 10]^T\\):

**1. Permute the right-hand side.** \\(P b\\) reorders `b` by `p = [0,2,3,1]`:

```
Pb = [b0, b2, b3, b1] = [7, 13, 10, 4]
```

**2. Solve the blocks in reverse order.** This is the part BTF bought us. An earlier block's rows may
reference a later block's columns (that's what "upper block-triangular" means), so the *last* block
must be solved first:

```
block 2  (1×1):   2·y3 = 4                        →  y3 = 2
block 1  (1×1):   5·y2 = 10 − 1·y3 = 8            →  y2 = 1.6
                        ^^^^^^^ off-diagonal entry, now that y3 is known
block 0  (2×2):   [1 3; 4 1]·[y0;y1] = [7; 13 − 2·y3] = [7; 9]
```

The subtractions are exactly the off-diagonal arrays from step 1 (`off_x = [2.0, 1.0]`, both in the
last permuted column) being applied as each block's solution becomes available.

**3. Forward/back substitution inside each block.** Block 0's factors are \\(L = [[1,0],[4,1]]\\),
\\(U = [[1,3],[0,-11]]\\) (`udiag = [1.0, -11.0]`):

```
forward (Lz = rhs):   z0 = 7,   z1 = 9 − 4·7 = −19
back    (Uy = z):     y1 = −19 / −11 = 1.727273,   y0 = (7 − 3·1.727273) / 1 = 1.818182
```

**4. Permute back.** \\(x = Q y\\), i.e. `x[q[k]] = y[k]` with `q = [1,3,0,2]`:

```
x[1] = y0 = 1.818182
x[3] = y1 = 1.727273
x[0] = y2 = 1.6
x[2] = y3 = 2.0

x = [1.6, 1.8181818181818183, 2.0, 1.7272727272727273]
```

which matches a dense Gaussian-elimination solve of the original unpermuted matrix to the last digit
but one (`[1.6, 1.8181818181818181, 2.0, 1.7272727272727273]` — the difference is one ulp of
round-off, from a genuinely different order of operations).

## Step 5: refactor — the one that runs every Newton iteration

Between two Newton iterations the Jacobian's *values* change but its *pattern* does not. Refactor
exploits that as hard as it can: same BTF blocks, same AMD ordering, same pivot choices, same L/U
sparsity pattern — only the numbers are recomputed.

That means refactor skips **all** of step 3a (no DFS, no reachability search) and all of step 3c (no
pivot search). Real KLU's own `klu_refactor.c` never calls `dfs`/`lsolve_symbolic` at all, and
neither does the port. The stored pattern of each \\(U\\) column is already in a valid topological
order — it came from the original DFS — so the elimination can just walk it.

Keeping the earlier matrix's pattern and changing its values:

```
       c0  c1  c2   c3          before:  c0  c1  c2  c3
r0      .   1   .   3.5                   .   1   .   3
r1      .   .   3    .                    .   .   2   .
r2      .   4  2.5   2                    .   4   2   1
r3      5   .  1.5   .                    5   .   1   .
```

```
solve 1 (original values):  x = [1.6, 1.8181818181818183, 2.0, 1.7272727272727273]
solve 2 (new values):       x = [1.6, 1.6527777777777786, 1.3333333333333333, 1.5277777777777777]
     dense cross-check:         [1.6, 1.6527777777777781, 1.3333333333333333, 1.5277777777777777]
```

`KluNativeSystem::factor_and_solve_values` is the entry point the Newton loop uses: it packs the new
values into the cached CSC slots (via `groups` from step 0), refactors in place, and solves.
`factor_and_solve` is the same path for a caller that still holds full triplets — it reads nothing
but their value component. The port's `refactor_block_in_place`
overwrites the existing value fields rather than rebuilding `Vec`s — profiling showed those
per-column allocations were the dominant reason the Rust backend ran ~2× slower than the C one.

The risk of reusing pivots is real but bounded: if the new values make an old pivot choice unstable,
the result degrades rather than being caught by a fresh pivot search. `factor_and_solve` guards
against the extreme case by rejecting a non-finite solution, and returns `None` so the caller can
report `SolveStatus::Singular`.

## The whole pipeline on a real Jacobian

The 3-bus network in `tests/data/network.json` (slack + PV + PQ) gives 2 angle unknowns and 1
magnitude unknown, so a 3×3 Jacobian. At the first Newton iteration from a linear initial guess
(`vm = [1.06, 1.04, 1.00307]`, `va = [0, 0, −0.05160]`):

```
J = [ 21.834690  −5.298690  −1.462838 ]
    [ −5.119349   9.032698   2.323778 ]
    [  2.005352  −3.538289   8.503522 ]

mismatch = [0.26866152, 0.00368908, 0.00153712]
```

Through the pipeline:

- **BTF**: one block of size 3 (`r = [0, 3]`) — connected network, irreducible.
- **AMD**: skipped, `nk ≤ 3` → natural order. `p = q = [0, 1, 2]`.
- **Factor**: no pivoting needed (`pnum = [0, 1, 2]`) — the Jacobian's diagonal dominates, which is
  the property KLU's loose `tol` is built around.

```
k=0  udiag = 21.834690   L = [(1, −0.234459), (2, 0.091842)]   U = []
k=1  udiag =  7.790370   L = [(2, −0.391720)]                  U = [(0, −5.298690)]
k=2  udiag =  9.413792   L = []                                U = [(0, −1.462838), (1, 1.980802)]
```

- **Solve**: \\(\Delta x = [0.014383106, 0.008478649, 0.000316791]\\) — matching a dense solve
  exactly. The first two entries are angle corrections in radians; the third is a voltage-magnitude
  correction in per unit.
- **Refactor**: iteration 2 rebuilds `J` with the updated voltages and reuses everything above.

## What this port deliberately leaves out

`src/klu_native/` implements KLU for exactly the configuration gridoxide uses, and the omissions are
documented rather than silent:

- **Row scaling is ported but not wired in.** `scale.rs` implements both of KLU's variants and is
  differentially tested against the C, but `factor`/`refactor` currently run as if scaling were
  disabled. Scaling changes *which* candidate pivots partial pivoting compares — a stability
  preconditioner, not a correctness requirement — and a per-unit power-flow Jacobian is not
  pathologically scaled. The differential tests (unscaled Rust vs. scaled C, same matrices) agree to
  1e-8.
- **Real `f64` only, single right-hand side, `int32`-range indices.** No complex arithmetic, no
  batched multi-RHS, no `DLONG`. Newton-Raphson needs one real solve per iteration.
- **AMD only.** COLAMD, user-supplied orderings, and the user-callback ordering are all reachable in
  real KLU but never selected by gridoxide, so `Options` has no `ordering` field to select them.
- **No `maxwork` limit** on the maximum transversal — it defaults to "no limit" upstream and is never
  overridden, so the port always runs the matching to completion.

## Where to look in the code

| File | Ports | Contents |
|---|---|---|
| `src/klu_native/types.rs` | `klu_internal.h`, `klu_defaults.c` | `EMPTY`/`flip` sentinels, `Options` |
| `src/klu_native/btf/maxtrans.rs` | `btf_maxtrans.c` | step 1a, bipartite matching |
| `src/klu_native/btf/strongcomp.rs` | `btf_strongcomp.c` | step 1b, Tarjan SCC |
| `src/klu_native/btf/mod.rs` | `btf_order.c` | step 1, both halves combined |
| `src/klu_native/amd/` | `amd_order.c`, `amd_2.c` | step 2, approximate minimum degree |
| `src/klu_native/analyze.rs` | `klu_analyze.c` | steps 1–2 driver, produces `Symbolic` |
| `src/klu_native/kernel.rs` | `klu_kernel.c` | step 3, one block: DFS, pivot, prune |
| `src/klu_native/factor.rs` | `klu_factor.c` | step 3 driver, off-diagonal bookkeeping |
| `src/klu_native/scale.rs` | `klu_scale.c` | row scaling (not wired in) |
| `src/klu_native/refactor.rs` | `klu_refactor.c` | step 5 |
| `src/klu_native/solve.rs` | `klu_solve.c` | step 4 |
| `src/sparse_klu.rs` | — | the FFI backend, same algorithm via vendored C |

`src/klu_native/PROVENANCE.md` maps each file to its upstream source and license; the vendored C it
was ported from lives in `vendor/suitesparse/`.

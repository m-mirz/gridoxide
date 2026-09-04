# The Linear Optimization of Range Actions

A range action has a degree of freedom — an angle, a number of megawatts — so it can be optimized
rather than searched. This page is the linear program that does it: what the variables are, what each
row means, why the whole thing is wrapped in an outer loop, and a worked example small enough to
solve on paper.

## The keystone: one linearization

Everything rests on writing each CNEC's flow as an affine function of the set-points:

\\[
F(c) \;=\; f_n(c) \;+\; \sum_r \sigma_n(r, c)\,\bigl[A(r) - \alpha_n(r)\bigr]
\\]

with

- \\(A(r)\\) the set-point of range action \\(r\\) — degrees for a phase shifter, MW for a redispatch;
- \\(\alpha_n(r)\\) where that action sits at the start of outer iteration \\(n\\);
- \\(f_n(c)\\) the flow at that operating point;
- \\(\sigma_n(r, c)\\) the sensitivity \\(\partial F(c) / \partial A(r)\\).

Both \\(f_n\\) and \\(\sigma_n\\) are recomputed each outer iteration. The subscript is the whole
reason there *is* an outer iteration — see [below](#why-it-iterates).

### Where the sensitivities come from

For a **redispatch** the sensitivity is a PTDF difference: shifting one MW from bus \\(j\\) to bus
\\(k\\) changes branch \\(c\\)'s flow by \\(\text{PTDF}_{c,j} - \text{PTDF}_{c,k}\\), scaled by the
action's distribution keys. DC flow is linear in injection, so this is exact.

For a **phase shifter** the derivative has two terms, and omitting either is a silent error rather
than a loud one:

\\[
\frac{\partial F_b}{\partial \alpha_s} \;=\;
\underbrace{b_s \bigl[\text{PTDF}_{b,\,f(s)} - \text{PTDF}_{b,\,t(s)}\bigr]}_{\text{indirect, through the angles}}
\;\;-\;\; \underbrace{b_s \,\bigl[\, b = s \,\bigr]}_{\text{direct, on the shifting branch itself}}
\\]

for a shifter on branch \\(s\\) with susceptance \\(b_s\\) between buses \\(f(s)\\) and \\(t(s)\\).

The indirect term exists because `btheta` puts a shift on the right-hand side as an injection of
\\(+b_s\alpha\\) at `from` and \\(-b_s\alpha\\) at `to`, so the response is exactly what those two
injections produce. The direct term exists because the shifting branch's own flow is
\\(b_s(\theta_f - \theta_t - \alpha)\\), in which \\(\alpha\\) appears explicitly, contributing
\\(-b_s\\) to its own sensitivity and to no other branch's.

Drop the direct term and you get a sensitivity that is correct everywhere except on the one branch
the operator is actually moving — which is exactly the branch most likely to be the CNEC.

`tests/rao_linear_test.rs` checks the analytic derivative against a finite difference on every
branch, which is the load-bearing test of this layer: everything else here is bookkeeping around one
quantity, and if that quantity is wrong the optimizer confidently moves shifters the wrong way.

A useful sanity check falls out of the same test: a **radial** branch never responds to any shift. It
carries whatever its bus demands regardless of any shifter, because there is no parallel path for a
circulating flow to take.

## The program

Per perimeter, with \\(n\\) range actions, the problem has \\(3n + 1\\) columns:

| Column | Variable | Bounds | Objective coefficient |
|---|---|---|---|
| \\(3k\\) | \\(A(r_k)\\), the set-point | the action's own range | 0 |
| \\(3k+1\\) | \\(\Delta^{+}(r_k)\\), upward movement | \\([0, \infty)\\) | \\(+\rho_k\\) |
| \\(3k+2\\) | \\(\Delta^{-}(r_k)\\), downward movement | \\([0, \infty)\\) | \\(+\rho_k\\) |
| \\(3n\\) | \\(MM\\), the minimum margin | \\((-\infty, +\infty)\\) | \\(-1\\) |

and these rows:

**One movement row per action**, tying the set-point to its two non-negative movements:

\\[
A(r_k) \;-\; \Delta^{+}(r_k) \;+\; \Delta^{-}(r_k) \;=\; \alpha_n(r_k)
\\]

Splitting the movement in two is what lets \\(\vert A - \alpha \vert\\) be penalized linearly. At the
optimum at most one of the pair is nonzero, because both carry a positive cost.

**One balance row**, if any redispatch action's distribution keys do not sum to zero:

\\[
\sum_r \bigl(\Delta^{+}(r) - \Delta^{-}(r)\bigr) \cdot \sum_d \text{key}_d(r) \;=\; 0
\\]

An action whose keys sum to zero moves power between buses and is unaffected by this row. One whose
keys do not sum to zero creates or destroys power, and this row is what stops it being used alone —
while still allowing it *alongside* another that cancels it, which is a case a per-action check would
forbid and a real CRAC contains. Without this row the optimizer happily invents generation and
reports a margin no network could achieve.

**Two margin rows per optimized CNEC**, one per finite bound:

\\[
MM \;\le\; \bigl(f^{+}(c) - F(c)\bigr)\,k_c,
\qquad\qquad
MM \;\le\; \bigl(F(c) - f^{-}(c)\bigr)\,k_c
\\]

Substituting the linearization makes each of these an ordinary linear row in \\(A\\) and \\(MM\\).
\\(F(c)\\) never gets a column of its own: it appears only in these two rows, so eliminating it halves
the problem for no loss.

**The objective** is

\\[
\min\ \ -MM \;+\; \sum_r \rho_r \bigl(\Delta^{+}(r) + \Delta^{-}(r)\bigr)
\\]

The penalties \\(\rho_r\\) are small — 0.01 per degree of phase shift, 0.001 per MW of redispatch —
so they only break ties. But they are not zero, or the optimizer will happily move every shifter by a
rounding error's worth for no gain, and the plan becomes unreadable.

### The bounds are separate, not one magnitude used twice

A CNEC with only a lower threshold gets only the second row. Adding the first — which is what a
symmetric \\(\vert F \vert \le \text{limit}\\) does — constrains the flow in a direction the CRAC left
free, so the optimizer refuses set-points that are perfectly legal. An infinite bound contributes no
row at all.

### The objective unit is not cosmetic

\\(k_c\\) above is the scale factor that expresses this CNEC's margin in the objective's unit: 1 for
megawatts, and this CNEC's own amperes-per-MW for amperes. Scaling the row is what makes the unit
mean anything, and two CNECs at different voltages scale differently — which is why the two units can
rank the same pair of candidate networks in opposite orders.

The reference does not make this a setting. `RaoUtil.getFlowUnit` returns megawatts for a DC load flow
and **amperes for an AC one**, so the objective follows the flow model, and every threshold stated
"in the objective's unit" — the minimum-impact thresholds among them — follows with it.

### One definition of the margin, not two

The limits this program optimizes against come from
[`evaluate`](./margins.md), not from re-reading the CRAC's thresholds here. That is not tidiness. A
CRAC states thresholds in MW, in amperes, or as a fraction of a rated current, and each needs a
different conversion; doing that conversion twice invites the optimizer to maximize a quantity nobody
measures. It did, briefly — the LP read an ampere threshold as MW and so optimized against a limit
some 40% adrift of the real one, while the evaluator scored it correctly. Every margin stayed
self-consistent and the answer was simply wrong.

## Monitored elements, the other kind of constraint

A CRAC labels each CNEC with two independent flags. `optimized` puts it in the objective: make this
better. `monitored` says something else entirely: whatever you do elsewhere, **do not ruin this**. A
CNEC can carry either, both, or neither, and the second is an *MNEC*.

The two cannot both be maximized. Unloading one branch loads another, so a program told to maximize
every margin at once would have no answer. The reference makes the second a **penalized soft
constraint**: an MNEC may be pushed past its limit, and the objective pays for every megawatt (or
ampere) of it.

### "Not worse" is not "not negative"

The rule is

\\[
v(c) \;=\; \max\bigl(0,\; \min(0,\; m_0(c) - d) \;-\; m(c)\bigr)
\\]

with \\(m_0\\) the margin the **untouched** network had, \\(m\\) the margin now, and \\(d\\) the
*acceptable decrease* — 50 by default, in the objective's unit.

The inner \\(\min(0, \cdot)\\) is the part to read twice. The floor an MNEC is held to is zero, or
its own initial margin less \\(d\\), whichever is **lower**:

| where it started | floor | reading |
|---|---|---|
| \\(m_0 \ge d\\) | \\(0\\) | all of its margin is available, not just \\(d\\) of it |
| \\(0 \le m_0 < d\\) | \\(m_0 - d\\) | it may be pushed *through* its threshold, by the part of \\(d\\) it had not used |
| \\(m_0 < 0\\) | \\(m_0 - d\\) | already overloaded: \\(d\\) worse and no worse, and no obligation to repair it |

Reading it as a flat "no more than \\(d\\) worse" gets the first row wrong by the entire initial
margin. Reading it as "never negative" gets the other two wrong by \\(d\\). The reference wrote one
scenario per row — 5.2.1.2, 5.2.1.4 and 5.2.1.3 — and all three are in the vendored gate.

### In the program

Each monitored CNEC earns a column \\(V(c) \ge 0\\) and, for each bound the CRAC states, one row:

\\[
F(c) - V(c) \;\le\; \max\bigl(f^{+}(c),\; f_0(c) + d\bigr) - a
\qquad
F(c) + V(c) \;\ge\; \min\bigl(f^{-}(c),\; f_0(c) - d\bigr) + a
\\]

with \\(f_0\\) the initial flow and \\(a\\) the constraint-adjustment coefficient, zero by default.
The objective gains \\(\sum_c \pi\, V(c)\\) at the configured violation cost \\(\pi\\), 10 by
default. An MNEC gets **no** \\(MM\\) row: it is not something to improve.

The same penalty enters the objective the *search tree* ranks leaves by, and both are needed. The LP
rows stop a set-point degrading an MNEC; the objective term stops a topological action doing it,
because nothing else in the tree looks at MNECs at all.

### The baseline is the run's starting point, not the perimeter's

\\(m_0\\) and \\(f_0\\) come from the network before **any** remedial action, preventive ones
included. A curative MNEC is judged against the margin it had with nothing applied, so the baseline is
measured once by `castor::run` on the untouched network and carried into every perimeter. Measuring it
per perimeter would judge a curative MNEC against whatever the preventive stage left it at — which is
exactly the degradation the constraint exists to forbid.

## Why it iterates

The linearization is exact in DC for a redispatch, because DC flow is linear in injection. It is
**not** exact for a phase shifter, for a reason that is easy to state backwards.

The sensitivity itself is constant in DC: it depends on the network's susceptances, not on the shift.
What is not linear is the **tap-to-angle map**. A tap changer offers a finite set of angles, so the
continuous optimum lands between two of them, and rounding puts the network somewhere the LP did not
ask for. Sensitivities computed at the old point are then slightly wrong about the new one.

So the loop is: solve, apply, recompute, and **keep the result only if the true minimum margin
improved**. The acceptance test is run on the evaluator's margin, not on the LP's objective value —
the LP's own number is the answer to the linearized problem, which is the thing under suspicion.

## A worked example

The [chapter's network](./index.md#the-worked-example-used-throughout), the ±700 MW CNEC on BE2–DE1
from [the previous page](./margins.md#a-worked-margin), and one range action: the phase shifter
BBE1AA1–BBE2AA1, free over its full ±16 taps.

### The program, written out

One action, one CNEC, two finite bounds: 4 columns and 4 rows.

\\[
\begin{align*}
\text{minimize } \quad & -MM + 0.01\,(\Delta^{+} + \Delta^{-}) \\\\
\text{subject to } \quad
& A - \Delta^{+} + \Delta^{-} = -3.116 \\\\
& MM \le 700 - \bigl(814.0 - 100.79\,[A - (-3.116)]\bigr) \\\\
& MM \le \bigl(814.0 - 100.79\,[A + 3.116]\bigr) + 700
\end{align*}
\\]

The set-point is written in the CRAC's sign convention, in which the file's tap −8 reads as −3.116°
and increasing \\(A\\) *reduces* the BE2–DE1 flow. (gridoxide's internal transformer angle is the
negation of IIDM's, which is what a CRAC is written in; `tap_table` does the negation once, on the
way in.)

### Solving it on paper

Write \\(\delta = A + 3.116\\) for the movement in degrees. The flow is
\\(F = 814.0 - 100.79\,\delta\\) and the two bounds give

\\[
MM \le 700 - F = -114.0 + 100.79\,\delta,
\qquad
MM \le F + 700 = 1514.0 - 100.79\,\delta
\\]

\\(MM\\) is the lower of two lines, one rising and one falling, so the maximum sits where they cross:

\\[
-114.0 + 100.79\,\delta = 1514.0 - 100.79\,\delta
\quad\Longrightarrow\quad
\delta = \frac{1628.0}{2 \times 100.79} = 8.076°
\\]

which is \\(F = 0\\) and \\(MM = 700\\) MW — the flow driven to zero, both bounds equally far away.
That is \\(A = 8.076 - 3.116 = 4.960°\\).

### What the taps allow

4.960° is not a tap. The UCTE `##R` record gives a symmetric 16-position changer whose nearest
positions are

| Tap | Angle |
|---|---|
| 12 | 4.6727° |
| **13** | **5.0617°** |
| 14 | 5.4504° |

Rounding to the nearest gives tap 13, and the re-evaluation is what decides whether that was the
right rounding:

\\[
\begin{align*}
\text{tap } 13:\quad & \delta = 8.178°,\quad F = 814.0 - 824.3 = -10.3\ \text{MW},\quad m = 689.7 \\\\
\text{tap } 12:\quad & \delta = 7.789°,\quad F = 814.0 - 785.1 = +28.9\ \text{MW},\quad m = 671.1
\end{align*}
\\]

Note that at tap 13 the flow has crossed zero and the **lower** bound is now the binding one —
\\(m = \min(700 - (-10.3),\ -10.3 + 700) = \min(710.3,\ 689.7)\\). A margin computed as
\\(\text{limit} - \vert F \vert\\) would have given the same answer here, because the bounds are
symmetric; on a one-sided CNEC it would not.

### An action that cannot be where it is

Before any of that, a range action whose **starting** set-point is already outside its own range is
dropped from the perimeter. A CRAC saying "this device may only be at positions it is not at" is not
describing a tightly-constrained lever; it is describing no lever, and no movement the optimizer
chooses can make the statement true.

This only bites when an earlier perimeter has moved the device, and then it matters more than it
sounds — because several range actions may name the **same** network element. The reference's
`SL_ep15us11-3case2_withPstCra` declares four on one phase shifter and names one of them
`useless_pst`, permitting tap 0 and nothing else; an automaton has put that shifter on tap −8 by the
time the curative perimeter runs. Kept, it is not inert: it becomes a second column driving a device
that already has one, pinned to a position the machine is not at, and the perimeter moves nothing and
reports that nothing helped.

### Three ways to say how far it may move

A range is a pair of bounds and an **anchor**, and the CRAC names the anchor per range:

| kind | anchored on | means |
|---|---|---|
| `absolute` | nothing | the bounds are tap positions |
| `relativeToInitialNetwork` | the tap in the network as imported | "no more than *n* taps from where the file had it" |
| `relativeToPreviousInstant` | the tap this perimeter began at | "no more than *n* taps from whatever the plan already did" |

An action carries several ranges and they are **intersected**, so the binding one wins. The two
relative kinds coincide for a preventive perimeter, whose previous instant *is* the network as
imported, and part company for a curative one — which is the whole reason a CRAC bothers to write
both. `SL_ep13us5case3` in the reference's own suite declares all three on one shifter: absolute
\([-16, 16]\), ten taps of the imported network's 5, and ten taps of the preventive answer \(-5\).
The intersection is \([-5, 5]\), narrower than any of them alone.

The anchor for the last kind is measured **once, on the network the perimeter was handed** — before
any leaf has applied a set-point and before the outer iteration has moved anything. It is not the
live tap. Anchoring on the live tap lets the box walk one width per iteration, until the answer bears
no relation to what the CRAC allowed; and reading the kind as absolute, which is what an unhandled
range kind degenerates to, silently shrinks the permission instead. On scenario 1.3.4.3 that second
failure stops the optimizer at tap 10 where 15 was allowed, with five taps of travel it never knew it
had — and nothing about it is visible in a margin, because the answer stays feasible,
self-consistent and worse.

### What the code does

```console
$ gridoxide rao tests/data/ucte/3nodes_pst.uct \
      --crac docs/examples/pst-worked-example.crac.json

preventive perimeter (1 state(s)): -114.0 -> 689.7 MW (+803.7), 0 leaf/leaves
  SET    PST-BE to tap 13 (5.062 deg, was -3.116)

SECURE: worst margin 689.7 MW (was -114.0)
```

Tap 13, 5.062°, margin 689.7 MW. `0 leaf/leaves` because this CRAC declares no network actions, so
the [search tree](./search.md) has nothing discrete to try and the answer is the LP's alone.

Two things this small case shows that a larger one hides:

- **The optimizer does not stop at secure.** A margin of 0 was reachable at \\(\delta = 1.131°\\),
  roughly tap −5. The default objective is to *maximize* the worst margin, so it goes to 689.7
  instead. Whether that is the right behaviour is a policy question the CRAC answers — see
  [secure is enough](./search.md#stopping-when-secure-is-enough).
- **The penalty is invisible here and would not be with two shifters.** \\(0.01 \times 8.178 = 0.08\\)
  against an objective of 689.7. It exists to break ties, and among equally good answers the one that
  moves least wins.

## Discrete taps, optionally

`TapModel::Continuous` (the default) optimizes the angle as a real number and rounds, as above. It
needs only an LP, so it runs on gridoxide's own interior-point solver.

`TapModel::Discrete` optimizes the tap itself as an integer variable. That needs a MIP backend, and
`IpmSolver` refuses rather than relaxing the integrality silently — see
[the OPF solver boundary](../opf/index.md#the-solver-boundary), which this layer shares.

## What is not here

**Network actions.** This layer moves continuous set-points; choosing which discrete actions to take
is [the search tree's](./search.md) job, and the two interleave rather than run in sequence.

**Cross-perimeter range actions in one problem.** One perimeter at a time: the preventive perimeter
is solved, its decisions are fixed, and each curative perimeter is solved against them. That is the
CASTOR decomposition, not an approximation of it — but it means a curative set-point cannot be
*traded against* a preventive one inside a single LP. What the ranges below chain is the **bound**,
not the variable.

**HVDC range actions** are recognised and skipped. A counter trade has no network sensitivity at all,
which is why the reference leaves it out of its LP too.

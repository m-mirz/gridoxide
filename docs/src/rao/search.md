# The Search Tree and the CASTOR Decomposition

Range actions have a degree of freedom and go to [an LP](./linear.md). Network actions do not: a
switch is open or closed and there is no gradient between the two. So the discrete half is
**searched**, and the perimeters that searching happens in are chosen by a decomposition that is
itself part of the answer.

## The greedy tree

```text
evaluate(root); optimize_range_actions(root)
for depth in 0..D:
    for each available action not yet applied:
        leaf = root + action
        evaluate(leaf)
        optimize_range_actions(leaf)      <- the full LP, every leaf
    keep the best leaf if it improved enough; else stop
```

With \\(N\\) available actions and a depth limit \\(D\\), that is at most

\\[
\sum_{d=0}^{D-1} (N - d) \;\approx\; ND - \tfrac{D(D-1)}{2}
\\]

leaves — linear in \\(N\\) for fixed \\(D\\), against the \\(\binom{N}{D}\\) of an exhaustive search.
The price is the usual one: greedy picks the best single action at each level, and the best *pair* is
not always two actions that are individually best. `SearchResult::leaves` reports the count actually
evaluated, and it is the number to watch when a case gets big.

The default depth is 2.

### Every leaf re-runs the whole linear optimization

This is where all the time goes, which makes it tempting to skip. Do not.

A topological action changes the sensitivities \\(\sigma(r, c)\\) the phase shifters are optimized
against. Choose the topology first and the set-points afterwards, and an action that looks poor on its
own never gets the chance to be the best one once the shifters are re-tuned around it. The two
decisions interact, so they have to be made together.

## When a candidate is good enough to take

Two thresholds, both applied, the stricter binding. A candidate is taken only if it improves the
objective by at least

\\[
\Delta \;\ge\; \max\bigl(\, \varepsilon_{abs},\;\; \varepsilon_{rel} \times \vert \text{objective} \vert \,\bigr)
\\]

with \\(\varepsilon_{abs}\\) the CRAC's `absolute_min_impact` and \\(\varepsilon_{rel}\\) its
`relative_min_impact`.

Both default to zero, which takes any improvement at all. They exist because a real study does not
want a remedial action reported for a gain of 0.3 MW: the action has a cost the optimizer cannot see,
and an operator asked to switch a busbar for a rounding error will stop trusting the tool. Both are
measured in [the objective's unit](./linear.md#the-objective-unit-is-not-cosmetic), so a threshold of
275 means 275 MW or 275 A depending on the flow model.

### Stopping when secure is enough

`stop_at_target` replaces "maximize" with "reach this value and stop". The reference's `SECURE_FLOW`
objective is exactly this, with a target of zero: **secure is enough**.

It is not a weaker version of maximizing, and it changes two things:

- A network that already has a positive margin gets **no remedial action at all**.
- A candidate that reaches security is taken whether or not it clears the minimum-impact thresholds.

Twelve of the reference's own AC scenarios use it, and one asserts that *zero* actions are used on a
network gridoxide would happily have improved. Spending remedial actions to gain margin nobody asked
for is a different answer, not a better one. The
[worked example](./linear.md#a-worked-example) shows the other default in action: it drives a −114 MW
margin to +689.7 MW when +0 would have done.

### Skipping actions far from the problem

`skip_far_actions: Some(k)` discards candidate actions more than \\(k\\) country boundaries away from
the most limiting element; `Some(0)` keeps only actions in the same country as the worst CNEC.

This exists because an operator in one control area cannot generally be asked to act for an overload
in another — and it **changes the answer**. Without it the search takes actions the reference never
offers itself and reports a better margin than the problem actually allows. It needs
`Network::bus_countries`; with no countries there is no notion of far and the filter passes
everything.

### Determinism under parallelism

Candidates are evaluated in a fixed order and ties break on the action id, so a run reproduces
itself. This matters more than it sounds: a search tree that returns a different answer each time
cannot be regression-tested, and an operator cannot be told why yesterday's study disagreed with
today's.

## Which actions are on the table

Not every action a CRAC contains is available in every state. Its **usage rules** say when, and they
come in two kinds.

`onInstant` and `onContingencyState` are topological — a state alone answers them. `onFlowConstraint`
and `onFlowConstraintInCountry` are not: they say the action is available *only if some CNEC is
actually constrained*, which takes flows. A TSO writes those to mean "I will open this line, but only
if that line is overloaded" — an action nobody would take otherwise, and one no operator would be
offered otherwise.

Three details decide the answer, and each of them is a way to get it wrong:

**Any rule, not every rule.** An action is available when *one* of its rules is activated. So a
conditional rule is another way in, never a restriction: an action carrying both `onInstant:
preventive` and `onFlowConstraint` is available in every preventive state whatever the flows say.

**Constrained means margin ≤ 0**, in the objective's unit. Not `< 0` — a CNEC sitting exactly on its
threshold authorizes its action — and the unit matters for the same reason it matters to the
objective, since a margin that is negative in amperes can be positive in megawatts.

**Measured once, at the perimeter's starting point, and never re-derived.** This is the one that
looks like a bug and is not. An action authorized by an overload keeps its authority even after some
other action relieves that overload. Re-deriving availability inside each leaf would let the
candidate set change underneath the search — a different problem at every depth, whose answer depends
on the order the actions happened to be tried. The reference names a scenario after it: 2.4.1.2,
"onConstraint RAs with a constraint triggered by another preventive RA, **no reevaluation**".

Getting this wrong is expensive and flattering. Before the flow half existed, gridoxide offered every
conditional action unconditionally, and on that scenario used three actions for +97 A where the
reference uses one and reports −45. A gate reading only margins would have called that an
improvement.

## Usage limits

A CRAC may cap how much may be done at all, per instant: a total `max_ra`, a number of TSOs allowed to
act at all (`max_tso`), and per-TSO caps on topological actions, PSTs, remedial actions and elementary
actions. These are constraints on the *plan*, not on any one CNEC, and they are what keeps an
optimizer from proposing a coordinated fourteen-action manoeuvre across five countries because it
gained 20 MW.

### One budget, spent by both halves

The limits count *remedial actions*, and a phase shifter that moves is one exactly as much as an
opened line is. So the two halves of the search spend a single allowance between them and neither can
be limited on its own. On scenario 2.6.1.3 the whole curative allowance is one action and the
reference spends it on the shifter, while gridoxide — counting nothing — opened a line *and* moved
the shifter, then reported a margin for a plan the CRAC forbids.

Enforcement is split to match. Growing the tree, a candidate is rejected before it is ever evaluated
if the set it would make exceeds `max_ra`, `max_topo_per_tso`, `max_ra_per_tso` or
`max_elementary_per_tso` — a plan the CRAC forbids is not a worse answer, it is not an answer. What
those network actions leave over becomes the leaf's **budget** for range actions, and the subtraction
is asymmetric on purpose: `max_ra` and `max_ra_per_tso` count everything, so topology eats into them,
while `max_pst_per_tso` counts only shifters and is untouched by topology.

The budget is a cardinality constraint on which range actions may move, which the reference expresses
with a binary per range action and a MIP. gridoxide does not need one: the largest vendored CRAC has
four range actions in a perimeter and every one that declares a limit has two, so **enumerating the
admissible subsets answers the same question exactly** on the LP solver already in hand. It runs only
when the unconstrained answer turns out inadmissible, so the ordinary case pays one comparison.

`max-tso` is deliberately absent. A CRAC may declare it and the reference ignores it outright — "a
max-tso limit can no longer be defined and will be ignored" — so honouring it would apply a
constraint the reference does not.

## When a curative perimeter stops

A curative perimeter is not asked for the best answer it can find. It is asked whether the
post-contingency state is at least as good as the one the preventive stage already accepted, and it
stops as soon as it is.

That is the reference's rule, and it is unconditional: `TreeParameters` gives every curative perimeter
`AT_TARGET_OBJECTIVE_VALUE`, where the preventive perimeter gets it only under `SECURE_FLOW`. The
target is

\\[ \text{objective} \;\ge\; \text{preventive objective} \;+\; \delta \\]

with \\(\delta\\) the `curative-min-obj-improvement`, and \\(\max(\cdot, 0)\\) applied on top when
`enforce-curative-security` is set. The reasoning is operational rather than mathematical: curative
actions are carried out under time pressure by people who did not plan them, so an extra 40 A bought
by a third switching operation is not worth having.

Two details decide how much of the answer this moves. The reference's **default \\(\delta\\) is
zero** — "better than preventive at all" is the ordinary rule, and the 10000 in its own test
configurations is what turns the rule *off*. And the criterion is tested on the root leaf **before**
that leaf's range actions are optimized, so a perimeter that already qualifies is left entirely
alone: no network action and no tap movement either. That is the difference between two curative
remedial actions and none.

## An action is applied whole, or refused

A network action is a set of elementary actions taken **together** — "split this busbar" is one
decision, not six. An action containing an elementary action gridoxide cannot express is therefore
rejected outright rather than applied partially. Applying half of one produces a network the CRAC
never described, and an answer that references an action whose effect was not what the optimizer
measured.

## CASTOR: which perimeters there are

The search answers one perimeter. Deciding *which perimeters there are* and *in what order* is the
difference between a set of independent answers and a plan.

**The preventive perimeter** is the base case together with every outage state. Both are secured by
the same actions, because an outage instant is too soon for anyone to do anything: whatever protects
it must already have been done.

**Each contingency then gets its own curative perimeters**, one per curative instant that has an
action available, solved in chronological order with the preventive decisions applied and each
instant's result fixed before the next begins.

```text
                    ┌──────────────────────────────────────────┐
   preventive       │ base case + every outage state           │  one set of actions
                    └──────────────────┬───────────────────────┘
                                       │  decisions carried forward
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
        contingency A            contingency B            contingency C
        auto → curative          auto → curative          auto → curative
        (its own perimeters, chronological, independent of the others)
```

### How the decisions are carried forward

A curative perimeter inherits an open set: the network file's own out-of-service circuits, whatever
the preventive stage opened, and whatever the automatons opened or closed. That set is passed to the
search **as a list**, and this is the one implementation detail in this layer worth stating in a
book, because the obvious alternative is wrong in a way that produces no error.

The alternative is to bake the set into the network — write a 10⁹ Ω impedance into a copy of the
lines and search the result. It reads as equivalent. It is not, because a **closing** remedial action
is expressed by removing a branch *from the open set*:

```text
open = base_open ∪ action.open ∖ action.close
```

With the set already spent on the impedances, `base_open` is empty, the removal removes nothing, and
the branch stays at 10⁹ Ω whatever the CRAC says. The action is not refused — it evaluates as a
change that does nothing, loses to every candidate that does something, and the perimeter reports
that no remedial action was worth taking. Every margin downstream stays perfectly self-consistent
while describing a network in which the standby circuit was never reconnected.

Against the reference's suite that was **21 curative closes declined out of 21 offered**, while the
same actions were taken 8 times out of 8 in the preventive perimeter and 3 out of 3 by automatons —
which is how it was found. The automaton simulator had always carried its open set as a list; only
the curative path had not.

The same rule governs what a perimeter hands the next one: the effective set is the search's own
result, never a union with what went in. A union is safe only while nothing can ever be closed, since
it puts back exactly the branch the closing action just shut.

### Why curative actions are not optimized jointly

Refusing to put every contingency's curative actions into one problem is **not** an approximation for
speed. Curative actions for different contingencies are never taken together — only one contingency
happens — so optimizing them jointly would let the answer trade one against another, which is
meaningless. The decomposition is the correct model, not a relaxation of one.

### The pull-forward rule

A curative CNEC for which **no curative action exists** cannot be secured after the fact, so it is
moved into the preventive perimeter.

Without this the preventive optimization is free to park a flow between the PATL (the permanent limit)
and the TATL (the temporary one) — acceptable at the outage instant, when the temporary limit applies,
and permanently overloaded afterwards with nothing available to fix it. It is the one rule in this
layer that changes an answer rather than just organising the work, and `Plan::pulled_forward` reports
where it applied.

## Automatons: simulated, never optimized

The `auto` instant is not a decision. A protection scheme fires when its trigger condition is met,
whether or not that helps anything else, and the job is to reproduce what the equipment does rather
than to choose what it should do.

Automatons fire in ascending order of their stated `speed`, **in batches**, and the trigger conditions
are re-evaluated between batches but not within one. Both halves are load-bearing:

- Re-evaluating *between* batches is why a fast automaton that relieves an overload stops a slower one
  from ever seeing the condition that would have triggered it.
- **Not** re-evaluating *within* a batch is why an automaton whose CNEC is healthy when the batch
  begins stays out of it, even if a sibling firing alongside pushes that CNEC into overload.
  Equipment that sampled the grid simultaneously would not see that, and the reference does not fire
  it.

Range actions are the exception, and the reference is explicit about the asymmetry: *first* all
automatic network actions are applied, *then* automatic range actions one by one for as long as CNECs
remain overloaded. A range action has a set-point to size against the flows as they stand, so it
necessarily sees the state its predecessors left. Its set-point comes from a formula rather than an
LP:

\\[
A_{\text{new}} \;=\; A_{\text{current}} \;+\; \operatorname{sign}\bigl(F(c)\bigr)\,
\frac{\min\bigl(0,\ m(c)\bigr)}{\sigma}
\\]

taking \\(c\\) as the worst-overloaded CNEC the action is watching: shift just far enough to clear it,
capped by the action's own range. \\(\min(0, m)\\) is zero on a healthy CNEC, so a triggered action
whose CNEC is no longer overloaded moves nothing.

## Reading the plan

```console
$ gridoxide rao tests/data/ucte/TestCase12Nodes.uct \
      --crac tests/data/rao/crac-for-12nodes.json --depth 2

preventive perimeter (2 state(s)): -182.3 -> -82.9 MW (+99.4), 3 leaf/leaves
  APPLY  Open line NL1-NL2
  SET    PRA_PST_BE to tap -16 (-6.228 deg, was -0.000)

curative after Contingency_FR1_FR3: -89.2 -> -89.2 MW (+0.0), 1 leaf/leaves
  (nothing available helps)

INSECURE: worst margin -89.2 MW (was -182.3)
```

Each perimeter reports the states it secures, the worst margin before and after, and the leaves it
cost. The exit code is 0 when every perimeter ends secure and 1 when any does not, so the command can
gate a pipeline the same way `gridoxide security` does.

This run shows both halves of an honest answer. The preventive perimeter improves by 99.4 MW using a
topological action and a phase shifter together — the PST went to its limit at tap −16 — and is still
short of secure. The curative perimeter then reports that nothing available helps: it evaluated its
one candidate, took nothing, and left the margin where it found it. `(nothing available helps)` is a
result, not a failure; it is the answer when the actions on offer cannot reach the problem.

`PerimeterPlan` additionally carries the buses and transformers **as the perimeter leaves them**,
which together with `open_branches` is the network its figures describe. That is what lets a caller
re-derive any quantity the plan does not itself report — a per-CNEC margin, say — instead of
reconstructing a network that drifts from the one the searcher measured. A redispatch lives *only* in
the buses, so a consumer that carries the transformers and forgets the buses reproduces a network in
which no injection ever moved.

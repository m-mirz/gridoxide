# Closing the curative search gap

Status: **answered**, 2026-09-02. Written 2026-08-30 against `43a5a33`, after re-running the Cucumber
gate and reading the reference's search, which **falsified the premise this plan was going to be
built on**. §2 is that finding; the plan followed from it rather than from `RAO_PLAN.md` §15's item 1.

> **Outcome.** §2's refutation stands and §1's measurement was right about where to look. Phases 1
> and 2 — a candidate trace, then a table classifying every curative mismatch by why the action was
> declined — were **not needed**: reading `castor::curative_search` against `automaton.rs` gave the
> answer directly, and it is one of the four outcomes phase 1 was built to distinguish. A curative
> perimeter's inherited open set was applied by writing `OPEN_BRANCH_Z` into the lines, so a
> **closing** action — which is expressed by removing a branch from that set — had nothing to remove
> and evaluated as a change that does nothing. Closes matched 8 of 8 in preventive and 3 of 3 in
> auto, and 0 of 21 in curative; that asymmetry is the whole diagnosis. Fixed 2026-09-02, worth 68
> assertions on the sixteen-node file (727 → 795), recorded as defect 20 in `RAO_PLAN.md` §8.3.
>
> Phase 4's residue question is answered with it: with the closes taken, `remedial action X is used`
> failures fall from 30 to 2, so there is **no** body of scenarios where every action was considered
> and correctly judged and the answer is still out of reach. Nothing here justifies a combinatorial
> search. Phase 4's cheap form and phase 5 are **built**, 2026-09-02: `SearchOptions` carries
> `predefined_combinations` and `curative_max_depth`, the harness reads both, and each is gated by a
> test of its own since the corpus cannot exercise either — every vendored configuration carries
> `"predefined-combinations": []` and sets the two depths the same. Neither moves an assertion, which
> is expected and is the whole point: they close the reference's own mechanisms, so a configuration
> that used one would no longer be silently scored against a different search.
>
> The new largest cause is different in kind and is tracked in `RAO_PLAN.md` §15 item 1b: a curative
> perimeter's **range actions start from the wrong point**, reverting the preventive taps rather than
> carrying them forward.

> **What this is.** 167 of 1 072 checkable assertions in the reference's own Cucumber suite do not
> match. `plans/RAO_PLAN.md` names the dominant cause as *action combinations beyond the greedy
> chain* — "the reference blooms combinations at each depth" — and ranks it first among what is
> left. That is wrong, and building it would be building the wrong thing.

---

## 1. The measurement

Re-run today, unchanged from the recorded figures: **138/142** DC, **192/203** AC on TestCase12Nodes,
**727/883** AC on TestCase16Nodes. Grouped by scenario family, which is where the signal is:

| family | matched | **mismatched** | unsupported | what it covers |
|---|---|---|---|---|
| 1.3 | 372 | **85** | 38 | preventive + curative optimization, simple to complex |
| 2.6 | 103 | **31** | 22 | curative RA usage limits |
| 1.2 | 72 | **25** | 7 | automatons |
| 2.2 | 54 | 9 | 11 | |
| 3.2 | 20 | 9 | 6 | |
| 1.4 | 3 | 6 | 2 | |
| 5.2 | 22 | 2 | 21 | MNEC |
| 0.1, 0.2, 2.1, 2.4, 5.3, 5.5 | 264 | **0** | 64 | |
| **total** | **905** | **167** | **171** | |

Three families hold 141 of the 167. **1.2 is the automaton simulator, not the search** — a separate
defect, out of scope here and worth its own look (§7).

## 2. Why the premise is false

`RAO_PLAN.md` §15 says the reference "blooms *combinations* at each depth" while gridoxide "extends a
single best chain", and calls this "not a missing rule but a different search, and therefore a
different size of job". Two checks, and both refute it.

**The reference's search is greedy too.** `SearchTreeBloomer.bloom` returns exactly two things: the
*predefined* combinations configured in the RAO parameters, and one candidate per individual network
action. It does not enumerate subsets. Its breadth is **configuration**, not algorithm.

**No vendored configuration supplies a single predefined combination.** Every
`tests/data/rao/features/RaoParameters*.json` carries `"predefined-combinations": []`.

So on precisely the scenarios that fail, the reference reaches its answer with the same greedy chain
gridoxide has. Whatever the cause is, it is not combination depth — and a plan to build a
combinatorial search would have spent the largest item on the list closing a gap that is not there.

## 3. What the failures actually look like

In family 1.3 the preventive perimeter matches and the curative one does not. From 1.3.4.7, which is
representative:

```text
ok    3 action(s) used (expected 3) Preventive
ok    tap of `pst_fr` Some(-5) (expected -5) Preventive
ok    margin on `FFR2AA1  DDE3AA1  1 - preventive` Some(200.481) (expected 200)
DIFF  0 action(s) used (expected 1) After { co1_fr2_fr3_1, curative }
DIFF  `close_fr1_fr2_cra` used: false          After { co1_fr2_fr3_1, curative }
DIFF  tap of `pst_fr` Some(5) (expected -5)    After { co1_fr2_fr3_1, curative }
```

The signature repeats across 1.3.3.4, 1.3.4.3, 1.3.5.4 and others:

- **preventive is exact** — the right actions, the right taps, the margins to the unit;
- **curative takes too few actions**, often zero where one is expected;
- the missing one is a **topological close** (`close_fr1_fr5`, `close_fr1_fr2_cra`);
- the range actions then settle somewhere else, so every downstream margin differs too.

That last point matters for counting: one unfound action produces five or six mismatched assertions,
so 85 mismatches in 1.3 is a much smaller number of distinct decisions.

## 4. What has been ruled out

Each of these was a plausible cause and each is checked, so the diagnosis does not start from zero:

- **Combination depth** — §2.
- **Leaf evaluation.** `search.rs` re-runs the full LP on *every* leaf, and its module doc says why:
  a topological action changes the sensitivities the shifters are optimized against. So a candidate
  that only pays off once the PSTs are re-tuned is already evaluated correctly.
- **Availability.** `close_fr1_fr5` carries `onStateUsageRules: [{instant: curative, contingency:
  co1_fr2_fr3_1, usageMethod: available}]`. It is offered in the state where it is expected.
- **Expressibility.** The action closes `FFR1AA1  FFR5AA1  1`, and gridoxide models a UCTE file's
  initially-open branches — `view.initially_open`, which the harness already threads through.
- **`skip-actions-far-from-most-limiting-element`** is `false` in this configuration, so the country
  filter is not removing it.
- **Search depth** is `2147483647` in the configuration, clamped to 3.

## 5. One defect found on the way, and it is small

`tests/rao_cucumber_test.rs:530` reads `max-preventive-search-tree-depth` and sets `max_depth` from
it. **`max-curative-search-tree-depth` is never read at all.** Both are `2147483647` in the
configurations examined so far, so this is not the cause of §3 — but it is a parameter the harness
silently ignores, and a configuration that sets the two differently would be scored against the
wrong search.

## 6. The plan

The shape follows from §4: the cheap hypotheses are exhausted, so the first phase buys **evidence**
rather than a fix. This is the discipline `plans/RAO_PLAN.md` §8.3 already used to find nineteen
defects — the gate is a measurement tool, and what it currently measures is "wrong", not "why".

### Phase 1 — Make the search say why it declined ~~(not built; answered by reading the code)~~

`SearchResult` records what was chosen. It does not record what was considered and rejected, which
is the only thing that distinguishes the remaining hypotheses. Add a per-candidate trace, behind the
existing options rather than a feature:

```rust
pub struct CandidateTrace {
    pub action: usize,
    pub depth: usize,
    pub objective_before: f64,
    pub objective_after: Option<f64>,   // `None` if the leaf never evaluated
    pub outcome: CandidateOutcome,
}

pub enum CandidateOutcome {
    Taken,
    NotOffered { reason: NotOffered },   // usage rule, filter, already applied
    Inexpressible { element: String },   // an elementary action the network cannot carry
    Evaluated { improvement: f64 },      // considered, lost to a better one
    BelowThreshold { improvement: f64, needed: f64 },
    TargetAlreadyReached,
}
```

The distinction that matters is `NotOffered` / `Inexpressible` (the action never had a chance)
against `Evaluated` / `BelowThreshold` (it was measured and judged). Those are different defects with
different fixes, and today's output cannot tell them apart.

**Gate:** on 1.3.3.4, the trace names `close_fr1_fr5` and gives one of the outcomes above. Whichever
it is, that is the finding.

### Phase 2 — Classify every curative mismatch by that outcome ~~(not built)~~

Extend the Cucumber harness to print, for each mismatching scenario, the trace for the actions the
reference used and gridoxide did not. Then group as §1 groups margins:

| outcome | scenarios | assertions | what it means |
|---|---|---|---|
| `Inexpressible` | | | a modelling gap in the network, not the search |
| `NotOffered` | | | a usage-rule or filter defect |
| `BelowThreshold` | | | thresholds or objective sign |
| `Evaluated`, lost | | | the evaluation disagrees with the reference's |

**This table is the deliverable of phase 2**, and it decides phases 3 and 4. Writing them now would
be guessing — which is exactly what produced the item this plan replaces.

### Phase 3 — Fix by cause, largest first ~~(done: one cause, defect 20)~~

One fix per cause, each with the gate re-run and the family table recorded before and after, so a
change that moves 1.3 and breaks 2.4 is visible immediately. `RAO_PLAN.md` §8.3's nineteen defects
were found this way and every one of them was "internally consistent and externally wrong".

### Phase 4 — Only then, combinations ~~(cheap form built; the large form is not justified)~~

If the phase-2 table shows a residue of scenarios where gridoxide considered every action, judged
each correctly, and still could not reach the reference's answer, *that* residue is the combination
gap — and it can be sized before anything is built. §2 says it is not what is in front of us; it does
not say it is nothing.

The cheap form is what the reference itself has: read `predefined-combinations` from the RAO
parameters and offer each as one candidate. That is a parameter gridoxide ignores, it is a day's
work, and it closes the reference's own mechanism exactly. Enumerating subsets is a different and
much larger thing, and nothing yet justifies it.

### Phase 5 — The small one from §5 ~~(done)~~

Read `max-curative-search-tree-depth`, and give the curative perimeters their own depth. Gated by a
configuration that sets the two differently.

## 7. Deliberately out of scope

- **Family 1.2 (25 mismatches)** — the automaton simulator. A different subsystem with a different
  cause; folding it in would confuse two measurements.
- **Second-preventive optimization** — its scenarios are excluded from the corpus outright, so the
  gate is silent on it and no amount of this work will move a number.
- **Loop flows and relative margins**, for the reasons `RAO_PLAN.md` §11 gives.
- **`TapModel::Discrete`** — declared and unbuilt, with no gate to validate it against, since the
  reference's own default is the continuous model this already matches at 138/142.

## 8. What this plan is wrong about

Unknown, and that is the point of phase 1. What is *established* is §1's measurement, §2's
refutation, §3's signature and §4's five exclusions. Everything from phase 2 onward is conditional on
a table that does not exist yet, and the plan is written so that filling it in redirects the work
rather than embarrassing it.

`RAO_PLAN.md` §15 item 1 should be struck when phase 2 lands, whatever it says.

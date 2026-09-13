# The Remedial Action Problem

## Motivation

[Power flow](../powerflow/index.md) says what the network is doing. A contingency analysis says what
it would do if something tripped. Neither says what to *do about it*, and by itself neither is
actionable: an operator handed a list of post-contingency overloads still has to decide which switch
to open, which phase shifter to move, and how far.

Remedial action optimization — RAO — is that decision, posed as an optimization. Given

- a network,
- a list of contingencies to survive,
- the elements whose loading matters and the limits they must respect,
- the actions an operator is permitted to take, and when,

it returns a **plan**: which actions to take before the fact, and which to hold in reserve for each
contingency. It is the calculation behind European capacity calculation and coordinated security
analysis, and the reference implementation is
[powsybl-open-rao](https://github.com/powsybl/powsybl-open-rao), whose algorithm — CASTOR — this
module follows.

## The vocabulary, because the acronyms are load-bearing

| Term | What it is |
|---|---|
| **CRAC** | Contingencies, Remedial Actions and Additional Constraints — the document defining the problem. Everything below comes out of it. |
| **Contingency** | The simultaneous loss of one or more elements. More than one element makes it N-k. |
| **Instant** | A moment in the chronology after a contingency: `preventive`, `outage`, `auto`, `curative`. |
| **State** | A (contingency, instant) pair. The preventive state has no contingency. |
| **CNEC** | Critical Network Element and Contingency — one monitored element *in one state*. The same line yields a different CNEC per state, with different limits. |
| **Threshold** | A limit on a CNEC's flow, in MW, amperes, or a fraction of \\(i_{max}\\). |
| **Range action** | A control with a continuum of settings: a phase shifter's angle, a redispatch in MW. |
| **Network action** | A discrete control: open a switch, change a topology. |
| **Perimeter** | A set of states optimized together, sharing one set of decisions. |

The instant chronology is what makes this more than one optimization:

\\[
\text{preventive} \;\longrightarrow\; \text{outage} \;\longrightarrow\; \text{auto} \;\longrightarrow\; \text{curative}
\\]

An **outage** instant is too soon for anyone to act, so whatever protects it must already have been
done preventively. An **auto** instant is what automatic devices do, which is forced and simulated
rather than chosen — a protection scheme fires whether or not it helps. Only **preventive** and
**curative** instants carry decisions.

## The problem

Write \\(x\\) for the vector of discrete decisions (which network actions are taken) and \\(a\\) for
the vector of continuous set-points (range actions). For a state \\(s\\), each CNEC \\(c\\) carries a
flow \\(F_s(c;\, x, a)\\) and a pair of limits \\(f^{-}(c) \le F \le f^{+}(c)\\).

The **margin** of a CNEC is its distance to whichever limit is nearer:

\\[
m(c) \;=\; \min\bigl(\, f^{+}(c) - F(c),\;\; F(c) - f^{-}(c) \,\bigr)
\\]

and the objective is to make the worst one as large as possible:

\\[
\max_{x,\,a}\ \ \min_{s}\ \min_{c \in s}\ m_s(c)
\\]

subject to the range actions' own bounds, the power flow, and whatever usage rules the CRAC places on
each action.

Three properties of that statement drive everything in this module:

**It is a max-min, not a sum.** The objective is the *worst* margin, so improving an already-healthy
CNEC is worth nothing. This is what makes it expressible as a linear program — see
[the linear optimization](./linear.md) — and it is also why a plan can look like it "did nothing"
while being optimal.

**Negative margins are the normal case.** A margin is negative exactly when a limit is exceeded, so
the objective is meaningful on an insecure network and the sign of the answer is the security
verdict. `gridoxide security` exits 1 when any margin is negative, so it can gate a pipeline.

**\\(x\\) is discrete and \\(a\\) is continuous.** There is no gradient between an open switch and a
closed one, so the two halves need different machinery: a search tree over \\(x\\), and a linear
program over \\(a\\) at every node of it. They interleave rather than run in sequence, because a
topological change moves the sensitivities the phase shifters are optimized against.

## The three layers

| Layer | Answers | Code | Page |
|---|---|---|---|
| Evaluation | where does it hurt, and by how much | `rao::evaluate` | [Margins](./margins.md) |
| Linear optimization | what should the continuous controls be | `rao::linear` | [The LP](./linear.md) |
| Search and decomposition | which discrete actions, in which perimeter | `rao::search`, `rao::castor` | [The search tree](./search.md) |

Evaluation is a deliverable on its own. gridoxide could run a contingency analysis before any of this
existed, but it could not say whether the result was *acceptable*, because nothing told it what the
limits were or which elements anyone cared about.

## Why the flows are DC

Every margin, sensitivity and candidate score in the search is computed on the
[DC linearization](../powerflow/dc.md), and the whole post-contingency sweep comes out of one
factorization: `DcSensitivity::multi_outage_flows` answers an N-k outage with a Woodbury update
rather than a re-solve. On `case9241pegase` that is 0.40 ms against 13.4 ms.

A search tree evaluates thousands of candidates, so that ratio is the difference between a tool that
runs in a control room and one that does not. An AC gate is available
(`FlowModel::Ac`, `gridoxide rao --validate-ac`) for scoring and for the final verdict, at the cost of
a Newton-Raphson solve per outer iteration.

## The worked example used throughout

The next three pages share one network, `tests/data/ucte/3nodes_pst.uct` — five buses, three
countries, six branches, one phase shifter — small enough that every number on them can be checked
with a calculator.

```text
                 ┌─── FFR1AA1 ───┐
                 │   (+500 MW)    │
        2 lines  │                │ 1 line
                 │                │
            BBE1AA1 ──[PST]── BBE2AA1
            (+500 MW)              │
                                   │ 1 line
                                   │
                              DDE1AA1 ──── DDE2AA1
                             (−1000 MW)     (radial)
```

Every line has \\(x = 10\ \Omega\\). The nominal voltage is 380 kV — UCTE encodes it in the seventh
character of the node code, where `1` means 380 kV, and the `400.00` in the node record is a
regulation reference rather than the base. With \\(S_{base} = 100\\) MVA the impedance base is
\\(380^2/100 = 1444\ \Omega\\), so

\\[
x_{pu} = \frac{10}{1444} = 0.006925, \qquad b = \frac{1}{x_{pu}} = 144.4 \text{ p.u.}
\\]

for every line — which is exactly what `dc_branches` reports. The phase shifter's own reactance is
zero in the file and is clamped to the
[zero-impedance threshold](../powerflow/zero_impedance_branches.md), making it an ideal shifter to
within 0.02%.

## Running it

Assess first, optimize second. The two are separate commands because the first answers a question
worth asking on its own:

```bash
# Where does it hurt?
gridoxide security tests/data/ucte/3nodes_pst.uct \
    --crac docs/examples/pst-worked-example.crac.json

# What should we do about it?
gridoxide rao tests/data/ucte/3nodes_pst.uct \
    --crac docs/examples/pst-worked-example.crac.json --depth 2

# ... under the reference's own settings, rather than gridoxide's defaults.
gridoxide rao tests/data/ucte/3nodes_pst.uct \
    --crac docs/examples/pst-worked-example.crac.json \
    --parameters RaoParameters.json
```

`--parameters` reads OpenRAO's own `RaoParameters` document: the objective (including `MIN_COST`),
the flow model, MNEC handling, the second-preventive execution condition, the curative stop
criterion and the search thresholds. Without it a run is max-min-margin on DC with no MNECs and no
second pass, which is a fraction of what the optimizer does — and until §8.24 that was the *only*
thing the binary could be asked for, so the gate validated behaviour no user could reach. Settings
the file states and gridoxide does not read are ignored rather than refused, and
`src/rao/parameters.rs` lists which and why.

Both take a UCTE `.uct`, an IIDM `.xiidm`, or a **CGMES** model — for CGMES, the directory holding
the profile set or any one profile beside the others:

```bash
gridoxide security path/to/SmallGrid-Merged --crac smallgrid.crac.json
```

A CRAC names a CGMES element by the mRID of its `ConductingEquipment`, which is what powsybl gives an
IIDM element converted from CGMES. Both need `--features rao` plus an importer (`ucte`, `iidm` or
`cgmes`). The CRAC may be OpenRAO JSON — the
format has 24 versions across the reference checkout's 428 files and the reader spans them — or
gridoxide's own `<network>.rao.json` companion.

## What is validated

197 scenarios from powsybl-open-rao's own Cucumber suite are vendored under
`tests/data/rao/features/`, with the CRACs and parameter files they name. They are the only check in
this repository that gridoxide did not write for itself: they state margins to the decimal and name
which remedial actions should be used, and their authors wrote them to judge a different
implementation.

**1830 of 1862 checkable assertions match**, at the reference's own tolerance of `max(5, 1.5%)` in
whichever unit the step is written — margins, flows per side, taps, thresholds, named actions, action
counts, set-points, objective values, security statuses and **which optimization steps ran**, across
two flow models and three networks. **Nothing is skipped**: every step the corpus states is checked,
so the ratio is the whole of it rather than the part that was convenient.

| file | | |
|---|---|---|
| `dc_scenarios.feature` | 180 of 186 | |
| `ac_scenarios.feature` | 276 of 282 | TestCase12Nodes |
| `ac_scenarios_16nodes.feature` | 963 of 977 | TestCase16Nodes |
| `second_preventive.feature` | **123 of 123** | |
| `min_cost.feature` | 288 of 294 | costly optimization |

Two of the five were vendored **before** the capability they test existed, which is the order the
rest of this was built in and the only one that works. The second-preventive corpus scored 45 of 108
with nothing implemented; the costly one scored 165. In both cases the gate then found every defect
one scenario at a time, and a corpus vendored after the fact only ever confirms what its author
already believed.

All 32 assertions that do not match are **recorded disagreements rather than a backlog**, in ten
scenarios. There is no open defect left in the corpus. Each has been measured on the reference's *own* objective, in the unit its own
configuration selects and with its own MNEC violation cost applied, and in none of them is gridoxide
worse: some are its `BestTapFinder` rounding a set-point on minimum margin alone, blind to a virtual
cost its own javadoc warns about; two are gridoxide securing the network more cheaply than the
reference asks under `MIN_COST`, which is the objective working rather than a defect; the rest are
ties reached by a different route. The gate names each with the measurement behind it and refuses to
let a scenario disagree without one — see `plans/RAO_PLAN.md` §8.6.

The last open defect — 1.4.1.5 and 1.4.1.6, a curative perimeter spending three actions where the
reference spends one — closed in §8.25. Seven mechanisms had been implemented, measured and refuted
for it; the eighth was **read** out of the reference's own source rather than guessed, which took a
morning where the seven had taken weeks. `second_preventive.feature` went 118 → 123.

## What is not here

- **Loop flows and relative margins.**
- **HVDC and counter-trade range actions**, recognised and skipped. gridoxide models a DC network
  but nothing connects it to a range action yet; a counter trade has no network sensitivity at all,
  which is why the reference leaves it out of its own LP too. Neither appears in any CRAC the
  vendored corpus loads, so building either would be building blind.
- **Angle and voltage CNECs**, counted rather than modelled — the same boundary the reference draws,
  which checks them in a separate monitoring pass after the fact. gridoxide has copied the exclusion
  from the LP and **not** that monitoring pass, so a CRAC leaning on them is not fully answered. One
  consequence is worth stating because nothing else does: a usage rule conditioned on an angle or
  voltage constraint names a CNEC that is never evaluated, so it can never fire. That is
  conservative — the action is not offered rather than offered freely — but it is a silence, not a
  decision the optimizer makes.
- **An integer tap.** `pst-model: APPROXIMATED_INTEGERS` is read by nothing and `TapModel::Discrete`
  is declared and unbuilt — measured rather than assumed. Of 37 vendored configurations 5 ask for it,
  governing 24 scenarios, and the reference built eight of them as controlled pairs ("copy of 2.6.2.x
  with MIP for PSTs"). All 24 match with the continuous model and rounding, so building the integer
  tap could win nothing here and could only lose. `plans/RAO_PLAN.md` §8.21.
- **Multi-timestamp (MARMOT) runs.**
- **A `relativeToPreviousInstant` curative range inside the second preventive problem.** The rest of
  that problem is here. After the curative stage, the preventive perimeter is optimized again with
  every CNEC in front of it and the automaton and curative *switching* held applied, and the result
  is kept only if the whole plan it leads to is better. The curative *range* actions get a column of
  their own in it — `A(r, s)`, a set-point per action per state — so the pass can see a curative
  shifter paying for a preventive push; the column is scoped to the states it governs and is not
  reported, because the curative perimeter that follows decides it properly.

  A range written **relative to the previous instant** is expressed too, as of §8.23, and the way it
  had to be is the interesting part: its window is relative to another *column* — the preventive
  set-point the same problem is deciding — so it is stated as a **row**, `min ≤ A(r,s) − A(r) ≤ max`,
  rather than as the column's own bounds. §8.11 declined such a column on the grounds that the LP
  carries one bound pair per column and no row coupling two of them, which was true of a bound and
  not of a row. It changes no answer in the vendored corpus; what it removes is a case where the
  second pass could not see a curative shifter at all.

  What the reference additionally does is *keep* the curative decisions afterwards rather than
  re-deriving them. That is a separate change, and §8.9 records it costing 2 and fixing nothing.

  What that pass holds is exact about two things it used to get wrong, and both were worth more than
  any of the architecture around them. A curative **set-point** is deliberately *not* held: the pass
  optimizes one network standing in for every state, so anything held in it is held in the preventive
  and outage states too, where a curative decision is not in force. For a switch that is the price of
  letting the pass see the curative CNECs at all. For a set-point it is a stale iterate — chosen
  against preventive decisions the second pass exists to discard, and recomputed the moment it
  returns — and holding one lets the two passes settle into a fixed point neither can leave. A
  curative **close** *is* held, which took noticing: the held set is built by adding the branches a
  curative perimeter leaves open, and a close is invisible to an add-only rule, so the second pass
  optimized against an overload the curative stage had already removed.

  And the switching is held **only in the states that see it**. This is the one perimeter that spans
  every state at once, so one network for all of them puts a curative branch into the preventive and
  outage states, where no curative decision has been taken, and reads their CNECs somewhere they do
  not live. `evaluate::Held` carries the difference as a delta and the objective, the base flows and
  the sensitivities all read a CNEC in its own state's network — which is why an outage CNEC and a
  curative one under the same contingency no longer share a linearization. §8.8, §8.9 and §8.10 have
  the measurements.
- **Several perimeters' set-points as variables in one LP.** Each perimeter is solved against the
  previous one's fixed decisions, which is the CASTOR decomposition rather than a shortcut past it.
  The `relativeToPreviousInstant` range kind *is* honoured — what chains across perimeters is the
  bound, not the variable. See [the linear chapter](./linear.md#three-ways-to-say-how-far-it-may-move).

The AC evaluation **distributes the slack**, weighted by generation, because every configuration the
reference ships does. All three importers retain the per-bus generation it needs. It is not a refinement: when a contingency islands a generator the imbalance is
that machine's whole output, and where it reappears decides the flows. On one vendored fixture a
single slack sits at the end of a tie into the country that just lost 1000 MW, so it pushes the
entire make-up back through that country and over the line being measured — 1165 MW against a true
1000. Weighting by *net* injection rather than generation puts 70% of it back in the same place and
barely helps, which is why the setting looked innocent for a long time.

Angle and voltage CNECs are counted rather than modelled, which is what the reference does too: it
checks them in a separate monitoring pass after the fact rather than putting them in its LP.

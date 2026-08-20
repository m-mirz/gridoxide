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
```

Both need `--features rao` plus an importer (`ucte` or `iidm`). The CRAC may be OpenRAO JSON — the
format has 24 versions across the reference checkout's 428 files and the reader spans them — or
gridoxide's own `<network>.rao.json` companion.

## What is validated

22 scenarios from powsybl-open-rao's own Cucumber suite are vendored under
`tests/data/rao/features/`, with the CRACs and parameter files they name. They are the only check in
this repository that gridoxide did not write for itself: they state margins to the decimal and name
which remedial actions should be used, and their authors wrote them to judge a different
implementation.

All 124 checkable assertions match, at the reference's own tolerance of `max(5 MW, 1.5%)` — every
margin, every tap, every named action, every action count and every security status.

## What is not here

- **Loop flows, MNECs and relative margins.** MNECs are read and excluded from the objective, which
  is correct, but nothing yet stops an action from degrading one.
- **Costly optimization** — minimizing the price of the actions rather than maximizing margin.
- **HVDC range actions**, recognised and skipped: gridoxide models a DC network but nothing connects
  it to a range action yet.
- **Second-preventive optimization** and multi-timestamp (MARMOT) runs.
- **Multi-perimeter range actions** — the `relativeToPreviousInstant` range kind needs several states
  in one problem.

Angle and voltage CNECs are counted rather than modelled, which is what the reference does too: it
checks them in a separate monitoring pass after the fact rather than putting them in its LP.

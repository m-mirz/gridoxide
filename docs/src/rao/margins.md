# CNECs, Thresholds and Margins

Everything the optimizer maximizes is defined on this page, and none of it is difficult. It is
nevertheless where a RAO is most likely to be quietly wrong, because a threshold arrives in one of
four units, may be one-sided, may be stated per terminal, and is tightened by a reliability margin
before anyone sees it. Get any of that wrong and every subsequent number stays perfectly
self-consistent while describing a limit nobody wrote.

## From a threshold to a pair of bounds

A CNEC carries a list of thresholds. Each states a `min`, a `max`, or both, in one of

| Unit | Meaning | Conversion to MW |
|---|---|---|
| `megawatt` | active power on the branch | none |
| `ampere` | current | \\(P = \sqrt{3}\, U\, I\\) |
| `percentImax` | a **fraction** of the CNEC's own \\(i_{max}\\) | \\(I = \phi \cdot i_{max}\\), then as above |
| `degree`, `kilovolt` | angle and voltage CNECs | not expressible on a flow |

The ampere conversion is the one worth writing out, because dropping the \\(\sqrt{3}\\) is a 73%
error that still looks entirely plausible:

\\[
P_{pu} \;=\; \frac{\sqrt{3}\; U_{rated}\; I}{S_{base}}, \qquad\qquad
I \;=\; \frac{P_{MW}\cdot 10^{6}}{\sqrt{3}\; U}
\\]

with \\(U\\) line-to-line, in volts. `ratings::current_to_power_pu` exists as a named function purely
so that \\(\sqrt{3}\\) has one home.

`percentImax` is a fraction and **not** a percentage, despite the name — `1.0` means 100%, as the
reference's own `ThresholdAdder` javadoc states. Reading it as a percentage makes every such
threshold a hundred times too tight.

### Thresholds accumulate, they do not replace

A CNEC with several thresholds is bounded by the tightest of each side, taken independently:

\\[
f^{+} \;=\; \min_t\ \bigl(\text{max}_t\bigr) - \text{FRM} - \text{charge}, \qquad
f^{-} \;=\; \max_t\ \bigl(\text{min}_t\bigr) + \text{FRM} + \text{charge}
\\]

Each side remembers the voltage its own binding threshold named, so an ampere margin can be converted
back at the right voltage even when a CNEC's thresholds disagree about which that is.

**FRM** is the CRAC's reliability margin (`frm`), a fixed MW cushion held back from every limit to
cover the difference between the model and the network. **charge** is the headroom already consumed
by reactive flow, and is zero under DC — see [the AC case](#the-ac-case) below.

### The bounds are not assumed symmetric

A CRAC routinely writes a one-sided threshold. `min: -1500, max: null` means *no more than 1500 A in
the reverse direction, and nothing at all in the forward one*. Collapsing that to
\\(\vert F \vert \le 1500\\) invents a constraint nobody wrote, in the direction the flow is most
likely to go.

So `Bounds` starts at \\((-\infty, +\infty)\\) and only ever narrows, and a side that no threshold
mentions stays infinite and contributes no row to the LP. A CNEC whose thresholds are all
inexpressible — a `percentImax` threshold on a CNEC whose \\(i_{max}\\) the CRAC never stated, say —
constrains nothing at all, which is better than constraining it with a number nobody wrote.

## The margin

\\[
m(c) \;=\; \min\bigl(\, f^{+} - F,\;\; F - f^{-} \,\bigr)
\\]

against the **signed** flow \\(F\\), not its magnitude. For a symmetric pair of bounds this reduces to
the familiar

\\[
m(c) \;=\; \text{limit} - \vert F \vert
\\]

and for a one-sided pair that identity does **not** hold. The margin is the quantity to trust;
`CnecResult` keeps `limit_mw` alongside it for display only, and the two disagree by construction on a
one-sided CNEC.

The sign carries the verdict: \\(m < 0\\) is an overload, exactly, and by exactly the amount reported.

### Two ends of one flow

`CnecResult` reports `flow_mw` at side one and `side_two` — active power and current — at the other,
and the pair follows one rule: **both are measured in the same direction along the branch**. Power
entering at side one, power *leaving* at side two. So the two differ by the branch's own losses,
which is the comparison anyone asking for both ends wants, and under DC they are equal to the last
bit because a linear model has no losses to spend.

The alternative reading — report the power *entering* at each end, which is what powsybl's
`Terminal::getP` gives and what the AC evaluator computes internally — negates side two, and the two
figures then differ by roughly twice the flow rather than by a fraction of a megawatt. Both readings
agree on magnitude, so nothing but a per-side expectation from outside can tell them apart.

One consequence worth stating because it looks like an inconsistency: `current_a` and `side_two.1`
are **magnitudes**, since a current is one and a threshold is compared against one. A flow *reported*
in amperes is signed, by the active power at that terminal — the reference divides the signed
megawatts by \(\sqrt{3}\,U\) — so `-1444.0 A` names a direction rather than a smaller number. The
evaluator keeps the magnitude and the caller applies the sign, which is why the two live apart.

### One margin, in two units

`CnecResult` reports `margin_mw` and `margin_a`, and the second is *not* a unit conversion of the
first at some convenient voltage — it is the same margin expressed in the units the binding threshold
was written in, converted at the voltage that threshold named. Two CNECs at different voltages
convert differently, which is precisely why the two units can rank the same pair of candidate
networks in opposite orders:

> A 400 kV CNEC and a 225 kV one with equal MW margins do not have equal ampere margins.

That is why the objective's unit is a setting at all — see
[the LP's objective unit](./linear.md#the-objective-unit-is-not-cosmetic).

Concretely, the ampere margin is

\\[ m_A(c) \;=\; f^{+}_A \;-\; I \\]

— the limit expressed in amperes less the **current** — and not \\(m_{MW}\\) divided by
\\(\sqrt{3}\,U\\). For a threshold already written in amperes the two are the same number, because
the charge above has taken reactive flow and the voltage deviation off the megawatt limit and
converting back undoes exactly that. For a threshold written in **megawatts** they are not:
\\(\text{limit} - \vert P \vert\) carries neither effect and \\(I\\) carries both.

The voltage that turns a megawatt limit into amperes is the **network's** nominal, not the CRAC's
`nominalV`. A CRAC states `nominalV` to say what an *ampere* threshold was written against; a
megawatt threshold was written against nothing, and the reference reads the CNEC's nominal voltage
off the network. On a UCTE 380 kV line whose CRAC says 400 that is 150 A on a 2000 MW limit — enough
to change which remedial action the search takes when the objective is in amperes.

## A worked margin

Take the network from [the chapter introduction](./index.md#the-worked-example-used-throughout) and
put a single CNEC on the BE2–DE1 tie, with a two-sided megawatt threshold of ±700 MW, no reliability
margin, and no contingency:

```json
"thresholds" : [ { "unit" : "megawatt", "max" : 700.0, "min" : -700.0, "side" : 1 } ]
```

### The flow, by hand

The phase shifter's reactance is clamped to nearly zero, so BE1 and BE2 are one electrical node
separated only by the shift \\(\alpha\\). Set that aside for a moment — take \\(\alpha = 0\\) — and
the network reduces to three nodes B, D, F with

\\[
b_{BD} = 144.4, \qquad b_{BF} = 2 \times 144.4 = 288.8, \qquad b_{DF} = 144.4
\\]

and injections \\(P_B = +5\\), \\(P_D = -10\\), \\(P_F = +5\\) p.u. The radial DE1–DE2 branch carries
nothing and can be dropped. Taking \\(\theta_B = 0\\), the two remaining DC equations are

\\[
\begin{align*}
288.8\,\theta_D - 144.4\,\theta_F &= -10 \\\\
-144.4\,\theta_D + 433.2\,\theta_F &= +5
\end{align*}
\\]

Since \\(144.4/433.2 = 1/3\\) exactly, eliminating \\(\theta_F\\) gives
\\(240.667\,\theta_D = -8.3333\\), so

\\[
\theta_D = -0.034626\ \text{rad}, \qquad \theta_F = \frac{5 + 144.4\,\theta_D}{433.2} = 0
\\]

\\(\theta_F = 0\\) exactly: by symmetry France sits at Belgium's angle, its 500 MW flows straight to
Germany over the DE1–FR1 line, and Belgium's 500 MW flows over BE2–DE1. So with the shifter at
neutral,

\\[
F_{BD} = 144.4 \times (0 - (-0.034626)) = 5.000\ \text{p.u.} = 500\ \text{MW}
\\]

### What the shifter adds

The file leaves the PST at tap −8, which the UCTE `##R` record turns into a shift of
\\(\alpha = 0.054387\\) rad \\(= 3.116°\\) in gridoxide's internal convention. (A CRAC states the
same angle as −3.116°, since IIDM's sign is the opposite; the conversion happens once, on the way in,
and [the next page](./linear.md#the-program-written-out) works in the CRAC's frame.) A shift drives a
circulating flow around the loop
BE1–BE2–DE1–FR1–BE1, whose series reactance is

\\[
x_{loop} = \frac{1}{144.4} + \frac{1}{144.4} + \frac{1}{288.8} = 0.017313\ \text{p.u.}
\\]

so the loop susceptance is \\(b_{loop} = 57.76\\) p.u. and the circulating flow is
\\(b_{loop}\,\alpha\\):

\\[
\frac{\partial F_{BD}}{\partial \alpha} = 57.76 \times \frac{\pi}{180} = 1.0081\ \text{p.u./deg}
= 100.81\ \text{MW/deg}
\\]

against the 100.7896 MW/deg the code computes — the 0.02% gap is the phase shifter's clamped, not
quite zero, reactance. So

\\[
F_{BD} = 500 + 100.79 \times 3.116 = 814.0\ \text{MW}
\\]

### The margin

\\[
m = \min(700 - 814.0,\ \ 814.0 + 700) = \min(-114.0,\ 1514.0) = -114.0\ \text{MW}
\\]

which is what the evaluator reports:

```console
$ gridoxide security tests/data/ucte/3nodes_pst.uct \
      --crac docs/examples/pst-worked-example.crac.json

preventive / base case: 1 monitored, worst margin -114.0 MW
  OVERLOAD BE2-DE1 - preventive         flow     814.0  limit     700.0  by    114.0 MW

INSECURE: 1 overload(s) across 1 perimeter(s)
```

Note which bound binds: the upper one. Had the threshold been one-sided in the other direction, the
margin would have been +1514 MW and this network perfectly secure — the same flow, the same element,
a different question.

## The AC case

Under `FlowModel::Ac` two things change that the linear model cannot see, and both act on an
**ampere** threshold only — an MW threshold binds active power, which is exactly what the evaluator
reports either way.

**Current is measured at the voltage the bus is actually running at.** A bus at 1.05 p.u. carries a
given MW at 5% less current than nominal, so a DC study reading its ampere thresholds at nominal
voltage is conservative there and optimistic wherever voltage has sagged.

**Reactive flow consumes thermal headroom.** An ampere threshold binds \\(\vert S \vert\\), not
\\(\vert P \vert\\). Rather than change what a margin means, the reactive part is charged against the
limit:

\\[
\text{charge} \;=\; \max\left(0,\ \ \sqrt{P^2 + Q^2}\ \frac{U_{written}}{U_{actual}} \;-\; \vert P \vert \right)
\\]

taken off **both** bounds, since the headroom is unavailable in either direction. The two effects
arrive in one expression because they are the same conversion: the flow is re-expressed as the
active power that would draw the same current at the voltage the threshold was written against.

The point of doing it this way is that \\(m = \text{limit} - \vert P \vert\\) still holds and an MW
margin remains an MW margin. The alternative — reporting a margin on \\(\vert S \vert\\) — would mean
two runs of the same study report margins that are not comparable depending on the flow model.

## Resolution: the failure that produces no error

Before any of this, the CRAC's network-element ids have to be matched to gridoxide's branch and bus
indices, and that step is fallible in an interesting way. An element that fails to resolve is not a
small problem:

- a **CNEC** that silently disappears is a constraint the optimizer will never see;
- a **contingency** that fails to resolve is an outage it will never simulate;
- a **range action** whose generators fail to resolve is a control dropped from the optimization
  while every margin still looks right.

All three produce a confident answer to a different question, so `Resolution` reports every
unresolved id rather than treating a miss as an error or as nothing. Unresolved ids are legitimate —
a CRAC written for a merged model names elements outside any single file — which is exactly why they
have to be *reported* instead of either failing or being ignored.

Two matching subtleties, both of which look like sloppiness and are not:

- Ids are matched exactly, then with surrounding whitespace trimmed. UCTE element ids are fixed-width
  and carry padding (`"BBE1AA1  BBE2AA1  1"`), and a CRAC written against the same network may or may
  not preserve it.
- Bus ids additionally match with a trailing `_generator` or `_load` removed, because powsybl names a
  UCTE node's generator `<node>_generator` and a CRAC uses the name powsybl gave it. CNECs name
  branches and redispatches name buses; those are different id spaces, and resolving one against the
  other silently yields nothing.

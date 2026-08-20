# Two Buses, One Congested Line

[The OPF page](./index.md) says the prices are usually the reason to run it. This one derives where
they come from, on the smallest network that has any: two buses, two generators, one line, one
number that changes everything.

## The system

```text
   bus 1                      bus 2
   ┌───┐      line, x=0.1     ┌───┐
   │ G1│──────────────────────│ G2│
   └───┘   rate = R MW        └───┘
   $10/MWh                    $30/MWh
   0..200 MW                  0..200 MW
                                │
                            100 MW load
```

Committed as `docs/examples/two-bus-lmp.json` and its `.opf.json` companion. Both generators are
linear-cost — \\(c_2 = 0\\) — so this is an LP and the answer is provably optimal, not locally
optimal.

The document numbers its nodes 1 and 2, which this page follows; the CLI prints gridoxide's own
zero-based bus indices, so `bus 0` below is node 1.

## The program

With bus 1 as the angle reference the DC-OPF has three variables — \\(P_1\\), \\(P_2\\),
\\(\theta_2\\) — and reads

\\[
\begin{align*}
\min\quad & 10 P_1 + 30 P_2 \\\\
\text{s.t.}\quad & P_1 - b\,(\theta_1 - \theta_2) = 0 && (\lambda_1) \\\\
& P_2 - b\,(\theta_2 - \theta_1) = 100 && (\lambda_2) \\\\
& -R \le b\,(\theta_1 - \theta_2) \le R && (\mu^{-}, \mu^{+}) \\\\
& 0 \le P_1, P_2 \le 200
\end{align*}
\\]

The balance rows are written **generation minus outflow on the left, load on the right**, which is
not cosmetic: with load on the right-hand side each row's dual is
\\(\partial(\text{cost})/\partial(\text{load})\\) directly — which *is* the locational marginal price,
positive, in the sign anyone expects. The opposite orientation yields the negated quantity, and the
resulting sign error would appear in exactly the output most likely to be published.

## Case A: nothing binds

Take \\(R = 200\\) MW. The line can carry everything, so the cheap generator serves the whole load:

\\[
P_1 = 100,\quad P_2 = 0,\quad \text{cost} = \$1000/\text{h}
\\]

The prices follow from stationarity. With \\(\mu^{\pm} = 0\\) and \\(P_1\\) strictly between its
bounds, the \\(P_1\\) column gives \\(10 - \lambda_1 = 0\\), and the \\(\theta_2\\) column — which
appears in both balance rows with equal and opposite coefficients and in no binding limit — gives
\\(\lambda_1 = \lambda_2\\). So

\\[
\lambda_1 = \lambda_2 = \$10/\text{MWh}
\\]

**One price everywhere.** \\(P_2\\) sits at its lower bound with a positive reduced cost of
\\(30 - 10 = \$20\\)/MWh, which is exactly the amount by which it is too expensive to run.

```console
$ gridoxide opf docs/examples/two-bus-lmp.json

2 bus(es), 2 generator(s), 100.0 MW of demand
total cost: 1000.00 $/h

dispatch (MW):
  generator    0:    100.000
  generator    1:      0.000 (at min)

locational marginal price ($/MWh):
  uniform at 10.0000 — nothing is congested

no binding branch limits
```

A uniform price is the correct answer here, and it is also what an OPF prints when every branch limit
is accidentally unlimited. `OpfData::binding_capable_branches` exists as a named accessor for exactly
that reason: an all-unlimited case makes every congestion constraint vacuous and every price
identical, which looks like a working OPF right up until someone reads the prices.

## Case B: the line binds

Set `rate_a` to 60 MW. Now the cheap generator cannot serve more than 60, and the remaining 40 must
come from the expensive one:

\\[
P_1 = 60,\quad P_2 = 40,\quad \text{cost} = 600 + 1200 = \$1800/\text{h}
\\]

Both generators are now strictly inside their boxes, so stationarity fixes both prices directly:

\\[
\lambda_1 = 10, \qquad \lambda_2 = 30
\\]

Bus 2's price is set by the marginal unit *at bus 2*, because no more power can reach it from bus 1.
The \\(\theta_2\\) column now also touches the binding limit row, and its stationarity condition gives
the limit's own dual:

\\[
\mu^{+} = \lambda_2 - \lambda_1 = \$20/\text{MWh}
\\]

Three readings of that same number, all correct and all worth having:

- **The price spread.** \\(\lambda_2 - \lambda_1 = 20\\) — the congestion, in dollars.
- **The shadow price of capacity.** One more MW of rating on that corridor would save \\$20/h. This is
  the number that justifies a reinforcement, and it is why an OPF is run for planning and not only for
  dispatch.
- **The cost of the constraint.** Serving that marginal MW at bus 2 costs \\$30 instead of \\$10
  because the line is full.

```console
$ gridoxide opf docs/examples/two-bus-lmp.json

2 bus(es), 2 generator(s), 100.0 MW of demand
total cost: 1800.00 $/h

dispatch (MW):
  generator    0:     60.000
  generator    1:     40.000

locational marginal price ($/MWh):
  bus    0:    10.0000
  bus    1:    30.0000
  spread: 20.0000 (the congestion)

binding branch limits:
  branch    0: flow     60.000 of    60.000 MW, worth   20.0000 $/MWh to relieve
```

### Congestion rent

Settle every generator at its own bus's price and every load at its own:

\\[
\underbrace{100 \times 30}_{\text{load pays}} \;-\; \underbrace{60 \times 10 + 40 \times 30}_{\text{generators are paid}}
= 3000 - 1800 = \$1200/\text{h}
\\]

The market collects \\$1200/h more than it pays out. That surplus is the **congestion rent**, and it
is not an accounting error: it equals the shadow price times the flow,

\\[
\mu^{+} \times F = 20 \times 60 = \$1200/\text{h}
\\]

which is the general identity. It accrues to whoever owns the constrained transmission, and it is the
economic argument that a line worth building pays for itself.

Note also that the cheap generator is paid \\$10 while the system's marginal cost at the other end is
\\$30. Nothing is being taken from it: it is at its limit, and the limit is the line's, not its own.

## Why the duals are trustworthy

For a convex problem the KKT conditions are a **proof** of optimality, not a comparison against
another tool. Every pglib case in the test suite asserts stationarity, primal and dual feasibility,
and complementary slackness — the four conditions used informally above — so a dual that is reported
is a dual that was checked.

That machinery is what separates *"our model differs from theirs"* from *"our answer is wrong"*, and
in [the `case30` susceptance investigation](./index.md#which-susceptance--and-why-it-is-not-a-detail)
it is what made the discrepancy diagnosable at all: the certificate cleared the solver, which left the
formula.

## Where this stops being exact

The whole page rests on DC being a linear model, which makes the problem convex and the prices
unique. Two things break that:

**AC.** The AC-OPF is nonconvex, so its answer is a local optimum satisfying the first-order
conditions — which is the state of the art, and what every published AC-OPF objective means. Its
duals are still \\(\partial(\text{cost})/\partial(\text{load})\\) at that point, and still useful,
but "the" price is no longer a well-defined global quantity.

**Integrality.** A binary decision — a unit commitment, a discrete tap — destroys convexity outright
and with it the meaning of a dual. `IpmSolver` refuses an integer variable rather than relaxing it
silently, because a relaxed answer that is reported as an optimum is worse than no answer.

## Reproducing this

```bash
cargo run --release --features opf -- opf docs/examples/two-bus-lmp.json
```

The committed `.opf.json` has `rate_a: 60.0`, giving Case B. Change it to `200.0` for Case A. The
network document itself is unchanged between the two — congestion is a property of the limit, not of
the grid.

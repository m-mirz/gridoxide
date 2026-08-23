# Area Interchange Control

## Motivation

[Distributed slack](./distributed_slack.md) drives one number to one target: the slack's output to
its own schedule. Interconnected systems schedule more than that. Each control area agrees a **net
position** with its neighbours — how much it will export across its tie lines over the hour — and
dispatches its own generators to meet it. An area importing 300 MW when it agreed to import 200 has
every tie-line flow around it wrong, for exactly the reason a single slack absorbing an unscheduled
few hundred megawatts does.

## The formulation

An area's **interchange** is the active power leaving it across its own boundary branches, measured
at its own side of each:

\\[ X_a = \sum_{b \in \partial a} P_b^{(a\text{-side})} \\]

A branch is a boundary of area \\(a\\) when exactly one of its ends is in \\(a\\) — including the
case where the other end is in no area at all. The knob is the one distributed slack already uses:
the participating generators' `p_spec`.

## One area's position is dependent, and it has to be

Both ends of a tie are measured *into* the branch, so they do not cancel — they sum to what the tie
dissipates:

\\[ \sum_a X_a = +\ell_{\text{tie}} \\]

This is the easy thing to get backwards. It is not that one area's export is the other's import: the
loss falls between them and belongs to neither.

The consequence is structural. A set of agreed net positions sums to zero, but the achievable sum is
\\(\ell_{\text{tie}}\\), which nobody knows before the solve. Counting the freedom:

| | |
|---|---|
| Knobs | each area's aggregate participant schedule — \\(N\\) |
| Conditions wanted | \\(N\\) interchange targets, **plus** the slack on its own schedule — \\(N+1\\) |

Over-determined by one, always. So **the area holding the slack is the dependent one**: its own
target is not enforced, and its condition is that the slack produces its schedule instead. Every
other area's position is met exactly, and the slack's area absorbs the tie losses.

That is what a real interconnection does — one area's position ends up a residual rather than an
agreement — and gridoxide reports it as such rather than implying it. `AreaInterchangeReport::dependent`
names the area, and its `residual` entry is how far its nominal target was missed.

## The sign

For an area that is not the slack's, exporting above target means it should generate less, so
\\(X_a - X_a^{target}\\) is **subtracted** from its participants by normalized weight.

For the slack's area, write \\(\delta_s\\) for the slack's excess over its own schedule. Its
participants must take that excess up, so the quantity subtracted is \\(-\delta_s\\).

## It generalizes distributed slack rather than sitting beside it

One area covering the whole network has no boundary branches, so \\(X = 0\\); that area *is* the
slack's area, so its condition is \\(-\delta_s\\); and the loop becomes distributed slack exactly.

That is asserted, not asserted-about: `area_interchange_with_one_area_is_distributed_slack` runs both
loops on one network and compares the per-bus shifts and the solved voltages. A flipped sign in the
derivation above makes it diverge instead of agree.

It is also why the two **must not both be configured** — `run_power_flow` refuses rather than
dispatching the same generators twice — and why powsybl-open-loadflow's own
`AcAreaInterchangeControlOuterLoop` constructs a `DistributedSlackOuterLoop` as its no-area fallback
and its `filterInconsistentOuterLoops` removes the latter when the former is present.

## Where the areas come from

**Nowhere yet, and that is the gap.** `AreaDefinition` takes a bus-to-area assignment and a target
per area, and no importer supplies either. The pieces exist on the input side — CGMES has
`ControlArea`, and `ucte::UcteImport::bus_countries` is already read and is exactly an area
assignment — but nothing wires them up, and no format in the tree carries a scheduled net position
at all. So this is a Rust-API control today: the caller states the areas and the schedule.

`AreaInterchange::measure` is public for the same reason, and is useful without the loop: "what is
this area actually exchanging" is worth asking of a solved network without controlling it.

## Deliberately out of scope

- **Several slacks per area, and areas spanning islands.** The first area found holding a slack is
  the dependent one; an island with its own reference balances through its own participants.
- **Boundary-point areas**, where a bus sits on a tie whose flow the area does not count toward its
  own position. powsybl handles this by attributing the slack across the areas that *do* count it.
- **A per-area tolerance**, and powsybl's separate `areaInterchangePMaxMismatch` beside
  `slackBusPMaxMismatch`. One tolerance covers both here.

## See also

- [Outer Loops](./outer_loops.md) — the layer this runs on
- [Distributed Slack](./distributed_slack.md) — the degenerate case

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
other area's position is met exactly, and the slack's area takes the residual.

That residual is the tie losses when the areas partition the network — but only then. A bus in *no*
area is on the boundary of whichever areas it touches, so anything it injects lands in the residual
too. CGMES makes this the normal case rather than an edge one: a `TieFlow` names the boundary node,
which in a merged model is the X-node two areas' lines meet at, and that node belongs to neither.
On MicroGrid the five X-nodes inject nothing, so the whole 53 MW residual is the boundary lines'
own losses — which are large because it is a truncated-boundary model, not because anything is
wrong.

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

All three importers supply them, and they supply different amounts.

**CGMES states both halves.** `ControlArea` carries the schedule as
`netInterchange`, and a `TieFlow` per boundary terminal states the boundary.
`cgmes::cgmes_control_areas` turns that into an assignment and a target.

CGMES defines an area by its boundary and never lists what is inside, so
membership is **derived**: take the branches the tie flows name, cut them, and
compute the connected components of what remains. Each component containing an
area's seed is that area. A component reached from two areas' seeds is a
contradiction in the file's own boundary, reported rather than arbitrated.

One detail is worth stating because getting it wrong is quiet. A `TieFlow` names
the terminal at the **boundary**, not inside the area — in a merged model that
is the X-node where two areas' lines meet, and `TN_Border_AL11` carries both
`NL-Line_1`'s and `BE-Line_3`'s tie flow. What belongs to the area is the
*equipment*; the node is shared. So the seed is the cut branch's **far** end.
Seeding from the named end put every area's seed on the same border node, which
showed up as five contested buses and one area owning nothing at all.

The sign matters as much. CGMES states `netInterchange` as an *import* —
"positive sign means flow in to the area" — and `AreaDefinition::targets` is an
export, so the importer negates. Checked against the fixture's own data rather
than against the specification alone: SmallGrid declares a 210 MW position and
its published state already sits at 210.271 MW measured, which it could not with
the sign flipped.

**IIDM states membership directly, and it is the only one that does.**
`<iidm:area>` lists the voltage levels inside it as `<voltageLevelRef>`, so
nothing is derived — a bus belongs to whichever area claims any of its nodes'
voltage levels, and a bus merging nodes from two areas is a contradiction the
report counts. `interchangeTarget` gives the schedule where a file states one,
in the same load sign convention CGMES uses ("negative is export, positive is
import", per `Area.java`'s own javadoc), so the same negation applies.

`<areaBoundary>` elements are counted and not read: gridoxide derives the
boundary from membership, so a stated one is redundant here. An `areaType` other
than `ControlArea` — a `BiddingZone`, say — partitions the network for a
different purpose and is skipped, counted so its absence is visible.

**UCTE states membership only.** `##Z<cc>` sub-headers are a bus-to-area
assignment and nothing more, exposed as `UcteImport::country_areas`. No UCTE
file anywhere states a scheduled net position, so the caller supplies the
targets — and the default of zero asks each country to serve its own load, which
is the honest reading of "no interchange agreed". The twelve-node case is a real
four-country interconnection whose own state has BE exporting 2000 MW and DE
importing 2500.

Both are reachable from the command line:

```
gridoxide solve <network> --area-interchange
```

### A cross-format check worth having

The twelve-node case exists as both `.uct` and `.xiidm`, and
`ucte_and_iidm_agree_exactly_on_the_twelve_node_case` already establishes they
are the same network. So the areas must agree too — derived from `##Z` country
codes on one side and stated as `<voltageLevelRef>` on the other, by entirely
separate code. They do: BE +2000 MW, DE −2500, FR +1000, NL −500, to 1e-6.

### Which participants, and why it differs from distributed slack

`AreaDefinition::uniform` participates `Slack` and `PV` buses only, matching
`SlackDistribution::uniform`. That policy is right for distributed slack, which
is **frequency response**: a machine not under governor control does not pick up
imbalance.

A net position is not frequency response. It is met by **redispatch**, and a
generator held at a fixed active set-point is precisely the machine an operator
redispatches. `AreaDefinition::by_generation` weights by `p_spec` at every bus
instead, and the command line uses it.

What the wrong policy costs is visible in powsybl's own `two_area_case.xiidm`:
both of AREA2's generators are `voltageRegulatorOn="false"` and so arrive as
`PQ` buses, leaving that area with no participant at all under `uniform` and its
−400 MW schedule unreachable. Under `by_generation` it reaches −400.000 exactly.

`AreaInterchange::measure` is public for a related reason, and is useful without
the loop: "what is this area actually exchanging" is worth asking of a solved
network without controlling it.

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

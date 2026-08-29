# Reading IIDM Input

IIDM is the native serialization of powsybl-core's network model — `.xiidm`
files — and the second of the two formats published remedial-action test
material comes in, the other being [UCTE-DEF](./ucte.md).

```rust
let net = gridoxide::iidm::read("nordic32.xiidm")?;
let report = gridoxide::run_power_flow(
    net.buses.clone(), &net.lines, &net.transformers, &net.shunts,
    PowerFlowOptions::default(),
);
```

Behind the `iidm` feature. It costs one crate (`quick-xml`, already in the tree
as a transitive dependency of the CGMES decoder) and no system library, so like
`ucte` and `opf` it is built and tested in CI.

## Version tolerance is the design constraint

The IIDM fixtures available for testing span **eleven schema versions**, `1_0`
through `1_16`, and pypowsybl 1.16 writes `1_17`. The model moved underneath
them. Limits are the clearest case:

```xml
<!-- older -->                     <!-- newer -->
<iidm:currentLimits1               <iidm:operationalLimitsGroup1 id="DEFAULT">
    permanentLimit="721.7"/>           <iidm:currentLimits permanentLimit="5000.0"/>
                                   </iidm:operationalLimitsGroup1>
```

and `danglingLine` became `boundaryLine` along the way.

So the importer accepts **any** `1_*` namespace, understands both spellings, and
**skips elements it does not recognise** rather than failing. A parser pinned to
one version reads a fifth of the available material and breaks against the next
powsybl release.

What it does not do is skip silently. Every unrecognised element type is counted
and named:

```text
note: skipped 4 `hvdcLine` element(s)
note: skipped 2 `slackTerminal` element(s)
note: 1 branch(es) omitted as disconnected
```

## One topology, two spellings

IIDM voltage levels come in two kinds, and this importer folds them into one
representation. A `busBreakerTopology` names its buses; a `nodeBreakerTopology`
numbers its nodes and connects them with switches. Both become nodes of a single
[`NodeBreakerTopology`](../cgmes/node_breaker.md), and the bus view is whatever
[`bus_view`] makes of it — for a bus-breaker file with no switches, exactly the
buses the file declared; for a node-breaker file, the connected components of
its closed switches.

That is not merely tidy. It is why a topological remedial action will be
expressible on an IIDM network at all: the switches survive import as
first-class objects instead of being resolved away, so `RetentionPolicy` picks
the view rather than the parser fixing it.

```rust
let options = IidmOptions { retention: RetentionPolicy::RetainAll, ..Default::default() };
let net = gridoxide::iidm::read_with(path, &options)?;
net.topology.switches.len();     // 17 on the node-breaker fixture
net.view.n_buses();              // more than under MergeAll — the switches now split buses
```

## Boundary nodes

A dangling (boundary) line runs from a real bus to a boundary node — what UCTE
calls an X-node. gridoxide keeps that boundary as a **real bus** rather than
collapsing the half-line into an injection, for two reasons: it is what the
[UCTE importer](./ucte.md) does with an X-node, so the two agree structurally on
the same network; and it makes tie lines fall out for free. The two halves of a
tie line share a `pairingKey`, so they land on the same boundary bus with no
pairing logic at all — which covers both the newer form (a `tieLine` naming two
`danglingLine`s) and the older one (both halves inline as `_1`/`_2` attributes).

A boundary node belongs to no declared voltage level and so has no `nominalV`.
It adopts the voltage of whatever it attaches to; without that the per-unit base
would default to 1 V and every quantity through the boundary would be nonsense.

## How far it agrees

Two gates, and the first is the stronger.

**Cross-format.** Three fixtures are pypowsybl exports of `.uct` files the
[UCTE suite](./ucte.md#how-far-it-agrees-with-pypowsybl) already checks against
pypowsybl. gridoxide reads the same network through two parsers sharing no code
and the answers must agree. On the twelve-node case, with and without the phase
shifter at tap 16 of 16, they are **bit-identical** — zero deviation across all
16 branches. On the case with 400/225 transformers and X-nodes the worst
deviation is **6e-12 MW**, once the seven boundary half-lines whose terminals
the two formats declare in opposite order are accounted for.

That last point is a labelling difference, not a physical one: a UCTE `##L`
record may name the X-node first where the IIDM boundary line names the real bus
first, which flips the reported sign of a flow without changing anything.

**Against pypowsybl**, on fixtures written as IIDM in the first place — which is
what covers node-breaker topology, temporary limits and the older spellings,
none of which UCTE can express. `nordic32` (52 buses, 80 branches) agrees to
1.2e-3 MVar on quantities of several hundred MVar. The residual is the same
known model difference the UCTE page quantifies: IIDM places a transformer's
magnetizing shunt entirely on side 2 while `network::branch_calc_param` splits
it equally between both ends.

## Not modelled

- **HVDC.** `hvdcLine` and `vscConverterStation` are parsed enough to be
  counted and are then skipped with a note. gridoxide has a real DC-side network
  (`src/dc.rs`) reachable from CGMES, and wiring IIDM into it is its own piece of
  work.
- **Extensions.** `slackTerminal`, `referenceTerminal`, `mergedXnode`,
  `busbarSectionPosition`, virtual hubs — all named in `notes` and skipped.
  Notably `slackTerminal` is where powsybl records a chosen slack, so gridoxide
  currently picks its own (largest generation, preferring a voltage-regulating
  bus, ties broken on the bus label) and says so in `notes`.
- **Asymmetric line shunts.** IIDM states `g1`/`b1` and `g2`/`b2` separately;
  `Line` carries one total that the π-model splits equally. Every line in every
  vendored fixture is symmetric, and any that is not is counted in `notes`
  rather than silently averaged.


## Individual generators and loads stay addressable

The importer folds every injection into its bus's net `p_spec`/`q_spec`, which is all a power flow
needs and is not reversible afterwards. `IidmImport::injections` keeps the correspondence back:
each generator and load by its own IIDM id, the bus it sits on, and the injection the file states.

That is what lets something address an *individual* machine rather than a bus. A Dynawo dynamic
model attaches to a generator by `staticId`, and a dynamic study needs each machine's own terminal
power rather than its bus's total — see [Reading Dynamic Data](../dynamics/input.md).

# Reading UCTE-DEF Input

UCTE-DEF is the fixed-column text format the continental-European TSOs used to
exchange network snapshots before CGMES. It is superseded, and it is still the
format most published remedial-action and capacity-calculation test material is
written in — which is the practical reason gridoxide reads it.

```rust
let net = gridoxide::ucte::read("TestCase12Nodes.uct")?;
let report = gridoxide::run_power_flow(
    net.buses.clone(), &net.lines, &net.transformers, &net.shunts,
    PowerFlowOptions::default(),
);
```

Behind the `ucte` feature. Unlike `cgmes` it costs nothing to enable: the format
is plain text, so the importer is pure `std` with no dependency, and it is built
and tested in CI.

## What it brings that the other importers do not

Two things, and they are the reason this importer exists at all rather than a
conversion script.

**Branch ratings.** A `##L` record ends in a current limit and a `##T` record
carries one too:

```text
BBE1AA1  BBE2AA1  1 0 0.0000 10.000 0.000000   5000
//                                             ^^^^ permanent rating, amperes
```

Nothing in [`types::Line`] or [`types::Transformer`] has anywhere to put that —
the only rating in the crate before this was the single `rate_a` in the
companion OPF document. It lands in
[`ratings::BranchLimits`](../reference/provenance.md), indexed by the same flat
branch index as everything else:

```rust
let b = net.branch_ids.iter().position(|id| id.starts_with("BBE1AA1  BBE2AA1")).unwrap();
assert_eq!(net.limits[b].patl_a, Some(5000.0));
```

**Tap tables.** A `##R` record *is* a tap changer:

```text
BBE2AA1  BBE3AA1  1                    -0.68 90.00 16  0        SYMM
//                                     ^^^^^ ^^^^^ ^^ ^^        ^^^^
//                                     du%   theta  n  n'       kind
```

so a phase shifter arrives as a `TapChanger` with all 33 positions, not as one
complex number:

```rust
let pst = net.tap_changers[0].as_ref().unwrap();
assert_eq!((pst.low, pst.high(), pst.position), (-16, 16, 0));
pst.angle_deg(16);                    // the shift at the top tap
```

## Conventions, and why each is worth stating

Every one of these is silent when wrong — the file still loads and the network
still solves, just to a different answer.

**Generation is negative.** UCTE writes generation as a negative number in a
field called "active power generation". Net injection is `(−generation) − load`,
and the permissible-generation fields are negated the same way, which swaps
which of each pair is the lower bound.

**Nominal voltage comes from the node code, not from the record.** Character 7
of the 8-character code is a voltage class — `1` is 380 kV, `2` is 220 kV. The
`voltage` field in the record is the set-point a regulating node holds, commonly
400 kV on a 380 kV node. That is a per-unit target of 1.0526, not a different
base.

**Transformers are reversed relative to the file.** UCTE refers a transformer's
impedance to its node-1 side and puts the tap changer on node 2. gridoxide's
`network::branch_calc_param` — the MATPOWER convention — wants the series
admittance at `to` and the complex ratio at `from`. So `from` is UCTE node 2 and
`to` is node 1, which is the same swap powsybl's own importer makes. Getting it
wrong is invisible on a 400/400 phase shifter and a 3% error on a 400/225 unit.

**Files are Latin-1 and columns are bytes.** One accented character in a node
name would shift every later field if the line were decoded as UTF-8 first.

**The slack is chosen, not read.** UCTE has a node type for it — `3`, "U and θ
constant" — and none of the vendored fixtures uses it, because the tools that
wrote them let the load flow pick. gridoxide picks the largest generator,
preferring one that regulates voltage, ties broken on the node code so two
imports of one file agree. The choice is recorded in `notes`, and
`SlackPolicy::Node` overrides it.

## What the importer decides, it says out loud

`UcteImport::notes` carries everything that was a decision rather than a
reading, and `out_of_service` names every branch left out:

```text
note: 8 X-node(s) kept as ordinary buses; boundary-line semantics are not modelled
note: 2 closed busbar coupler(s) imported as near-zero-impedance branches
note: 3 branch(es) omitted as out of operation
note: no `U and theta constant` node in file; slack chosen as `BBE2AA1 ` (largest generation)
```

A silently dropped element is how an importer produces a plausible wrong answer,
so nothing is dropped silently.

## How far it agrees with pypowsybl

Every fixture is solved and compared against pypowsybl, element by element
(`tests/ucte_test.rs`). On networks whose transformers declare no magnetizing
admittance the agreement is **exact to solver tolerance** — below 1e-9 on
per-unit voltage, 1e-7 degrees on angle, and 1e-5 MW on every branch terminal —
including a case with 400/225 transformers, X-nodes, and a phase shifter parked
at tap 16 of 16.

Where a transformer *does* declare a magnetizing admittance, angles differ by
around 2e-4 degrees and reactive flows by a few tenths of a MVar. That is one
known model difference and nothing else: IIDM places the whole shunt on side 2,
while `network::branch_calc_param` splits it equally between both ends — a Γ
model against a π model. Zeroing just those fields drops every figure back to
2e-9 degrees and 1.1e-7 MW, which is how the cause was established rather than
assumed. The choice is the crate's, not this importer's: the CGMES and PGM paths
split the shunt the same way.

## Not modelled

- **X-nodes are ordinary buses.** powsybl merges the two half-lines meeting at
  an X-node into one tie line; gridoxide keeps the X-node. On the buses the two
  have in common the answers agree, but gridoxide reports more buses and more
  branches, and an X-node carrying its own load or generation is not treated as
  a boundary injection.
- **Out-of-service branches are omitted**, not carried as openable elements.
  They are listed in `out_of_service` so a later phase can reconnect them.
- **`##TT`, `##E` and any other block are skipped** with a note. None appears in
  any vendored fixture.

# Worked-example inputs

Small input documents the book's worked-example pages compute against. They exist so every number
printed on those pages can be reproduced by running one command, and so the arithmetic done by hand
there has something to be checked against.

They are **not** test fixtures. Nothing in `tests/` reads them; they are deliberately minimal rather
than representative, and each is sized so that a reader with a calculator can follow every step.

| File | Used by | What it is |
|---|---|---|
| `one-node-slg.json` | [A Fault Current, by Hand](../src/short_circuit/worked_example.md) | One 10 kV node, one source with `z01_ratio: 3`, a line-to-ground fault through 0.1 + j0.1 Ω. |
| `one-node-2ph.json` | same | The same system with a two-phase fault, to show the zero-sequence component vanishing. |
| `two-bus-lmp.json` + `.opf.json` | [Two Buses, One Congested Line](../src/opf/worked_example.md) | Two buses, a cheap and an expensive generator, one line whose rating decides whether the prices separate. |
| `pst-worked-example.crac.json` | [the remedial-action chapter](../src/rao/index.md) | A one-CNEC, one-phase-shifter CRAC for `tests/data/ucte/3nodes_pst.uct`. |

## Reproducing

```bash
cargo build --release --features rao,ucte,opf

./target/release/gridoxide short-circuit docs/examples/one-node-slg.json --scaling max
./target/release/gridoxide short-circuit docs/examples/one-node-2ph.json --scaling max
./target/release/gridoxide opf docs/examples/two-bus-lmp.json
./target/release/gridoxide security tests/data/ucte/3nodes_pst.uct \
    --crac docs/examples/pst-worked-example.crac.json
./target/release/gridoxide rao tests/data/ucte/3nodes_pst.uct \
    --crac docs/examples/pst-worked-example.crac.json
```

`examples/doc_numbers.rs` prints the quantities no CLI subcommand exposes — per-branch DC
susceptances and the phase-shift distribution factors the remedial-action pages derive by hand:

```bash
cargo run --release --features ucte,rao --example doc_numbers tests/data/ucte/3nodes_pst.uct
```

The `two-bus-lmp.opf.json` companion is committed with `rate_a: 60.0`, the congested case. Set it to
`200.0` for the uncongested one; the network document is the same either way.

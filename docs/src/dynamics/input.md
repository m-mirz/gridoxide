# Reading Dynamic Data

Three readers, all producing the same internal representation. Nothing downstream of them knows
which format a case arrived in.

## gridoxide JSON

The native format, and the one the other two target. A `dynamics` section alongside the ordinary
`buses` and `lines`:

```json
{
  "buses": [ … ], "lines": [ … ],
  "dynamics": {
    "s_base": 100.0, "f_nom": 50.0,
    "fixed_buses": [2],
    "units": [
      { "id": "G1", "bus": 0,
        "machine": { "model": "gen_round", "h": 5.0, "d": 1.0, "ra": 0.003,
                     "xd": 1.8, "xq": 1.7, "xdp": 0.30, "xqp": 0.55,
                     "xdpp": 0.22, "xqpp": 0.25, "xl": 0.15,
                     "td0p": 8.0, "tq0p": 0.4, "td0pp": 0.03, "tq0pp": 0.05,
                     "mbase": 100.0 },
        "avr": { "model": "sexs", "k": 200.0, "ta": 0.1, "tb": 1.0, "te": 0.05 },
        "gov": { "model": "tgov1", "r": 0.05, "t1": 0.5, "t2": 1.0, "t3": 5.0, "dt": 0.0 } }
    ],
    "loads": [ { "id": "L1", "bus": 1, "zip": [0.3, 0.3, 0.4], "cutoff": 0.5 } ],
    "events": [
      { "kind": "bus_fault",   "t": 1.0, "bus": 1 },
      { "kind": "clear_fault", "t": 1.1, "bus": 1 },
      { "kind": "branch_trip", "t": 1.1, "branch": 1 }
    ]
  }
}
```

The model parameter blocks **are** the models' own parameter structs. That is deliberate: the file
format cannot drift away from what the models take, because a renamed field is a missing-field parse
error rather than a silently defaulted zero.

`speed_voltages` (default `false`) chooses the machine formulation for the whole case — see
[The Model Library](./models.md).

`fixed_buses` names buses held at constant voltage for the whole run — an infinite bus, represented
*exactly* rather than approximated by a large inertia or a small source impedance. It is optional: a
system with no fixed bus is perfectly well posed, since every machine's rotor angle is an absolute
state and the whole system's frequency is free to move.

A device may state its own `p` and `q`; omitting both makes it take whatever its bus's solved
injection has left. See [Initialization](./initialization.md) for why that matters.

## PSS/E `.dyr`

The format most published dynamic cases ship in, and the one ANDES reads — so the same file can
drive both gridoxide and a reference, which removes a class of "the two tools were given different
data" disagreement before it starts.

Free-format records terminated by `/`, spanning any number of lines:

```text
     1 'GENROU' '1 '  8.0  0.03  0.4  0.05  5.0  1.0
                      1.8  1.70  0.30  0.55  0.22  0.15  0.1  0.3  /
     1 'SEXS'   '1 '  0.10  1.00  200.0  0.05  -5.0  5.0  /
```

**A `.dyr` cannot stand alone.** Two things every machine model needs are simply not in it — the MVA
rating and the armature resistance, which are the `.raw`'s `MBASE` and `ZSORCE` — and neither is the
bus numbering. All three are demanded from the caller rather than invented, and the error message
says where they live.

Two parsing details matter. A record ends at the `/`, not at the newline, so a three-line `GENROU`
is one record with fourteen parameters rather than three truncated ones. And a leading `/` is a
comment only when no record is open — mid-record the same character is the terminator, and files do
put it on a line of its own.

`SEXS`'s first field is the **ratio** `T_a/T_b`, not a time constant. Misreading it gives a
plausible exciter an order of magnitude too fast.

## Dynawo `.dyd` / `.par` / `.crv`

Dynawo is MPL-2.0, RTE-maintained, ships validated cases, and reads IIDM — which
[gridoxide's IIDM importer](../import/iidm.md) already handles. A `.dyd` lists `blackBoxModel`
entries naming a Modelica `lib`, a `parId` into a `.par`, and optionally a `staticId` tying it to an
IIDM element; a `.par` holds named parameter sets; a `.crv` says what to record.

Because a `.par` is addressed by parameter **name**, the only real risk is looking up a name no file
uses — and a hand-written fixture would agree with whatever the reader happened to expect. So the
test fixtures are copied verbatim from Dynawo's own repository, and the gates assert against the
file's own literal values.

`GeneratorSynchronousFourWindings*` maps onto the sixth-order machine parameter for parameter, with
`generator_SNom` as the machine base. `...ProportionalRegulations` maps onto the proportional
regulator and governor exactly. `GeneratorSynchronousThreeWindings*` maps onto the fifth-order
salient-pole machine — its parameter set carries no `XpqPu` and no `Tpq0`, so the reader keys on the
**library name** rather than on which parameters happen to be present, which is the difference
between reading a model and guessing one from its data.

### A whole case, in one call

`dyd::load_case` reads both halves together — the IIDM network through
[gridoxide's IIDM importer](../import/iidm.md), the models through the reader above — and returns a
system ready to integrate. The `staticId` on each `blackBoxModel` is the correspondence between
them.

Each machine's **own** terminal power is recovered exactly rather than apportioned. A power flow
produces a bus's total; the IIDM states every load's `p0` and `q0`, and those are precisely what went
into that bus's specification, so

```text
machine's injection = bus's solved injection − the loads the file states there
```

holds for both active and reactive power, at a `PV` bus as much as a `PQ` one. That closes for a
Dynawo case the one hazard no downstream gate can catch — a wrong device/load split is
self-consistent and therefore silent. **Two machines on one bus** is the case it cannot resolve:
their share of the bus's solved reactive power is genuinely not in the file, only the total is, so
it is refused by name.

Tap-changing load models, and anything else attached to a static element that is not a generator,
are treated as static and **named** — the network is right, but their dynamics are not there.

**One conversion is inferred rather than read**, and is flagged in the source: `governor_KGover` is a
gain on the machine's own `governor_PNom`, while the proportional governor wants one on the network
base, so the reader applies `k = KGover · PNom / s_base`.

## Limits are carried through

`SEXS`'s `EMIN`/`EMAX`, `TGOV1`'s `VMIN`/`VMAX`, Dynawo's `voltageRegulator_EfdMinPu`/`MaxPu` and
its `governor_PMin`/`PMax` all reach the models. Two details are worth knowing: `TGOV1` states
**`VMAX` before `VMIN`**, so reading them in field order gives a valve limited upside-down that
therefore never moves; and Dynawo's power limits are in MW, so they divide by the network base while
its field-voltage limits go through unchanged — which is sound because gridoxide's own
initialization reproduces Dynawo's `efdPu` exactly.

## Saturation, stated twice and incompatibly

PSS/E gives two points on a curve, `S(1.0)` and `S(1.2)`. Dynawo gives an exponential
characteristic, `md`/`mq`/`nd`/`nq`. The two are not convertible without committing to a curve
shape, and no model here implements saturation. Both readers parse it, neither uses it, and both
**report a nonzero value** — because ignoring it changes answers, and a silent drop is exactly what
surfaces later as a small unexplained disagreement with a reference.

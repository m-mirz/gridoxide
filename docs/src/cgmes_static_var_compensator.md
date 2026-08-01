# CGMES: StaticVarCompensator

## Motivation

A Static Var Compensator (SVC) is a shunt-connected, power-electronically-controlled reactive power
source: instead of a fixed shunt capacitor/reactor bank, it continuously adjusts its own reactive
injection to hold the voltage at its connection point near a setpoint, within a capacitive/inductive
rating. Electrically it behaves like a voltage-controlled bus for load-flow purposes — the same
\\(\vert V_k \vert\\)-known, \\(Q_k\\)-unknown PV formulation the [Powerflow](./powerflow.md) page already
describes for a generator, just without any active-power term.

CGMES represents one as its own `StaticVarCompensator` class (a `RegulatingCondEq`, the same base every
`SynchronousMachine` and `ExternalNetworkInjection` also derive from), carrying:

- `capacitiveRating` / `inductiveRating` — the SVC's reactive range,
- `slope` — a droop coefficient for voltage-vs-reactive-power regulation,
- `q` (SSH) — a starting/fallback reactive power value,
- an optional `RegulatingControl` reference — the same voltage-mode-target mechanism
  `SynchronousMachine` uses.

## The concepts

### 1. Ratings are reactances, not powers

`capacitiveRating`/`inductiveRating` read like power quantities ("...at maximum capacitive reactive
power") but are documented, and universally treated by real tools, as **reactance ratings in ohms**. They
have to be converted to a susceptance, and from there to a reactive-power rating, rather than used
directly as a Mvar value:

\\[ B = \frac{1}{X_{rating}}, \qquad Q \approx V^2 B \\]

The deciding evidence is that powsybl-core's CGMES importer computes exactly `1 / rating` to get the
susceptance it stores. If `capacitiveRating` were already a Mvar quantity, taking its reciprocal to get a
susceptance would make no dimensional sense. (See "A bug this caught" below for what happens if you
believe the doc text instead.)

An absent or zero rating conventionally means *unlimited* rather than *zero* — mapped to \\(\pm\infty\\)
(or `±Double.MAX_VALUE`), the same "no rating means no limit" convention tap-changer `xMin`/`xMax` uses.

### 2. Three regulation behaviors, in increasing fidelity

**Hard voltage pin.** An SVC that is actively regulating in voltage mode is treated exactly like a PV
bus: its controlled bus's \\(\vert V \vert\\) is fixed at the target and its \\(Q\\) is the free
variable, clamped to \\([B_{min}V^2,\ B_{max}V^2]\\). This is the same mechanism as a generator's voltage
control — the SVC contributes no \\(P\\) term, which is the only structural difference.

**Droop (slope).** A real SVC doesn't hold voltage exactly; it regulates along a droop characteristic, so
that absorbing more reactive power comes with a slightly lower terminal voltage. The linearized form folds
directly into the voltage equation as an extra term rather than being a post-hoc correction:

\\[ V + \text{slope} \cdot Q_{SVC} = V_{target} \\]

**Standby / dead-band.** An SVC may sit idle as a fixed susceptance while voltage stays inside a
dead-band, only entering active regulation when voltage leaves it. Because that decision depends on the
solved voltage, it can't be made before the solve — it needs an outer loop that toggles the bus between a
fixed-susceptance PQ shunt and an active PV pin between passes, the same architectural pattern the
[Reactive Power Limits](./q_limits.md) page describes for Q-limit switching, applied here to decide
*whether* to regulate at all rather than *how far* a limit was exceeded.

A tool that implements only the hard pin still solves the common case correctly; droop and standby refine
it.

### 3. The regulated bus need not be the SVC's own bus

`RegulatingControl.Terminal` is independent of the equipment's own terminal, so an SVC can regulate a
*remote* bus. Two consequences for anything converting one: the bus whose voltage gets pinned is the
control terminal's bus, while the per-unit base for converting the ohm ratings into a Q limit is the
*SVC's own* physical bus's rated voltage. The two differ whenever regulation is remote.

### 4. Not regulating? Fall back to a fixed injection

An SVC with `controlEnabled = false`, a disabled `RegulatingControl`, or no `RegulatingControl` at all is
not a voltage-controlled bus. The fallback is the SSH `q` value as a plain fixed reactive injection —
an ordinary PQ contribution.

## Where this fits in gridoxide today

`src/cgmes.rs`'s `StaticVarCompensator` conversion (added alongside the `ACLineSegment.gch` fix — see
the [Shunt Conductance](./cgmes_shunt_conductance.md) page — after this fixture's own SVC was found to
be silently dropped entirely) mirrors `SynchronousMachine`'s existing `RegulatingControl`-driven PV
upgrade:

- If the SVC has an enabled, voltage-mode `RegulatingControl`, the controlled bus (concept 3 — possibly
  remote) is promoted from `PQ` to `PV` and pinned to the target voltage: concept 2's hard pin.
- Otherwise it falls back to concept 4's fixed Q injection from the SSH `q` value, using the same sign
  convention `SynchronousMachine.q` already established empirically (no negation — see that code's own
  comment for why the doc text alone isn't trustworthy here).

`q_min`/`q_max` follow concept 1 — a reactance rating converted to a per-unit reactive-power limit,
\\(Q \approx V^2 B \approx B_{pu}\\) at \\(V \approx 1\\) pu (the same flat-voltage approximation
`SynchronousMachine`'s own `min_q`/`max_q` already make), anchored to the SVC's own physical bus's
`u_rated`:

```rust
let z_base = own_bus.map(|b| buses[b].u_rated * buses[b].u_rated / s_base_va);
buses[controlled_bus].q_min = match (sc.inductive_rating, z_base) {
    (Some(x), Some(zb)) if x != 0.0 => zb / x,
    _ => -f64::INFINITY,
};
buses[controlled_bus].q_max = match (sc.capacitive_rating, z_base) {
    (Some(x), Some(zb)) if x != 0.0 => zb / x,
    _ => f64::INFINITY,
};
```

Two deliberate simplifications versus concept 2's fuller model, both consistent with gridoxide's existing
scope elsewhere:

- **No droop/slope.** `StaticVarCompensator.slope` is not read at all — every regulating SVC is a hard
  voltage pin, the same simplification gridoxide already makes for `SynchronousMachine`.
- **No standby/monitoring mode.** A non-regulating SVC falls back to its fixed `q` injection permanently;
  there's no outer loop that would later switch it back into regulation if voltage left some dead-band,
  since gridoxide's plain `newton_raphson` doesn't run any outer loop for SVCs at all (only for PV→PQ
  switching, and only when `newton_raphson_enforcing_q_limits` is used instead of the default solver).

## Tool reference

| Tool | Rating storage (§1) | Regulation (§2) | Remote (§3) |
|---|---|---|---|
| **gridoxide** | ohm rating → per-unit Q limit at \\(V=1\\), anchored to the SVC's own bus (`src/cgmes.rs`) | hard pin only | ✅ |
| powsybl-core | `bMin`/`bMax` in siemens, via `getB() = 1 / rating` in `StaticVarCompensatorConversion.java`; zero/absent rating → `±Double.MAX_VALUE` | data model only: `voltageSetpoint`, `reactivePowerSetpoint`, `RegulationMode` (`VOLTAGE`/`REACTIVE_POWER`). No `slope` field in core IIDM — droop is the optional `VoltagePerReactivePowerControl` extension (added only if `slope >= 0`), dead-band the `StandbyAutomaton` extension | ✅ |
| powsybl-open-loadflow | consumes `getBmin()`/`getBmax()` as `ReactiveLimits` (`LfStaticVarCompensatorImpl`) | all three: `BUS_TARGET_V` hard pin by default; droop folded into that same equation by `AcEquationSystemCreator.createGeneratorLocalVoltageControlEquation` when the extension and solver flag are both present; standby dead-band via the dedicated `MonitoringVoltageOuterLoop` | ✅ `VoltageControl.controlledBus` |
| VeraGrid | stepped `Bmin`/`Bmax` on its `ControllableShunt` | regulates a `control_bus`'s voltage to `Vset` | ✅ |
| pandapower | `create_svc` (plus `create_tcsc`, `create_ssc` — the broadest FACTS coverage of the tools surveyed) | voltage setpoint regulation | ❌ |

power-grid-model and lightsim2grid have no SVC concept at all: PGM's only shunt-connected component is
`Shunt`, a fixed admittance, and lightsim2grid's `ShuntContainer` is a fixed injection stamped straight
into the Y-bus diagonal. Both are consistent with their domains — SVCs are a transmission-level device.

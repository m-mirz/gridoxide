# RatioTapChanger.RatioTapChangerTable

## Motivation

An ordinary (voltage-magnitude-only, non-phase-shifting) tap changer's simplest CGMES representation is
`RatioTapChanger`'s own `stepVoltageIncrement` — a single percentage, giving a ratio linear in tap step:
\\(\text{ratio} = 1 + (\text{step} - \text{neutralStep}) \cdot \text{stepVoltageIncrement}/100\\). Real
transformers don't always tap this uniformly, though — non-uniform winding turns per step, or step-
dependent leakage reactance, are both real physical effects a single linear coefficient can't capture.
CGMES accommodates this with an *optional* `RatioTapChanger.RatioTapChangerTable` reference: a plain
`RatioTapChanger` may carry this alongside its own `stepVoltageIncrement`, pointing to a table of
`RatioTapChangerTablePoint` rows giving each step's ratio (and optionally r/x/g/b deviation) explicitly,
rather than via the linear formula.

ENTSO-E's Svedala conformance fixture (national-scale substation-area model, 53 `PowerTransformer`s) uses
this extensively: all 11 of its `RatioTapChanger`s reference a table.

## The concepts

### 1. An optional table alongside a formula — not a separate class

This is architecturally different from `PhaseTapChangerTabular`, and the difference drives everything else
on this page. `PhaseTapChangerTabular` is its own distinct CGMES *class*: if its table lookup fails there
is nothing else to fall back on, so a failed lookup is a genuine data error. `RatioTapChangerTable` is
just an optional *reference* that a `RatioTapChanger` may or may not carry — the same object always has
`stepVoltageIncrement` sitting right there.

The consequence is a precedence rule rather than an error path: **try the table, fall back to the formula**
when the reference is absent, the table is empty, or the table is invalid. A missing or unusable table is
a normal, expected condition, not a failure.

### 2. What a table point can override

Each `RatioTapChangerTablePoint` carries `ratio` plus optional `r`, `x`, `g`, `b` deviations (in percent)
for its step — the same `TapChangerTablePoint` shape `PhaseTapChangerTabular`'s own points use. A tool can
therefore honor anywhere from just the ratio up to the full impedance deviation, which is a scope choice
independent of concept 1's precedence rule.

### 3. Full step table vs. current step only

Whether a converter needs to materialize the whole table depends on what it does downstream. A tool that
exports models, or that moves taps during the solve, needs every step. A tool that only ever solves the
snapshot it was handed needs exactly one row — the one matching the SSH `step` — and can look it up
directly.

## Where this fits in gridoxide today

`ratio_tap_table` (`src/cgmes.rs`) implements concept 1's precedence rule, with concept 3's single-step
lookup (gridoxide only ever needs *this* step's effect, since it neither exports step tables nor moves
taps — see the [PhaseTapChangerLinear](./phase_tap_changer_linear.md) page for the latter):

```rust
fn ratio_tap_table(ds: &CimDataset, table_mrid: &str, step: i64, xtx: f64) -> Option<TapEffect> {
    for pt_mrid in by_type(ds, "RatioTapChangerTablePoint") {
        let pt: &cimstructs::RatioTapChangerTablePoint = get(ds, pt_mrid)?;
        let Some(owner) = &pt.ratio_tap_changer_table else { continue };
        if owner.mrid != *table_mrid || pt.base.step != Some(step) {
            continue;
        }
        let ratio = pt.base.ratio.unwrap_or(1.0);
        let x_pct = pt.base.x.unwrap_or(0.0);
        return Some(TapEffect { tap: Complex::new(ratio, 0.0), x_override: Some(xtx * (1.0 + x_pct / 100.0)) });
    }
    None
}
```

Returning `None` (rather than an error) when the table or a matching point for the current step isn't
found is the deliberate difference from `phase_tap_tabular`'s own hard-error behavior — exactly concept 1:
the caller falls back to the linear `stepVoltageIncrement` formula instead of treating a missing/invalid
table as a data error.

On concept 2, gridoxide reads only `ratio` and `x`. `r`/`g`/`b` deviations have no representation in
`TapEffect` at all (`{ tap: Complex<f64>, x_override: Option<f64> }` — no `r_override`, no `g`/`b`
override), a pre-existing simplification across *every* tap-changer conversion in this file, not something
specific to this feature.

### Validated against Svedala — but not a strong before/after signal on this particular fixture

Solving ENTSO-E's Svedala conformance case end-to-end (191 buses after this fixture's own 3-winding
transformer star-bus synthesis on a different fixture with the same feature)
converges cleanly in 6 Newton-Raphson iterations, matching 108 published `SvVoltage` values with a mean
absolute error of ~0.53% and a worst case of ~4.35%.

Disabling the table lookup (forcing every `RatioTapChanger` back onto its linear fallback, as a direct
A/B comparison) barely moves either number — this fixture's own tables happen to have ratios that are
*exactly* linear already (confirmed directly against the raw EQ XML: table step 1 gives ratio 0.88,
matching \\(1 + (1-13)\times 1/100 = 0.88\\) exactly, all the way through step 25). So Svedala doesn't
demonstrate a large accuracy win from this feature specifically — its real value is CIM-spec correctness
for the (real, if less common) case where a table genuinely diverges from the linear approximation, which
this particular fixture just doesn't happen to exercise.

## Tool reference

| Tool | Precedence (§1) | Fields read (§2) | Table scope (§3) |
|---|---|---|---|
| **gridoxide** | table first, linear formula on absent/missing point (`ratio_tap_table` returns `None`) | `ratio`, `x` | current SSH step only |
| powsybl-core | table first, formula on absent/empty/invalid table — `CgmesRatioTapChangerBuilder.addSteps()` dispatches to `addStepsFromTable` or `addStepsFromLowHighIncrement`, with an explicit `isTableValid()` check between | `ratio`, `r`, `x`, `g`, `b` per row | full step table, materialized for export |

Of the tools surveyed, only these two have a CGMES `RatioTapChangerTable` path at all: power-grid-model
and lightsim2grid have no CGMES import, and VeraGrid's and pandapower's CIM importers weren't surveyed for
this specific reference.

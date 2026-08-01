# CGMES: PhaseTapChangerLinear

## Motivation

A phase-shifting transformer (PST) uses its tap changer to shift the voltage *angle* across the
transformer, not (only) its magnitude — a way to directly control how much active power flows through a
particular path in a meshed network, independent of the voltage-magnitude control an ordinary tap changer
provides. CGMES models four distinct ways a phase tap changer's per-step behavior can be specified, each
its own concrete class: `PhaseTapChangerSymmetrical` and `PhaseTapChangerAsymmetrical` (trigonometric
formulas), `PhaseTapChangerTabular` (an explicit per-step lookup table), and `PhaseTapChangerLinear` — the
simplest of the four.

`PhaseTapChangerLinear` is exactly what its name says: ratio is always exactly 1.0 (a pure phase shifter,
no magnitude change at all), and angle is *linear* in tap step —
\\(\alpha = (\text{step} - \text{neutralStep}) \cdot \text{stepPhaseShiftIncrement}\\) — a mathematical
approximation of a real PST's behavior, per the CIM class's own doc comment, rather than a physically
derived model like Symmetrical/Asymmetrical's trigonometric curves. ENTSO-E's conformance suite has two
dedicated test configurations for exactly this class, `PST_PhaseTapChangerLinear_Type1`/`_Type2` —
without support for it, neither could be solved correctly at all.

## The concepts

### 1. Per-step behavior: formula vs. table, normalized at import

The four CGMES classes are four *specifications* of the same underlying thing: for a given tap position,
what complex ratio \\(\rho e^{j\alpha}\\) (and what r/x/g/b deviation) does the transformer have? The
distinction only needs to exist while reading CGMES. Once each class's own rule is evaluated, everything
downstream can work from a uniform flat representation — a per-step table of \\(\{\rho, \alpha, r, x, g,
b\}\\) plus a scalar tap position selecting a row, with the originating class kept only as a property for
round-trip export.

For `PhaseTapChangerLinear` the rule is the two lines from the Motivation: \\(\rho = 1\\) and
\\(\alpha = (\text{step} - \text{neutralStep}) \cdot \text{stepPhaseShiftIncrement}\\), evaluated for
every step from `lowStep` to `highStep`.

A converter that only ever needs *one* step's effect — the current SSH position, because it never exports
a step table — can evaluate the same rule for that single step and skip building the table at all.

### 2. Reactance varies with tap position

A phase shifter's leakage reactance is not constant across its tap range. When `xMin`/`xMax` are present,
\\(x\\) is interpolated between them by a trigonometric rule shared between the Linear and Symmetrical
cases:

\\[ x(\alpha) = x_{min} + (x_{max} - x_{min}) \left( \frac{\sin(\alpha/2)}{\sin(\alpha_{max}/2)} \right)^{2} \\]

where \\(\alpha_{max}\\) is the largest angle reachable across the changer's own step range. This value
supersedes the `PowerTransformerEnd.x` from the EQ profile — the reason phase tap changers need an
`x`-override path at all, where an ordinary ratio tap changer usually doesn't.

### 3. Who moves the tap: fixed input vs. control outer loop

Regardless of which CGMES class produced it, the phase-shift angle is a *fixed parameter* within any
single Newton-Raphson solve — it is not an unknown, and appears in the Jacobian only if a tool wants
sensitivities with respect to it. Two designs then exist for choosing that parameter's value:

- **Fixed at conversion.** The tap position from the input snapshot (CGMES SSH `step`) is baked in once
  before the solve, and only an explicit external call changes it between solves.
- **Moved by an outer loop.** Solve, check whether the controlled branch's active-power or current flow
  matches its target, adjust the tap position, re-solve — the same architectural pattern the
  [Reactive Power Limits](./q_limits.md) page describes for Q-limit switching. Since tap positions are
  discrete, an "incremental" refinement computes a continuous \\(dP/d\alpha_1\\) sensitivity from the
  Jacobian to estimate how many positions to move at once, bounded to prevent oscillation.

This choice is orthogonal to concepts 1 and 2: a tool can model all four CGMES flavors faithfully and
still never move a tap, or move taps aggressively while supporting only one flavor.

## Where this fits in gridoxide today

gridoxide takes concept 3's fixed-input option: tap position is chosen once at conversion time from the
current SSH `step`, and there's no outer loop anywhere in the solver that moves a tap to hit an
active-power or current target. Every tap changer conversion (`RatioTapChanger`, all four
`PhaseTapChanger` variants) computes the complex ratio/angle for *whatever step CGMES's SSH profile
already says it's at* and bakes that into the transformer's `tap` field before Newton-Raphson ever runs.

`phase_tap_linear` (`src/cgmes.rs`) is concepts 1 and 2 for the single current step — no step table is
built, since nothing in gridoxide exports one:

```rust
fn phase_tap_linear(ptc: &cimstructs::PhaseTapChangerLinear, mrid: &str, xtx: f64) -> Result<TapEffect, CgmesError> {
    ...
    let angle_rad_at = |s: f64| -> f64 { ((s - neutral) * inc_deg).to_radians() };

    let alpha = angle_rad_at(step);
    let tap = Complex::from_polar(1.0, alpha);

    let alpha_max = (low..=high).map(|s| angle_rad_at(s as f64)).fold(f64::MIN, f64::max);
    let x_override = match (x_min_max(ptc.x_min, ptc.x_max, xtx), alpha_max != 0.0) {
        (Some((x_min, x_max)), true) => {
            let ratio = (alpha / 2.0).sin() / (alpha_max / 2.0).sin();
            Some(x_min + (x_max - x_min) * ratio * ratio)
        }
        (Some(_), false) => Some(0.0),
        (None, _) => None,
    };

    Ok(TapEffect { tap, x_override })
}
```

The x-interpolation helper is shared with `phase_tap_symmetrical`, matching concept 2's "one rule for both
classes" — generalized to take raw `xMin`/`xMax` rather than a `PhaseTapChangerNonLinear` reference, since
`PhaseTapChangerLinear` is a structurally distinct, shallower CGMES class carrying its own same-named
fields, not a sibling subtype of Symmetrical/Asymmetrical.

## Tool reference

| Tool | Per-step model (§1) | x interpolation (§2) | Tap movement (§3) |
|---|---|---|---|
| **gridoxide** | all four CGMES flavors, evaluated for the current SSH step only — no step table | ✅ shared Linear/Symmetrical helper (`phase_tap_linear`, `phase_tap_symmetrical`) | fixed at conversion; no outer loop |
| powsybl-core | all four, normalized at import into one flat `PhaseTapChangerStep` table (`{rho, alpha, r, x, g, b}`); `CgmesPhaseTapChangerBuilder.addSteps()` dispatches on `isLinear()`/`isTabular()`/`isAsymmetrical()`/`isSymmetrical()`, originating class kept only as a property for SSH round-trip | ✅ `getStepXforLinearAndSymmetrical`, shared by `addStepsLinear()` and `addStepsSymmetrical()` | data model only — carries a tap position, doesn't move it |
| powsybl-open-loadflow | consumes the normalized table; nothing downstream distinguishes a Linear-derived changer from any other (regulation mode is `CURRENT_LIMITER` vs `ACTIVE_POWER_CONTROL`, not the CGMES class) | inherited from the imported table | ✅ `PhaseControlOuterLoop` / `AcIncrementalPhaseControlOuterLoop`, the latter using a \\(dP/d\alpha_1\\) sensitivity to size discrete moves |
| lightsim2grid | a single fixed per-transformer angle `shift_` (radians) alongside the magnitude `ratio_` — structurally a fixed ratio+angle tap, no per-step flavors | ❌ | fixed input; changeable only via `GridModel::change_shift_trafo(...)` between solves. Its pandapower import rejects "ideal phase shifter" transformers outright (`RuntimeError("Ideal phase shifters are not modeled...")`) |
| power-grid-model | n/a — no CGMES import | n/a | ✅ `TapChangingStrategy` outer loop |
| VeraGrid | not surveyed (has a CIM importer) | not surveyed | ✅ `control_taps_phase` |
| pandapower | not surveyed (has a CIM importer) | not surveyed | ✅ `control.DiscreteTapControl`/`ContinuousTapControl` |

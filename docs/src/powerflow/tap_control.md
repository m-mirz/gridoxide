# Transformer Tap Control

## Motivation

Every steady-state result in this book before this chapter — power flow, contingency,
sensitivity, OPF, state estimation, the RAO gate — was computed with transformer taps that never
move. That is a modelling assumption, and four of the five tools in the
[feature comparison](../reference/feature_comparison.md) do not make it.

A real on-load tap changer holds something. Given a target and a deadband, it steps its position
until the quantity it regulates is inside:

\\[ \text{find } t \in \{t_{\min},\dots,t_{\max}\} \ \text{ s.t. } \
   \left|V_c(t) - V^{\text{target}}\right| \le \tfrac{1}{2}\,\text{deadband} \\]

with \\(V_c\\) the voltage at the **controlled** bus, which need not be either of the
transformer's own. A phase shifter substitutes a monitored branch's active power for the voltage.

## The strategy, and why this one

powsybl-open-loadflow offers three strategies for voltage control. Two put the continuous ratio
inside the Newton system and round afterwards. The third steps the discrete position directly,
from the sensitivity of the controlled quantity to the tap:

\\[ \Delta\rho = \frac{V^{\text{target}} - V_c}{\partial V_c/\partial\rho},
   \qquad
   \Delta\alpha = \frac{P^{\text{target}} - P_b}{\partial P_b/\partial\alpha} \\]

The incremental one is what gridoxide implements, for a reason specific to this crate: those
derivatives are already [`ac_sensitivity::AcSensitivity`](../sensitivity/ac.md), with
`Variable::TransformerRatio` against `Function::VoltageMagnitude` and `Variable::PhaseShift`
against `Function::BranchActivePower`, validated against a central-difference re-solve of the
full nonlinear power flow. The continuous variants would need the ratio inside the Jacobian
pattern and a different `n_unknowns`.

One consequence is worth stating. `AcSensitivity` carries a **direct** \\(\partial f/\partial p\\)
term for a tapped branch's own flow, on top of the chain rule. A phase shifter usually regulates
its own flow, which is exactly that configuration, and dropping the direct term flips the sign of
the answer on the regulated branch while leaving every other branch correct.

## Three guards, all load-bearing

**Deadband.** Every enabled control in the CGMES conformity fixtures carries one — Svedala's are
±1.3% to ±1.9% of nominal, FullGrid's phase control is ±17.5 MW on a −65 MW target. A loop that
ignores the deadband hunts, and a loop that hunts does not terminate.

**Insensitivity filter.** Below `MIN_SENSITIVITY` (powsybl's own 0.05) a controller is declared
unable to reach its quantity. Without it \\(\Delta\rho\\) is a division by nearly zero and the tap
slams to a limit.

**Direction-change budget.** A tap that has reversed `MAX_DIRECTION_CHANGES` times is oscillating
between two positions that straddle the target with none inside the deadband. That is a property
of the network and the deadband, not a solver failure.

The sensitivity is exact only at the converged state, so `max_tap_shift` caps how far a
controller may move in one pass — trading passes for stability.

## What a controller reports

| Outcome | Meaning |
|---|---|
| `InDeadband` | Inside. The success case. |
| `Closest` | The best position the table offers, target still out of reach. |
| `AtLimit` | It asked for a ratio or angle the changer cannot produce at all. |
| `Insensitive` | It cannot move its own controlled quantity. |
| `Hunting` | It reversed direction too often to be converging. |
| `Unfinished` | The run ended before it settled. |

`Closest` is common rather than exceptional, and distinguishing it from `InDeadband` matters. A
tap moves in finite steps, so a deadband narrower than one step's worth of effect is unreachable
from every position. ENTSO-E's own `PST_PhaseTapChangerLinear_Type2` asks for zero flow within
5·10⁻⁴ pu on a shifter whose closest position leaves 0.11 pu — reporting that as `InDeadband`
claims success at a flow 200 times the tolerance.

## Several controllers, one target

Parallel transformers in a substation regulate one bus; FullGrid has two phase shifters holding
one branch's power. Both loops group by the quantity being held and size the controllers against
each other — most authority first, each one's achieved effect subtracted before the next is
sized. Sizing them independently doubles the correction and sets both oscillating.

This is deliberately *not* the "last writer wins" behaviour the comparison table records for
gridoxide's generator-side shared voltage control. It is also the cheaper half of that problem,
since taps sit outside the Newton system.

## Where the data comes from

| Importer | Tap table | Regulating control |
|---|---|---|
| CGMES | every position of all four `PhaseTapChanger` flavours plus `RatioTapChangerTable` | `TapChangerControl`, in `voltage` and `activePower` modes |
| IIDM | `<step>` elements | `regulating`, `targetV`/`regulationValue`, `targetDeadband`, `regulationMode` |
| UCTE-DEF | `##R` tap fields | `##R` columns 33–38 (kV) and 58–63 (MW) |
| PGM, native JSON | — | — |

CGMES is where the gate is, which is the opposite of the [RAO](../rao/index.md) work: there, UCTE
and IIDM carried everything and CGMES carried nothing. Not one of the 207 `##R` records in the
vendored UCTE corpus populates a target column, and the committed IIDM fixtures have no regulating
changer either.

A CGMES `PhaseTapChangerLinear`/`Symmetrical`/`Asymmetrical` carrying `xMin`/`xMax` genuinely
changes **reactance** as it moves. `TapChanger::series` carries that per position, and
`set_position` writes it alongside the ratio — moving the phase shift while leaving the impedance
at the exported position is a wrong answer that still moves the flow in the right direction.

## Validation

Three kinds, in increasing order of how much they depend on anything outside the repository.

**Analytic.** A two-bus network whose right position is computable from the step table. The loop
must find it from every starting position, and find the *same* one regardless of where it
started.

**Against an exhaustive sweep.** Solve at every position in range and check the loop's answer
against what the sweep says. This is what catches a sign error in \\(\partial V/\partial\rho\\):
a loop that moved the tap the wrong way would still change the voltage, still terminate, and still
look plausible.

**Against the conformity fixtures.** On Svedala all eleven controllers go from 2–5% outside their
deadbands to inside, and the solve still converges. A per-controller sweep, every other tap held
where the loop left it, confirms each chosen position.

### A finding worth recording

The obvious gate — Svedala's SSH tap positions equal its SV ones, so a converged loop must leave
all eleven alone — has a **false premise**. The fixture's own published solution sits 2–5% away
from the targets its own SSH declares, against half-deadbands of 1–2%. The published state does
not satisfy the controls the document records, and a loop that left those taps alone would be the
defective one.

This says nothing bad about the fixture: a conformity model exists to exercise a profile's
classes, not to be a converged operating point. But it does mean Svedala cannot serve as evidence
that the loop leaves *correct* taps alone. SmallGrid, PowerFlow and MiniGrid carry that instead —
between them ten tap tables, one disabled control and no live one, so nothing may move.

## Deliberately out of scope

- powsybl's two continuous-then-round voltage strategies (§ above).
- `reactivePower`-mode tap control, and IIDM's `CURRENT_LIMITER` phase mode, which holds a
  current. Both are recognised and counted rather than silently read as something else.
- Shunt-section control: the same outer loop over a different discrete device.
- Tap control inside contingency and batch solves, and in DC.
- Taps as OPF decision variables — a regulating tap and an optimized tap are different questions.

## See also

- [Outer Loops](./outer_loops.md) — the layer this runs on
- [AC Sensitivity Analysis](../sensitivity/ac.md) — where \\(\partial V/\partial\rho\\) comes from

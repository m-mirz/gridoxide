# Security Analysis and the SAR

`gridoxide::security_analysis` (`--features cgmes`) takes a CGMES model and a decoded CO dataset, solves
the base case and each contingency, holds each terminal's current against its `CurrentLimit`s, and
writes what it finds as a dataset of the `SecurityAnalysisResult` (SAR) profile.

```python
import gridoxide

analysis = gridoxide.security_analysis(profiles, co_files)      # sharing="capability"
print(analysis.solved, analysis.stalled, analysis.violations()[:1])
xml = analysis.to_sar(publisher=..., is_version_of=..., rights_holder=...,
                      generated_at=..., title=..., description=...)
```

```bash
gridoxide cgmes-security profiles/*.xml --contingencies co/*.xml --sar out.xml \
    --publisher <iri> --version-of <iri> --rights-holder <name> --generated-at <time>
```

The building blocks are public: `read_contingencies`, `current_limits`, `terminal_flows`, `monitor`,
`machines`, `capability_weights`. `solve` solves one state as the analysis solves each: it returns the
buses, each component's outcome, each `TopologicalNode`'s bus, the terminal flows, and what the options
did (the buses reactive limits took off voltage control, the shift sharing the imbalance gave each bus,
the machines it took past their limits).

## What it reports

A SAR holds the violations, a violating terminal's per limit exceeded, plus the base case of each
terminal a contingency violates. A contingency's own equipment is not monitored: it carries nothing.
`PowerFlowResult.value` is in percent, as the profile defines it. A contingency that does not converge
has no result, as the profile has no place for one; `stalled` lists them.

A TATL allows more than the PATL for a while, so one at or below its PATL is not a limit, and is left
out. Otherwise terminals are held against every limit they have.

## Who takes the imbalance

What a state schedules seldom equals what it needs, and the difference has to land somewhere. That
choice can move the answer more than anything else here, and no CGMES attribute settles it:

| `sharing` | Who | Note |
|---|---|---|
| `capability` (default) | the machines in service, by `GeneratingUnit.maxOperatingP` | |
| `reference` | the angle reference the model declares | loads what is near it |
| `equal` | every generator bus alike | ignores machine size |

`GeneratingUnit.normalPF` would be the model's own statement, but in ReliCapGrid it is scaled per IGM, so
it cannot share across a merged model. Generator limits do not stop a share; `past_limits` lists
where one goes past them. None of these is a dispatch: reserves and control areas are not modelled.

## Limits

- Reactive limits are off by default (`enforce_q_limits`): a model whose voltage targets they cannot all
  hold stalls with them, in the base case and so everywhere.
- Only current limits are monitored, not voltage or power limits.
- Remedial actions are not applied.
- A `contingentStatus` other than `outOfService` is an error.

See [the 2026 SV-IOP](./sv_iop_2026.md) for how this does on ReliCapGrid.

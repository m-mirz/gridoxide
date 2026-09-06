# CRAC fixtures

Five OpenRAO JSON CRACs, taken verbatim from [powsybl-open-rao][rao] test
resources and covered by its **MPL-2.0** licence, not gridoxide's Apache-2.0 —
see `docs/src/reference/provenance.md`.

[rao]: https://github.com/powsybl/powsybl-open-rao

The format has **24 versions** across the reference checkout's 428 CRAC files,
and it renamed things between them. These five are chosen to span that:

| File | Format version | What it is there to catch |
|---|---|---|
| `crac-v1.1.json` | 1.1 | An early 1.x file, with no `instants` list at all — the four instants were fixed. |
| `crac-for-rao-result-v1.8.json` | 1.5 | The 1.x elementary-action spellings `pstSetpoints` and `injectionSetpoints`, which do not end in `Actions` and were therefore invisible to an earlier version of the reader *and* to its own unknown-key detector. |
| `crac-v2.1.json` | 2.1 | Early 2.x. |
| `crac-v2.6.json` | 2.6 | The richest: two curative instants, RA usage limits, all four range-action families, relative ranges, and a PST tap-to-angle map. |
| `crac-v2.9.json` | 2.9 | The newest spellings. |

**All five contain bare `NaN`**, which Java's Jackson writes for an absent rated
current and which no JSON parser accepts. 98 of the 428 CRACs in the reference
checkout do. `crac_json::parse` neutralises those literals — outside strings
only — on a retry, which is the difference between reading a third of the corpus
and reading all of it.

## `features/` — the external gate

Scenarios copied verbatim from powsybl-open-rao's own Cucumber suite, together
with the CRACs and `RaoParameters` files they name. Also MPL-2.0.

They are the only check in this repository that gridoxide did not write for
itself. A finite-differenced derivative proves a derivative; a
screening-versus-resolve comparison proves two of gridoxide's own paths agree.
These state margins to the decimal and name which remedial actions should be
used, and their authors wrote them to judge a different implementation.

**Five corpora, 197 scenarios, 1702 of 1862 checkable assertions.** Each file
states its own selection in its header; the split is by what the scenarios
exercise rather than by where they came from upstream.

| File | Scenarios | Score | Covers |
|---|---|---|---|
| `dc_scenarios.feature` | 25 | 180 / 186 | `@dc`, eleven networks |
| `ac_scenarios.feature` | 38 | 276 / 282 | `@ac` on TestCase12Nodes |
| `ac_scenarios_16nodes.feature` | 93 | 963 / 977 | `@ac` on TestCase16Nodes |
| `second_preventive.feature` | 15 | 118 / 123 | the second preventive pass |
| `min_cost.feature` | 26 | 165 / 294 | costly optimization (`MIN_COST`) |

The first three exclude the features that have a corpus of their own, which is
why the other two exist. `min_cost.feature` is the odd one: it was vendored
**before** the capability, so its score is a recorded *before* figure rather
than an achievement — the order `plans/RAO_PLAN.md` records as the only one that
works, and the one that found every defect in the second-preventive corpus.

Tolerance is the reference's own, `max(5, 1.5%)` in whichever unit the step is
written (`RaoSteps.flowMegawattTolerance`). Of the 160 assertions that differ,
**26 are recorded disagreements** — measured on OpenRAO's own objective, with
gridoxide never worse — and the gate refuses to let a scenario disagree without
one; see `RECORDED_DISAGREEMENTS` in `tests/rao_cucumber_test.rs`.

Getting here took **thirty-two defects**, every one internally consistent and
externally wrong, all listed in `plans/RAO_PLAN.md` §8.3 and §8.8–§8.12. Two
worth knowing when adding scenarios: `Given network file is "..." for CORE CC`
is **not** decoration — it rewrites every voltage level (380 kV to 400, 220 to
225), and since the nominal voltage is the per-unit base that moves every
susceptance by 11%. And `the initial margin on cnec` is a different step from
`the margin on cnec ... after PRA`; reading them as one compares the optimized
answer against the starting point.

Steps are unmodified, including the file paths — the harness resolves them by
basename, so the text stays as its authors wrote it.

**Nothing is skipped.** Every step the corpus states is checked, so the ratio is
the whole of it rather than the part that was convenient. It was not always so,
and the last one to go was worth the trouble: five `setpoint of RangeAction`
steps sat skipped as "not comparable", and wiring them up uncovered two defects
in redispatch, one a hundredfold sensitivity error that had passed for years
(§8.12). A skip that states what it is waiting for is a defect report nobody has
read yet.

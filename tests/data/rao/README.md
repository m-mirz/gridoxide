# CRAC fixtures and the external gate

Nothing from [powsybl-open-rao][rao] is stored here. Its CRACs and Cucumber
suite (MPL-2.0) live in the `benchmark-grids` submodule under
`powsybl-open-rao/`; `../openrao-fixtures.txt` maps the CRAC names below to
their paths there. What this directory holds is gridoxide's selection of the
suite's scenarios: `dc_scenarios.txt`, `ac_scenarios.txt` and
`ac_scenarios_16nodes.txt`.

[rao]: https://github.com/powsybl/powsybl-open-rao

The format has **24 versions** across the reference checkout's 428 CRAC files,
and it renamed things between them. These five, which `tests/rao_crac_test.rs`
reads, are chosen to span that:

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

## The external gate

Scenarios from powsybl-open-rao's own Cucumber suite, which
`tests/rao_cucumber_test.rs` reads unmodified from the submodule and selects by
the ids in the three `.txt` files.

They are the only check in this repository that gridoxide did not write for
itself. A finite-differenced derivative proves a derivative; a
screening-versus-resolve comparison proves two of gridoxide's own paths agree.
These state margins to the decimal and name which remedial actions should be
used, and their authors wrote them to judge a different implementation.

The selection is every scenario in that suite that is `@dc` and `@rao` (and not
`@ac`), uses a JSON CRAC, and needs none of loop flows, relative margins, costly
optimization, HVDC, second-preventive or MARMOT — the features `src/rao/` does
not implement. That is 25 of roughly 500, across eleven networks.

**138 of 142 checkable assertions match**, at the reference's own tolerance
(`max(5 MW, 1.5%)`, from its `RaoSteps.flowMegawattTolerance`). The four that do
not are two phase-shifter taps in the MNEC scenarios 5.2.1.3 and 5.2.1.4, and
the two margins that follow from them: the reference's own tap rounding declines
to reconsider a tap that pays an MNEC penalty, and gridoxide's answer scores
better on the reference's objective than the reference's does. See
`BASELINE_MATCHED_AC` in `tests/rao_cucumber_test.rs` for the arithmetic.

Getting there took thirteen fixes, listed in `tests/rao_cucumber_test.rs`. Two
worth knowing about when adding scenarios: `Given network file is "..." for
CORE CC` is **not** decoration — it rewrites every voltage level (380 kV to 400,
220 to 225), and since the nominal voltage is the per-unit base that moves every
susceptance by 11%. And `the initial margin on cnec` is a different step from
`the margin on cnec ... after PRA`; reading them as one compares the optimized
answer against the starting point.

Step paths resolve the way the reference's own `CommonTestData` resolves them:
networks under `files/cases/`, CRACs under `files/crac/`, parameters under
`files/configurations/`. Not by basename, since one name can mean two files:
the suite's `common/TestCase12Nodes.uct` differs from the `commons` module's
copy by one column's alignment.

Not every step is checkable. Those that are not are **printed as skipped**
rather than dropped, so the list can be shrunk deliberately: what remains is
second-preventive bookkeeping (`the execution details should be`), per-side
flows the result does not carry, and assertions against a written-out network
file. Skipped steps are excluded from both halves of the ratio — counting them
as passes would flatter it and counting them as failures would be a lie.

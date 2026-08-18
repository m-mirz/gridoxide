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

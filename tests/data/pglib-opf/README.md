*Data from the [IEEE PES Power Grid Library — Optimal Power Flow](https://github.com/power-grid-lib/pglib-opf),
v23.07, Benchmark Group "Typical Operations".*
*Licensed **CC-BY-4.0** — https://creativecommons.org/licenses/by/4.0/ — used
here with attribution and unmodified.*

---

# OPF benchmark cases

The fixture set for optimal power flow. MATPOWER `.m` files, read by
`python/gridoxide/matpower.py`, which emits both the PGM network document and
the companion `*.opf.json` carrying costs, generator limits and branch ratings.

## Why these and not the cases already in `benchmark-grids/`

Because those cannot test an OPF. `tests/data/benchmark-grids/matpower/` holds
power-*flow* benchmarks that happen to carry `gencost`, and on the three
smallest every branch is unrated:

| Case | Branches | `rateA = 0` (unlimited) |
|---|---|---|
| `benchmark-grids` `case14` | 20 | **20** |
| `benchmark-grids` `case118` | 186 | **186** |
| `benchmark-grids` `case300` | 411 | **411** |
| **pglib** `case14_ieee` | 20 | **0** |
| **pglib** `case118_ieee` | 186 | **0** |

With no branch limits nothing ever binds, every locational marginal price is
identical, and the congestion half of the problem is not exercised at all. A
fixture set that cannot make a constraint bind cannot test an optimizer.

pglib exists to fix exactly that: the cases are curated *for* OPF, with
meaningful generation limits, line ratings and costs. Every branch in all five
cases here is rated.

Note that pglib's `case14_ieee` is **not** the `case14` in `benchmark-grids` —
same underlying network, deliberately different limits. They must not be
conflated.

## The cases

| File | Buses | Branches | Generators |
|---|---|---|---|
| `pglib_opf_case3_lmbd.m` | 3 | 3 | 3 |
| `pglib_opf_case5_pjm.m` | 5 | 6 | 5 |
| `pglib_opf_case14_ieee.m` | 14 | 20 | 5 |
| `pglib_opf_case30_ieee.m` | 30 | 41 | 6 |
| `pglib_opf_case118_ieee.m` | 118 | 186 | 54 |

`case3_lmbd` and `case5_pjm` are small enough to reason about by hand, which is
what makes them useful when a larger case disagrees and the question is *which
term*.

## Reference objectives

From pglib's own [`BASELINE.md`](https://github.com/power-grid-lib/pglib-opf/blob/master/BASELINE.md),
produced with PowerModels.jl and IPOPT — independent of anything in this
repository, and the external gate `plans/OPF_PLAN.md` §7 describes.

| Case | DC ($/h) | AC ($/h) |
|---|---|---|
| `case3_lmbd` | 5.6959e+03 | 5.8126e+03 |
| `case5_pjm` | 1.7480e+04 | 1.7552e+04 |
| `case14_ieee` | 2.0515e+03 | 2.1781e+03 |
| `case30_ieee` | 7.4728e+03 | 8.2085e+03 |
| `case118_ieee` | 9.3101e+04 | 9.7214e+04 |

**Both columns are published, which matters more than it looks.** An earlier
draft of `OPF_PLAN.md` claimed pglib publishes AC objectives only, and
concluded that DC-OPF would have to ship without an external number to point
at — leaning entirely on KKT certificates and analytic cases. That was wrong:
the DC column is right there, so phase 3 has a published external reference
after all.

Two caveats on using them. PowerModels' DC formulation need not match this
one in every convention — reference-bus handling and whether line limits apply
to the DC approximation are both places implementations differ — so a small
gap is a question to investigate rather than an immediate failure. And the AC
column is a *local* optimum found by an interior-point method, so at phase 6 a
disagreement may mean a different local solution rather than a bug; that
comparison must report feasibility alongside the objective.

**The first caveat paid off immediately.** Phase 3's initial run matched four
cases to better than 0.04% and `case30_ieee` to only 0.42%. The convention
that differed was not one of the two guessed above but the *susceptance
formula*: PowerModels builds its DC model from the full series admittance
(`b = x/(r²+x²)`), while MATPOWER's `makeBdc` — and gridoxide's DC power flow,
and pandapower, and lightsim2grid — use `b = 1/x`. On resistive branches these
differ materially, and `case30_ieee` happens to be congested on one
(`r = 0.0192, x = 0.0575`, a 10% overstatement), so the error landed directly
on a binding constraint. Adopting the series form for OPF brought every case
inside 0.03%. `src/opf/dc.rs` carries the full table and reasoning.

The transferable point: matching a published objective to 0.4% is *not*
reassurance. It was the one case that disagreed by ten times the others that
exposed a formula wrong in all five.

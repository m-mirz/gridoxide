# Dynawo example fixtures

`IEEE14.dyd`, `IEEE14.par` and `IEEE14.crv` are copied verbatim from
[dynawo/dynawo](https://github.com/dynawo/dynawo),
`examples/DynaWaltz/IEEE14/IEEE14_GeneratorDisconnections/`.

Copyright (c) 2015-2019, RTE (http://www.rte-france.com). Licensed under the
**Mozilla Public License, v. 2.0** — see the header each file carries.
SPDX-License-Identifier: MPL-2.0.

They are here so `tests/dynamics_dyd_test.rs` reads the parameter names a real
Dynawo case uses rather than names written from memory, which is the whole risk
in a keyword-addressed format. The matching `IEEE14.iidm` is deliberately *not*
vendored: this suite gates the reader, not a full case, and the static half is
phase 5's business.

Dynawo's repository also ships `reference/outputs/curves/curves.csv` for this
case — its own solver's trajectories, committed. That is the comparison data
phase 5 needs, and it means the Dynawo gate does **not** require a Dynawo
install, contrary to what `plans/RMS_PLAN.md` §8 phase 0 assumed.

Note that this case's own timeline records `PMIN : activation` on two
generators: its governors hit their minimum-power limits. gridoxide models no
limits (see `src/dynamics/models/avr.rs`), so this particular case is known in
advance to diverge for a stated reason.

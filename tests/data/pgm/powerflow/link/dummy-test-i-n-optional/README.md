*SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>*  
*SPDX-License-Identifier: MPL-2.0*

*Adopted from Power Grid Model tests/data/power_flow/dummy-test-i-n-optional*

---

# Power Flow Test Case: dummy-test-i-n-optional

3 node, 1 line, 1 link, 2 source, 1 sym_load, 1 asym_load, 1 shunt

Adopted for its `link` component: a zero-impedance connection that power-grid-model
models as a branch with a large fixed admittance, and whose own flow it reports. That
reported flow is why gridoxide stamps a link rather than merging its endpoints — a merge
would delete the branch these numbers describe. See
`docs/src/powerflow/zero_impedance_branches.md`.

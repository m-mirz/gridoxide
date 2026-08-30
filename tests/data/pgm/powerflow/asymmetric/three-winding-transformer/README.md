*SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>*  
*SPDX-License-Identifier: MPL-2.0*

*Adopted from Power Grid Model tests/data/power_flow/three-winding-transformer*

---

# Component Test Case: Three-Winding Transformer, asymmetric

Test case for validation of the three-winding transformer in **asymmetric** power flow.

This is the only asymmetric reference answer for a three-winding transformer anywhere in the
vendored corpus, which is why it is here. gridoxide models a three-winding transformer as three
two-winding legs to a synthesized star node — power-grid-model's own star-equivalent, in
`three_winding_transformer.hpp::convert_to_two_winding_transformers` — and in the phase domain the
legs need their *winding configurations*, which the symmetric model never had to carry:

- **T1** is node 1 to the star, `wye_n`/`wye_n` with clock 0,
- **T2** is node 2 to the star, `winding_2`/`winding_1` with clock `12 − clock_12`,
- **T3** is node 3 to the star, `winding_3`/`winding_1` with clock `12 − clock_13`.

Those are what determine the **zero sequence**, and a balanced case cannot see them. This fixture's
windings are delta / wye_n / wye_n with both clocks at 11, so the zero-sequence paths of the three
legs genuinely differ, and an implementation that got them wrong would still reproduce the symmetric
answer.

The batch walks ten tap positions with all three sides in service.

```txt
node_1 ──┐
node_2 ──┼── three_winding_transformer_4 (star)
node_3 ──┘
source_5 on node_1
```

`params.json` in the reference states `rtol = 1e-4`, `atol = 1e-5`.

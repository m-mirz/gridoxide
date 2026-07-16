*SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>*  
*SPDX-License-Identifier: MPL-2.0*

*Adopted from Power Grid Model tests/data/power_flow/pandapower/components/symmetric/three_winding_transformer*

---

# Component Test Case: Three Winding Transformer

Test case for validation of the three winding transformer component for symmetrical power flow calculations in
pandapower.
The implementation of both the libraries are similar.
Three 2-winding transformers are used in star connection for creating a 3-winding transformer.

- A three-winding transformer can have multiple states from the node end status being open or closed.
  However only the all working status is being tested here for the sake of simplicity.

The circuit diagram is as follows:

```txt
source_7--node_1--three_winding_transformer_3--node_2
                                        |
                                        node_3
```

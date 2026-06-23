*SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>*  
*SPDX-License-Identifier: MPL-2.0*

*Adopted from Power Grid Model tests/data/power_flow/pandapower/components/asymmetric/line*

---

# Component Test Case: Asymmetric Line

Test case for validation of the line component for asymmetric (three-phase unbalanced) power flow.

The circuit topology is identical to the symmetric line test, with an unbalanced load replacing the symmetric load:

```txt
source_4--node_1--line_3--node_2--line_6--node_5              (Line from_status=to_status=1)
                          node_2--line_7--node_5--asym_load_9  (Line from_status=0)
                          node_2--line_8--node_5               (Line to_status=0)
                          node_2--line_10--node_5              (Line from_status=to_status=0)
```

The asymmetric load at node_5 specifies different P and Q per phase (10 kW/9 kW/8 kW and 2 kVAR/1.52 kVAR/1 kVAR).

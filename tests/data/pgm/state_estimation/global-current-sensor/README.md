*SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>*  
*SPDX-License-Identifier: MPL-2.0*

*Adopted from Power Grid Model tests/data/state_estimation/current-sensor/global-current-sensor*

---

# global-current-sensor

2 node, 1 line, 1 source, 1 sym_load, 1 sym_voltage_sensor, 1 sym_current_sensor

The current sensor uses `angle_measurement_type`: 1 (global).
Identical to its sibling fixture but for that field, and they converge to different states.

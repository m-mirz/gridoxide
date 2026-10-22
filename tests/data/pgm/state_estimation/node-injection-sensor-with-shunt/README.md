# State Estimation Test Case: node-injection-sensor-with-shunt

3 node, 2 line, 1 source, 2 sym_load, 1 shunt, 3 sym_voltage_sensor, 2 sym_power_sensor

Not adopted from Power Grid Model: written for gridoxide, with power-grid-model
1.13.185 as the reference. Node 2 carries a load and a shunt and a node-injection
sensor (`measured_terminal_type` 9). The sensors read power-grid-model's own power
flow of the network exactly: |V| at every node, and at nodes 2 and 3 the node
`p`/`q` of that power flow, which at node 2 is the load alone. power-grid-model
counts a shunt as part of the network, not of a node's injection; its estimate
from these sensors is the power flow state (to 2e-16 p.u.), which
`sym_output.json` holds.

# gridoxide-cli

`gridox` is the command line of [gridoxide](https://github.com/m-mirz/gridoxide): AC and DC power
flow, state estimation, short circuit, AC sensitivity, QV curves and continuation, dynamics, DC and
AC optimal power flow, security analysis and remedial-action optimisation, over MATPOWER (`.m`),
UCTE-DEF (`.uct`), PowSyBl IIDM (`.xiidm`), power-grid-model JSON and CGMES.

```bash
cargo install gridoxide-cli
gridox --help
gridox solve case118.m
```

Every feature that needs no system
library is on by default; `opf-highs`, `opf-ipopt`, `klu`, `pardiso`, `smallsignal-arpack` and
`gpu` are forwarded to gridoxide under the same names. See the
[book](https://m-mirz.github.io/gridoxide/getting_started/building.html#running) for each
subcommand.

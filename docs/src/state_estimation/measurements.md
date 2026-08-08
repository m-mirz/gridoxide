# Measurements and What They Mean

Everything on this page is about turning sensors into the rows of \\(z\\) and \\(H\\). It is the part
of state estimation with the least mathematics and the most opportunity for a silent, confident,
wrong answer — a flipped sign or a misattributed quantity does not fail, it converges to something
plausible.

## One measurement is one scalar

A `Measurement` is a single scalar observation, not a sensor. Newton-Raphson WLS treats
\\(\sigma_P\\) and \\(\sigma_Q\\) as independent, so a power sensor becomes *two* rows; a voltage
sensor becomes one or two depending on whether it carries an angle. A voltage sensor that does carry
an angle is a phasor (PMU) measurement, and it is what makes the global phase determinate.

## Aggregation runs in two opposite directions

Redundant sensors are merged before estimation rather than passed through as extra rows, and the two
cases move opposite ways:

**Several sensors observing one quantity** merge by inverse variance:

\\[
z = \frac{\sum_k z_k / \sigma_k^2}{\sum_k 1 / \sigma_k^2}, \qquad
\sigma^2 = \frac{1}{\sum_k 1 / \sigma_k^2}
\\]

The result is *more* certain than any input. A quantity you measure twice is one you know better.

**Several appliances making up one bus injection** sum, and their variances sum with them:

\\[
S = \sum_k S_k, \qquad \sigma^2 = \sum_k \sigma_k^2
\\]

The result is *less* certain than its parts, because independent errors accumulate.

Conflating these is easy and costly. An early version of gridoxide's aggregation merged per *bus*
rather than per *appliance*, which turned two sensors watching one load into two loads and doubled
the injection. The ordering that works is: merge per appliance first, then sum across appliances at
the bus.

## Sign conventions are not uniform

Taken from power-grid-model's reference-direction rules, which differ by sensor type:

| Sensor | Reference direction | Positive means |
|---|---|---|
| Branch terminal | branch | power flows *from the node into the branch* |
| Load, shunt | load | power flows *from the node into the appliance* (consumption) |
| Source, generator | generator | power flows *into the node* |
| Node injection | generator | net injection into the node |

Branch measurements pass through unchanged, since that is already the convention
`branch_flow::terminal_flow` uses. Loads and shunts are negated to become injections.

## Where gridoxide's model differs from power-grid-model's

This is the subtlety that most affects correctness, and it is not a sign issue — it is a difference
in *what the quantity is*.

power-grid-model treats a source and a shunt as appliances at a node, so their power counts toward
that node's injection. gridoxide models both **structurally**: a source becomes a virtual slack bus
feeding through an impedance branch, and a shunt becomes a Y-bus diagonal entry. Neither therefore
appears in `network::power_injections` at that bus at all — by Kirchhoff's current law, the net
injection at a source node with no load is *zero*, because the source's power arrives through a
branch that is part of the network.

Each of the three needs its own measurement function:

| power-grid-model sensor | gridoxide's model | \\(h(x)\\) |
|---|---|---|
| `source` | virtual slack bus behind an impedance branch | that branch's flow, negated |
| `shunt` | Y-bus diagonal entry | \\(-\vert V \vert^2 \overline{y_{sh}}\\) |
| `node` (injection) | — | bus injection **plus** both of the above |

Using the plain bus injection for any of them is wrong by construction. This was found rather than
predicted: `tests/measurement_residual_test.rs` evaluates every measurement function at the state
power-grid-model published and reported a **63-sigma** disagreement on exactly that quantity, with
the model saying 0 and the sensor saying 2.4 p.u.

## Zero-injection buses

A bus with no load and no generator injects exactly nothing. That is a property of the network, not
an observation of it: no sensor, no noise, no uncertainty.

The common shortcut is to feed it in as a pseudo-measurement of zero with a very small \\(\sigma\\).
It works, and it is why so many estimators are described as ill-conditioned — the weight matrix then
spans many orders of magnitude and \\(G\\) squares that spread. power-grid-model ships fixtures named
`ill-conditioned-by-line-meshed` and `ill-conditioned-by-link-meshed` for precisely this failure.

gridoxide enforces it as a hard equality constraint instead, via the Lagrangian stationarity
conditions:

\\[
\begin{bmatrix} G & C^{T} \\\\ C & 0 \end{bmatrix}
\begin{bmatrix} \Delta x \\\\ \lambda \end{bmatrix} =
\begin{bmatrix} H^{T} W r \\\\ -c(x) \end{bmatrix}
\\]

The augmented matrix is symmetric *indefinite* rather than positive definite, which rules out a
Cholesky-style solver — but every gridoxide backend is a general sparse LU, so this costs assembly
work only. A constraint row turns out to be the same bus-injection partials an injection
*measurement* would produce; the difference between a constraint and a measurement is entirely in how
the system consumes the row.

Two details worth knowing:

- **Which buses qualify is read off the input document**, not off `Bus::p_spec`. A state-estimation
  document leaves `p_specified` unset, so an unmeasured load looks exactly like zero injection in the
  converted network while being nothing of the sort. Getting this backwards would constrain real
  loads to zero.
- **Sources and shunts do not disqualify a bus**, because gridoxide models both structurally and
  neither appears in that bus's injection. The virtual slack buses *are* excluded, since that is
  where a source's unknown power enters.

The fixture that pins this down is `node-injection-sensor-and-zero-injection`: an injection sensor
reading 0.1 p.u. on a node with no appliance attached. power-grid-model requires at least one
appliance for such a sensor to mean anything, so it overrides the reading and reports both buses at
exactly 1.0∠0. Without the constraint, weighted least squares fits the sensor *perfectly* by driving
that node to \\(\sqrt{2} \angle -45°\\) — a 41% overvoltage on a bus with nothing connected to it,
and an objective of \\(4 \times 10^{-28}\\). By its own criterion, a flawless answer.

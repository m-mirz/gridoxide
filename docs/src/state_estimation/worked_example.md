# A Weighted Least Squares Estimate, Worked

[The problem page](./index.md) states the algebra; this one carries it out. It has three parts: a
linear estimate small enough to do with a calculator, the measurement Jacobian written out for the
nonlinear case, and what gridoxide reports on a real fixture.

The first part is **pen-and-paper arithmetic, not gridoxide output** — gridoxide has no DC estimator,
and a DC estimate is the only version of this whose every intermediate fits on a page. It is exactly
the algebra `se::nr` runs, with \\(h\\) linear so that \\(H\\) is constant and one step is exact.

## Part 1: three buses, four measurements, one step

### The network

Three buses in a triangle, every branch with \\(x = 0.1\\) p.u., so \\(b = 10\\) throughout. Bus 1 is
the angle reference, so the unknowns are

\\[
x = \begin{bmatrix} \theta_2 \\\\ \theta_3 \end{bmatrix}, \qquad n = 2
\\]

In DC, a branch flow is \\(P_{ij} = b_{ij}(\theta_i - \theta_j)\\) and nothing depends on voltage
magnitude at all.

### The measurements

Three, so the redundancy is \\(m - n = 1\\).

| \\(i\\) | Quantity | \\(h_i(x)\\) | \\(z_i\\) (p.u.) | \\(\sigma_i\\) |
|---|---|---|---|---|
| 1 | flow 1→2 | \\(10(\theta_1 - \theta_2) = -10\,\theta_2\\) | 0.20 | 0.01 |
| 2 | flow 1→3 | \\(-10\,\theta_3\\) | 0.42 | 0.01 |
| 3 | flow 3→2 | \\(10(\theta_3 - \theta_2)\\) | −0.18 | 0.02 |

The three disagree, which is the whole point. Flows 1 and 2 imply \\(\theta_2 = -0.020\\) and
\\(\theta_3 = -0.042\\), and those two together imply a 3→2 flow of \\(-0.22\\) — not the \\(-0.18\\)
the third meter reports. No state reproduces all three.

### \\(H\\), \\(W\\) and \\(G\\)

\\(h\\) is linear, so \\(H = \partial h/\partial x\\) is constant:

\\[
H = \begin{bmatrix} -10 & 0 \\\\ 0 & -10 \\\\ -10 & 10 \end{bmatrix},
\qquad
W = \begin{bmatrix} 10^4 & & \\\\ & 10^4 & \\\\ & & 2.5 \times 10^3 \end{bmatrix}
\\]

Note the third meter's weight: \\(1/0.02^2 = 2500\\), a quarter of the others', because it is trusted
half as far. Then

\\[
G = H^{T} W H = \begin{bmatrix} 1.25 \times 10^6 & -2.5 \times 10^5 \\\\ -2.5 \times 10^5 & 1.25 \times 10^6 \end{bmatrix},
\qquad
H^{T} W z = \begin{bmatrix} -15500 \\\\ -46500 \end{bmatrix}
\\]

Each entry is worth checking once by hand, because the pattern is the whole trick:
\\(G_{11} = 10^4(-10)^2 + 2500(-10)^2 = 1.25\times 10^6\\) — meters 1 and 3 both see \\(\theta_2\\) —
while \\(G_{12} = 2500 \cdot (-10)(10) = -2.5 \times 10^5\\) comes from meter 3 alone, the only one
that sees both unknowns. **\\(G\\)'s sparsity is the measurement graph, not the network graph.**

### The estimate

Solving \\(G\,\hat{x} = H^{T} W z\\) — one step, because the problem is linear:

\\[
\hat{\theta}_2 = -\frac{31}{1500} = -0.0206\overline{6}, \qquad
\hat{\theta}_3 = -\frac{31}{750} = -0.0413\overline{3}
\\]

Neither matches what any pair of meters said on its own. The estimate splits the difference, weighted:

\\[
h(\hat{x}) = \begin{bmatrix} 0.20\overline{6} \\\\ 0.41\overline{3} \\\\ -0.20\overline{6} \end{bmatrix},
\qquad
r = z - h(\hat{x}) = \begin{bmatrix} -0.00\overline{6} \\\\ +0.00\overline{6} \\\\ +0.02\overline{6} \end{bmatrix}
\\]

The residual does **not** vanish, and it never will: the measurements contradict each other, so
\\(r \ne 0\\) at the optimum. This is the single most important difference from a power-flow Newton
loop, which drives its mismatch to zero because an exact solution exists. Convergence here is tested
on the size of the *step*, never on the residual.

Note also where the residual landed: the third meter, the least trusted one, absorbs four times as
much of the disagreement as either of the others. That is the weighting doing its job.

### The objective and the chi-squared test

\\[
J(\hat{x}) = r^{T} W r = 0.4\overline{4} + 0.4\overline{4} + 1.7\overline{7} = \frac{8}{3} = 2.667
\\]

with \\(m - n = 1\\) degree of freedom. The 5% critical value of \\(\chi^2(1)\\) is 3.841, and
\\(2.667 < 3.841\\), so the test does not reject. The three meters disagree about as much as meters of
their declared accuracy would be expected to.

### Normalized residuals, and why they are all equal

\\[
\Omega = R - H G^{-1} H^{T}, \qquad
r_i^{N} = \frac{\vert r_i \vert}{\sqrt{\Omega_{ii}}}
\\]

With

\\[
G^{-1} = \begin{bmatrix} 8.\overline{3} \times 10^{-7} & 1.\overline{6} \times 10^{-7} \\\\ 1.\overline{6} \times 10^{-7} & 8.\overline{3} \times 10^{-7} \end{bmatrix}
\\]

the diagonal of \\(\Omega\\) comes out as

| \\(i\\) | \\(R_{ii} = \sigma_i^2\\) | \\((HG^{-1}H^{T})_{ii}\\) | \\(\Omega_{ii}\\) | \\(r_i^{N}\\) |
|---|---|---|---|---|
| 1 | \\(10^{-4}\\) | \\(8.\overline{3} \times 10^{-5}\\) | \\(1.\overline{6} \times 10^{-5}\\) | **1.633** |
| 2 | \\(10^{-4}\\) | \\(8.\overline{3} \times 10^{-5}\\) | \\(1.\overline{6} \times 10^{-5}\\) | **1.633** |
| 3 | \\(4 \times 10^{-4}\\) | \\(1.\overline{3} \times 10^{-4}\\) | \\(2.\overline{6} \times 10^{-4}\\) | **1.633** |

All three are identical, and that is not a coincidence of the numbers chosen. With one degree of
freedom the residual vector is confined to a one-dimensional subspace, so every normalized residual
is \\(\pm\sqrt{J}\\) — here \\(\sqrt{8/3} = 1.633\\) exactly.

**The lesson is the one worth taking away from this whole page.** At redundancy 1, bad data can be
*detected* and never *identified*: whichever meter is broken, the analysis accuses all three equally.
Identification needs redundancy 2 or more, and a measurement whose residual has no variance at all —
\\(\Omega_{ii} = 0\\) — is *critical*: the estimate is forced to reproduce it exactly, so an error in
it cannot be detected by any amount of analysis. Those are skipped rather than given a meaningless
normalized residual.

Notice also that \\(\Omega_{ii} < \sigma_i^2\\) for every meter. That is why dividing by \\(\sigma\\)
alone is not enough: a redundantly measured quantity spreads its error across its neighbours and so
*under*-shows in its own raw residual.

## Part 2: the nonlinear case, written out

In AC the flows are not linear in the state, \\(H\\) changes every iteration, and the estimate is
Gauss-Newton:

\\[
G(x^{(k)})\ \Delta x = H(x^{(k)})^{T} W\, r(x^{(k)}), \qquad x^{(k+1)} = x^{(k)} + \Delta x
\\]

The state layout has no PV buses — a generator's voltage is estimated, not asserted — so every bus
contributes a magnitude and every bus but the angle reference contributes an angle:

\\[
x = \bigl[\ \theta_0 \ldots \theta_{N-1} \text{ except the reference},\ \ \vert V_0 \vert \ldots \vert V_{N-1} \vert\ \bigr],
\qquad n = 2N - 1
\\]

against power flow's \\(n_{angle} + n_{pq}\\). Confusing the two layouts is the easiest way to produce
a Jacobian that is subtly and consistently wrong, so `se::jacobian::StateLayout` owns the mapping and
nothing else indexes by hand.

### One formula for every power measurement

A textbook writes a separate Jacobian block for a bus injection, a branch terminal flow, a shunt draw
and so on. gridoxide writes one, because they are the same thing five ways: a current that is linear
in the complex bus voltages, evaluated against the voltage at one bus.

\\[
I = \sum_k c_k V_k, \qquad S = V_{at}\,\overline{I}
\\]

The pair \\((c, at)\\) is all that distinguishes them:

| Measurement | \\(c\\) | \\(at\\) |
|---|---|---|
| Bus injection at \\(i\\) | the Y-bus row \\(i\\) | \\(i\\) |
| Branch terminal flow | that branch's own half-row | the terminal's bus |
| Shunt draw | a single negated admittance | the shunt's bus |

and the derivatives follow once, for all of them:

\\[
\frac{\partial S}{\partial \theta_k} = -j\,V_{at}\,\overline{c_k V_k} \;+\; [\,k = at\,]\; j S,
\qquad
\frac{\partial S}{\partial \vert V_k \vert} = V_{at}\,\overline{c_k e^{j\theta_k}} \;+\; [\,k = at\,]\; \frac{S}{\vert V_{at} \vert}
\\]

with \\(\operatorname{Re}\\) giving the \\(P\\) row and \\(\operatorname{Im}\\) the \\(Q\\) row.

### It reduces to the textbook expressions

Take the injection case — \\(c = Y_{i\cdot}\\), \\(at = i\\) — and a bus \\(k \ne i\\). Writing
\\(Y_{ik} = G_{ik} + jB_{ik}\\) and \\(\theta_{ik} = \theta_i - \theta_k\\),

\\[
V_i \overline{Y_{ik} V_k} = \vert V_i \vert \vert V_k \vert
\bigl[\,G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik}
\;+\; j\bigl(G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik}\bigr)\bigr]
\\]

and multiplying by \\(-j\\) swaps the parts, giving

\\[
\frac{\partial P_i}{\partial \theta_k} = \vert V_i \vert \vert V_k \vert \bigl(G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik}\bigr),
\qquad
\frac{\partial Q_i}{\partial \theta_k} = -\vert V_i \vert \vert V_k \vert \bigl(G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik}\bigr)
\\]

which are exactly the off-diagonal power-flow Jacobian entries from
[the power flow page](../powerflow/index.md#powerflow-with-power-mismatch-function-and-polar-coordinates).
The \\([k = at]\\) terms recover the diagonal ones. A bus-injection *measurement* has the same row a
bus-injection *mismatch* does; what differs is the system that consumes it.

`se::jacobian`'s finite-difference test is the real guard on all of this, and it is applied to the
unified form on every measurement kind rather than to each closed form separately.

### Structural zeros are kept

Every coefficient produces an entry in \\(H\\), including one whose numerical value is zero.
\\(H\\)'s sparsity pattern has to depend on the topology alone and not on the state: dropping a zero
would shrink the row at a flat start — where many \\(\sin(\theta_i - \theta_k)\\) vanish exactly — and
grow it again next iteration, invalidating the cached symbolic factorization that makes the
[solver reuse](../solvers/backends.md) worthwhile.

## Part 3: what the code reports

power-grid-model's `1os2msr` fixture: three nodes, two lines, one source, three voltage sensors with
angles, and seven power sensors.

```console
$ gridoxide estimate tests/data/pgm/state_estimation/1os2msr/input.json

4 bus(es), 20 measurement(s) after aggregation
Converged in 4 iteration(s)
Objective J(x) = 3.413330e-19

Estimated voltages:
  node 1: |V| = 1.023912 p.u., angle = -0.747976 deg
  node 2: |V| = 1.024067 p.u., angle = -1.010546 deg
  node 3: |V| = 1.023650 p.u., angle = -1.156362 deg

Observability: rank 8 of 8 unknown(s)
  fully observable

Bad data: chi-squared 6.8267e-19 on 14 dof, p = 1.0000e0
  not rejected at 5%
```

Four numbers there are worth reading carefully.

**Four buses, not three.** gridoxide models a source structurally, as a virtual slack bus feeding
through an impedance branch, so a three-node document has four buses. power-grid-model has no such
bus in its state space, which is why a network it considers fully observable can still leave
gridoxide with surplus unknowns. See
[where the models differ](./measurements.md#where-gridoxides-model-differs-from-power-grid-models).

**Eight unknowns, not seven.** \\(n = 2N\\) here rather than \\(2N - 1\\), because this fixture's
voltage sensors carry angles. A network measured only in magnitudes and powers is invariant under a
global phase shift and needs a reference pinned or \\(G\\) is singular however many measurements
there are; a phasor measurement supplies an absolute angle, that invariance disappears, and pinning a
reference on top of one would be a false constraint rotating the whole estimate away from the data.
gridoxide pins a reference exactly when no angle is measured — compare `1os2msr-no-angle`, the same
network with 12 degrees of freedom instead of 14.

**Twenty measurements from ten sensors.** A `Measurement` is a single scalar, not a sensor:
Newton-Raphson WLS treats \\(\sigma_P\\) and \\(\sigma_Q\\) as independent, so each of the seven power
sensors becomes two rows and each of the three angle-carrying voltage sensors becomes two.

**\\(J = 3.4 \times 10^{-19}\\), not \\(\approx m - n\\).** On real telemetry the objective would sit
near its degrees of freedom. It sits at zero here because power-grid-model generated these readings
from the true state without adding noise — there is nothing for a correct estimate to disagree with.
A near-zero statistic on *real* data would itself be suspicious: it would suggest the declared sigmas
are far too large.

**14 degrees of freedom from 20 measurements and 8 unknowns.** The count is \\(m - n + k\\): a
variable that was pinned rather than estimated never consumed a degree of freedom, and each equality
constraint gives one back. \\(20 - 8 + k = 14\\) says two zero-injection constraints were in force —
enforced as hard constraints rather than as very-high-weight pseudo-measurements, which is what keeps
\\(W\\) from spanning more orders of magnitude than the physics requires.

Run the same thing on the fixture built to fail and the machinery from Part 1 comes back with
something to say:

| Fixture | \\(\chi^2\\) | dof | Rejected at 5% |
|---|---|---|---|
| `1os2msr` | 3.4e-19 | 14 | no |
| `transmission-case` | 2.4e-7 | 48 | no |
| `node-injection-sensor-and-zero-injection` | **2.0e4** | 4 | **yes** |

The last one's worst suspect is its injection sensor at a normalized residual of exactly **100.00** —
the 100-sigma conflict that fixture is built around, recovered as a number, by exactly the
\\(\Omega\\) calculation done by hand above.

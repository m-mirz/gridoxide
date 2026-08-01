# Observability and Bad Data

A converged estimate is not a correct one. Weighted least squares will absorb a broken sensor by
bending the state toward it, and will happily return an answer for a network where half the
quantities were never determined at all. Both failures look exactly like success. These two analyses
are what distinguish them.

## Observability: what the measurements determine

An unobservable system is not a numerical accident, and treating it as one produces the least useful
diagnostic available — a factorization that failed, with no indication why. The useful answer names
the unknowns nobody is watching.

There are two distinct failures.

**Structural.** A column of \\(H\\) that is identically zero: no measurement function mentions that
unknown at all. The estimator has to find these anyway, since they make \\(G\\) singular on their own.
In gridoxide they are usually the virtual slack buses synthesized per source — a network
power-grid-model considers fully observable can still leave gridoxide with surplus unknowns, because
PGM has no such bus in its state space.

**Numerical.** A column that is *present* but linearly dependent on the others. Two branch-flow
measurements on a radial feeder with no injection measurement between them constrain the same
combination of angles twice and leave another combination free. Nothing about the sparsity pattern
gives this away; it takes a rank computation.

\\(G = H^{T} W H\\) is symmetric positive *semi*definite by construction, and the standard
rank-revealing factorization for that class is a Cholesky with symmetric diagonal pivoting: take the
largest remaining diagonal at each step, stop when it falls below tolerance, and the number of steps
is the rank. gridoxide uses `faer`'s blocked implementation for the factorization but makes the rank
*decision* itself — faer stops at \\(\varepsilon n\\), around \\(10^{-16}\\) relative, which is the
threshold for "not numerically positive definite". Observability wants a much looser one: a direction
determined only at the \\(10^{-12}\\) level is not usefully determined, and calling it observable
would be the more misleading answer.

The cost is that \\(G\\) is densified, making this \\(O(n^3)\\) time and \\(O(n^2)\\) memory. The
analysis refuses above `DENSE_LIMIT` and says so, rather than quietly allocating gigabytes on a
transmission grid; a sparse rank-revealing method is the natural follow-up.

Unknowns pinned this way are held at their starting value rather than estimated, and reported. That
is deliberate: an unknown nothing observes cannot be recovered, and moving it would be inventing
information. It also improves the failure mode — an under-measured system now solves its observable
part and names the rest, instead of failing outright.

## Bad data: whether the measurements agree

Two questions, answered separately.

### Is there bad data at all?

Under the assumption that each error is independent and normal with the declared \\(\sigma\\), the
objective is chi-squared distributed:

\\[
J = r^{T} W r \sim \chi^2(m - n + k)
\\]

with \\(m\\) measurements, \\(n\\) estimated state variables and \\(k\\) equality constraints — each
constraint gives a degree of freedom back, and a pinned unobservable variable never consumed one. A
\\(J\\) far out in the tail says the residuals are too large to be noise.

Rejection says *something* is wrong, not which measurement. A test that does not reject is also not a
clean bill of health: a single moderate error, or several that partly cancel, can sit inside the
threshold.

### Which measurement?

The largest normalized residual. Raw residuals are not comparable — 0.1 is enormous against a
\\(\sigma\\) of 0.001 and negligible against a \\(\sigma\\) of 1 — and dividing by \\(\sigma\\) alone
is still not enough, because a redundantly measured quantity spreads its error across its neighbours
and so *under*-shows in its own residual. The right scale is the residual's own standard deviation:

\\[
r_i^{N} = \frac{\vert r_i \vert}{\sqrt{\Omega_{ii}}}, \qquad
\Omega = R - H G^{-1} H^{T}
\\]

conventionally compared against 3. \\(\Omega\\) is computed through the *augmented* system, so the
zero-injection constraints count: a constrained estimate has less freedom to move, which changes how
much of an error surfaces in the residual rather than in the state.

\\(\Omega_{ii}\\) costs one linear solve per measurement, so the full diagonal costs more than the
estimate itself. gridoxide shortlists candidates by the cheap proxy \\(\vert r_i \vert / \sigma_i\\)
and computes \\(\Omega_{ii}\\) exactly for the worst 20, configurable. This is an approximation with
a real failure mode, not a free shortcut: because \\(\Omega_{ii}\\) varies per measurement, the true
worst can in principle sit outside the shortlist. Raising the limit to \\(m\\) removes the doubt at
the corresponding cost.

A measurement whose residual has no variance at all is *critical* in the usual terminology — the
estimate is forced to reproduce it exactly, so an error in it cannot be detected by any amount of
analysis. Those are skipped rather than assigned a meaningless normalized residual.

## What the fixtures show

Run over power-grid-model's own state-estimation fixtures:

| Fixture | \\(\chi^2\\) | dof | Rejected at 5% |
|---|---|---|---|
| `1os2msr` | 3.4e-19 | 14 | no |
| `1os2msr-no-angle` | 9.6e-22 | 12 | no |
| `inf-measurement-with-injection` | 1.1e-20 | 2 | no |
| `transmission-case` | 2.4e-7 | 48 | no |
| `node-injection-sensor-and-zero-injection` | **2.0e4** | 4 | **yes** |

The last one's worst suspect is its injection sensor at a normalized residual of exactly **100.00** —
the 100-sigma conflict that fixture is built around, recovered as a number.

One caveat when reading those figures: the consistent fixtures produce \\(\chi^2 \approx 0\\) rather
than \\(\chi^2 \approx \text{dof}\\), because power-grid-model generated their readings from the true
state without adding noise. There is nothing for a correct estimate to disagree with. Real telemetry
would sit near its degrees of freedom, and a near-zero statistic on real data would itself be
suspicious — it would suggest the declared sigmas are far too large.

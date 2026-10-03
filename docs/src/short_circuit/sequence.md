# Symmetrical Components and the Fault Equations

[The problem page](./index.md) says gridoxide solves in the phase domain and reports in both. This
page is the mathematics of that: the transform, the sequence networks IEC 60909 is written in, the
boundary conditions each fault type imposes, and how those conditions are stamped into a phase-domain
matrix.

## The transform

Any three phase quantities decompose into three balanced sets. With
\\(a = e^{j2\pi/3} = 1\angle 120°\\),

\\[
\begin{pmatrix} X_a \\\\ X_b \\\\ X_c \end{pmatrix}
=
\underbrace{\begin{pmatrix}
1 & 1 & 1 \\\\
1 & a^2 & a \\\\
1 & a & a^2
\end{pmatrix}}_{A}
\begin{pmatrix} X_0 \\\\ X_1 \\\\ X_2 \end{pmatrix},
\qquad
\begin{pmatrix} X_0 \\\\ X_1 \\\\ X_2 \end{pmatrix}
= \frac{1}{3}
\begin{pmatrix}
1 & 1 & 1 \\\\
1 & a & a^2 \\\\
1 & a^2 & a
\end{pmatrix}
\begin{pmatrix} X_a \\\\ X_b \\\\ X_c \end{pmatrix}
\\]

Three useful identities fall straight out of \\(1 + a + a^2 = 0\\):

\\[
X_0 = \tfrac{1}{3}\bigl(X_a + X_b + X_c\bigr), \qquad
X_a = X_0 + X_1 + X_2, \qquad
\sum_{\text{phases}} X = 3X_0
\\]

The first is the whole reason zero sequence matters: it is the average of the three phases, so it is
non-zero exactly when the three do not sum to zero — which for currents means **current is returning
through the ground**. No ground path, no zero sequence. That single fact explains most of the
qualitative behaviour on this page.

The three sequences are named for what they do to a machine: **positive** rotates the normal way,
**negative** rotates backwards, **zero** does not rotate at all.

`shortcircuit::fortescue` is the \\(A^{-1}\\) above, and it is the exact inverse of the \\(A\\) that
`network::fortescue_to_phase` builds the phase-domain Y-bus with, so a result projected into
sequences and pushed back returns where it started.

## Why the sequences decouple

For a network whose three phases are symmetric, \\(A^{-1} Y_{abc} A\\) is block-diagonal:

\\[
Y_{012} = \begin{pmatrix} Y_0 & & \\\\ & Y_1 & \\\\ & & Y_2 \end{pmatrix}
\\]

Three independent networks, each a third the size, none talking to the others. That is what makes
the sequence domain the natural place to write a standard in, and it is why IEC 60909 states its
formulae in \\(Z_0, Z_1, Z_2\\).

For a passive network \\(Z_2 = Z_1\\): a line does not care which way the phasors rotate. \\(Z_0\\) is
different — usually two to four times \\(Z_1\\) for an overhead line, because the return path is the
earth rather than a conductor. gridoxide takes it from the input directly: a source's `z01_ratio` is
the multiplier \\(Z_0 / Z_1\\), and lines carry `r0`/`x0` of their own.

The couplings are what the fault introduces. A fault is the one place the three phases are *not*
treated alike, so it appears in the sequence domain as an interconnection between the three otherwise
independent networks — and the pattern of that interconnection is the fault type.

## The four fault types as boundary conditions

Reduce each sequence network to its Thévenin equivalent at the faulted bus — \\(Z_1, Z_2, Z_0\\)
looking in, with the pre-fault voltage \\(E = c\,U_n/\sqrt{3}\\) driving the positive-sequence one —
and every fault type becomes a way of wiring the three together.

| Fault | Phase-domain conditions | Sequence connection | \\(I_a\\) (or \\(I_b\\)) |
|---|---|---|---|
| Three-phase | \\(U_a = U_b = U_c = Z_f I\\) | positive only | \\(\dfrac{E}{Z_1 + Z_f}\\) |
| Line-to-ground (a) | \\(I_b = I_c = 0\\), \\(U_a = Z_f I_a\\) | all three **in series** | \\(\dfrac{3E}{Z_1 + Z_2 + Z_0 + 3Z_f}\\) |
| Two-phase (b–c) | \\(I_a = 0\\), \\(I_b = -I_c\\), \\(U_b - U_c = Z_f I_b\\) | 1 and 2 **in parallel**, 0 absent | \\(\dfrac{-j\sqrt{3}\,E}{Z_1 + Z_2 + Z_f}\\) |
| Two-phase-to-ground | \\(I_a = 0\\), \\(U_b = U_c = Z_f(I_b + I_c)\\) | 1 in series with (2 ∥ (0 + 3\\(Z_f\\))) | — |

Three things in that table are worth stating out loud.

**The \\(3\\) in the line-to-ground numerator and the \\(3Z_f\\)** are the same \\(3\\). All of
\\(I_a\\) flows through the fault impedance, but \\(I_a = 3I_0 = 3I_1\\) because the three sequence
currents are equal, so referred to one sequence network the impedance appears as \\(3Z_f\\).

**Two-phase has no zero-sequence term at all**, because there is no ground return. This is the
sharpest available check on a short-circuit result: if a two-phase fault reports a zero-sequence
component, something is wrong.

**Where \\(Z_0 = Z_1 = Z_2\\), a line-to-ground fault draws exactly the same current as a
three-phase one** — \\(3E/3Z = E/Z\\). That degeneracy is easy to hit in a test fixture (PGM's
`z01_ratio` defaults to 1) and easy to mistake for a bug. Where \\(Z_0 > Z_1\\), which is the normal
case, the line-to-ground current is *smaller*; near a solidly grounded transformer, where \\(Z_0\\)
can be less than \\(Z_1\\), it is larger, and that is why the line-to-ground case sometimes sizes the
breaker.

## The phase domain, and why gridoxide uses it

The sequence formulae above require reducing the network to a Thévenin equivalent at the fault, per
sequence, per fault location. The phase domain instead keeps the whole \\(3N \times 3N\\) nodal
system

\\[
I_N = Y_{bus}\, U_N, \qquad
U_N = \bigl[\,U_{1a}, U_{1b}, U_{1c},\ U_{2a}, \ldots\,\bigr]^{T}
\\]

and writes the fault's boundary conditions into it directly, by rewriting rows and columns at the
faulted bus. Two reasons this is the better choice here:

- Unbalanced faults fall out without special-casing. A fault type is a different set of rows to
  rewrite, not a different network to assemble.
- It is what power-grid-model does, so its fifteen short-circuit fixtures cross-validate gridoxide's
  numerics directly.

\\(Y_{bus}\\) is built by transforming each element's sequence admittances into phase coordinates
with \\(A\\), which for a symmetric element gives the familiar circulant

\\[
Y_{abc} = A \begin{pmatrix} y_0 & & \\\\ & y_1 & \\\\ & & y_2 \end{pmatrix} A^{-1}
= \begin{pmatrix} y_s & y_m & y_m \\\\ y_m & y_s & y_m \\\\ y_m & y_m & y_s \end{pmatrix},
\quad
y_s = \tfrac{y_0 + 2y_1}{3},\ \ y_m = \tfrac{y_0 - y_1}{3}
\\]

with \\(y_2 = y_1\\). The mutual term \\(y_m\\) vanishes exactly when \\(y_0 = y_1\\), which is the
statement that a network with identical sequence impedances has no phase coupling at all.

### Stamping the fault

An **impedance** fault and a **bolted** fault are not the same formula evaluated at a limit, and the
code does not treat them that way.

An impedance fault *adds* admittance. For a line-to-ground fault on phase \\(p\\) at bus \\(k\\), one
entry:

\\[
Y[3k{+}p,\ 3k{+}p] \mathrel{+}= y_f
\\]

A two-phase fault between \\(p_1\\) and \\(p_2\\) ties them to each other rather than to ground, so it
stamps the usual four-entry branch:

\\[
Y[p_1,p_1] \mathrel{+}= y_f,\quad Y[p_2,p_2] \mathrel{+}= y_f, \quad
Y[p_1,p_2] \mathrel{-}= y_f,\quad Y[p_2,p_1] \mathrel{-}= y_f
\\]

A **bolted** fault has \\(y_f = \infty\\), and letting an infinity into the matrix produces `NaN`, not
a large current. So the faulted bus's equations are **replaced**: its voltage is known to be zero, so
the unknown in that position becomes the injected current instead. In the matrix that is a column
erased and a \\(-1\\) written on the diagonal, with the corresponding right-hand side set to zero.

The two-phase case needs different surgery again, because its constraint is \\(U_{p_1} = U_{p_2}\\)
rather than either being zero: the two columns are *folded* together rather than erased, and one row
becomes the difference equation. Two-phase-to-ground needs both operations at once — the phases are
tied to each other *and* the pair is tied to ground — which is why it takes the column surgery even
with a finite impedance.

**The practical consequence for input data**: `r_f = x_f = 0` means bolted, and that is the default
when a `fault` omits them. A fault that draws no current is not something the component can express,
because it would not be a fault.

### The cost of the phase domain

The phase domain needs the network to have a path to ground. An ungrounded network has a singular
zero-sequence system in the phase domain — it does not in the sequence domain, where the
zero-sequence network is simply absent — and comes back as `ShortCircuitError::Singular`.

In practice a real network is grounded somewhere; the case that bites is a small test model with a
delta winding on both sides of a node. gridoxide follows power-grid-model in adding a tiny artificial
susceptance to ground such a winding, purely so the matrix stays non-singular. It is a numerical
device, not physics, and its exact value is arbitrary — a point that matters when comparing against
reference data, as `tests/data/pgm/short_circuit/README.md` records.

## Reading a result

Every result carries the sequence view alongside the phase quantities, and it is the fastest way to
tell whether an answer is sane:

| Fault | \\(X_0\\) | \\(X_1\\) | \\(X_2\\) |
|---|---|---|---|
| Three-phase | ≈ 0 | large | ≈ 0 |
| Line-to-ground | present | present | present |
| Two-phase | **exactly 0** | present | present |
| Two-phase-to-ground | present | present | present |

A second useful check: a delta winding blocks zero-sequence current. If a node sits behind one and
still shows zero-sequence voltage, the transformer's zero-sequence branch is wrong. Both properties
are asserted directly in `python/tests/test_short_circuit.py`.

[The worked example](./worked_example.md) carries all of this through on numbers small enough to
check by hand.

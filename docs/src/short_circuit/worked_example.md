# A Fault Current, by Hand

One node, one source, one fault. Everything on this page can be checked with a calculator, and every
figure quoted from gridoxide was produced by running the command shown.

## The system

```text
        ┌──── Z_s ────┐
   E ───┤             ├──── node 1 (U_n = 10 kV)
        └─────────────┘            │
                                 ┌─┴─┐
                                 │Z_f│   r_f = x_f = 0.1 Ω
                                 └─┬─┘
                                  ─┴─
```

A single 10 kV node fed by a source of short-circuit power \\(S_k'' = 100\\) MVA with an R/X ratio of
0.1, faulted through 0.1 + j0.1 Ω. That is power-grid-model's
`single_node_source_three_phase_c_maximum` fixture verbatim:

```json
"node":   [{"id": 1, "u_rated": 10000.0}],
"source": [{"id": 2, "node": 1, "status": 1, "u_ref": 1.0, "sk": 1e8, "rx_ratio": 0.1}],
"fault":  [{"id": 3, "status": 1, "fault_type": 0, "fault_phase": -1,
            "fault_object": 1, "r_f": 0.1, "x_f": 0.1}]
```

## Step 1: the source impedance

A short-circuit power is a statement about impedance:

\\[
Z_s = \frac{U_n^2}{S_k''} = \frac{(10^4)^2}{10^8} = 1.000\ \Omega
\\]

Then the R/X ratio splits it. With \\(R = 0.1X\\) and \\(\vert Z \vert = X\sqrt{1 + 0.1^2}\\),

\\[
X_s = \frac{1.000}{\sqrt{1.01}} = 0.995037\ \Omega, \qquad
R_s = 0.1\,X_s = 0.099504\ \Omega
\\]

**The voltage factor \\(c\\) does not appear here.** \\(c\\) scales the source's EMF, not its
impedance, and it *replaces* `u_ref` rather than multiplying it — a source declared at 1.02 p.u.
contributes \\(c\\), not \\(1.02c\\). That is precisely what isolates the result from the operating
point, and it is invisible in every reference fixture because they all set `u_ref = 1.0`.

## Step 2: the driving voltage

IEC 60909 replaces every source EMF with a line-to-neutral \\(cU_n/\sqrt{3}\\). Taking
\\(c = c_{max} = 1.10\\) for a network above 1 kV:

\\[
E = \frac{1.10 \times 10^4}{\sqrt{3}} = 6350.853\ \text{V}
\\]

## Step 3: the three-phase current

A three-phase fault is balanced, so only the positive-sequence network carries anything, and it is a
single loop:

\\[
Z_1 + Z_f = (0.099504 + 0.1) + j(0.995037 + 0.1) = 0.199504 + j1.095037\ \Omega
\\]

\\[
\vert Z_1 + Z_f \vert = \sqrt{0.039802 + 1.199106} = \sqrt{1.238908} = 1.113063\ \Omega
\\]

\\[
I_k'' = \frac{E}{\vert Z_1 + Z_f \vert} = \frac{6350.853}{1.113063} = \mathbf{5705.75\ A}
\\]

And the faulted node does not sit at zero volts, because the fault is not bolted — it sits at the
drop across \\(Z_f\\):

\\[
U_1 = I_k''\,\vert Z_f \vert = 5705.75 \times 0.141421 = 806.90\ \text{V}
\\]

which as a per-unit of the line-to-neutral base \\(10^4/\sqrt{3} = 5773.50\\) V is

\\[
u_1 = \frac{806.90}{5773.50} = 0.13976\ \text{p.u.}
\\]

### What the code says

```console
$ gridoxide short-circuit \
      tests/data/pgm/short_circuit/single_node_source_three_phase_c_maximum/input.json \
      --scaling max

1 node(s), 1 fault(s), 1 source(s), voltage scaling = c_max (largest current)

fault currents (A):
  fault      3: a =      5705.75, b =      5705.75, c =      5705.75

node voltages (p.u.):
  node      1 : a =  0.13976, b =  0.13976, c =  0.13976

symmetrical components of node voltage (p.u.):
  node      1: zero =  0.00000, positive =  0.13976, negative =  0.00000
```

Both hand figures to every digit printed, and the sequence line is the expected signature: positive
only, zero and negative at machine noise. That fixture's published expectation is
5705.7468249952872 A, so the agreement is not just with gridoxide but with power-grid-model.

### The `c_min` study is a different question

Rerun with `--scaling min` and \\(c\\) becomes 1.00, so \\(E = 5773.50\\) V and
\\(I = 5187.04\\) A — 9.1% lower, exactly the ratio \\(1.00/1.10\\), because \\(c\\) enters linearly
and the impedance does not move. That number is not a less accurate version of the first; it is the
answer to the protection-sensitivity question, where the first answers the breaker-rating question.

## Step 4: an unbalanced fault

Change two things — make the fault line-to-ground on phase a, and give the source a realistic
\\(Z_0 = 3Z_1\\):

```json
"source": [{"id": 2, "node": 1, "status": 1, "u_ref": 1.0, "sk": 1e8,
            "rx_ratio": 0.1, "z01_ratio": 3.0}],
"fault":  [{"id": 3, "status": 1, "fault_type": 1, "fault_phase": 1,
            "fault_object": 1, "r_f": 0.1, "x_f": 0.1}]
```

(committed as `docs/examples/one-node-slg.json`). Now all three sequence networks are in series:

\\[
Z_1 + Z_2 + Z_0 = (1 + 1 + 3)\,Z_1 = 5Z_1 = 0.497519 + j4.975186\ \Omega
\\]

\\[
Z_1 + Z_2 + Z_0 + 3Z_f = 0.797519 + j5.275186, \qquad \vert \cdot \vert = 5.335131\ \Omega
\\]

\\[
I_a = \frac{3E}{\vert Z_1 + Z_2 + Z_0 + 3Z_f \vert} = \frac{3 \times 6350.853}{5.335131} = \mathbf{3571.15\ A}
\\]

Smaller than the three-phase current, which is the usual ordering once \\(Z_0 > Z_1\\). Note that
setting `z01_ratio` back to 1 would have made the two *identical* — \\(3E/3Z_1 = E/Z_1\\) — a
degeneracy easy to hit in a fixture and easy to mistake for a bug.

### The sequence voltages, also by hand

With the three networks in series the sequence currents are equal:
\\(I_0 = I_1 = I_2 = I_a/3 = 1190.38\\) A. The zero and negative sequence networks have no source, so
their voltages are pure drops:

\\[
\vert U_0 \vert = I_0 \vert Z_0 \vert = 1190.38 \times 3.000 = 3571.15\ \text{V} = 0.61854\ \text{p.u.}
\\]
\\[
\vert U_2 \vert = I_2 \vert Z_2 \vert = 1190.38 \times 1.000 = 1190.38\ \text{V} = 0.20618\ \text{p.u.}
\\]

The positive-sequence one needs the phase kept, because it is a difference rather than a product:
\\(U_1 = E - I_1 Z_1\\), and \\(I_1 Z_1\\) leads \\(E\\)'s reference by
\\(\arg Z_1 - \arg Z_{total} = 84.29° - 81.40° = 2.89°\\). So

\\[
U_1 = 1.10 - 0.20618\angle 2.89° = 0.89408 - j0.01039, \qquad \vert U_1 \vert = 0.89414\ \text{p.u.}
\\]

Taking the magnitudes naively — \\(1.10 - 0.20618 = 0.89382\\) — is wrong in the fourth decimal, which
is a good demonstration that the small angle is real and not rounding.

```console
$ gridoxide short-circuit docs/examples/one-node-slg.json --scaling max

fault currents (A):
  fault      3: a =      3571.15, b =         0.00, c =         0.00

node voltages (p.u.):
  node      1 : a =  0.08747, b =  1.36844, c =  1.33922

symmetrical components of node voltage (p.u.):
  node      1: zero =  0.61854, positive =  0.89414, negative =  0.20618
```

All three sequence magnitudes to five decimals. Two more things are visible in that output and worth
naming:

- \\(u_a = 0.08747\\) is again \\(I_a \vert Z_f \vert\\) in per-unit: \\(3571.15 \times 0.141421 /
  5773.50\\). The faulted phase collapses; the healthy phases **rise**, to 1.37 and 1.34 p.u. That
  neutral-shift overvoltage on the unfaulted phases is a real effect and is why insulation is rated
  against it.
- \\(I_b = I_c = 0\\) exactly, which is the boundary condition the fault type imposes, recovered
  rather than assumed.

## Step 5: a fault with no ground return

Change nothing but the fault type — two-phase between a and b, clear of ground
(`docs/examples/one-node-2ph.json`, which is the file above with `fault_type: 2, fault_phase: 4`):

\\[
I = \frac{\sqrt{3}\,E}{\vert Z_1 + Z_2 + Z_f \vert}
= \frac{\sqrt{3} \times 6350.853}{\vert 0.299007 + j2.090074 \vert}
= \frac{11000.0}{2.111354} = \mathbf{5209.93\ A}
\\]

and gridoxide agrees to the last digit printed. The number to look at, though, is the one that is not
there:

```text
symmetrical components of node voltage (p.u.):
  node      1: zero =  0.00000, positive =  0.57990, negative =  0.52099
```

**Zero sequence is exactly zero**, because \\(Z_0\\) does not appear in the formula at all. There is
no ground return, so there is nowhere for zero-sequence current to go, and the source's
`z01_ratio: 3.0` is simply irrelevant to this fault. This is the sharpest single check available on a
short-circuit result — see
[reading a result](./sequence.md#reading-a-result).

## What the arithmetic here does *not* cover

Every number above is an **initial symmetrical** short-circuit current \\(I_k''\\). The quantities a
switchgear datasheet actually lists — the peak current \\(i_p\\), the breaking current \\(I_b\\), the
thermal equivalent \\(I_{th}\\) — are derived from \\(I_k''\\) by further standard factors that
gridoxide does not implement, because no fixture in this tree can validate them. See
[deliberate omissions](./index.md#deliberate-omissions).

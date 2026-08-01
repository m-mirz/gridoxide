# CGMES: Line Shunt Conductance (`ACLineSegment.gch`)

## Motivation

The [Powerflow](./powerflow.md) page's Y-bus construction models each line as a π-equivalent: a series
impedance plus a shunt admittance split evenly across both ends. Almost every power-flow tool's
documentation talks about that shunt term purely in terms of *susceptance* — line charging capacitance,
the reactive effect of a long line's own capacitance to ground — because for the overwhelming majority
of real lines the shunt's real part (conductance, representing corona loss or leakage) is negligible or
simply zero.

CGMES's `ACLineSegment` schema doesn't assume that: alongside `bch` (susceptance) it defines `gch`
(conductance) as a first-class, independent field, "of the entire line section" like every other
`ACLineSegment` electrical attribute. Most real conformance fixtures leave it at zero, but not all —
ENTSO-E's own MicroGrid-BE-MAS fixture has two lines (`BE-Line_6`, `BE-Line_2`) with non-negligible `gch`,
together worth several MW of real power at nominal voltage.

## The concepts

Every tool that models AC lines at all has shunt conductance somewhere in its branch admittance — this
isn't a case of some supporting it and others not. What differs is how it's *parameterized*, and how
visible a distinct "conductance" concept is in the data model. Three variants, all carrying the same
physical information:

### 1. A direct conductance field, per line or per end

The most explicit form: a siemens value stored next to the susceptance. Two sub-variants matter for
conversion:

- **One value for the whole line** — CGMES's own form, `gch` "of the entire line section". A converter
  must split it across the π-model's two ends, conventionally evenly: \\(g_1 = g_2 = g_{ch}/2\\), exactly
  the way \\(b_{ch}\\) is already split.
- **One value per end** (`g1`/`g2`) — strictly more general, since it can represent an asymmetric line
  whose two ends carry different shunts. A whole-line value maps into it trivially by halving; the reverse
  direction loses information unless the two ends happen to be equal.

### 2. One complex per-end shunt admittance

Instead of naming conductance separately, store a single complex number per end, \\(h = g + jb\\), and
stamp it directly into the Y-bus diagonal: \\(y_{11} = y_s + h_{or}\\). The conductance is the real part
— genuinely present and genuinely solved, just never given its own name. A grep for "conductance" in such
a codebase finds nothing, which is a naming fact, not a capability one.

### 3. Derived from capacitance and a loss tangent

A physically-motivated alternative parameterization: store the shunt capacitance \\(c_1\\) and a dielectric
loss tangent \\(\tan\delta_1\\), and derive the complex shunt from both:

\\[ y_{shunt} = \omega c_1 \tan\delta_1 + j\,\omega c_1 \\]

so \\(g = \omega c_1 \tan\delta_1\\). Same information, expressed the way a cable datasheet expresses it.
Converting *into* this form from a raw siemens pair means backing out the tangent as
\\(\tan\delta_1 = g_{ch}/b_{ch}\\) rather than mapping the two fields across directly — and a converter
whose source format has no loss-tangent equivalent has no way to produce a nonzero conductance at all.

## Where this fits in gridoxide today

Before this fix, `types::Line` had a `b_shunt` field and nothing else — `src/cgmes.rs`'s `ACLineSegment`
conversion read `bch` but never `gch`, even though its own comment already documented `gch` as one of the
fields "of the entire line section" (a stale comment that got ahead of the code, not the other way
around). `network::build_ybus` had no way to stamp a real shunt term even if the conversion had wanted
to.

The fix uses concept 1's whole-line form with the even split, since that's the same convention gridoxide's
own `bch` handling already used and `types::Line` already has half-open-line self-loop folding logic that
only needed a second field threaded through it:

```rust
pub struct Line {
    pub from: usize,
    pub to: usize,
    pub r: f64,
    pub x: f64,
    pub b_shunt: f64, // total line charging
    #[serde(default)]
    pub g_shunt: f64, // total shunt conductance (CGMES ACLineSegment.gch; usually 0)
}
```

```rust
// build_ybus: split shunt admittance (conductance + susceptance) equally to both ends of line
let y_shunt_half = Complex::new(ln.g_shunt / 2.0, ln.b_shunt / 2.0);
y.add(ln.from, ln.from, y_line + y_shunt_half);
y.add(ln.to, ln.to, y_line + y_shunt_half);
```

`#[serde(default)]` keeps every existing native-JSON and PGM-JSON fixture working unchanged. The PGM
importer always sets `g_shunt: 0.0`: PGM's own line schema is concept 3, and `PgmLine` has no `tan1` field
to derive a conductance from — the same "no data, so no effect" stance the rest of that converter takes.

### Why this mattered more than "a couple of MW out of a large network"

The missing MW didn't just make voltages a *little* off everywhere — it concentrated almost entirely
into one bus's *angle*, and nowhere else. `BE-Line_6`/`BE-Line_2` feed directly into the substation
hosting MicroGrid-BE-MAS's `StaticVarCompensator` (see the
[StaticVarCompensator](./cgmes_static_var_compensator.md) page), a voltage-magnitude-pinned bus.
A pinned bus can absorb a *reactive*-power mismatch by adjusting its own Q injection, but it has no
equivalent slack for an *active*-power one — so the several MW this fix restores had, before the fix,
nowhere to go but that one bus's angle. Cross-validated against pypowsybl's own independent CGMES import
(`scripts/bench/cross_validate_cgmes_microgrid_be.py`, with both tools pinned to the same reference bus
so their angles are directly comparable): worst angle deviation across the whole fixture dropped from
0.34° to 0.07° once `gch` was included — a five-fold improvement concentrated almost entirely at that
one substation, exactly where the missing real power was actually flowing in.

## Tool reference

| Tool | Parameterization | Where |
|---|---|---|
| **gridoxide** | 1 — whole-line `g_shunt`, split evenly in the Y-bus stamp | `types::Line::g_shunt`, `network::build_ybus`; read from `gch` by `src/cgmes.rs`, always 0 from the PGM importer |
| powsybl-core | 1 — per-end `g1`/`g2` alongside `b1`/`b2` (`MutableLineCharacteristics.java`); CGMES import splits evenly: `.setG1(gch / 2).setG2(gch / 2)` | `ACLineSegmentConversion.java` → `AbstractBranchConversion.convertBranch` |
| powsybl-open-loadflow | 1 — same per-end `getG1()`/`getG2()`, genuinely in the solved equations, not inert metadata | `AbstractBranchAcFlowEquationTerm` (P/Q mismatch terms), `AcBranchVector` (vectorized evaluator), `LfAsymLineAdmittanceMatrix` |
| lightsim2grid | 2 — one complex per-end shunt `h_or`/`h_ex`, stamped as `yac_11_ = ys + h_or`; no separately named conductance field anywhere in the C++ core | `element_container/LineContainer.hpp`, `TwoSidesContainer_rxh_A.hpp`; its powsybl import builds `h_or = g1 + j·b1`, confirming round-trip agreement with the model above |
| power-grid-model | 3 — `c1` + `tan1`, with \\(g = \omega c_1 \tan\delta_1\\) feeding the same `y1_shunt_`/`y0_shunt_` terms | `component/line.hpp`. A CGMES→PGM converter must compute `tan1 = gch / bch` |

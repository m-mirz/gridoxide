# Dynamics model library: TODO

Starting points for a plan, taken from a scan of Dynawo's `examples/`
corpus (sparse clone, 25 MB, no Dynawo install). Its 74 `.dyd` cases use
69 model libraries, and DynaSwing alone ships 63 sets of reference curves.

## Fix first

- [ ] `src/dynamics/dyd.rs` matches the generator library by prefix and
  suffix, and the regulator branch has no `else`. A library such as
  `GeneratorSynchronousFourWindingsTGov1SexsPss2a` is built as a bare
  machine with constant field voltage and mechanical power, and no
  warning is raised. Raise `DydWarning::UnsupportedLib` for any regulator
  suffix that is not recognised, as the `.dyr` reader already does.

## Models, by Dynawo cases that ship reference curves

- [ ] `TGov1Sexs`: map the suffix onto the existing `Tgov1` and `Sexs`
  models (`DynaSwing/ENTSOE/TestCase2`). Check first that the `.par`
  names and per-unit bases line up.
- [ ] PSS2A (`ENTSOE/TestCase1`, `ENTSOE/TestCase3`,
  `SingleMachineSystem/SynchronousMachineI8`). Also serves `IEEEST`-style
  stabilisers on the `.dyr` side.
- [ ] VRKundur, then PssKundur (`Kundur_Example13/KundurExample13_VR_NoPss`
  and `_VR_Pss`). The existing Dynawo gate already covers this directory.
- [ ] PmConst and VRNordic (`DynaWaltz/Nordic`).
- [ ] GoverPropVRPropInt (`DynaFlow/IEEE14/IEEE14_DisconnectLine`, no
  reference curves).
- [ ] Check that `Step` and `DoubleStep` map onto `EventSpec`, since the
  DynaSwing cases depend on them.

## Out of scope for now

- Converter-interfaced generation (IEC/WECC wind, PV, BESS, grid-forming):
  the largest block in the corpus. It is a different device class and
  needs a plan of its own.
- `LoadAlphaBeta` and `LoadAlphaBetaRestorative` loads.
- Current-limit, under-voltage and tap-changer-blocking automatons.
- Saturation.

# gridoxide on the 2026 SV-IOP test cases

What gridoxide does for the grid calculations of the [2026 SV-IOP report](https://www.entsoe.eu/Documents/CIM_documents/Market%20CIM/SV_IOP_Report_2026.pdf)
(Network Code Profiles 2.5, ReliCapGrid), and with what result. The section numbers follow the
report's, and only the use cases that need a calculation are here. Import, validation and export of the
datasets are cimoxide's, in [its own report](https://github.com/m-mirz/cimoxide/blob/main/scripts/relicapgrid/cimoxide_SV_IOP_2026.md),
which this one builds on. Every statement here is backed by a test in this repository, but for the two marked
as not. Each use case's **Test** line names what backs it; [Finding the tests](#finding-the-tests) says how they fit
together.

## Versions

| | |
|---|---|
| gridoxide | 0.4.0, commit `766c5a8` (branch `cgmes-relicapgrid`), Python bindings built with the `cgmes` feature |
| Dataset | [entsoe/relicapgrid](https://github.com/entsoe/relicapgrid) `cgmes-3.0_ncp-2.5_tc-2.0` @ `f09bb34` (2026-09-11), the submodule `tests/data/relicapgrid` |
| CIM input and SAR export | `cimoxide-model` 0.6.0, a dependency of the `cgmes` feature |
| Test reference | [cimoxide](https://pypi.org/project/cimoxide/) 0.6.0 from PyPI, a test-only dependency |

## How gridoxide was tested

gridoxide did not take part in the event. These are its own runs, afterwards, on a later commit of the dataset,
with the report's results known, so they do not stand beside the participants' on equal terms.

It was tested through its Python bindings and in Rust. The Python tests read the data through cimoxide,
independently of gridoxide's own use of `cimoxide-model`, so that what is solved is compared with what another
tool decodes. A converged answer is not taken as a result: each is held against something the data says.

| Check | What gridoxide does | Compared with | Test |
|---|---|---|---|
| Power flow | `PowerFlowModel.from_cgmes` and `solve` on each case | the case's own SvVoltage | `test_power_flow` |
| Flows | `PowerFlowModel.terminal_flows` | the SvPowerFlow of IGMs that agree with their SV | `test_flows_agree_with_the_published_flows` |
| An outage | `PowerFlowModel.with_out_of_service` | the same outage written into an SSH through cimoxide | `test_the_outage_equals_one_written_into_the_ssh` |
| The SAR | `gridoxide.security_analysis`, `to_sar` | cimoxide's validation, beside the SAR ReliCapGrid supplies | `test_cimoxide_finds_nothing_on_the_results_or_new_on_the_header` |

## Finding the tests

| File | What it holds |
|---|---|
| `python/tests/test_relicapgrid_power_flow.py` | PF_UC1, and BDS_UC1's merge |
| `python/tests/test_relicapgrid_contingencies.py` | CO_UC1's execution |
| `python/tests/test_relicapgrid_security_analysis.py` | SAR_UC1 |
| `python/tests/_relicapgrid.py` | what they share: the file groups, reading the data through cimoxide |
| `tests/cgmes_relicapgrid_test.rs` | the importer's behaviour on this data, and the analysis's counts, in Rust |
| `src/security_analysis.rs` | the analysis, with its unit tests |

Problems a power flow finds in the data are pinned in `test_relicapgrid_power_flow.KNOWN`, each with its cause. A
new entry there is a new difference to look at.

```bash
git submodule update --init tests/data/relicapgrid
maturin develop --release --features python,cgmes
pip install 'cimoxide>=0.6' pytest
python -m pytest -q python/tests -k relicapgrid
cargo test --release --features cgmes relicapgrid
```

## Overall results

| | Files or cases | Result | Test |
|---|---|---|---|
| Power flow | 17 cases (report Annex 2) | 13 solve with reactive limits, 6 IGMs agreeing with their SV; 4 stall on their data and converge without limits | `test_power_flow` |
| Boundary | 8 IGMs | merging the boundary changes no voltage | `test_the_boundary_changes_nothing` |
| Contingencies | 325, from 7 CO files | 312 solve on the CGM without limits, the reference taking the imbalance; 13 of Espheim's stall, unexplained | `test_every_contingency_runs_on_the_cgm` |
| Security analysis | the CGM, 325 contingencies | 304 solve, the machines sharing the imbalance; 13 base-case and 4,003 post-contingency violations | `test_the_counts` |
| SAR | 1 dataset, 4,071 results | cimoxide reports nothing on its results, and nothing on its header the supplied SAR's does not have | `test_cimoxide_finds_nothing_on_the_results_or_new_on_the_header` |

---

## 1. Power Flow on ReliCapGrid

### 1.1. PF_UC1: Power Flow ReliCapGrid model

**Test.** `test_relicapgrid_power_flow.py`: `test_power_flow`, on `case_files`, `solve` and `compare`, with each
problem's cause in `KNOWN`; `test_stalled_cases_converge_without_reactive_limits`.

**Report.** Run power flow on each IGM, then on growing combinations of IGMs, then on the whole CGM (§1.1, Annex 2).

**What gridoxide does.**

1. `PowerFlowModel.from_cgmes` on the case: an IGM's EQ, SSH, TP and SV with the common data; a combination's IGMs
   together; the CGM as ReliCapGrid's group `cgmes-CGM-Jotunheim`.
2. `solve(enforce_q_limits=True)`, Newton-Raphson with reactive limits. The four that stall are solved without them too.
3. An IGM or the CGM has to agree with its own SvVoltage, within 1% and 0.5° per component. A combination has no
   state of its own, each IGM's SV describing that IGM alone, so it has to hold one angle reference per component and
   voltages within 0.8 to 1.2 p.u.

**Result.** solved, on the terms below. All 17 converge without reactive limits.

| Case | With reactive limits | Largest residual (p.u.) | Largest difference from the SV | Cause |
|---|---|---|---|---|
| Belgovia | agrees | 1e-7 | 0.03%, 0.03° | |
| Britheim | agrees | 7e-7 | 0.14%, 0.21° | |
| DC-Espheim-Svedala | agrees | 4e-10 | 0.06%, 0.12° | |
| Galia | agrees | 2e-10 | 0.08%, 0.27° | |
| Nordheim | agrees | 0 | none | one bus, the angle reference: nothing to solve |
| Svedala | agrees | 2e-9 | 0.68%, 0.25° | |
| DC-Nordheim-Galia | solves, differs | 6e-12 | 0.10%, 7° | data: converter signs, below |
| Portheim | solves, out of range | 6e-12 | none, at 1.75 p.u. | data: the SV has 700 kV on a 400 kV base |
| Espheim | stalls | | | data: below; converges without limits (2e-7) |
| Nordheim + DC-Nordheim-Galia | solves | 6e-12 | | |
| Svedala + DC-Espheim-Svedala | solves | 1e-7 | | |
| Combination 1 (Belgovia + Svedala) | solves | 1e-12 | | |
| Combination 2 (+ Galia) | solves | 1e-12 | | |
| All except Espheim | solves, out of range | 1e-7 | | Portheim, as alone |
| Espheim + DC-Espheim-Svedala | stalls | | | as Espheim; converges without limits (2e-7) |
| Combination 3 (+ Espheim + Britheim) | stalls | | | as Espheim; converges without limits (6e-7) |
| Jotunheim (CGM) | stalls | | | data: below; converges without limits (3e-7) |

Scaling the load in an SSH is noticed (`test_notices_a_broken_hypothesis`), and files exported by cimoxide solve the
same (`test_exported_files_solve_the_same`).

**Notes.**

- **Espheim.** Its SSH has TannrsCk SM and two ExternalNetworkInjections hold a node at 220 kV, where its SV has them
  hold 231 kV, within their reactive limits; the SV is otherwise consistent with the EQ and SSH. With the SSH's target
  the limits cannot all hold, and Newton stalls. With 231 kV it solves with limits, and without them to the SV's voltages
  (`test_espheim_solves_with_the_target_its_sv_holds`).
- **The CGM.** Jotunheim's SV holds 38 of its 72 voltage controls off their SSH_2 targets, up to 1.32 times
  (ERIKSHAMN_G1: 20.9 kV, 27.7 in the SV), so it is no solution of that SSH_2, and with those targets the limits cannot
  all hold. It converges without them, at 0.77 p.u. at the lowest and up to 24% from that SV.
- **HVDC signs.** A DC-Nordheim-Galia converter's `targetPpcc` is +70 where its `ACDCConverter.p`, and the SV, are
  -70.01; Britheim writes both with one sign. This is the convention the report's §4.1 asks to settle.
- cimoxide checks that each IGM carries the data a power flow needs: see [its PF_UC1](https://github.com/m-mirz/cimoxide/blob/main/scripts/relicapgrid/cimoxide_SV_IOP_2026.md#11-pf_uc1-power-flow-relicapgrid-model).

**Against the participants.** Not a test: counted by hand from the report's Annex 2. Its "Yes" is not defined, and does
not say whether reactive limits were on.

| Tool | Solved | Reported as failed | Not reported |
|---|---|---|---|
| VeraGrid | 17 | 0 | 0 |
| gridoxide, no reactive limits | 17 | 0 | 0 |
| gridoxide, with reactive limits | 13 | 4 | 0 |
| PowerFactory | 12 | 3 | 2 |
| ODMS | 10 | 2 | 5 |
| CIMdesk | 7 | 3 | 7 |

In the report only VeraGrid solves Espheim.

## 2. Boundary, Common and Reference data

### 2.9.1. BDS_UC1: Boundary Data Set Importing

**Test.** `test_relicapgrid_power_flow.py`: `test_the_boundary_changes_nothing`.

**Report.** Load the boundary dataset, merge it with the IGMs, check that matching nodes and terminals connect, with no
duplication and no topology gap at the borders.

**What gridoxide does.** Solves each of the eight IGMs that solve with the six boundary files added.

**Result.** Merging the boundary changes no bus and no voltage.

**Notes.** The checks the report asks for are cimoxide's: see [its BDS_UC1](https://github.com/m-mirz/cimoxide/blob/main/scripts/relicapgrid/cimoxide_SV_IOP_2026.md#291-bds_uc1-boundary-data-set-importing).

## 3. Contingencies

### 3.8.1. CO_UC1: Importing, executing, updating and exporting

**Test.** `test_relicapgrid_contingencies.py`. `pytest python/tests/test_relicapgrid_contingencies.py`.

**Report.** Import the contingencies, resolve them against the network model, execute them, update them, export the
updated CO dataset in compliance with the profile. Execution does not permanently modify the base network state.

**What gridoxide does.** The execution. `PowerFlowModel.with_out_of_service` solves the network without each
contingency's equipment, all `outOfService`, and leaves the base model as it was.

1. For Table 3's equipment, of eight classes, `test_an_outage_removes_its_equipment_and_nothing_else` shows what each
   takes away: a branch only the admittances between its ends, a shunt only its bus's, a load or machine only its
   scheduled power, an HVDC converter or line the link's power at both ends.
2. `test_the_outage_equals_one_written_into_the_ssh` reaches each by another road, `Equipment.inService` set in the
   SSH through cimoxide, with the same voltages.
3. `test_execution_leaves_the_base_network_alone` checks the report's expected outcome.
4. Every contingency of the CO files runs, 325 from 7 files.

**Result.** executed.

| Network | Reactive limits | Solve | Stall | Test |
|---|---|---|---|---|
| Belgovia, Britheim, DC-Espheim-Svedala, Svedala (108) | yes | all | | `test_every_contingency_of_an_igm_solves` |
| Galia (12) | yes | 11 | | as above; `CO_HVDC`'s equipment is another IGM's |
| Belgovia + Svedala, for `Svedala-Belgovia_CO.xml` (4) | yes | all | | `test_the_contingencies_between_two_igms_solve_on_both` |
| Espheim (201) | no, as its base stalls with them | 188 | 13 | `test_espheims_contingencies` |
| Espheim at the 231 kV its SV holds (201) | yes | 172 | 29 | `test_espheims_contingencies_with_the_voltage_its_sv_holds` |
| Espheim at the 231 kV its SV holds (201) | no | 191 | 10 | as above |
| The CGM (325), the reference taking the imbalance | no, as its base stalls with them | 312 | 13, Espheim's | `test_every_contingency_runs_on_the_cgm` |

**Notes.**

- The 13 trip heavily loaded lines of a base already at 0.77 p.u. Correcting Espheim's target removes only three of them,
  so that contradiction does not explain them. Newton not converging does not prove there is no solution: **these are
  not explained.**
- Table 3 names two ties the wrong way round, and its N-2 differs from the data. The files say which equipment goes out,
  and they are what runs: see [cimoxide's CO_UC1](https://github.com/m-mirz/cimoxide/blob/main/scripts/relicapgrid/cimoxide_SV_IOP_2026.md#381-co_uc1-importing-executing-updating-and-exporting),
  which also has the import, update and export.
- Which of these violate a limit is [SAR_UC1](#911-sar_uc1)'s question. Remedial actions are not applied.

## 9. Security Analysis Results (SAR)

### 9.1.1. SAR_UC1

**Test.** `test_relicapgrid_security_analysis.py`; `security_analysis_of_the_cgm` in Rust.

**Report.** Export a SAR: the tool produces security analysis results and exports them; the export is checked against the
profile, and for consistency.

**What gridoxide does.** `gridoxide.security_analysis` solves the CGM's base case and each contingency, without reactive
limits as its base stalls with them, and holds each terminal's current against its limits; `to_sar` writes the violations,
and the base case of each terminal a contingency violates. See [Security Analysis and the SAR](./security_analysis.md).

**Result.** produced, and validated by cimoxide.

| | Results |
|---|---|
| Contingencies solved / not converging | 304 / 21, Espheim's; the profile has no place for these, so they are not in the SAR |
| Base-case violations of PATL | 13 |
| Contingency results | 4,003, on 303 contingencies |
| Of those, on a limit the base case does not violate | 214, on 81 contingencies and 55 terminals |
| Base cases of other terminals, given for comparison | 55, for 4,071 results in the dataset |
| Machines shared past their limits | none, in the base case or any contingency solved |

The largest of the 214 are `68-81` at 192% and `65-68` at 178% after `CO190_Aaa_1_`, `CT11_T401` at 167% after
`CO31_CT112_T401`, and `GA-TR2_1` at 162% after `CO3`.

**How far the numbers can be taken.**

- *The flows.* `test_flows_agree_with_the_published_flows` compares them with SvPowerFlow on the four IGMs that agree with
  their SV: a median of at most 0.5 MVA, at most 12 MVA. On the CGM they are up to 746 MVA off its SV, which is no solution
  of its SSH_2.
- *The state.* The CGM's base is what its SSH_2 gives without reactive limits, as low as 0.75 p.u.
- *The imbalance.* `SSH_2` schedules some 530 to 650 MW more than the CGM takes, and who takes that changes the answer
  (`test_the_answer_depends_on_who_takes_the_imbalance`):

  | `sharing` | Solved / stalled | Base-case violations | `BO-TR2_3`, behind the reference: contingencies, worst |
  |---|---|---|---|
  | `capability`, by `maxOperatingP` (default) | 304 / 21 | 13 | 303, 127% |
  | `reference`, the CGM's one reference | 312 / 13 | 17 | 311, 596% |
  | `equal`, the same for every generator bus | 287 / 38 | 21 | 286, 127% |

  None is a dispatch. `reference` is how the CGM's SV was made, which therefore fits it best, and cannot validate the
  others: there only two machines leave their schedule, and BO-G1, scheduled at 52.8 MW, runs at -481.8 MW
  (`test_the_cgms_state_leaves_the_imbalance_on_one_machine`). `normalPF` is scaled per IGM, so Belgovia's 2 units (700 MW)
  would take the share of Svedala's 38 (16,750 MW) (`test_normal_pf_is_scaled_per_igm`).
- *No outside check.* The supplied SAR's one base case agrees with every choice: a dead-end stub under 2 MW, set by local
  data, so it says little for the solver (`test_the_supplied_sars_base_case_cannot_tell_them_apart`). Its one contingency
  result is on the line that contingency trips, so there is nothing to compare.
- *The limits.* Only PATL. All 578 TATLs are 500 A and none is above its PATL, so they cannot be what they say
  (`test_every_tatl_in_the_data_is_500_a_and_none_above_its_patl`). Remedial actions are not applied.

**Validation.** `test_cimoxide_finds_nothing_on_the_results_or_new_on_the_header`: in the model with the CO files, cimoxide
reports nothing on a result, and nothing on the header the supplied SAR's header does not have. A broken SAR is reported
(`test_a_result_without_its_value_is_reported`).

**Notes.**

- Not a test: triplets 0.2.0, ReliCapGrid's CI tool, was run once on this output in place of the supplied SAR, in
  cimoxide's harness, and reported nothing on its results and nothing new on its header. It would bring ReliCapGrid's whole
  validation into this repository.
- What cimoxide does with the SAR ReliCapGrid supplies is in [its SAR_UC1](https://github.com/m-mirz/cimoxide/blob/main/scripts/relicapgrid/cimoxide_SV_IOP_2026.md#911-sar_uc1).

## Annex 1: Test Use Cases Results

Only the use cases that need a calculation. `pytest -k relicapgrid` runs them.

| Use case | gridoxide 0.4.0 | Test |
|---|---|---|
| **PF_UC1** | solved: 13 of 17 cases with reactive limits, all without | `test_relicapgrid_power_flow.py` |
| **BDS_UC1** | simulate: the boundary changes nothing | `test_the_boundary_changes_nothing` |
| **CO_UC1** | execute: 312 of 325 on the CGM | `test_relicapgrid_contingencies.py` |
| **SAR_UC1** | simulate, export | `test_relicapgrid_security_analysis.py` |

The rest of the report's Annex 1 is [cimoxide's](https://github.com/m-mirz/cimoxide/blob/main/scripts/relicapgrid/cimoxide_SV_IOP_2026.md#annex-1-test-use-cases-results).

## Annex 2: Issues found in the data

| Where | Issue |
|---|---|
| ReliCapGrid data | Espheim's SSH holds a node at 220 kV through three controls; its SV holds it at 231 kV. |
| ReliCapGrid data | Jotunheim's SV holds 38 of its 72 voltage controls off their SSH_2 targets, up to 1.32 times. |
| ReliCapGrid data | The CGM's `SSH_2` schedules some 530 to 650 MW too much; its SV leaves that on one machine, BO-G1 at -481.8 MW. |
| ReliCapGrid data | A DC-Nordheim-Galia converter's `targetPpcc` and `ACDCConverter.p` have opposite signs; Britheim's have one. |
| ReliCapGrid data | Portheim's SV has every node at 700 kV on a 400 kV base (1.75 p.u.). |
| ReliCapGrid data | All 578 TATLs are 500 A and none is above its PATL; 558 are below it. |
| ReliCapGrid data | `GeneratingUnit.normalPF` is scaled per IGM (100 for most, 200 for Espheim, 1 for Britheim), so it cannot share across a CGM. |

What is found in the files themselves, such as the supplied SAR's `value` and its result on a tripped line, or the report's
Table 3 differing from the data, is in [cimoxide's Annex 2](https://github.com/m-mirz/cimoxide/blob/main/scripts/relicapgrid/cimoxide_SV_IOP_2026.md#annex-2-issues-found-in-the-data-the-descriptions-the-apl-and-triplets).

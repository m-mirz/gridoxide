# gridoxide on the 2026 SV-IOP

What gridoxide does for the grid calculations of the [2026 SV-IOP report](https://www.entsoe.eu/Documents/CIM_documents/Market%20CIM/SV_IOP_Report_2026.pdf)
(Network Code Profiles 2.5, ReliCapGrid): the power flows of PF_UC1, the execution of CO_UC1 and producing
the SAR of SAR_UC1. Import, validation and export of the datasets are cimoxide's, in [its own
report](https://github.com/m-mirz/cimoxide/blob/main/scripts/relicapgrid/cimoxide_SV_IOP_2026.md). gridoxide did
not take part in the event: these are its own runs, afterwards, on a later dataset, with the report's
results known.

## Finding the tests

| File (`python/tests/`) | What it holds |
|---|---|
| `test_relicapgrid_power_flow.py` | PF_UC1 |
| `test_relicapgrid_contingencies.py` | CO_UC1's execution |
| `test_relicapgrid_security_analysis.py` | SAR_UC1 |
| `_relicapgrid.py` | what they share: the file groups, reading the data through cimoxide |

Rust: `tests/cgmes_relicapgrid_test.rs`. The data is the submodule `tests/data/relicapgrid`, at the commit
cimoxide's results are for, `f09bb34`. The Python tests read it through cimoxide, independently of
gridoxide's own use of `cimoxide-model`, and validate with it; it is their one test dependency.

```bash
git submodule update --init tests/data/relicapgrid
maturin develop --release --features python,cgmes
pip install 'cimoxide>=0.6' pytest
python -m pytest python/tests -k relicapgrid
```

## Results against the report's Annex 2

The report's Annex 2, with gridoxide beside the tools that took part. Its "Yes" is not defined, and does not say
whether reactive limits were on; here each case is solved with them, and all converge without.

| Tool | Solved | Reported as failed | Not reported |
|---|---|---|---|
| VeraGrid | 17 | 0 | 0 |
| gridoxide, no reactive limits | 17 | 0 | 0 |
| gridoxide, with reactive limits | 13 | 4 | 0 |
| PowerFactory | 12 | 3 | 2 |
| ODMS | 10 | 2 | 5 |
| CIMdesk | 7 | 3 | 7 |

## PF_UC1: Power Flow ReliCapGrid model

**Test.** `test_relicapgrid_power_flow.py`: `test_power_flow`, on `case_files`, `solve` and `compare`, with
each problem's cause in `KNOWN`.

The report runs power flow on each IGM, then on growing combinations, then on the whole CGM. Convergence
alone is not taken as the result. An IGM or the CGM has to agree with its own SvVoltage (within 1% and 0.5°
per component). A combination has no state of its own, each IGM's SV describing that IGM alone, so it has
to hold one angle reference per component and voltages within 0.8 to 1.2 p.u. Each case solves with
reactive limits; the four that stall are also solved without them
(`test_stalled_cases_converge_without_reactive_limits`).

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

Merging the six boundary files with an IGM changes no bus and no voltage
(`test_the_boundary_changes_nothing`, BDS_UC1). Scaling the load in an SSH is noticed
(`test_notices_a_broken_hypothesis`), and files exported by cimoxide solve the same
(`test_exported_files_solve_the_same`).

What is left is in the data:

- **Espheim.** Its SSH has TannrsCk SM and two ExternalNetworkInjections hold a node at 220 kV, where its SV
  has them hold 231 kV, within their reactive limits; the SV is otherwise consistent with the EQ and SSH.
  With the SSH's target the limits cannot all hold, and Newton stalls. With 231 kV it solves with limits, and
  without them to the SV's voltages (`test_espheim_solves_with_the_target_its_sv_holds`). In the report only
  VeraGrid solves Espheim.
- **The CGM.** Jotunheim's SV holds 38 of its 72 voltage controls off their SSH_2 targets, up to 1.32 times
  (ERIKSHAMN_G1: 20.9 kV, 27.7 in the SV), so it is no solution of that SSH_2, and with those targets the
  limits cannot all hold. It converges without them, at 0.77 p.u. at the lowest and up to 24% from that SV.
- **HVDC signs.** A DC-Nordheim-Galia converter's `targetPpcc` is +70 where its `ACDCConverter.p`, and the SV,
  are -70.01; Britheim writes both with one sign. This is the convention the report's §4.1 asks to settle.

## CO_UC1: execution

**Test.** `test_relicapgrid_contingencies.py`. `model.with_out_of_service` applies each.

The report's Table 3 spans eight classes of equipment. For each,
`test_an_outage_removes_its_equipment_and_nothing_else` shows what it takes away: a branch only the admittances
between its ends, a shunt only its bus's, a load or machine only its scheduled power, an HVDC converter or
line the link's power at both ends. `test_the_outage_equals_one_written_into_the_ssh` reaches each by another
road, `Equipment.inService` set in the SSH through cimoxide, with the same voltages, and
`test_execution_leaves_the_base_network_alone` checks the report's expected outcome.

Every contingency of the CO files, 325 of them from 7 files:

| Network | Reactive limits | Solve | Stall | Test |
|---|---|---|---|---|
| Belgovia, Britheim, DC-Espheim-Svedala, Svedala (108) | yes | all | | `test_every_contingency_of_an_igm_solves` |
| Galia (12) | yes | 11 | | as above; `CO_HVDC`'s equipment is another IGM's |
| Belgovia + Svedala, for `Svedala-Belgovia_CO.xml` (4) | yes | all | | `test_the_contingencies_between_two_igms_solve_on_both` |
| Espheim (201) | no, as its base stalls with them | 188 | 13 | `test_espheims_contingencies` |
| Espheim at the 231 kV its SV holds (201) | yes | 172 | 29 | `test_espheims_contingencies_with_the_voltage_its_sv_holds` |
| Espheim at the 231 kV its SV holds (201) | no | 191 | 10 | as above |
| The CGM (325), the reference taking the imbalance | no, as its base stalls with them | 312 | 13, Espheim's | `test_every_contingency_runs_on_the_cgm` |

The 13 trip heavily loaded lines of a base already at 0.77 p.u. Correcting Espheim's target removes only three
of them, so that contradiction does not explain them. Newton not converging does not prove there is no
solution: **these are not explained.**

## SAR_UC1: producing the SAR

**Test.** `test_relicapgrid_security_analysis.py`; `security_analysis_of_the_cgm` in Rust. See
[Security Analysis and the SAR](./security_analysis.md) for what it does.

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

- *The flows.* `test_flows_agree_with_the_published_flows` compares them with SvPowerFlow on the four IGMs that
  agree with their SV: a median of at most 0.5 MVA, at most 12 MVA. On the CGM they are up to 746 MVA off
  its SV, which is no solution of its SSH_2.
- *The state.* The CGM's base is what its SSH_2 gives without reactive limits, as low as 0.75 p.u.
- *The imbalance.* `SSH_2` schedules some 530 to 650 MW more than the CGM takes, and who takes that changes the
  answer (`test_the_answer_depends_on_who_takes_the_imbalance`):

  | `sharing` | Solved / stalled | Base-case violations | `BO-TR2_3`, behind the reference: contingencies, worst |
  |---|---|---|---|
  | `capability`, by `maxOperatingP` (default) | 304 / 21 | 13 | 303, 127% |
  | `reference`, the CGM's one reference | 312 / 13 | 17 | 311, 596% |
  | `equal`, the same for every generator bus | 287 / 38 | 21 | 286, 127% |

  None is a dispatch. `reference` is how the CGM's SV was made, which therefore fits it best, and cannot validate
  the others: there only two machines leave their schedule, and BO-G1, scheduled at 52.8 MW, runs at -481.8 MW
  (`test_the_cgms_state_leaves_the_imbalance_on_one_machine`). `normalPF` is scaled per IGM, so Belgovia's 2 units
  (700 MW) would take the share of Svedala's 38 (16,750 MW) (`test_normal_pf_is_scaled_per_igm`).
- *No outside check.* The supplied SAR's one base case agrees with every choice: a dead-end stub under 2 MW, set
  by local data, so it says little for the solver (`test_the_supplied_sars_base_case_cannot_tell_them_apart`). Its
  one contingency result is on the line that contingency trips, so there is nothing to compare.
- *The limits.* Only PATL. All 578 TATLs are 500 A and none is above its PATL, so they cannot be what they
  say (`test_every_tatl_in_the_data_is_500_a_and_none_above_its_patl`). Remedial actions are not applied.

**Validation.** `test_cimoxide_finds_nothing_on_the_results_or_new_on_the_header`: in the model with the CO files,
cimoxide reports nothing on a result, and nothing on the header the supplied SAR's header does not have. A broken
SAR is reported (`test_a_result_without_its_value_is_reported`). triplets 0.2.0, ReliCapGrid's CI tool, was run
once on this output in place of the supplied SAR, in cimoxide's harness: nothing on its results, nothing new on
its header. It is not a test here, as it would bring ReliCapGrid's whole validation into this repository.

## Issues found in the data

| Where | Issue |
|---|---|
| ReliCapGrid data | Espheim's SSH holds a node at 220 kV through three controls; its SV holds it at 231 kV. |
| ReliCapGrid data | Jotunheim's SV holds 38 of its 72 voltage controls off their SSH_2 targets, up to 1.32 times. |
| ReliCapGrid data | The CGM's `SSH_2` schedules some 530 to 650 MW too much; its SV leaves that on one machine, BO-G1 at -481.8 MW. |
| ReliCapGrid data | A DC-Nordheim-Galia converter's `targetPpcc` and `ACDCConverter.p` have opposite signs; Britheim's have one. |
| ReliCapGrid data | Portheim's SV has every node at 700 kV on a 400 kV base (1.75 p.u.). |
| ReliCapGrid data | All 578 TATLs are 500 A and none is above its PATL; 558 are below it. |
| ReliCapGrid data | `GeneratingUnit.normalPF` is scaled per IGM (100 for most, 200 for Espheim, 1 for Britheim), so it cannot share across a CGM. |

Issues in the files themselves, such as the supplied SAR's `value` and its result on a tripped line, or the report's
Table 3 differing from the data, are in cimoxide's report.

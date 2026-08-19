# Scenarios taken verbatim from powsybl-open-rao's own Cucumber suite.
#
# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# These are the external gate of `plans/RAO_PLAN.md` §8.3: expectations written
# by the reference implementation's authors, against inputs it ships, stating
# margins to the decimal and naming which remedial actions should be used.
# Nothing here was written by gridoxide, which is the entire point — every other
# check in this repository is one gridoxide wrote for itself.
#
# The selection is every scenario in that suite that is `@dc` and `@rao`, uses a
# JSON CRAC and the TestCase12Nodes network, and needs none of loop flows, MNECs,
# relative margins, costly optimization, HVDC, second-preventive or MARMOT — the
# features `src/rao/` does not implement. That is 8 of roughly 500.
#
# Each scenario's `Scenario:` line names the file it came from. Steps are
# unmodified, including the file paths: the harness resolves them by basename
# against `tests/data/rao/features/`, so the text stays exactly as its authors
# wrote it.

Feature: gridoxide against powsybl-open-rao's own expectations

  @fast @rao @dc @preventive-only @max-min-margin
  Scenario: 0.1.1.3.1  [0_import_export/0_1_import/0_1_1_cnec/0_1_1_3_transformers.feature]: Handle transformers on a small test case in DC
    Given network file is "epic15/TestCase12Nodes_with_2_voltage_levels_1.uct" for CORE CC
    Given crac file is "epic15/SL_ep15us3case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 3 remedial actions are used in preventive
    Then the remedial action "open_be1_fr1" is used in preventive
    Then the remedial action "open_be1_be2" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then the worst margin is 79.12 MW
    Then the margin on cnec "BBE2AA2  BBE2AA1  2 - preventive" after PRA should be 79.12 MW
    Then the margin on cnec "BBE2AA2  BBE2AA1  2 - co_fr - outage" after PRA should be 136.19 MW
    Then the margin on cnec "BBE1AA1  BBE1AA2  1 - preventive" after PRA should be 192.82 MW
    Then the margin on cnec "FFR3AA1  FFR3AA2  1 - preventive" after PRA should be 195.34 MW
    Then the margin on cnec "FFR3AA1  FFR3AA2  1 - co_fr - outage" after PRA should be 207.61 MW
    Then the margin on cnec "FFR1AA2  FFR1AA1  5 - preventive" after PRA should be 293.2 MW
    Then the margin on cnec "BBE1AA1  BBE1AA2  1 - co_fr - outage" after PRA should be 296.74 MW
    Then the margin on cnec "FFR1AA2  FFR1AA1  5 - co_fr - outage" after PRA should be 544 MW

  @fast @rao @dc @contingency-scenarios @max-min-margin
  Scenario: 1.2.4.3.1  [1_multi_step_optimisation/1_2_automatons/1_2_4_additional_tests.feature]: Test get highest functional cost worst CNEC is a curative after 1PRAO
    Given network file is "epic15/TestCase12Nodes_15_11_5_3_2.uct"
    Given crac file is "epic15/crac_15_11_5_3_1.json"
    Given configuration file is "epic15/RaoParameters_ep15us11-5-3-3.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -773.0 MW on cnec "be1_be3_co2 - BBE1AA11->BBE3AA11  - co2_de1_de3 - curative"

  @fast @rao @dc @contingency-scenarios @max-min-margin
  Scenario: 1.2.4.3.2  [1_multi_step_optimisation/1_2_automatons/1_2_4_additional_tests.feature]: Test get highest functional cost worst CNEC is auto after 1ARAO
    Given network file is "epic15/TestCase12Nodes_15_11_5_3_2.uct"
    Given crac file is "epic15/crac_15_11_5_3_2.json"
    Given configuration file is "epic15/RaoParameters_ep15us11-5-3-3.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -1945.45 MW on cnec "de2_nl3_co1 - DDE2AA11->NNL3AA11  - co1_de1_de2 - auto"

  @fast @rao @dc @contingency-scenarios @max-min-margin
  Scenario: 1.2.4.3.3  [1_multi_step_optimisation/1_2_automatons/1_2_4_additional_tests.feature]: Test get highest functional cost worst CNEC is curative after 1CRAO
    Given network file is "epic15/TestCase12Nodes_15_11_5_3_2.uct"
    Given crac file is "epic15/crac_15_11_5_3_3.json"
    Given configuration file is "epic15/RaoParameters_ep15us11-5-3-3.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -543.5 MW on cnec "be1_be3_co1 - BBE1AA11->BBE3AA11  - co1_de1_de2 - curative"

  @fast @rao @dc @contingency-scenarios @max-min-margin
  Scenario: 1.2.4.5  [1_multi_step_optimisation/1_2_automatons/1_2_4_additional_tests.feature]: RaoResult AFTER PRA fixed for curative CNECs, without 2P
    Given network file is "epic15/TestCase12Nodes_15_11_5_1.uct"
    Given crac file is "epic15/crac_15_11_5_1.json"
    Given configuration file is "epic15/RaoParameters_ep15us11-5-3-3.json"
    When I launch rao
    Then the remedial action "open_de1_de2_open_nl2_be3 - prev" is not used in preventive
    Then the remedial action "open_de2_nl3 - co1 - auto" is not used after "co1_fr2_de3" at "auto"
    Then the remedial action "close_fr2_de3 - co1 - auto" is used after "co1_fr2_de3" at "auto"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_de3" at "curative"
    Then the margin on cnec "be1_be3_co1 - BBE1AA11->BBE3AA11  - co1_fr2_de3 - curative" after PRA should be -302.38 MW
    Then the margin on cnec "be1_be3_co1 - BBE1AA11->BBE3AA11  - co1_fr2_de3 - auto" after ARA should be -223.44 MW
    Then the margin on cnec "be1_be3_co1 - BBE1AA11->BBE3AA11  - co1_fr2_de3 - curative" after CRA should be 414.58 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"

  @fast @rao @dc @contingency-scenarios @max-min-margin
  Scenario: 1.2.5.1  [1_multi_step_optimisation/1_2_automatons/1_2_5_complex_cases.feature]: Automatons simulated by batches, speed-wise
    Complex case with 5 automatons that have different speeds:
    - FR1-FR2-1 is overloaded so FR1-FR2-3 is closed
    - The previous automaton activation solved the constraint so closing FR1-FR2-4 is not triggered
    - The two PSTs are triggered as their respective monitored lines are overloaded
    - Finally, FR5-FR6-2 is closed as FR5-FR6-1 is closed is still overloaded
    Given network file is "epic15/TestCase8Nodes_15_11_6_1.uct"
    Given crac file is "epic15/crac_15_11_6_1.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 0 remedial actions are used in preventive
    Then 4 remedial actions are used after "co_fr1_fr2_2" at "auto"
    Then the remedial action "close_fr1_fr2_3" is used after "co_fr1_fr2_2" at "auto"
    Then the remedial action "pst_fr3_fr4" is used after "co_fr1_fr2_2" at "auto"
    Then the tap of PstRangeAction "pst_fr3_fr4" should be 2 after "co_fr1_fr2_2" at "auto"
    Then the remedial action "pst_fr7_fr8" is used after "co_fr1_fr2_2" at "auto"
    Then the tap of PstRangeAction "pst_fr7_fr8" should be -3 after "co_fr1_fr2_2" at "auto"
    Then the remedial action "close_fr5_fr6_2" is used after "co_fr1_fr2_2" at "auto"
    Then the worst margin is 1 MW

  @fast @rao @dc @redispatching @preventive-only @max-min-margin
  Scenario: 2.3.1.1.a  [2_remedial_actions/2_3_redispatching/2_3_1_basic.feature]: Extremely basic redispatching on 2 nodes network - maxMargin
  Two nodes containing one generator each, and linked by an overloaded line.
  One generator produces 1000, the other produces -1000.
  The objective is to maximize the min margin => shut down both generators.
  Redispatching action's chosen setpoint = 0: on FRR1AA1 -> setpoint * 1 = 0, on FRR2 -> setpoint * -1 = 0.
    Given network file is "epic93/2Nodes.uct"
    Given crac file is "epic93/crac-93-1-1.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the worst margin is 300 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the initial margin on cnec "cnecFr1Fr2Preventive" should be -700 MW
    Then 1 remedial actions are used in preventive
    Then the remedial action "redispatchingAction" is used in preventive
    Then the setpoint of RangeAction "redispatchingAction" should be 0.0 MW in preventive
    Then the margin on cnec "cnecFr1Fr2Preventive" after PRA should be 300 MW

  @fast @rao @dc @redispatching @preventive-only @max-min-margin
  Scenario: 2.3.1.1.bis  [2_remedial_actions/2_3_redispatching/2_3_1_basic.feature]: Extremely basic redispatching on 2 nodes network - maxMargin - load
  Exact same situation but with loads instead of generators.
    Given network file is "epic93/2Nodes_load.uct"
    Given crac file is "epic93/crac-93-1-1-load.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the worst margin is 300 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the initial margin on cnec "cnecFr1Fr2Preventive" should be -700 MW
    Then 1 remedial actions are used in preventive
    Then the remedial action "redispatchingAction" is used in preventive
    Then the setpoint of RangeAction "redispatchingAction" should be 0.0 MW in preventive
    Then the margin on cnec "cnecFr1Fr2Preventive" after PRA should be 300 MW

  @fast @rao @dc @redispatching @preventive-only @max-min-margin
  Scenario: 2.3.1.3  [2_remedial_actions/2_3_redispatching/2_3_1_basic.feature]: Unbalanced redispatching
  Only one redispatching action available: with a key equal to 1 on FR1 and -0.7 on FR2.
  The sum of the key do not sum up to 1. It's impossible to respect injection balance constraint with a variation != 0.
  The RAO chose to not apply the action.
    Given network file is "epic93/3Nodes.uct"
    Given crac file is "epic93/crac-93-1-3.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the worst margin is -267 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the initial margin on cnec "cnecFr1Fr2Preventive" should be -267 MW
    Then 0 remedial actions are used in preventive
    Then the setpoint of RangeAction "redispatchingAction" should be 1000.0 MW in preventive
    Then the margin on cnec "cnecFr1Fr2Preventive" after PRA should be -267.0 MW

  @fast @rao @dc @redispatching @preventive-only @max-min-margin
  Scenario: 2.3.1.4  [2_remedial_actions/2_3_redispatching/2_3_1_basic.feature]: Multiple redispatching actions keep network balanced - max min margin
  Both redispatching actions have distribution keys that do not sum to 0, the actions have to be activated together to
  compensate each other, and the result is optimal when all generators are shut down.
  Injection balance constraint is satisfied 1000*(1+1)-1000*(1+1)=0
    Given network file is "epic93/4Nodes.uct"
    Given crac file is "epic93/crac-93-1-4.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the worst margin is 500 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "redispatchingActionFR1FR3" is used in preventive
    Then the remedial action "redispatchingActionFR2FR4" is used in preventive
    Then the setpoint of RangeAction "redispatchingActionFR1FR3" should be 0.0 MW in preventive
    Then the setpoint of RangeAction "redispatchingActionFR2FR4" should be 0.0 MW in preventive
    Then the margin on cnec "cnecFr1Fr2Preventive" after PRA should be 500.0 MW

  @fast @rao @dc @preventive-only @secure-flow
  Scenario: 3.2.1.0.b  [3_objective_functions/3_2_max_min_margin/3_2_1_max_min_margin.feature]: use relevant number of decimals for margin and cost logging
  This test is used as a reference for positive margin stop criterion, for comparison with the tests with max margin stop criterion.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic5/SL_ep5us1b.json"
    Given configuration file is "common/RaoParameters_posMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -0.0001 MW with a tolerance of 0.00000001 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -0.0001 MW
    Then 0 remedial actions are used in preventive

  @fast @rao @dc @preventive-only @secure-flow
  Scenario: 5.1.1.1.1  [5_special_features/5_1_loopflow/5_1_1_loopflow_computation/5_1_1_1_basic_computation.feature]: optimise network action without loop flow limitation
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic7/crac_lf_rao_2.json"
    Given configuration file is "common/RaoParameters_posMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 92.0 MW
    Then 1 remedial actions are used in preventive

  @fast @rao @dc @preventive-only @max-min-margin
  Scenario: 5.1.2.3.1  [5_special_features/5_1_loopflow/5_1_2_loopflow_in_rao/5_1_2_3_linear_rao_loopflow_limitation.feature]: linear RAO without LF limitation
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic7/crac_lf_rao_1.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 218.0 MW
    Then the margin on cnec "FFR1AA1  FFR2AA1  1 - preventive" after PRA should be 218.0 MW
    Then the tap of PstRangeAction "PRA_PST_BE" should be -16 in preventive

  @fast @rao @dc @preventive-only @max-min-margin
  Scenario: 5.1.2.4.1  [5_special_features/5_1_loopflow/5_1_2_loopflow_in_rao/5_1_2_4_search_tree_loopflow_limitation.feature]: Simple search tree RAO without LF limitation - MEGAWATT
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic7/crac_lf_rao_3.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -160.0 MW
    Then the worst margin is -160.0 MW on cnec "FFR2AA1  DDE3AA1  1 - preventive"
    Then the tap of PstRangeAction "PRA_PST_BE" should be -16 in preventive
    Then 2 remedial actions are used in preventive
    Then the remedial action "Open FR1 FR2" is used in preventive
    Then the remedial action "PRA_PST_BE" is used in preventive

  @fast @rao @dc @preventive-only @max-min-margin
  Scenario: 5.1.2.4.5  [5_special_features/5_1_loopflow/5_1_2_loopflow_in_rao/5_1_2_4_search_tree_loopflow_limitation.feature]: Complex search tree RAO without LF limitation
    Given network file is "common/TestCase12Nodes2PSTs.uct"
    Given crac file is "epic7/crac_lf_rao_4.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -250.0 MW
    Then the worst margin is -250.0 MW on cnec "FFR1AA1  FFR2AA1  1 - preventive"
    Then the tap of PstRangeAction "PRA_PST_BE" should be -16 in preventive
    Then the tap of PstRangeAction "PRA_PST_DE" should be 0 in preventive
    Then 2 remedial actions are used in preventive
    Then the remedial action "PRA_PST_BE" is used in preventive
    Then the remedial action "Open_BE1_BE3" is used in preventive

  @fast @rao @dc @preventive-only @max-min-margin
  Scenario: 5.2.1.1  [5_special_features/5_2_mnec/5_2_1_linear_rao.feature]: reference run, no MNEC
  The flow on CNEC "NNL2AA1  NNL3AA1  1 - preventive" is increased because the CNEC is not limiting.
    Given network file is "common/TestCase12Nodes.uct" for CORE CC
    Given crac file is "epic11/ls_mnec_linearRao_ref.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the tap of PstRangeAction "PRA_PST_BE" should be -16 in preventive
    Then PST "BBE2AA1  BBE3AA1  1" in network file with PRA is on tap -16
    Then the value of the objective function after CRA should be -224.0
    Then the worst margin is 224.0 MW on cnec "FFR1AA1  FFR2AA1  1 - preventive"
    Then the initial flow on cnec "NNL2AA1  NNL3AA1  1 - preventive" should be 833.3 MW on side 1
    Then the flow on cnec "NNL2AA1  NNL3AA1  1 - preventive" after PRA should be 949.0 MW on side 1

  @fast @rao @dc @preventive-only @max-min-margin
  Scenario: 5.3.2.1.1.1  [5_special_features/5_3_ac_dc_behaviour/5_3_2_dc/5_3_2_1_max_min_margin.feature]: MW thresholds in DC mode and min margin in MW
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic4/SL_ep4us2_4MR_MW.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 22 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - curative" after PRA should be 22 MW
    Then the value of the objective function after CRA should be -22.0
    Then the tap of PstRangeAction "PRA_PST_BE" should be 5 in preventive
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - outage" after PRA should be 22.4 MW
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 24.1 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 44.0 MW

  @fast @rao @dc @preventive-only @max-min-margin
  Scenario: 5.3.2.1.1.2  [5_special_features/5_3_ac_dc_behaviour/5_3_2_dc/5_3_2_1_max_min_margin.feature]: MW thresholds in AC mode and min margin in MW
  Same data as 5.3.2.1.1, but the computation is in AC.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic4/SL_ep4us2_4MR_MW.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 18.0 MW
    Then the value of the objective function after CRA should be -18.0
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - curative" after PRA should be 18 MW
    Then the value of the objective function after CRA should be -18.0
    Then the tap of PstRangeAction "PRA_PST_BE" should be 5 in preventive
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - outage" after PRA should be 22.4 MW
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 24.1 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 44.0 MW

  @fast @rao @dc @preventive-only @max-min-margin
  Scenario: 5.3.2.1.2.1  [5_special_features/5_3_ac_dc_behaviour/5_3_2_dc/5_3_2_1_max_min_margin.feature]: A thresholds in DC mode and min margin in MW
  Same inputs as 5.3.2.1.1.1, but the thresholds are defined in A in the CRAC.
    Given network file is "common/TestCase12Nodes.uct" for CORE CC
    Given crac file is "epic4/SL_ep4us2_4MR_A.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 15.07 MW
    Then the value of the objective function after CRA should be -15.07
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 15.07 MW
    Then the tap of PstRangeAction "PRA_PST_BE" should be 4 in preventive
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - outage" after PRA should be 23.38 MW
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 15.07 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 45.12 MW

  @fast @rao @dc @preventive-only @max-min-margin
  Scenario: 5.3.2.1.3.1  [5_special_features/5_3_ac_dc_behaviour/5_3_2_dc/5_3_2_1_max_min_margin.feature]: mixed thresholds in DC mode and min margin in MW
  Same inputs as 5.3.2.1.1.1, but some thresholds are defined in A in the CRAC (and others in MW).
    Given network file is "common/TestCase12Nodes.uct" for CORE CC
    Given crac file is "epic4/SL_ep4us2_4MR_mixed.json"
    Given configuration file is "common/RaoParameters_maxMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 18.52 MW
    Then the value of the objective function after CRA should be -18.52
    Then the tap of PstRangeAction "PRA_PST_BE" should be 4 in preventive
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 18.52 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - outage" after PRA should be 23.38 MW

  @fast @rao @dc @preventive-only @secure-flow
  Scenario: 5.3.2.2.1.2  [5_special_features/5_3_ac_dc_behaviour/5_3_2_dc/5_3_2_2_secure_flow.feature]: secure with DC config
  Same as 5.3.2.2.1 but computation in DC
    Given network file is "common/TestCase12Nodes.uct" for CORE CC
    Given crac file is "epic4/SL_ep4us3.json"
    Given configuration file is "common/RaoParameters_posMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 32 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - curative" after PRA should be 32 MW
    Then the remedial action "PRA_PST_BE" is used in preventive
    Then the tap of PstRangeAction "PRA_PST_BE" should be 4 in preventive

  @fast @rao @dc @preventive-only @secure-flow
  Scenario: 5.3.2.2.2.2  [5_special_features/5_3_ac_dc_behaviour/5_3_2_dc/5_3_2_2_secure_flow.feature]: no failure with DC config
  Same case as 4.3.2.1 but in DC.
    Given network file is "epic4/US4-3-TestCase12Nodes-diverging.uct"
    Given crac file is "epic4/SL_ep4us3.json"
    Given configuration file is "common/RaoParameters_posMargin_megawatt_dc.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 32.0 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - curative" after PRA should be 32.0 MW
    Then the remedial action "PRA_PST_BE" is used in preventive
    Then the tap of PstRangeAction "PRA_PST_BE" should be 4 in preventive


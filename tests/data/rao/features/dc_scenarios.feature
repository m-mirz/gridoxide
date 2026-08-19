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


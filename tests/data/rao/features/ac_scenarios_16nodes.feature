# Scenarios taken verbatim from powsybl-open-rao's own Cucumber suite.
#
# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# The AC half of the external gate, on `common/TestCase16Nodes.uct` — 16 buses,
# 27 branches, four countries, and the network the largest single block of the
# reference's scenarios is written against. `ac_scenarios.feature` is the same
# selection on the twelve-node case.
#
# The selection is every scenario in that suite that is `@ac` and `@rao`, uses a
# JSON CRAC and this network, and needs none of loop flows, relative margins,
# costly optimization, HVDC, second-preventive or MARMOT — the features
# `src/rao/` does not implement. That is 93.
#
# Each scenario's `Scenario:` line names the file it came from. Steps are
# unmodified, including the file paths: the harness resolves them by basename
# against `tests/data/rao/features/`, so the text stays exactly as its authors
# wrote it.


Feature: gridoxide against powsybl-open-rao's sixteen-node AC expectations

  @fast @rao @ac @contingency-scenarios @secure-flow
  Scenario: 1.2.1.1: onConstraint automaton not applied
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic15/SL_ep15us11-2case1.json"
    Given configuration file is "epic15/RaoParameters_ep15us11-2.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 0 remedial actions are used in preventive
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 852.4 A
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 3184.3 A
    Then 0 remedial actions are used after "co2_be1_be3" at "auto"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" after ARA should be 92.4 A
    Then 0 remedial actions are used after "co2_be1_be3" at "curative"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 92.4 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - curative" after CRA should be 858.8 A
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 878.0 A

  @fast @rao @ac @contingency-scenarios @secure-flow
  Scenario: 1.2.1.2: onConstraint automaton applied
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic15/SL_ep15us11-2case2.json"
    Given configuration file is "epic15/RaoParameters_ep15us11-2.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 0 remedial actions are used in preventive
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 852.4 A
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 3184.3 A
    Then 1 remedial actions are used after "co2_be1_be3" at "auto"
    Then the remedial action "open_be1_be4" is used after "co2_be1_be3" at "auto"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" after ARA should be 16.4 A
    Then 1 remedial actions are used after "co2_be1_be3" at "curative"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 448.8 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - curative" after CRA should be 970.5 A
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 878.0 A

  @fast @rao @ac @contingency-scenarios @secure-flow
  Scenario: 1.2.1.3: OnContingencyState automaton applied
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic15/SL_ep15us11-2case3.json"
    Given configuration file is "epic15/RaoParameters_ep15us11-2.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 0 remedial actions are used in preventive
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 852.4 A
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 3184.3 A
    Then 1 remedial actions are used after "co2_be1_be3" at "auto"
    Then the remedial action "open_be1_be4" is used after "co2_be1_be3" at "auto"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" after ARA should be 16.4 A
    Then 1 remedial actions are used after "co2_be1_be3" at "curative"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 448.8 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - curative" after CRA should be 970.5 A
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 878.0 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.2.2.2: 1 auto PST
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic15/SL_ep15us11-3case2.json"
    Given configuration file is "epic13/RaoParameters_maxMargin_ampere_absolute_threshold.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_be" should be 0 in preventive
    Then the initial margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" should be -107.6 A
    Then the initial margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - auto" should be -41.2 A
    Then 1 remedial actions are used after "co2_be1_be3" at "auto"
    Then the remedial action "pst_be" is used after "co2_be1_be3" at "auto"
    Then the tap of PstRangeAction "pst_be" should be -8 after "co2_be1_be3" at "auto"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" after ARA should be 98.9 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - auto" after ARA should be 0.2 A
    Then the worst margin is -22 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.2.2.3: 2 auto range actions
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic15/SL_ep15us11-3case3.json"
    Given configuration file is "epic13/RaoParameters_maxMargin_ampere_absolute_threshold.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 5 in preventive
    Then the tap of PstRangeAction "pst_be" should be 0 in preventive
    Then the initial margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" should be -107.6 A
    Then the initial margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - auto" should be -41.2 A
    Then 2 remedial actions are used after "co2_be1_be3" at "auto"
    Then the remedial action "pst_fr" is used after "co2_be1_be3" at "auto"
    Then the remedial action "pst_be" is used after "co2_be1_be3" at "auto"
    Then the tap of PstRangeAction "pst_fr" should be 15 after "co2_be1_be3" at "auto"
    Then the tap of PstRangeAction "pst_be" should be -3 after "co2_be1_be3" at "auto"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" after ARA should be 9.0 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - auto" after ARA should be 146.5 A
    Then the worst margin is -22 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.2.2.4: auto range actions and topological range actions
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic15/SL_ep15us11-3case4.json"
    Given configuration file is "epic13/RaoParameters_maxMargin_ampere_absolute_threshold.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 5 in preventive
    Then the tap of PstRangeAction "pst_be" should be 0 in preventive
    Then the initial margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" should be -107.6 A
    Then the initial margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - auto" should be -41.2 A
    Then 3 remedial actions are used after "co2_be1_be3" at "auto"
    Then the remedial action "open_be1_be4" is used after "co2_be1_be3" at "auto"
    Then the remedial action "pst_fr" is used after "co2_be1_be3" at "auto"
    Then the remedial action "pst_be" is used after "co2_be1_be3" at "auto"
    Then the tap of PstRangeAction "pst_fr" should be 10 after "co2_be1_be3" at "auto"
    Then the tap of PstRangeAction "pst_be" should be -1 after "co2_be1_be3" at "auto"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" after ARA should be 15.3 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - auto" after ARA should be 65.7 A
    Then the worst margin is -22 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.2.2.5: Verify post-ARAO setpoint for automatic+curative range action
    # copy of test case Scenario: 1.2.2.2: 1 auto PST
    # except that pst_be is also preventive and curative
    # it should be used in preventive at tap position -2
    # in auto at -8
    # in curative at -16
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic15/SL_ep15us11-3case2_withPstCra.json"
    Given configuration file is "epic13/RaoParameters_maxMargin_ampere_absolute_threshold.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_be_pra" should be -2 in preventive
    Then the tap of PstRangeAction "pst_be_ara" should be -2 in preventive
    Then the tap of PstRangeAction "pst_be_cra" should be -2 in preventive
    Then the initial margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" should be -107.6 A
    Then the initial margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - auto" should be -41.2 A
    Then 1 remedial actions are used after "co2_be1_be3" at "auto"
    Then the remedial action "pst_be_ara" is used after "co2_be1_be3" at "auto"
    Then the tap of PstRangeAction "pst_be_pra" should be -8 after "co2_be1_be3" at "auto"
    Then the tap of PstRangeAction "pst_be_ara" should be -8 after "co2_be1_be3" at "auto"
    Then the tap of PstRangeAction "pst_be_cra" should be -8 after "co2_be1_be3" at "auto"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - auto" after ARA should be 98.9 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - auto" after ARA should be 0.2 A
    Then 1 remedial actions are used after "co2_be1_be3" at "auto"
    Then the remedial action "pst_be_cra" is used after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be_pra" should be -16 after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be_cra" should be -16 after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be_ara" should be -16 after "co2_be1_be3" at "curative"
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - curative" after CRA should be -58 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 305 A
    Then the worst margin is -58 A

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 1.3.2.1: Simple case with preventive remedial actions only
    If left alone, it would decide to use open_fr1_fr2, close_de3_de4, pst_fr, and pst_be to generate a minimum margin of 988 A
    If we impose the usage of close_fr1_fr5, it would then only add pst_be and generate a minimum margin of 999 A
    #
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us2case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "close_fr1_fr5" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 5 in preventive
    Then the tap of PstRangeAction "pst_be" should be -12 in preventive
    Then the worst margin is 999 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 1064 A
    Then the value of the objective function after CRA should be -999

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.2.2: Simple case with curative remedial actions only
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us6basecase.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -12 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 1000 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 1000 A
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 1301 A
    Then the value of the objective function after CRA should be -1000

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.2.3: Simple case with a mix of preventive and curative remedial actions
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us2case3.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    # In preventive exactly the same results as OSIRIS
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be -5 in preventive

    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 15 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 992 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 992 A
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 1495 A
    Then the value of the objective function after CRA should be -992

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 1.3.2.4: Complex case with preventive remedial actions only
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us2case4.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 4 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be -5 in preventive
    Then the tap of PstRangeAction "pst_be" should be -15 in preventive
    Then the worst margin is 302 A on cnec "BBE4AA1  FFR5AA1  1 - preventive"
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - preventive" after PRA should be 302 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 311 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 319 A
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 376 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 437 A
    Then the value of the objective function after CRA should be -302

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.2.5: Complex case with curative remedial actions only
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us2case5.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 3 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be -5 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -196 A on cnec "BBE2AA1  FFR3AA1  1 - preventive"
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be -196 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 316 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 321 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 441 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 501 A
    Then the value of the objective function after CRA should be 196

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.2.6: Complex case with a mix of preventive and curative remedial actions (1/3)
    Given network file is "common/TestCase16Nodes.uct" for CORE CC
    Given crac file is "epic13/SL_ep13us2case6.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 3 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be -15 in preventive
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be -5 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -244 A on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - preventive" after PRA should be 300 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 308 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -244 A
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 366 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 417 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 469 A
    Then the value of the objective function after CRA should be 244

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.2.7: Complex case with a mix of preventive and curative remedial actions (2/3)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us2case6.json"
    Given configuration file is "epic13/RaoParameters_maxMargin_ampere_absolute_threshold.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be -15 in preventive
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be -5 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 298 A on cnec "BBE4AA1  FFR5AA1  1 - preventive"
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - preventive" after PRA should be 298 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 304 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 405 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 319 A
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 376 A
    Then the value of the objective function after CRA should be -298

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.2.8: Complex case with a mix of preventive and curative remedial actions (3/3)
    Given network file is "common/TestCase16Nodes.uct" for CORE CC
    Given crac file is "epic13/SL_ep13us2case7.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 3 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the remedial action "close_fr1_fr5" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be -5 in preventive
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"

    Then the tap of PstRangeAction "pst_be" should be 16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 433 A on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 597 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 601 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 643 A

    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 433 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 440 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 571 A
    Then the value of the objective function after CRA should be -433

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.3.1: Simple case with preventive, outage and curative states
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 3 remedial actions are used in preventive
    Then the remedial action "close_de3_de4" is used in preventive
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 0 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 556 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 556 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 865 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 914 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 921 A
    Then the value of the objective function after CRA should be -556

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.3.2: Simple case, with 2 curative states
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case2.json"
    Given configuration file is "epic13/RaoParameters_maxMargin_ampere_absolute_threshold.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 9 after "co1_fr2_fr3_1" at "curative"
    Then 3 remedial actions are used after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co2_be1_be3" at "curative"
    Then the remedial action "open_be1_be4" is used after "co2_be1_be3" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co2_be1_be3" at "curative"
    Then the worst margin is 766 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 766 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 992 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 1124 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 1198 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - curative" after CRA should be 1281 A
    Then the value of the objective function after CRA should be -766

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.3.3: Simple case, with 2 curative states and on-contingency remedial actions
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case3.json"
    Given configuration file is "epic13/RaoParameters_maxMargin_ampere_absolute_threshold.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co1_fr2_fr3_1" at "curative"
    Then 2 remedial actions are used after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co2_be1_be3" at "curative"
    Then the remedial action "open_be1_be4" is used after "co2_be1_be3" at "curative"
    Then the worst margin is 753 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 753 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 865 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 1124 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - curative" after CRA should be 1229 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 1279 A
    Then the value of the objective function after CRA should be -753

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.3.4: Complex case, with several outage/curative states, and on-contingency remedial actions
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case4.json"
    Given configuration file is "epic13/RaoParameters_maxMargin_ampere_absolute_threshold.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then 1 remedial actions are used after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co2_be1_be3" at "curative"
    Then the worst margin is -184 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - outage" after PRA should be -184 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be -141 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - outage" after PRA should be 332 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 544 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 782 A
    Then the value of the objective function after CRA should be 184

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.3.5: Simple case, with two curative states, including one without CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case5.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 4 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 in preventive
    Then the remedial action "close_fr1_fr5" is used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then 3 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_de3_de4" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 469 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 469 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 875 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 1004 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - curative" after CRA should be 1031 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 1049 A
    Then the value of the objective function after CRA should be -469

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.3.6: Simple case, with two outage + curative states, including one without CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case6.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 in preventive
    Then 2 remedial actions are used after "co2_be1_be3" at "curative"
    Then the remedial action "open_be1_be4" is used after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co2_be1_be3" at "curative"
    Then the worst margin is -484 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be -484 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - outage" after PRA should be -184 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - outage" after PRA should be 232 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 525 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 649 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 549 A
    Then the value of the objective function after CRA should be 484

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.3.7: Simple case, with one outage and one curative state, but on two different contingencies
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case7.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "close_fr1_fr5" is used in preventive
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 15 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -115 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be -115 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 306 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - outage" after PRA should be 332 A
    Then the value of the objective function after CRA should be 115

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.3.8: Complex case, with several curative / outage states, and some curative states without CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case8.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then 1 remedial actions are used after "co3_fr1_fr3" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co3_fr1_fr3" at "curative"
    Then the worst margin is 418 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - outage" after PRA should be 418 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be 485 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 544 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - outage" after PRA should be 744 A
    Then the margin on cnec "FFR1AA1  FFR2AA1  1 - co3_fr1_fr3 - outage" after PRA should be 857 A
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 886 A
    Then the value of the objective function after CRA should be -418

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.3.9: Test case with no RA in the preventive perimeter
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case9.json"
    Given configuration file is "epic13/RaoParameters_maxMargin_ampere_absolute_threshold_12.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -8 after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 677 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 677 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 678 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 852 A
    Then the value of the objective function after CRA should be -677

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.4.1: Topological RA already applied in initial network : not available for optimization
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us4case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -522 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -522 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -184 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 145 A
    Then the value of the objective function after CRA should be 522

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.4.2: Topological RA available in preventive and curative : used in preventive
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us4case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "close_de3_de4" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -467 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -467 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -54 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 297 A
    Then the value of the objective function after CRA should be 467

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.4.3: Topological RA available in preventive and curative : used in curative
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us4case3.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 5 in preventive
    Then the tap of PstRangeAction "pst_be" should be 0 in preventive
    Then 3 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 15 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -315 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be -315 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -184 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 145 A
    Then the value of the objective function after CRA should be 315

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.4.4: Topological RA duplicated into PRA and CRA : PRA is activated
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us4case4.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "close_de3_de4_pra" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -467 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -467 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -54 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 297 A
    Then the value of the objective function after CRA should be 467

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.4.5: Topological RA duplicated into PRA and CRA : CRA is activated
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us4case5.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 5 in preventive
    Then the tap of PstRangeAction "pst_be" should be 0 in preventive
    Then 3 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5_cra" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 15 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -315 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be -315 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -184 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 145 A
    Then the value of the objective function after CRA should be 315

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.4.6: Topological RA with inverted CRA : PRA is not used, so the CRA is not available
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us4case6.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -184 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -184 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -55 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 145 A
    Then the value of the objective function after CRA should be 184

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.4.7: Topological RA with inverted CRA : line opened in preventive and closed in curative
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us4case7.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 3 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr2_pra" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be -5 in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr2_cra" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be -5 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 200 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 200 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 267 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 613 A
    Then the value of the objective function after CRA should be -200

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.5.1: Preventive and curative PST RA, with same taps in both states
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us5case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -582 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be -582 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -87 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 236 A
    Then the tap of PstRangeAction "pst_fr" should be 5 in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the value of the objective function after CRA should be 582

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.5.2: Preventive and curative PST RA, with different taps in each state
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us5case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 71 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be 71 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 370 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 777 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 786 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 1196 A
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be -16 in preventive
    Then the tap of PstRangeAction "pst_fr" should be 16 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 0 in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the value of the objective function after CRA should be -71

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.5.3: Preventive and curative PST RA, with an activation in the curative state limited by a RELATIVE_DYNAMIC range
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us5case3.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -99 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be -99 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 104 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 525 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 813 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 1183 A
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be -5 in preventive
    Then the tap of PstRangeAction "pst_fr" should be 5 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 0 in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the value of the objective function after CRA should be 99

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.5.4: Duplicated RA on the same PST, one being a PRA and the other one being a CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us5case4.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -47 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be -47 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 149 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 575 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 774 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 1179 A
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr_pra" should be -7 in preventive
    Then the tap of PstRangeAction "pst_fr_cra" should be 1 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 0 in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the value of the objective function after CRA should be 47

  @fast @rao @ac @contingency-scenarios @mnec @max-min-margin
  Scenario: 1.3.6.1: Simple case with a mix of preventive and curative remedial actions and a MNEC in preventive limited by threshold
    Given network file is "common/TestCase16Nodes.uct" for CORE CC
    Given crac file is "epic13/SL_ep13us2case5_with_mnec.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    # Without MNEC pst_fr is set to -5
    Then the tap of PstRangeAction "pst_fr" should be 2 in preventive
    # Margin of the limiting CNEC is slightly lower than in the original test case without MNEC
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 1483 A
    Then the initial flow on cnec "FFR1AA1  FFR2AA1  1 - preventive" should be 430 MW on side 1
    Then the initial margin on cnec "FFR1AA1  FFR2AA1  1 - preventive" should be 70 MW
    Then the margin on cnec "FFR1AA1  FFR2AA1  1 - preventive" after PRA should be 5 MW
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 14 after "co1_fr2_fr3_1" at "curative"
    Then the value of the objective function after CRA should be -999

  @fast @rao @ac @contingency-scenarios @mnec @max-min-margin
  Scenario: 1.3.6.5: Simple case with a mix of preventive and curative remedial actions and a MNEC in preventive limited by threshold
    Given network file is "common/TestCase16Nodes.uct" for CORE CC
    Given crac file is "epic13/SL_ep13us2case5_with_mnec.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    # Without MNEC pst_fr is set to -5
    Then the tap of PstRangeAction "pst_fr" should be 2 in preventive
    # Margin of the limiting CNEC is slightly lower than in the original test case without MNEC
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 1483 A
    Then the initial margin on cnec "FFR1AA1  FFR2AA1  1 - preventive" should be 70 MW
    Then the margin on cnec "FFR1AA1  FFR2AA1  1 - preventive" after PRA should be 5 MW
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 14 after "co1_fr2_fr3_1" at "curative"
    Then the value of the objective function after CRA should be -999

  @fast @rao @ac @contingency-scenarios @mnec @max-min-margin
  Scenario: 1.3.6.6: Simple case with a mix of preventive and curative remedial actions and MNECs in preventive and curative limited by threshold
    Given network file is "common/TestCase16Nodes.uct" for CORE CC
    Given crac file is "epic13/SL_ep13us2case6_with_mnec_curative.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 2 in preventive
    Then the initial margin on cnec "FFR1AA1  FFR2AA1  1 - preventive" should be 70 MW
    Then the margin on cnec "FFR1AA1  FFR2AA1  1 - preventive" after PRA should be 5 MW
    # Flow is -572 MW without RA, and threshold -700 MW.
    Then the initial margin on cnec "BBE1AA1  BBE2AA1  1 - co1_fr2_fr3_1 - curative" should be 127 MW
    # Here the margin should not be negative because the branch is a MNEC and initial margin was positive.
    # Flow is -643 MW with PRA and CRA (actually no CRA were activated in this test case), and threshold -700 MW. Margin is positive.
    Then the margin on cnec "BBE1AA1  BBE2AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 57 MW
    # 2 Remedial actions would have been used if the MNEC was not limiting
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 612 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the value of the objective function after CRA should be -612

  @fast @rao @ac @contingency-scenarios @mnec @max-min-margin
  Scenario: 1.3.6.7: Simple case with a mix of preventive and curative remedial actions and MNECs in preventive and curative limited by initial value
    Given network file is "common/TestCase16Nodes.uct" for CORE CC
    Given crac file is "epic13/SL_ep13us2case7_with_mnec_curative_initial.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_mip.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 2 in preventive
    Then the initial margin on cnec "FFR1AA1  FFR2AA1  1 - preventive" should be 70 MW
    Then the margin on cnec "FFR1AA1  FFR2AA1  1 - preventive" after PRA should be 5 MW
    # Flow is -572 MW without RA, and threshold -500 MW.
    Then the initial margin on cnec "BBE1AA1  BBE2AA1  1 - co1_fr2_fr3_1 - curative" should be -72 MW
    # Here the margin should not be below -122 MW because the initial margin is -72 MW (taking acceptable diminution parameter into account).
    # Flow is -643 MW with PRA and CRA, and threshold -700 MW. Margin is positive.
    Then the margin on cnec "BBE1AA1  BBE2AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -105 MW
    # Curative RA need to be used in order to respect the MNEC constraint (the MNEC is violated by 30 MW in the root leaf of the curative perimeter)
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -2 after "co1_fr2_fr3_1" at "curative"
    # The min margin is lower than in the previous case as the MNEC threshold has been tightened
    Then the worst margin is 705 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the value of the objective function after CRA should be -705

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.8.2: Full optimization in absolute margin with positive margin in curative
    # Curative limiting element of previous case has been removed so that limiting element in curative has a positive
    # absolute margin. This case is a reference for the following one.
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us8case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 3 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be -15 in preventive
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 12 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 300 A on cnec "BBE4AA1  FFR5AA1  1 - preventive"
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - preventive" after PRA should be 300 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 308 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 408 A
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 411 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 470 A
    Then the value of the objective function after CRA should be -300

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.9.1: Skip curative RAO
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us11case1.json"
    Given configuration file is "epic13/RaoParameters_stop_curative_at_preventive.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 301 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 680 A
    Then the value of the objective function after CRA should be -301

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.9.2: Stop curative RAO after root leaf optimization
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us11case1.json"
    Given configuration file is "epic13/RaoParameters_best_preventive_by_500.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 15 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 16 after "co1_fr2_fr3_1" at "curative"
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 301 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 971 A
    Then the value of the objective function after CRA should be -301

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.9.3: Stop curative RAO after reaching set difference with preventive
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us11case3.json"
    Given configuration file is "epic13/RaoParameters_best_preventive_by_628.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then 2 remedial actions are used after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co2_be1_be3" at "curative"
    Then the remedial action "open_be1_be4" is used after "co2_be1_be3" at "curative"
    Then the worst margin is 124 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 124 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 752.5 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 854 A
    Then the value of the objective function after CRA should be -124

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 1.3.9.4: Stop curative RAO after making perimeters secure
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us11case4.json"
    Given configuration file is "epic13/RaoParameters_best_preventive_by_300_secure.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 15 in preventive
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 8 after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co1_fr2_fr3_1" at "curative"
    Then 1 remedial actions are used after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co2_be1_be3" at "curative"
    Then the worst margin is -376.0 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be -376 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 2 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 115.0 A
    Then the value of the objective function after CRA should be 376

  @fast @rao @ac @max-min-margin
  Scenario: 1.4.4.1: Cost has not increased during RAO, do not fall back to initial solution (copy of 1.4.1.1.1)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/second_preventive_ls_1.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_forbid_cost_increase.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -144 A
    Then the value of the objective function after CRA should be 144

  @fast @rao @ac @max-min-margin
  Scenario: 1.4.4.2: Cost has increased during RAO, fall back to initial solution (copy of 1.4.1.1.2)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/SL_ep20us5case2.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_forbid_cost_increase.json"
    When I launch rao
    Then the execution details should be "First preventive fell back to initial situation"
    Then its security status should be "SECURED"
    Then 0 remedial actions are used in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 0 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 113 A
    Then the value of the objective function after CRA should be -113

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.2.1.5.1: One PST and no topo (copy of 2.6.2.3 with MIP for PSTs)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us10case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_mip.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 945 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 1301 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.2.1.5.2: No PST and one topo (copy of 2.6.2.4 with MIP for PSTs)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us10case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_mip.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 840 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.2.1.5.3: One PST and one topo, one CRA, chose PST (copy of 2.6.2.5 with MIP for PSTs)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us10case3.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_mip.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 945 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 1301 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.2.1.5.4: Two allowed CRAs (copy of 2.6.3.2 with MIP for PSTs)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us10case4.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_mip.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -12 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 5 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 1000 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.2.1.5.5: One allowed CRA (copy of 2.6.3.3 with MIP for PSTs)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case3.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_mip.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 945 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.2.1.5.6: One allowed CRA, BE PST not allowed (copy of 2.6.3.4 with MIP for PSTs)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us10case6.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_mip.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 0 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 5 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 840 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.2.1.5.7: One allowed TSO - BE PST not allowed
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us10case9.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_mip.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 3 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 2 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 998 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 2.4.1.1: Preventive onConstraint RA with a constraint on the base network
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us3case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the worst margin is -135 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be -135 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be -134 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be 48 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 308 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be 492 A

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 2.4.1.2: Preventive onConstraint RAs with a constraint triggered by another preventive RA, no reevaluation
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us3case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "close_fr1_fr5" is used in preventive
    Then the worst margin is -45 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be -45 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be -41 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be 81 A

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 2.4.1.3: Preventive onConstraint RAs with no constraint triggered
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us3case3.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "close_fr1_fr5" is used in preventive
    Then the worst margin is -45 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be -45 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be -41 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be 81 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.1.4: Curative onConstraint RA with a constraint right after applying the contingency
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us3case4.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 2 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -37 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -37 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -13 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.1.5: Curative onConstraint RA with a constraint triggered by another curative RA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us3case5.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 2 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -37 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -37 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -13 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.1.6: Curative onConstraint RA with no constraint triggered
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us3case6.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 0 remedial actions are used in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 43 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 43 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 80 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.1.7: Preventive and curative onConstraint RA with a constraint triggered on the base network
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us3case7.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "pst_be" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be 16 in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 82 A
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 82 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 245 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 345 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.1.8: Preventive and curative onConstraint RA with a constraint triggered after applying the contingency
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us3case8.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "pst_be" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be 16 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -1 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 63 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 63 A
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 82 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 71 A

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 2.4.2.1: Preventive onConstraint RA with constraint on base network
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us5case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the worst margin is -135 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be -135 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be -134 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be 48 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 308 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be 492 A

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 2.4.2.2: Preventive onConstraint RA with constraint triggered by another preventive RA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us5case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the remedial action "close_fr1_fr5" is used in preventive
    Then the worst margin is 45 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 45 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be 158 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be 264 A

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 2.4.2.3: Preventive onConstraint RA with no constraint triggered
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us5case3.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "close_fr1_fr5" is used in preventive
    Then the worst margin is -45 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be -45 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be -41 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after PRA should be 81 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.2.4: Curative onConstraint RA with constraint after contingency
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us5case4.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 2 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -37 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -37 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -13 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.2.5: Curative onConstraint RA with constraint triggered by another curative RA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us5case5.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 2 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -37 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -37 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -13 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.2.6: Curative onConstraint RA with no constraint triggered
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us5case6.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 0 remedial actions are used in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 43 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 43 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 80 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.2.7: Preventive and curative onConstraint RA with constraint triggered on base network
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us5case7.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "pst_be" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be 16 in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 82 A
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 82 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 245 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 345 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.2.8: Preventive and curative onConstraint RA with constraint triggered after contingency
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic16/SL_ep16us5case8.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "pst_be" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be 16 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -1 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 63 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 63 A
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 82 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 72 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.3.1: Flow constraint in country with no contingency
    # This is a copy of test case 2.4.2.7
    # pst_be is available after a flow constraint in BE, no contingency defined
    # So the same results as 2.4.2.7 are expected
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "extra_features/Crac_UR_1_1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "pst_be" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be 16 in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 82 A
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 82 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 245 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 345 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.4.3.2: Flow constraint in country only after a given contingency
    # This is a copy of previous case but pst_be is available after a flow constraint in BE, only after contingency co1_fr2_fr3_1
    # Since only the preventive CNEC is constrained initially, the PST shall not be available
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "extra_features/Crac_UR_1_2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 0 remedial actions are used in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -49 A
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be -49 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 693 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 79 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.1.1: Check that the maximum number of network actions per TSO is ignored in preventive 1
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us2case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 3 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be -15 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be -15 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -500 A on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - preventive" after PRA should be 300 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 308 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -500 A
    Then the margin on cnec "BBE4AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 326 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 334 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 371 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.1.2: Check that the maximum number of network actions per TSO is respected in curative - reference run
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us2case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be -8 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 254 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 254 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 450 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.1.3: Check that the maximum number of network actions per TSO is respected in curative
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us2case3.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be -14 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 254 A on cnec "BBE2AA1  FFR3AA1  1 - preventive"
    Then the margin on cnec "FFR1AA1  FFR3AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 437 A

  @fast @rao @ac @contingency-scenarios @search-tree-rao @max-min-margin
  Scenario: 2.6.1.4: Simple case, with 2 curative states
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us2case4.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co2_be1_be3" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co2_be1_be3" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co2_be1_be3" at "curative"
    Then the worst margin is 510 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - co2_be1_be3 - curative" after CRA should be 510 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co2_be1_be3 - curative" after CRA should be 904 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.1.5: Check that the maximum number of network actions per TSO is ignored in preventive 2
    # Copy of 1.3.2.6 test but with a configuration limiting curative topo per TSO
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us2case5.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.1.6: Check country filtering is well done in curative
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us2case6.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    # Without limitation it should use 3 french topological actions as it is limited at 2, il will choose belgian topo instead
    Then 3 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr4" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_be1_be4" is used after "co1_fr2_fr3_1" at "curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.2.1: One PST and one topo, two CRAs
    # <!> All RAs are hypothetically operated by "be"
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us3case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -12 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 1000 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 1301 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.2.2: Two PSTs and no topo
    # <!> All RAs are hypothetically operated by "be"
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us3case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_fr" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 15 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 972 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "BBE1AA1  FFR5AA1  1 - preventive" after PRA should be 1301 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.2.3: No PST and one topo
    # <!> All RAs are hypothetically operated by "be"
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us10case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 840 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.2.4: Two operators, no PST and one topo per operator
    # <!> All RAs are hypothetically operated by "be", except for "open_be1_be4" operated by "fr"
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us3case8.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_be1_be4" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 876 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @preventive-only @search-tree-rao @max-min-margin
  Scenario: 2.6.2.5: Test that the parameters are ignored in preventive
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us3case9.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 3 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be -15 in preventive
    Then the worst margin is 300.37 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.2.6: One PST, limiting element changes
    # <!> All RAs are hypothetically operated by "be"
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us3case10.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 3 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 11 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 5 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 399 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"
    Then the margin on cnec "FFR1AA1  FFR2AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 900 A

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.1: Three allowed CRAs
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -12 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 999.5 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.2: Two allowed CRAs
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case2.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_be" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -12 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 5 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 1000 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.3: One allowed CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case3.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 945 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.4: One allowed CRA, BE PST not allowed
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case4.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be 0 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 5 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 840 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.5: No allowed CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case5.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 679 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.6: Three topological CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case6.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 3 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_be1_be4" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 987 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.7: Two topological CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case7.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr2" is used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 973 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.8: One topological CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case8.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 839 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.9: One topological CRA, best FR topo not allowed
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case9.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 814 A on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative"

  @fast @rao @ac @contingency-scenarios @max-min-margin
  Scenario: 2.6.3.10: Test that the parameter is ignored in preventive
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic19/SL_ep19us5case10.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 3 remedial actions are used in preventive
    Then the remedial action "open_be1_be4" is used in preventive
    Then the remedial action "open_fr1_fr2" is used in preventive
    Then the tap of PstRangeAction "pst_be" should be -15 in preventive


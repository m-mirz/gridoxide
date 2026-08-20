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
# The selection is every scenario in that suite that is `@ac` and `@rao`, uses a
# JSON CRAC and the TestCase12Nodes network, and needs none of loop flows, MNECs,
# relative margins, costly optimization, HVDC, second-preventive or MARMOT — the
# features `src/rao/` does not implement. That is 35.
#
# These run the AC flow model (`evaluate::evaluate_ac`): every one of their
# configurations sets `"dc": false`, and most of their expectations are written
# in amperes rather than megawatts.
#
# Each scenario's `Scenario:` line names the file it came from. Steps are
# unmodified, including the file paths: the harness resolves them by basename
# against `tests/data/rao/features/`, so the text stays exactly as its authors
# wrote it.


Feature: gridoxide against powsybl-open-rao's own AC expectations

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 0.1.1.8.2: Reference run for tests with Xnodes
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "crac7/ls-ref.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -416.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -416.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 1028.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 DE-NL - outage" after PRA should be 1380.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - N-1 DE-NL - outage" after PRA should be 2832.0 A
    Then the tap of PstRangeAction "PST_BE" should be -16 in preventive
    Then 1 remedial actions are used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 0.1.3.4.1: Inverted PstRangeAction in Security Limit
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic90/SL_ep90us3case1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 26.0 A
    Then the margin on cnec "BBE2AA1  BBE3AA1  1 - preventive" after PRA should be 26.0 A
    Then the tap of PstRangeAction "PRA_PST_BE" should be 3 in preventive

    ## TODO: is this relevant as security limits are not used anymore?

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 0.1.3.4.2: Inverted PstSetpoint in Security Limit
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us2_selectionTopoRA_variant1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 83.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 83.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PST @1" is used in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 0.2.1.1.1: Secure optimization
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us2_selectionTopoRA.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "Open tie-line FR DE" is used in preventive
    Then the "upper" threshold on cnec "BBE2AA1  FFR3AA1  1 - preventive" should be 1500.0 A
    Then the initial flow on cnec "BBE2AA1  FFR3AA1  1 - preventive" should be 721.0 A on side 2
    Then the flow on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be -1444.0 A on side 2
    Then the "upper" threshold on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - outage" should be 1500.0 A
    Then the initial flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - outage" should be 652.0 A on side 2
    Then the flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - outage" after PRA should be -1444.0 A on side 2
    Then the initial flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - curative" should be 652.0 A on side 2
    Then the flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - curative" after PRA should be -1444.0 A on side 2
    Then the flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - curative" after CRA should be -1444.0 A on side 2

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 0.2.1.2.1
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic2/SL_ep2us2case1.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    Then I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the tap of PstRangeAction "PRA_PST_BE" should be 15 in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 2.1.1.1
  No remedial action, several unsecure CNECs.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us0_withoutRA.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -667.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -667.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - Defaut FR1 FR3 - outage" after PRA should be -598.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - Defaut FR1 FR3 - curative" after CRA should be -598.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - Defaut FR1 FR2 - outage" after PRA should be -389.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - Defaut FR1 FR2 - curative" after CRA should be -389.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 779.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - Defaut FR1 FR3 - outage" after PRA should be 848.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - Defaut FR1 FR3 - curative" after CRA should be 848.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - Defaut FR1 FR2 - outage" after PRA should be 1056.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - Defaut FR1 FR2 - curative" after CRA should be 1056.0 A

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 2.1.2.1: selection of topological action
    Two network actions and one PST setpoint action available, only one network action is activated.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us2_selectionTopoRA.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 56.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 56.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - outage" after PRA should be 56.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - curative" after CRA should be 56.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "Open tie-line FR DE" is used in preventive
    Then the "upper" threshold on cnec "BBE2AA1  FFR3AA1  1 - preventive" should be 1500.0 A
    Then the initial flow on cnec "BBE2AA1  FFR3AA1  1 - preventive" should be 721.0 A on side 2
    Then the flow on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be -1444.0 A on side 2
    Then the "upper" threshold on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - outage" should be 1500.0 A
    Then the initial flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - outage" should be 652.0 A on side 2
    Then the flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - outage" after PRA should be -1444.0 A on side 2
    Then the initial flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - curative" should be 652.0 A on side 2
    Then the flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - curative" after PRA should be -1444.0 A on side 2
    Then the flow on cnec "BBE2AA1  FFR3AA1  1 - Contingency FR1 FR3 - curative" after CRA should be -1444.0 A on side 2

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 2.1.2.2: selection of PST setpoint remedial action
  One network action and one PST setpoint action available, only the PST setpoint action is activated.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us2_selectionTopoRA_variant1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 83.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 83.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PST @1" is used in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 2.1.2.3: selection of PST setpoint remedial action but residual constraint
  One network action and one PST setpoint action available, only the PST setpoint action is activated but the situation
  remains unsecure.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us2_selectionTopoRA_variant2.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -417.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -417.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PST @1" is used in preventive
    Then the "upper" threshold on cnec "BBE2AA1  FFR3AA1  1 - preventive" should be 1500.0 A
    Then the initial flow on cnec "BBE2AA1  FFR3AA1  1 - preventive" should be 721.0 A on side 2
    Then the flow on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 471.1 A on side 2
    Then the "upper" threshold on cnec "FFR2AA1  DDE3AA1  1 - Contingency FR1 FR3 - outage" should be 1500.0 A
    Then the initial flow on cnec "FFR2AA1  DDE3AA1  1 - Contingency FR1 FR3 - outage" should be 2098.0 A on side 2
    Then the flow on cnec "FFR2AA1  DDE3AA1  1 - Contingency FR1 FR3 - outage" after PRA should be 1859.0 A on side 2
    Then the flow on cnec "FFR2AA1  DDE3AA1  1 - Contingency FR1 FR3 - outage" after CRA should be 1859.0 A on side 2

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 2.2.1.1.1: Optimization monitoring only the PST
  Basic case, the only RA is a PST range action - the CNECs are only defined for one network element.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic2/SL_ep2us2case1.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 778.0 A
    Then the margin on cnec "BBE2AA1  BBE3AA1  1 - preventive" after PRA should be 778.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PRA_PST_BE" is used in preventive
    Then the tap of PstRangeAction "PRA_PST_BE" should be 15 in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 2.2.1.1.2: Trade-off between various constraints
  Same as 2.2.1.1.1, except that CNECs are defined on two additional network elements.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic2/SL_ep2us2case2.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 39.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - outage" after PRA should be 39.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PRA_PST_BE" is used in preventive
    Then the tap of PstRangeAction "PRA_PST_BE" should be 4 in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 2.2.1.1.3: Unsecure solution
  Same as 2.2.1.1.2, except that the CNEC thresholds are more restrictive.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic2/SL_ep2us2case3.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -37.0 A
    Then the margin on cnec "BBE2AA1  BBE3AA1  1 - preventive" after PRA should be -37.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PRA_PST_BE" is used in preventive
    Then the tap of PstRangeAction "PRA_PST_BE" should be 2 in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 2.2.1.1.4: Range intersection for one PST
  Same as the previous cases, except that the max value of the absolute range of the RA is restricted from 16 to 3,
    and a relativeToInitialNetwork range is added (0 to 10).
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic2/SL_ep2us2case4.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 26.0 A
    Then the margin on cnec "BBE2AA1  BBE3AA1  1 - preventive" after PRA should be 26.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PRA_PST_BE" is used in preventive
    Then the tap of PstRangeAction "PRA_PST_BE" should be 3 in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 3.2.1.0.a: SECURE_FLOW objective function: secure initially, no RA applied
  This test is used as a reference for positive margin objective function,for comparison with the tests with MAX_MIN_MARGIN objective function.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic5/SL_ep5us1.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 500.0 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 500.0 MW
    Then 0 remedial actions are used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 3.2.1.1: MAX_MIN_MARGIN objective function: secure initially, optimize
  Same as 3.2.1.0.a, but with MAX_MIN_MARGIN objective function.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic5/SL_ep5us1.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 1000.0 MW
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 1000.0 MW
    Then 2 remedial actions are used in preventive
    Then the remedial action "Open tie-line FR1 FR2" is used in preventive
    Then the remedial action "Open tie-line FR1 FR3" is used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 3.2.1.2: MAX_MIN_MARGIN objective function: maximum depth reached
  Same as 3.2.1.1, but only one RA is allowed: the worst margin is lower than in 5.1.1.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic5/SL_ep5us1.json"
    Given configuration file is "epic5/RaoParameters_maxMargin_maxDepth.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 1149.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 693.0 MW
    Then 1 remedial actions are used in preventive
    Then the remedial action "Open tie-line FR1 FR2" is used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 3.2.1.3: MAX_MIN_MARGIN objective function: relative minimum impact threshold not reached
  Same as 3.2.1.1, but the parameter relative-minimum-impact-threshold is set at 50%. As the best RA improves the margin
  from 500 MW to 693 MW, the improvement is below 50%: no RA is activated.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic5/SL_ep5us1.json"
    Given configuration file is "epic5/RaoParameters_maxMargin_relativeMinImpact.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 871.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 500.0 MW
    Then 0 remedial actions are used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 3.2.1.4.a: MAX_MIN_MARGIN objective function: absolute minimum impact threshold reached
  Same as 3.2.1.1, but the parameter absolute-minimum-impact-threshold is set at 190 MW. As the best RA improves the margin
  from 500 MW to 693 MW, the improvement is more than 190 MW: the RA is activated.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic5/SL_ep5us1.json"
    Given configuration file is "epic5/RaoParameters_maxMargin_absoluteMinImpact275_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 1594.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 1000.0 MW
    Then 2 remedial actions are used in preventive
    Then the remedial action "Open tie-line FR1 FR2" is used in preventive
    Then the remedial action "Open tie-line FR1 FR3" is used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 3.2.1.4.b: MAX_MIN_MARGIN objective function: absolute minimum impact threshold not reached
  Same as 3.2.1.1, but the parameter absolute-minimum-impact-threshold is set at 195 MW. As the best RA improves the margin
  from 500 MW to 693 MW, the improvement is below 195 MW: no RA is activated.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic5/SL_ep5us1.json"
    Given configuration file is "epic5/RaoParameters_maxMargin_absoluteMinImpact280_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 871.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 500.0 MW
    Then 0 remedial actions are used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.2.2.1: reference run, no MNEC
    Given network file is "common/TestCase12Nodes.uct" for CORE CC
    Given crac file is "epic11/ls_mnec_networkAction_ref.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_ac.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the remedial action "Open line FR1- FR2" is used in preventive
    Then line "FFR1AA1  FFR2AA1  1" in network file with PRA has connection status to "false"
    Then the remedial action "PST BE setpoint" is used in preventive
    Then the worst margin is -207.7 A on cnec "FFR2AA1  DDE3AA1  1 - preventive"
    Then the flow on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be -2684.7 A on side 1

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.2.3.1: reference run, no MNEC
    Given network file is "common/TestCase12Nodes.uct" for CORE CC
    Given crac file is "epic11/ls_mixed_ref.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_ac.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the remedial action "Open line NL1-NL2" is used in preventive
    Then line "NNL1AA1  NNL2AA1  1" in network file with PRA has connection status to "false"
    Then the tap of PstRangeAction "PRA_PST_BE" should be -16 in preventive
    Then PST "BBE2AA1  BBE3AA1  1" in network file with PRA is on tap -16
    Then 2 remedial actions are used in preventive
    Then the worst margin is -106.6 A
    Then the flow on cnec "NNL2AA1  NNL3AA1  1 - preventive" after PRA should be 1648.7 A on side 1
    Then the flow on cnec "DDE1AA1  DDE2AA1  1 - Contingency FR1 FR3 - curative" after CRA should be -568.2 A on side 1
    Then the flow on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be -2372.5 A on side 1

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.3.1.2.1: topological RA, direct CNEC unsecure initially
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic3/SL_ep3us2_topo_direct.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 56.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 56.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "Open FR_DE" is used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.3.1.2.2: topological RA, opposite CNEC secure initially
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic3/SL_ep3us2_topo_opposite.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 779.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 779.0 A
    Then 0 remedial actions are used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.3.1.2.3: PST range action, direct CNEC
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic3/SL_ep3us2_pst_direct.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -416.1 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -416.1 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PST1" is used in preventive
    Then the tap of PstRangeAction "PST1" should be -16 in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.3.1.2.4: PST range action, opposite CNEC
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic3/SL_ep3us2_pst_opposite.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 1075.0 A
    Then the margin on cnec "DDE1AA1  DDE2AA1  1 - Contingency FR1 FR3 - outage" after PRA should be 1075.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PST1" is used in preventive
    Then the tap of PstRangeAction "PST1" should be 16 in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.3.2.1.2.2: A thresholds in AC mode and min margin in A
  Same data as 5.3.2.1.2.1, but the computation is in AC.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic4/SL_ep4us2_4MR_A.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 19.0 A
    Then the value of the objective function after CRA should be -19.0
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 19.0 A
    Then the tap of PstRangeAction "PRA_PST_BE" should be 4 in preventive
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 19.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - outage" after PRA should be 31.0 A
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - N-1 NL1-NL3 - outage" after PRA should be 51.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 63.0 A

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.3.2.1.3.2: mixed thresholds in AC mode and min margin in A
  Same data as 5.3.2.1.3.1, but the computation is in AC.
    Given network file is "common/TestCase12Nodes.uct" for CORE CC
    Given crac file is "epic4/SL_ep4us2_4MR_mixed.json"
    Given configuration file is "common/RaoParameters_maxMargin_ampere_ac.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 24 A
    Then the value of the objective function after CRA should be -24
    Then the tap of PstRangeAction "PRA_PST_BE" should be 4 in preventive
    Then the margin on cnec "NNL2AA1  BBE3AA1  1 - preventive" after PRA should be 24.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - outage" after PRA should be 31.5 A

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 5.3.2.2.1.1: secure with AC config
    Given network file is "common/TestCase12Nodes.uct" for CORE CC
    Given crac file is "epic4/SL_ep4us3.json"
    Given configuration file is "common/RaoParameters_posMargin_ampere.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 38.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - N-1 NL1-NL3 - curative" after PRA should be 38.0 A
    Then the remedial action "PRA_PST_BE" is used in preventive
    Then the tap of PstRangeAction "PRA_PST_BE" should be 4 in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 5.5.1.1: Simple case, only network actions inside country, positive margin
  The parameters mentioned in the US description only allow to use RAs inside the country of the limiting element,
  but the limiting element and the RAs are in the same country in any case.
    # TODO: this test is not so relevant because it does not test geographic filtering, as everything is in the same country
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us2_selectionTopoRA_variant1.json"
    Given configuration file is "epic91/RaoParameters_case_91_1_1.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 56.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 56.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "Open tie-line FR DE" is used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.5.1.2: Simple case, only network actions inside country, max margin
    # TODO: this description is written in comments, while in other files the descriptions are not inside comments
    # Before the first depth, the limiting cnec is a tie-line between FR and DE
    # So only RA "Open tie-line FR DE" can be used
    # At the end of the first depth RAO, the new limiting element is a tie-line between FR and BE
    # So the RA "PST @1" can now be used, but won't be because combining it with the first RA does not improve the solution
    # This results in a solution that is less optimal than using "PST @1" alone (83A, see test case 5.5.1.3 variant 1)
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us2_selectionTopoRA_variant1.json"
    Given configuration file is "epic91/RaoParameters_case_91_1_12.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 56.0 A
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - preventive" after PRA should be 56.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "Open tie-line FR DE" is used in preventive

  @fast @rao @ac @preventive-only @max-min-margin
  Scenario: 5.5.1.3: Simple case, one boundary can be passed, max margin
  Same case as 5.5.1.2, but more permissive: "max-number-of-boundaries-for-skipping-actions" is set to 1 instead of 0.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us2_selectionTopoRA_variant1.json"
    Given configuration file is "epic91/RaoParameters_case_91_1_3.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 83.0 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 83.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PST @1" is used in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 5.5.1.4: Simple case, one boundary can be passed, positive margin
  Same case as 5.5.1.3, but with positive margin.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic1/SL_ep1us2_selectionTopoRA_variant1.json"
    Given configuration file is "epic91/RaoParameters_case_91_1_3.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 83.88 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 83.88 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PST @1" is used in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 5.5.1.5: Another simple case, no boundary can be passed, positive margin
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic91/sl_ep91us1case5.json"
    Given configuration file is "epic91/RaoParameters_case_91_1_1.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -12.0 A
    Then the margin on cnec "DDE1AA1  DDE3AA1  1 - preventive" after PRA should be -12.0 A
    Then 0 remedial actions are used in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 5.5.1.6: Another simple case, one boundary can be passed, positive margin
  Same as 5.5.1.5, but more permissive: "max-number-of-boundaries-for-skipping-actions" is set to 1 instead of 0.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic91/sl_ep91us1case5.json"
    Given configuration file is "epic91/RaoParameters_case_91_1_6.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 11.0 A
    Then the margin on cnec "DDE1AA1  DDE3AA1  1 - preventive" after PRA should be 11.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "Open internal line FR" is used in preventive

  @fast @rao @ac @preventive-only @secure-flow
  Scenario: 5.5.1.7: Another simple case, two boundaries can be passed, positive margin
  Same as 5.5.1.5, but more permissive: "max-number-of-boundaries-for-skipping-actions" is set to 2 instead of 0.
    Given network file is "common/TestCase12Nodes.uct"
    Given crac file is "epic91/sl_ep91us1case5.json"
    Given configuration file is "epic91/RaoParameters_case_91_1_7.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the worst margin is 71.0 A
    Then the margin on cnec "DDE1AA1  DDE3AA1  1 - preventive" after PRA should be 71.0 A
    Then 1 remedial actions are used in preventive
    Then the remedial action "PST BE @1" is used in preventive


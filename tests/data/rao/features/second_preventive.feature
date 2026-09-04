# Second-preventive optimization, vendored verbatim from powsybl-open-rao's own
# Cucumber suite.
#
# Scored as its own file for the reason the other three are: this is a distinct
# capability, and a gain here must not hide a regression elsewhere.
#
# Excluded from the reference's 1.4 directory: every scenario whose CRAC is a
# CBCORA XML (1.4.1.7 and all of 1.4.2), which gridoxide does not read, and the
# loop-flow file, which plans/RAO_PLAN.md 11 puts out of scope. 1.4.4.1 and
# 1.4.4.2 already live in ac_scenarios_16nodes.feature -- they are the two where
# second preventive is configured and does *not* run.

Feature: second preventive optimization

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.1.1.1: Preventive network actions only
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/second_preventive_ls_1.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao
#    Then I export rao reports to "reports/reports_1_4_1_1_1.txt"
    Then the worst margin is 321 A
    Then 3 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be -5 in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 321 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 501 A
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.1.1.2: Same case as 1.4.1.1 with a limitation of 2 RAs in preventive
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/second_preventive_ls_1_2.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be -5 in preventive
    Then the tap of PstRangeAction "pst_be" should be 0 in preventive
    Then the worst margin is 295.6 A

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.1.1.3: Same case as 1.4.1.1.1 with pst_fr available in curative
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/second_preventive_ls_1_3.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 5 in preventive
    Then the tap of PstRangeAction "pst_fr" should be -5 after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then the worst margin is 321 A

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.1.1.4: Pst_fr limits relative to previous instant are the most impacting w.r.t relative to initial network
    # 2P is now always global, pst_fr is always optimized
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/second_preventive_ls_1_4.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "open_fr1_fr3" is used in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "pst_fr" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_fr" should be 1 in preventive
    Then the tap of PstRangeAction "pst_fr" should be -3 after "co1_fr2_fr3_1" at "curative"
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 210 A
    Then the worst margin is 210 A

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.1.2: Preventive and curative network actions 1/3
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us3case1.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao
    Then 2 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 4 in preventive
    Then 2 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -5 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 721 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 721 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 725 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after PRA should be 731 A
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.1.3: Preventive and curative network actions 2/3
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us4case2.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao
    Then 1 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -462 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -462 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -87 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 236 A
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "UNSECURED"

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.1.4: Preventive and curative network actions 3/3
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic13/SL_ep13us4case4.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao
    Then 1 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_be" should be -16 in preventive
    Then 0 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the tap of PstRangeAction "pst_be" should be -16 after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is -462 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be -462 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be -87 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 236 A
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "UNSECURED"

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.1.5: Duplicated RA on the same PST, one being a PRA and the other one being a CRA
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/crac_ep20us1case1_5.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao
    # Same: FARAO has better results than OSIRIS
    Then 1 remedial actions are used in preventive
    Then the remedial action "pst_fr_pra" is used in preventive
    Then the tap of PstRangeAction "pst_fr_pra" should be -7 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 43 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 43 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 385 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be 86 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 910 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 1148 A
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.1.6: Same test as 1.4.1.5. The CRA has a non relevant relative to previous instant range
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/crac_ep20us1case1_6.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao
    Then 1 remedial actions are used in preventive
    Then the remedial action "pst_fr_pra" is used in preventive
    Then the tap of PstRangeAction "pst_fr_pra" should be -7 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "close_fr1_fr5" is used after "co1_fr2_fr3_1" at "curative"
    # Same result as 1.4.1.5, the curative pst is not needed since preventive flow is limiting
    Then the worst margin is 43 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - preventive" after PRA should be 43 A
    Then the margin on cnec "FFR2AA1  DDE3AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 385 A
    Then the margin on cnec "FFR2AA1  FFR3AA1  2 - co1_fr2_fr3_1 - curative" after CRA should be 86 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 910 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - outage" after PRA should be 1148 A
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.4.3: Cost has not increased during RAO, do not run 2P (copy of 1.4.1.1.1)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/second_preventive_ls_1.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_2p_if_cost_increase.json"
    When I launch rao
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the worst margin is -144 A
    Then the value of the objective function after CRA should be 144

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.4.4: Cost has increased during RAO, run 2P (copy of 1.4.1.1.2)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/SL_ep20us5case2.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_2p_if_cost_increase.json"
    When I launch rao
    Then 1 remedial actions are used in preventive
    Then the tap of PstRangeAction "pst_fr" should be 2 in preventive
    Then 1 remedial actions are used after "co1_fr2_fr3_1" at "curative"
    Then the remedial action "open_fr1_fr3" is used after "co1_fr2_fr3_1" at "curative"
    Then the worst margin is 795 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 795 A
    Then the margin on cnec "FFR4AA1  DDE1AA1  1 - preventive" after CRA should be 800 A
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.4.5: Not enough time to run 2P (copy of 1.4.1.1.1)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/second_preventive_ls_1.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao with a time limit of -1 seconds
    Then the worst margin is -144 A
    Then the value of the objective function after CRA should be 144
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"

  @fast @rao @ac @second-preventive @max-min-margin
  Scenario: 1.4.4.6: Enough time to run 2P (copy of 1.4.1.1.1)
    Given network file is "common/TestCase16Nodes.uct"
    Given crac file is "epic20/second_preventive_ls_1.json"
    Given configuration file is "epic20/RaoParameters_maxMargin_ampere_second_preventive.json"
    When I launch rao with a time limit of 600 seconds
    Then the worst margin is 321 A
    Then the margin on cnec "FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 321 A
    Then the margin on cnec "FFR3AA1  FFR5AA1  1 - co1_fr2_fr3_1 - curative" after CRA should be 501 A
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"

  @fast @rao @dc @second-preventive @max-min-margin
  Scenario: 1.4.5.1: Fix PST CRA setpoints in global 2nd preventive
    Given network file is "epic20/TestCase12Nodes_20_6_1.uct"
    Given crac file is "epic20/crac_ep20us6case1.json"
    Given configuration file is "epic20/RaoParameters_20_6_1.json"
    When I launch rao
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "UNSECURED"
    Then the worst margin is -40.5 MW
    Then the margin on cnec "NNL3AA1  BBE1AA1  1 - Contingency NL3 BE1 2 - curative" after CRA should be -40.5 MW
    Then the tap of PstRangeAction "CRA_PST_DE" should be 0 after "Contingency NL3 BE1 2" at "curative"

  @fast @rao @dc @second-preventive @max-min-margin
  Scenario: 1.4.5.2: Fallback to first preventive after 2nd preventive
    Given network file is "epic20/TestCase12Nodes_20_6_2.uct"
    Given crac file is "epic20/crac_ep20us6case2.json"
    Given configuration file is "epic20/RaoParameters_20_6_2.json"
    When I launch rao
    Then the execution details should be "Second preventive fell back to first preventive results"
    Then its security status should be "SECURED"
    Then the worst margin is 100.0 MW
    Then the margin on cnec "BBE2AA1  FFR3AA1  1 - Contingency NL2 NL3 1 - auto" after ARA should be 100 MW

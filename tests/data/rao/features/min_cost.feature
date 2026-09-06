# Scenarios taken verbatim from powsybl-open-rao's own Cucumber suite.
#
# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# **Costly optimization** — the reference's `MIN_COST` objective, where the
# quantity minimized is the price of the plan rather than the margin it buys:
#
#     cost = violation_penalty x (every overload, summed) + (what the actions cost)
#
# a *sum* over CNECs where max-min-margin is a *min*, which is why it is a
# different objective rather than a penalty bolted onto the existing one.
#
# Vendored **before the capability exists**, which is the order
# `plans/RAO_PLAN.md` records as the only one that works: the second-preventive
# corpus was vendored first too, scored 45 of 108 with nothing implemented, and
# the gate then found every defect in it. A low score here is the deliverable,
# not a failure — it is the number the work is measured against.
#
# The selection is all three files of `3_objective_functions/3_4_min_cost/`, 26
# of their 27 scenarios. **3.4.13 is the one exclusion**: its CRAC is a CBCORA
# XML (`epic93/cbcora_93_2_2.xml`), needing a FlowBasedConstraintDocument
# importer that does not exist, and it is the suite's sole user of
# `When I launch rao at "<timestamp>"`. 3.4.1.2.bis is kept even though it is the
# only scenario here not tagged `@costly` — it is the deliberate `MAX_MIN_MARGIN`
# control for 3.4.1.2, and the contrast is the point of both.
#
# Each scenario's `Scenario:` line names the file it came from. Steps are
# unmodified, including the file paths: the harness resolves them by basename
# against `tests/data/rao/features/`, so the text stays exactly as its authors
# wrote it.

Feature: gridoxide against powsybl-open-rao's costly-optimization expectations

  @fast @preventive-only @costly @rao
  Scenario: 3.4.1.1  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Selection of cheapest of 3 equivalent network actions
  The network contains two nodes linked with 4 parallel lines: one of the lines is the overloaded CNEC,
  and the three other are open. 3 remedial actions with different costs: close one of the three lines.
    Given network file is "epic92/2Nodes4ParallelLines.uct"
    Given crac file is "epic92/crac-92-1-1.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 250.0 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr4" is used in preventive
    # Overload penalty (250 * 1000)
    Then the value of the objective function initially should be 250000.0
    # Activation of closeBeFr4 (10)
    Then the value of the objective function after PRA should be 10.0

  @fast @preventive-only @costly @rao
  Scenario: 3.4.1.2  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Selection of cheapest network action even if it does not maximize minimum margin
  Line BE-FR-3 has a higher resistance than line BE-FR-2 which means that closing the latter will lead
  to a higher margin on the optimized CNEC. However, closing line BE-FR-3 is cheaper and still secures
  the CNEC so it will be chosen by the RAO.
    Given network file is "epic92/2Nodes3ParallelLines_disconnected.uct"
    Given crac file is "epic92/crac-92-1-2.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 83.0 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr3" is used in preventive
    Then the value of the objective function initially should be 250000.0
    Then the value of the objective function after PRA should be 25.0

  @fast @preventive-only @max-min-margin @rao
  Scenario: 3.4.1.2.bis  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Duplicate of 3.4.1.2 in MAX_MIN_MARGIN mode
  The situation is the same as in 3.4.1.2 but the RAO maximizes the minimum margin.
  As activation costs are not taken in account, both lines will be closed.
    Given network file is "epic92/2Nodes3ParallelLines_disconnected.uct"
    Given crac file is "epic92/crac-92-1-2.json"
    Given configuration file is "epic92/RaoParameters_margin_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 464.29 MW
    Then the margin on cnec "cnecBeFrPreventive" after PRA should be 464.29 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "closeBeFr2" is used in preventive
    Then the remedial action "closeBeFr3" is used in preventive
    Then the value of the objective function initially should be 250.0
    # In MIN_MAX_MARGIN, the objective function is the opposite of the worst margin.
    Then the value of the objective function after PRA should be -464.29

  @fast @preventive-only @costly @rao
  Scenario: 3.4.1.3  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Selection of the two cheapest network actions
  Same as 3.4.1.1, but two network actions are needed to secure the CNEC.
    Given network file is "epic92/2Nodes4ParallelLines.uct"
    Given crac file is "epic92/crac-92-1-3.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 66.67 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 2 remedial actions are used in preventive
    Then the remedial action "closeBeFr3" is used in preventive
    Then the remedial action "closeBeFr4" is used in preventive
    # Overload penalty (600 * 1000)
    Then the value of the objective function initially should be 600000.0
    # Activation of closeBeFr3 (500) + activation of closeBeFr4 (220)
    Then the value of the objective function after PRA should be 720.0

  @fast @preventive-only @costly @rao
  Scenario: 3.4.1.4  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Selection of cheapest of 3 equivalent network actions but overload remains at the end of RAO
  Only one network action can be used (behavior set in the RAO parameters) so the RAO chooses the cheapest
  remedial action available to reduce the overload and thus the penalty cost. The total cost is:
  100 (overload in MW) * 10000 (penalty cost in currency/MW) + 220 (cost of the chosen remedial action)
    Given network file is "epic92/2Nodes4ParallelLines.uct"
    Given crac file is "epic92/crac-92-1-3.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_maxDepth1.json"
    When I launch rao
    Then the worst margin is -100.0 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr4" is used in preventive
    Then the value of the objective function initially should be 600000.0
    Then the value of the objective function after PRA should be 100220.0

  @fast @preventive-only @costly @rao
  Scenario: 3.4.1.5  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Sub-optimal case
  Closing line BE-FR-2 costs 1000 but solves the constraint immediately. As closing BE-FR-2 looks optimal
  at depth 1, the greedy search-tree keeps it at depth 2 but there is no need to apply additional remedial
  actions since the network is already secure.
  However, the optimal case is to close lines BE-FR-3 and BE-FR-4 successively (the order does not matter)
  for a total expense of 50. Yet, because of the penalty cost for overloads, the RAO still counts an over-cost
  of 100000 because closing BE-FR-3 or BE-FR-4 alone only reduce the minimum margin to -100 MW.
    #TODO: I reordered this text, but the last sentence above is still not clear to me
    Given network file is "epic92/2Nodes4ParallelLinesDifferentResistances.uct"
    Given crac file is "epic92/crac-92-1-5.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 66.67 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr2" is used in preventive
    Then the value of the objective function initially should be 600000.0
    Then the value of the objective function after PRA should be 1000.0

  @fast @costly @rao
  Scenario: 3.4.1.6  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Preventive and auto optimization - 1 scenario
  Overload in preventive (activation of 1 RA), no overload in auto (activation of a 2nd RA)
    Given network file is "epic92/2Nodes4ParallelLines2LinesClosed.uct"
    Given crac file is "epic92/crac-92-1-6.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 66.67 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 600000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr3" is used in preventive
    # Activation of closeBeFr3 (200) + overload penalty (100 * 1000)
    Then the value of the objective function after PRA should be 100200.0
    Then 1 remedial actions are used after "coBeFr2" at "auto"
    Then the remedial action "closeBeFr4" is used after "coBeFr2" at "auto"
    # Activation of closeBeFr3 (200) + activation of closeBeFr4 (850)
    Then the value of the objective function after ARA should be 1050.0

  @fast @costly @rao
  Scenario: 3.4.1.7  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Preventive and auto optimization - 2 scenarios
  Each scenario secured by a different RA in auto.
    Given network file is "epic92/2Nodes4ParallelLines3LinesClosed.uct"
    Given crac file is "epic92/crac-92-1-7.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 66.67 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 100000.0
    Then 1 remedial actions are used after "coBeFr2" at "auto"
    Then the remedial action "closeBeFr4" is used after "coBeFr2" at "auto"
    Then 1 remedial actions are used after "coBeFr3" at "auto"
    Then the remedial action "closeBeFr4" is used after "coBeFr3" at "auto"
    # Activation of closeBeFr4 after coBeFr2 (850) + activation of closeBeFr4 after coBeFr3 (850)
    Then the value of the objective function after ARA should be 1700.0

  @fast @costly @rao
  Scenario: 3.4.1.8  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Preventive and auto optimization - 2 scenarios with PRA
  Each scenario secured by a different RA in auto, after one PRA.
    Given network file is "epic92/2Nodes5ParallelLines3LinesClosed.uct"
    Given crac file is "epic92/crac-92-1-8.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 10.0 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 240000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr4" is used in preventive
    # Activation of closeBeFr4 (325) + overload penalty (73.33 * 1000)
    Then the value of the objective function after PRA should be 73658.0
    Then 1 remedial actions are used after "coBeFr2" at "auto"
    Then the remedial action "closeBeFr5" is used after "coBeFr2" at "auto"
    Then 1 remedial actions are used after "coBeFr3" at "auto"
    Then the remedial action "closeBeFr5" is used after "coBeFr3" at "auto"
    # Preventive cost (325) + activation of closeBeFr4 after coBeFr2 (850) + activation of closeBeFr4 after coBeFr3 (850)
    Then the value of the objective function after ARA should be 2025.0

  @fast @costly @rao
  Scenario: 3.4.1.9  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Preventive and curative optimization - 1 scenario - no auto instant
  The curative scenario is secured by one PRA and one CRA.
    Given network file is "epic92/2Nodes4ParallelLines2LinesClosed.uct"
    Given crac file is "epic92/crac-92-1-9.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 66.67 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 600000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr3" is used in preventive
    # Activation of closeBeFr3 (200) + overload penalty (100 * 1000)
    Then the value of the objective function after PRA should be 100200.0
    Then 1 remedial actions are used after "coBeFr2" at "curative"
    Then the remedial action "closeBeFr4" is used after "coBeFr2" at "curative"
    # Activation of closeBeFr3 (200) + activation of closeBeFr4 (850)
    Then the value of the objective function after CRA should be 1050.0

  @fast @costly @rao
  Scenario: 3.4.1.9.bis  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Preventive and curative optimization - 1 scenario - with auto instant
  The curative scenario is secured by one PRA and one CRA, but instants auto and preventive are unsecure.
    Given network file is "epic92/2Nodes4ParallelLines2LinesClosed.uct"
    Given crac file is "epic92/crac-92-1-9-bis.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 66.67 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 600000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr3" is used in preventive
    # Activation of closeBeFr3 (200) + overload penalty (100 * 1000)
    Then the value of the objective function after PRA should be 100200.0
    Then the value of the objective function after ARA should be 100200.0
    Then 1 remedial actions are used after "coBeFr2" at "curative"
    Then the remedial action "closeBeFr4" is used after "coBeFr2" at "curative"
    # Activation of closeBeFr3 (200) + activation of closeBeFr4 (850)
    Then the value of the objective function after CRA should be 1050.0

  @fast @costly @rao
  Scenario: 3.4.1.10  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Preventive, auto and curative optimization - 1 scenario
  One RA applied after each instant, only curative is secure.
    Given network file is "epic92/2Nodes5ParallelLines2LinesClosed.uct"
    Given crac file is "epic92/crac-92-1-10.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is 16.67 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 700000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr3" is used in preventive
    # Activation of closeBeFr2 (200) + overload penalty for curative cnec (200 * 1000)
    Then the value of the objective function after PRA should be 200200.0
    Then 1 remedial actions are used after "coBeFr2" at "auto"
    Then the remedial action "closeBeFr4" is used after "coBeFr2" at "auto"
    # Activation of closeBeFr2 (200) + activation of closeBeFr3 (1350) + overload penalty for curative cnec (33.33 * 1000)
    Then the value of the objective function after ARA should be 34883.33
    Then 1 remedial actions are used after "coBeFr2" at "curative"
    Then the remedial action "closeBeFr5" is used after "coBeFr2" at "curative"
    # Activation of closeBeFr2 (200) + activation of closeBeFr3 (1350) + activation of closeBeFr5 (850)
    Then the value of the objective function after CRA should be 2400.0

  @fast @costly @rao
  Scenario: 3.4.1.11  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Preventive, auto and curative optimization - 4 comprehensive scenarios
  4 scenarios are optimized in parallel:
  - scenario 1: no ARA and no CRA -> 50 MW overload
  - scenario 2: forced ARA and no CRA
  - scenario 3: no ARA and available CRA
  - scenario 4: forced ARA and available CRA
    Given network file is "epic92/2Nodes8ParallelLines5LinesClosed.uct"
    Given crac file is "epic92/crac-92-1-11.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is -50.00 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the value of the objective function initially should be 100000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr6" is used in preventive
    # Activation of closeBeFr6 (2500) + overload penalty (50 * 1000)
    Then the value of the objective function after PRA should be 52500.0
    Then 0 remedial actions are used after "coBeFr2" at "auto"
    Then 1 remedial actions are used after "coBeFr3" at "auto"
    Then the remedial action "closeBeFr7" is used after "coBeFr3" at "auto"
    Then 0 remedial actions are used after "coBeFr4" at "auto"
    Then 1 remedial actions are used after "coBeFr5" at "auto"
    Then the remedial action "closeBeFr7" is used after "coBeFr5" at "auto"
    # Activation of closeBeFr6 (2500) + activation of closeBeFr7 twice (2 * 60) + overload penalty (50 * 1000)
    Then the value of the objective function after ARA should be 52620.0
    Then 0 remedial actions are used after "coBeFr2" at "curative"
    Then 0 remedial actions are used after "coBeFr3" at "curative"
    Then 1 remedial actions are used after "coBeFr4" at "curative"
    Then the remedial action "closeBeFr8" is used after "coBeFr4" at "curative"
    Then 1 remedial actions are used after "coBeFr5" at "curative"
    Then the remedial action "closeBeFr8" is used after "coBeFr5" at "curative"
    # Activation of closeBeFr6 (2500) + activation of closeBeFr7 twice (2 * 60) + activation of closeBeFr8 twice (2 * 735) + overload penalty (50 * 1000)
    Then the value of the objective function after CRA should be 54090.0

  @fast @costly @rao
  Scenario: 3.4.1.12  [3_objective_functions/3_4_min_cost/3_4_1_network_actions.feature]: Preventive and auto optimization - curative overload
  4 scenarios are optimized in parallel with a curative overload each time:
  - scenario 1: no ARA and no CRA -> 50 MW overload
  - scenario 2: forced ARA and no CRA -> 66.67 MW overload
  - scenario 3: no ARA and available CRA -> 41.67 MW overload
  - scenario 4: forced ARA and available CRA -> 10 MW overload
    Given network file is "epic92/2Nodes8ParallelLines5LinesClosed.uct"
    Given crac file is "epic92/crac-92-1-12.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective.json"
    When I launch rao
    Then the worst margin is -66.67 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "UNSECURED"
    Then the value of the objective function initially should be 150000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "closeBeFr6" is used in preventive
    # Activation of closeBeFr6 (2500) + overload penalty (100 * 1000) on cnecBeFrCurative - coBeFr3
    Then the value of the objective function after PRA should be 102500.0
    Then 0 remedial actions are used after "coBeFr2" at "auto"
    Then 1 remedial actions are used after "coBeFr3" at "auto"
    Then the remedial action "closeBeFr7" is used after "coBeFr3" at "auto"
    Then 0 remedial actions are used after "coBeFr4" at "auto"
    Then 1 remedial actions are used after "coBeFr5" at "auto"
    Then the remedial action "closeBeFr7" is used after "coBeFr5" at "auto"
    # Activation of closeBeFr6 (2500) + activation of closeBeFr7 twice (2 * 60) + overload penalty on cnecBeFrCurative - coBeFr4 (75 * 1000)
    Then the value of the objective function after ARA should be 77620.0
    Then 0 remedial actions are used after "coBeFr2" at "curative"
    Then 0 remedial actions are used after "coBeFr3" at "curative"
    Then 1 remedial actions are used after "coBeFr4" at "curative"
    Then the remedial action "closeBeFr8" is used after "coBeFr4" at "curative"
    Then 1 remedial actions are used after "coBeFr5" at "curative"
    Then the remedial action "closeBeFr8" is used after "coBeFr5" at "curative"
    # Activation of closeBeFr6 (2500) + activation of closeBeFr7 twice (2 * 60) + activation of closeBeFr8 twice (2 * 735) + overload penalty on cnecBeFrCurative - coBeFr3 (66.67 * 1000)
    Then the value of the objective function after CRA should be 70756.67

  @fast @preventive-only @costly @rao
  Scenario: 3.4.2.1  [3_objective_functions/3_4_min_cost/3_4_2_range_actions.feature]: Change only necessary taps on preventive PST
  The RAO can increase the minimum margin by setting the tap of the PST on position -10
  but stops at position -5 because the network is secure and this saves expenses.
    Given network file is "epic92/2Nodes2ParallelLinesPST.uct"
    Given crac file is "epic92/crac-92-2-1.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst.json"
    When I launch rao
    Then the worst margin is 45.46 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 200000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr2" is used in preventive
    Then the tap of PstRangeAction "pstBeFr2" should be -5 in preventive
    Then the value of the objective function after PRA should be 55.0

  @fast @preventive-only @costly @rao
  Scenario: 3.4.2.2  [3_objective_functions/3_4_min_cost/3_4_2_range_actions.feature]: Two PSTs
  PST 1 has cheaper variation costs (5 per tap) but a higher activation price (100) so moving the 9 required taps would
  require a cost of 145. PST 2 is cheaper to activate (5) and more expensive to use (15 per tap) but leads to a total
  cost of 140 so it is chosen.
    Given network file is "epic92/2Nodes3ParallelLines2PSTs_v2.uct"
    Given crac file is "epic92/crac-92-2-2.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst.json"
    When I launch rao
    Then the worst margin is 31.15 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 263333.33
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr3" is used in preventive
    Then the tap of PstRangeAction "pstBeFr2" should be 0 in preventive
    Then the tap of PstRangeAction "pstBeFr3" should be -9 in preventive
    Then the value of the objective function after PRA should be 140.0

  @fast @costly @rao
  Scenario: 3.4.2.3  [3_objective_functions/3_4_min_cost/3_4_2_range_actions.feature]: Costly PST in preventive and curative
    A PRA and a CRA are activated on the same PST.
    Given network file is "epic92/2Nodes3ParallelLinesPST_v2.uct"
    Given crac file is "epic92/crac-92-2-3.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst.json"
    When I launch rao
    Then the worst margin is 11.73 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 430000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr3" is used in preventive
    Then the tap of PstRangeAction "pstBeFr3" should be -3 in preventive
    # Activation of pstBeFr3 (20) + 3 taps moved (3 * 7.5) + overload penalty (282.71 * 10000)
    Then the value of the objective function after PRA should be 282759.61
    Then 1 remedial actions are used after "coBeFr2" at "curative"
    Then the tap of PstRangeAction "pstBeFr3" should be -9 after "coBeFr2" at "curative"
    # Activation of pstBeFr3 twice (2 * 20) + 9 taps moved in total (3 * 7.5 + 6 * 7.5)
    Then the value of the objective function after CRA should be 107.5

  @fast @costly @rao
  Scenario: 3.4.2.4  [3_objective_functions/3_4_min_cost/3_4_2_range_actions.feature]: Free PST in preventive and curative
    The RA pstBeFr3 is not associated to activation or variation costs (contrarily to 3.4.2.3).
    Given network file is "epic92/2Nodes3ParallelLinesPST_v2.uct"
    Given crac file is "epic92/crac-92-2-4.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst.json"
    When I launch rao
    Then the worst margin is 11.73 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 430000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr3" is used in preventive
    Then the tap of PstRangeAction "pstBeFr3" should be -3 in preventive
    # Overload penalty (282.71 * 10000)
    Then the value of the objective function after PRA should be 282717.11
    Then 1 remedial actions are used after "coBeFr2" at "curative"
    Then the tap of PstRangeAction "pstBeFr3" should be -9 after "coBeFr2" at "curative"
    Then the value of the objective function after CRA should be 0

  @fast @costly @rao @second-preventive
  Scenario: 3.4.2.5  [3_objective_functions/3_4_min_cost/3_4_2_range_actions.feature]: PST in 2nd preventive optimization
  PST is moved to tap -9 straight from preventive optimization to cut curative activation costs.
  Same case as 3.4.2.3 but with 2nd preventive optimization allowed.
    Given network file is "epic92/2Nodes3ParallelLinesPST_v2.uct"
    Given crac file is "epic92/crac-92-2-3.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst_2P.json"
    When I launch rao
    Then the worst margin is 11.73 MW
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 430000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr3" is used in preventive
    Then the tap of PstRangeAction "pstBeFr3" should be -9 in preventive
    # Activation of pstBeFr3 (20) + 9 taps moved (9 * 7.5)
    Then the value of the objective function after PRA should be 87.5
    Then 0 remedial actions are used after "coBeFr2" at "curative"
    Then the value of the objective function after CRA should be 87.5

  @fast @costly @rao @multi-curative
  Scenario: 3.4.2.6  [3_objective_functions/3_4_min_cost/3_4_2_range_actions.feature]: Multi-curative costly optimization
  The same PST is moved in preventive optimization and at all curative states.
    Given network file is "epic92/2Nodes3ParallelLinesPST_v2.uct"
    Given crac file is "epic92/crac-92-2-6.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst.json"
    When I launch rao
    Then the worst margin is 32.13 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 450000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr3" is used in preventive
    Then the tap of PstRangeAction "pstBeFr3" should be -2 in preventive
    # Activation of pstBeFr3 (20) + 2 taps moved (2 * 5.0)
    Then the value of the objective function after PRA should be 351839.51
    Then 1 remedial actions are used after "coBeFr2" at "curative-1"
    Then the remedial action "pstBeFr3" is used after "coBeFr2" at "curative-1"
    Then the tap of PstRangeAction "pstBeFr3" should be -7 after "coBeFr2" at "curative-1"
    Then 1 remedial actions are used after "coBeFr2" at "curative-2"
    Then the remedial action "pstBeFr3" is used after "coBeFr2" at "curative-2"
    Then the tap of PstRangeAction "pstBeFr3" should be -9 after "coBeFr2" at "curative-2"
    Then 1 remedial actions are used after "coBeFr2" at "curative-3"
    Then the remedial action "pstBeFr3" is used after "coBeFr2" at "curative-3"
    Then the tap of PstRangeAction "pstBeFr3" should be -10 after "coBeFr2" at "curative-3"
    # Activation of pstBeFr3 4 times (4 * 20) + 10 taps moved (10 * 5.0)
    Then the value of the objective function after CRA should be 130.0

  @fast @costly @rao @second-preventive @multi-curative
  Scenario: 3.4.2.7  [3_objective_functions/3_4_min_cost/3_4_2_range_actions.feature]: Multi-curative costly optimization with 2P
  Same case as 3.4.2.6 but with second preventive optimization.
  The PST is moved to tap -10 straight from preventive optimization to cut activation expenses.
    Given network file is "epic92/2Nodes3ParallelLinesPST_v2.uct"
    Given crac file is "epic92/crac-92-2-6.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst_2P.json"
    When I launch rao
    Then the worst margin is 40.77 MW
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 450000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr3" is used in preventive
    Then the tap of PstRangeAction "pstBeFr3" should be -10 in preventive
    # Activation of pstBeFr3 (20) + 10 taps moved (10 * 5.0)
    Then the value of the objective function after PRA should be 70.0
    Then 0 remedial actions are used after "coBeFr2" at "curative-1"
    Then 0 remedial actions are used after "coBeFr2" at "curative-2"
    Then 0 remedial actions are used after "coBeFr2" at "curative-3"
    # Activation of pstBeFr3 (20) + 10 taps moved (10 * 5.0)
    Then the value of the objective function after CRA should be 70.0

  @fast @preventive-only @costly @rao
  Scenario: 3.4.3.1  [3_objective_functions/3_4_min_cost/3_4_3_exhaustive.feature]: Activate one topological action and one PST in preventive
  Two ways to secure the network:
  1. move PST to tap -5 => cost of 25
  2. (optimal) close line BE-FR 2 and move PST to tap -2 => cost of 20
    Given network file is "epic92/2Nodes3ParallelLinesPST2LinesClosed.uct"
    Given crac file is "epic92/crac-92-3-1.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst.json"
    When I launch rao
    Then the worst margin is 32.13 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 200000.0
    Then 2 remedial actions are used in preventive
    Then the remedial action "pstBeFr3" is used in preventive
    Then the tap of PstRangeAction "pstBeFr3" should be -2 in preventive
    Then the remedial action "closeBeFr2" is used in preventive
    Then the value of the objective function after PRA should be 20.0

  @fast @costly @rao
  Scenario: 3.4.3.2  [3_objective_functions/3_4_min_cost/3_4_3_exhaustive.feature]: Preventive and curative PST + curative topological action
  The PST is moved to tap -5 to secure the preventive perimeter for a total cost of 95
  (20 for activation + 5 * 15 for variation). Then, there are two ways to secure the curative perimeter:
  1. move PST to tap -8 => cost of 20 + 3 * 15 = 65
  2. (optimal) close line BE-FR 3 and move PST to tap -6 => cost of 10 + 20 + 1 * 15 = 45
    Given network file is "epic92/2Nodes4ParallelLinesPST3LinesClosed.uct"
    Given crac file is "epic92/crac-92-3-2.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst.json"
    When I launch rao
    # Worst margin on preventive CNEC
    Then the worst margin is 5.3 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 350000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr4" is used in preventive
    Then the tap of PstRangeAction "pstBeFr4" should be -5 in preventive
    # Activation of pstBeFr4 (20) + 5 taps moved (5 * 15) + overload penalty (104.54 * 1000)
    Then the value of the objective function after PRA should be 104638.64
    Then 2 remedial actions are used after "coBeFr2" at "curative"
    Then the remedial action "pstBeFr4" is used after "coBeFr2" at "curative"
    Then the tap of PstRangeAction "pstBeFr4" should be -6 after "coBeFr2" at "curative"
    Then the remedial action "closeBeFr3" is used after "coBeFr2" at "curative"
    # activation of closeBeFr3 (10)
    Then the value of the objective function after CRA should be 140

  @fast @costly @rao @second-preventive
  Scenario: 3.4.3.3  [3_objective_functions/3_4_min_cost/3_4_3_exhaustive.feature]: Preventive and curative PST + curative topological action with 2nd preventive optimization
  The PST is moved to tap -6 straight from preventive to only activate the remedial action once.
    Given network file is "epic92/2Nodes4ParallelLinesPST3LinesClosed.uct"
    Given crac file is "epic92/crac-92-3-2.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst_2P.json"
    When I launch rao
    # Worst margin on curative CNEC
    Then the worst margin is 13.02 MW
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 350000.0
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr4" is used in preventive
    Then the tap of PstRangeAction "pstBeFr4" should be -6 in preventive
    # Activation of pstBeFr4 (20) + 6 taps moved (6 * 15) + overload penalty (55.46 * 10000)
    Then the value of the objective function after PRA should be 55574.85
    Then 1 remedial actions are used after "coBeFr2" at "curative"
    Then the remedial action "closeBeFr3" is used after "coBeFr2" at "curative"
    # Activation of pstBeFr4 (20) + 6 taps moved in total (6 * 15) + activation of closeBeFr3 (10)
    Then the value of the objective function after CRA should be 120.0

  @fast @costly @rao
  Scenario: 3.4.3.4  [3_objective_functions/3_4_min_cost/3_4_3_exhaustive.feature]: Preventive and curative PST + curative topological action - 2 scenarios
  One preventive range action, then the same range action is used also in both curatives (along with a topological action),
  but no second preventive to choose preventive tap among the taps chosen in curative.
    Given network file is "epic92/2Nodes5ParallelLinesPST4LinesClosed.uct"
    Given crac file is "epic92/crac-92-3-4.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst.json"
    When I launch rao
    # Worst margin on preventive CNEC
    Then the worst margin is 3.73 MW
    Then the execution details should be "The RAO only went through first preventive"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 233333.33
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr5" is used in preventive
    Then the tap of PstRangeAction "pstBeFr5" should be -5 in preventive
    # Activation of pstBeFr4 (20) + 5 taps moved (5 * 15) + overload penalty (69.7 * 1000)
    Then the value of the objective function after PRA should be 69790.76
    Then 2 remedial actions are used after "coBeFr2" at "curative"
    Then the remedial action "pstBeFr5" is used after "coBeFr2" at "curative"
    Then the tap of PstRangeAction "pstBeFr5" should be -6 after "coBeFr2" at "curative"
    Then the remedial action "closeBeFr4" is used after "coBeFr2" at "curative"
    Then 2 remedial actions are used after "coBeFr3" at "curative"
    Then the remedial action "pstBeFr5" is used after "coBeFr3" at "curative"
    Then the tap of PstRangeAction "pstBeFr5" should be -7 after "coBeFr3" at "curative"
    Then the remedial action "closeBeFr4" is used after "coBeFr3" at "curative"
    # Activation of pstBeFr4 three times (3 * 20) + 8 taps moved in total (8 * 15) + activation of closeBeFr3 twice (2 * 10)
    Then the value of the objective function after CRA should be 200.0

  @fast @costly @rao @second-preventive
  Scenario: 3.4.3.5  [3_objective_functions/3_4_min_cost/3_4_3_exhaustive.feature]: Preventive and curative PST + curative topological action - 2 scenarios with 2nd preventive optimization
  To cut activation cost expenses, the PST is moved to tap -7 straight from preventive optimization.
    Given network file is "epic92/2Nodes5ParallelLinesPST4LinesClosed.uct"
    Given crac file is "epic92/crac-92-3-4.json"
    Given configuration file is "epic92/RaoParameters_dc_minObjective_discretePst_2P.json"
    When I launch rao
    # Worst margin on curative CNEC
    Then the worst margin is 21.8 MW
    Then the execution details should be "Second preventive improved first preventive results"
    Then its security status should be "SECURED"
    Then the value of the objective function initially should be 233333.33
    Then 1 remedial actions are used in preventive
    Then the remedial action "pstBeFr5" is used in preventive
    Then the tap of PstRangeAction "pstBeFr5" should be -7 in preventive
    # Activation of pstBeFr4 (20) + 5 taps moved (7 * 15) + overload penalty (4.26 * 10000)
    Then the value of the objective function after PRA should be 4386.91
    Then 1 remedial actions are used after "coBeFr2" at "curative"
    Then the remedial action "closeBeFr4" is used after "coBeFr2" at "curative"
    Then 1 remedial actions are used after "coBeFr3" at "curative"
    Then the remedial action "closeBeFr4" is used after "coBeFr3" at "curative"
    # Activation of pstBeFr4 (20) + 7 taps moved in total (7 * 15) + activation of closeBeFr3 twice (2 * 10)
    Then the value of the objective function after CRA should be 145.0

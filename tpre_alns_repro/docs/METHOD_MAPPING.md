# Manuscript-to-code mapping

This table follows the updated manuscript and supplementary information used to
prepare release 1.0.0.

| Source item | Implemented rule | Code |
|---|---|---|
| Supplementary Algorithm S2 | Uniform coordinates, demand/service/window draws, 5/8/12 stations, charger/power probabilities, `D_jt ~ Binomial(n_j, 0.05)` | `src/tpre_alns/instance.py` |
| Equations 3-5; Table S1 | Conditional occupation, hidden damage and residual capacity | `src/tpre_alns/scenarios.py` |
| Equations 6-8 | Available/occupied/failed classification and iterative stored queue delay | `entities.Scenario.state`, `evaluation._validated_wait` |
| Equations 9-11; 46-48 | Minute-consistent charging and chronological tariff allocation | `planning._charge_allocation_cost`, `evaluation._charge_cost_and_duration` |
| Equations 12-18 | Physical arrival, service start, customer waiting, work/rest propagation and synchronized stops | `planning.propagate_nominal_route`, `evaluation._station_stop` |
| Equations 19-41 | Start/terminal depot copies, load/time/battery/partial-charge planning rules | `planning.py`, `milp_reference.py` |
| Equations 42-55 | Exclusive wait/backup/local-repair/penalty records and route-position recourse cost | `evaluation._simulate_planned_route` |
| Equations 56-63 | Planning cost, scenario-dependent components, expectation, CVaR and one-time penalty | `evaluation.evaluate_solution`, `weighted_cvar` |
| Equations 64-67 | Signed cost increment, unrecovered infeasibility and station attribution | `features.build_route_scenario_sample` |
| Equations 68-79; Table S8 | 24->128->64 shared encoder, masked mean, absolute-difference fusion, three heads and multi-task loss | `evaluator.TwinBranchNet`, `TwinBranchRiskEvaluator` |
| Table S8 | 65,091 twin and 65,235 single-branch parameter counts | `evaluator.py`, `tests/test_features_search.py` |
| Table S9 | Exact 24 stop-level nominal/perturbed inputs | `features.FEATURE_NAMES`, `route_stop_features` |
| Equations 80-86; Table S9b | Frozen hand-crafted risk score and station vulnerability | `evaluator.HeuristicRiskEvaluator` |
| Equations 87-89 | Route/candidate risk and station selection | evaluator `score_solution` / `station_diagnostics` |
| Table S7 | Destroy, repair, deterministic restoration, local search and targeted repair | `alns.py` |
| Equation 91 | Fixed 10th/50th/90th severity scenario subset | `Scenario.severity`, `select_severity_scenarios` |
| Supplementary Algorithm S1 | Candidate counter, threshold updates, safeguard, complete acceptance, rewards and stopping | `TPREALNS.solve` |
| Section 3.5.2; Equations 94-95 | 600-s deterministic first-stage nominal MILP | `milp_reference.py` |
| Section 3.5.3; Table S2 | 25/50/100 scales, 50/500 scenarios and costs/vehicle/rest parameters | `configs/default.yaml` |
| Section 3.5.1 | Base-instance-first aggregation and disjoint seed domains | `experiments.py`, `scripts/make_summary_tables.py` |

## Clarifications encoded in the implementation

- The fusion uses `|h_s - h_0|`. Table S8, Figure 4 and the explanatory sentence
  describe an absolute difference, even though one displayed equation omits the
  absolute-value bars.
- A long pre-dispatch hold at the start-depot copy is represented as an
  algorithmic depot rest. This reconciles the zero start-copy clock with late
  customer windows while preserving the rule that early customer waiting counts
  toward continuous work.
- Iterative waiting is committed only when it terminates at an available
  endpoint and remains downstream-feasible. Otherwise the evaluator records the
  terminal backup/repair/penalty action, preserving the one-action record in
  Equation 43.
- Targeted backup alternatives are scored by a counterfactual replacement route
  and then stored as an optional backup record; the dispatched primary route is
  unchanged.

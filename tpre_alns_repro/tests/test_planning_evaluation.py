from __future__ import annotations

import unittest

from tpre_alns.entities import Solution
from tpre_alns.evaluation import evaluate_solution, weighted_cvar
from tpre_alns.instance import generate_synthetic_instance
from tpre_alns.planning import certify_and_restore_solution
from tpre_alns.scenarios import generate_scenarios


class PlanningEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = generate_synthetic_instance(8, 3, seed=7)
        self.raw = Solution(
            routes=[
                [0, 1, 8, self.instance.terminal_id],
                [
                    0,
                    2,
                    4,
                    6,
                    5,
                    3,
                    self.instance.terminal_id,
                ],
                [0, 7, self.instance.terminal_id],
            ]
        )

    def test_distinct_depot_copies_and_complete_customer_service(self) -> None:
        planning = certify_and_restore_solution(self.instance, self.raw)
        self.assertTrue(planning.feasible, planning.reason)
        visited = []
        for route in planning.solution.routes:
            self.assertEqual(route[0], 0)
            self.assertEqual(route[-1], self.instance.terminal_id)
            self.assertNotEqual(route[0], route[-1])
            visited.extend(
                node for node in route if self.instance.is_customer(node)
            )
        self.assertEqual(sorted(visited), sorted(self.instance.customer_ids))

    def test_objective_separates_planning_cost_from_tail(self) -> None:
        planning = certify_and_restore_solution(self.instance, self.raw)
        scenarios = generate_scenarios(self.instance, 5, seed=44)
        metrics = evaluate_solution(
            self.instance,
            planning.solution,
            scenarios,
            alpha=0.90,
            risk_aversion=0.50,
            planning=planning,
        )
        self.assertAlmostEqual(
            metrics.total_cost,
            metrics.planning_cost + metrics.expected_scenario_cost,
            places=8,
        )
        self.assertAlmostEqual(
            metrics.objective,
            metrics.total_cost + 0.50 * metrics.cvar_scenario_cost,
            places=8,
        )

    def test_weighted_cvar_finite_distribution(self) -> None:
        # At alpha=0.5, the upper half of equal-probability {0, 10} is 10.
        self.assertAlmostEqual(weighted_cvar([0.0, 10.0], alpha=0.5), 10.0)
        self.assertAlmostEqual(
            weighted_cvar([0.0, 10.0], [0.9, 0.1], alpha=0.9),
            10.0,
        )

    def test_late_customer_is_handled_by_start_depot_hold(self) -> None:
        late_customer = max(
            self.instance.customers, key=lambda customer: customer.tw_start
        )
        planning = certify_and_restore_solution(
            self.instance,
            Solution(
                routes=[
                    [0, late_customer.node_id, self.instance.terminal_id]
                ]
            ),
            require_all_customers=False,
        )
        self.assertTrue(planning.feasible, planning.reason)
        self.assertGreater(
            planning.solution.planned_rests.get((0, 0), 0.0), 0.0
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from tpre_alns.instance import generate_synthetic_instance
from tpre_alns.scenarios import (
    generate_scenarios,
    time_of_use_price,
)


class GenerationTests(unittest.TestCase):
    def test_algorithm_s2_ranges_and_reproducibility(self) -> None:
        first = generate_synthetic_instance(25, 5, seed=123)
        second = generate_synthetic_instance(25, 5, seed=123)
        self.assertEqual(first.customers, second.customers)
        self.assertEqual(first.stations, second.stations)
        for customer in first.customers:
            self.assertGreaterEqual(customer.x, 0.0)
            self.assertLessEqual(customer.x, 100.0)
            self.assertGreaterEqual(customer.y, 0.0)
            self.assertLessEqual(customer.y, 100.0)
            self.assertIn(int(customer.demand), range(10, 51))
            self.assertGreaterEqual(customer.service_time, 5.0)
            self.assertLessEqual(customer.service_time, 15.0)
            width = customer.tw_end - customer.tw_start
            self.assertGreaterEqual(width, 60.0)
            self.assertLessEqual(width, 180.0)
            self.assertGreaterEqual(customer.tw_start, 0.0)
            self.assertLessEqual(customer.tw_end, 1080.0)
        for station in first.stations:
            self.assertIn(station.chargers, {4, 6, 8})
            self.assertIn(station.charging_power_kw, {60.0, 120.0})
            self.assertEqual(len(station.reported_unavailable), 18)
            self.assertTrue(
                all(
                    0 <= unavailable <= station.chargers
                    for unavailable in station.reported_unavailable
                )
            )

    def test_scenario_mutual_exclusivity_and_wait_rule(self) -> None:
        instance = generate_synthetic_instance(8, 3, seed=9)
        scenarios = generate_scenarios(
            instance, 7, setting="extreme", seed=77
        )
        for scenario in scenarios:
            for station in instance.stations:
                for interval in range(instance.n_intervals):
                    reported = station.reported_at(interval)
                    occupied = int(
                        scenario.occupation[station.node_id][interval]
                    )
                    hidden = int(
                        scenario.hidden_damage[station.node_id][interval]
                    )
                    available = int(
                        scenario.available_capacity[station.node_id][interval]
                    )
                    self.assertEqual(
                        available,
                        max(
                            station.chargers
                            - reported
                            - occupied
                            - hidden,
                            0,
                        ),
                    )
                    self.assertLessEqual(
                        reported + occupied + hidden, station.chargers
                    )
                    waiting = float(
                        scenario.waiting_time[station.node_id][interval]
                    )
                    is_occupied = (
                        available == 0
                        and reported + hidden < station.chargers
                    )
                    self.assertEqual(waiting > 0.0, is_occupied)

    def test_tariff_boundaries(self) -> None:
        self.assertEqual(time_of_use_price(6 * 60), 0.45)
        self.assertEqual(time_of_use_price(7 * 60), 0.75)
        self.assertEqual(time_of_use_price(17 * 60), 1.20)
        self.assertEqual(time_of_use_price(22 * 60), 0.45)


if __name__ == "__main__":
    unittest.main()

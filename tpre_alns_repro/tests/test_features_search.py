from __future__ import annotations

import unittest

import numpy as np

from tpre_alns.alns import ALNSConfig, TPREALNS
from tpre_alns.evaluator import (
    single_branch_parameter_count_formula,
    twin_parameter_count_formula,
)
from tpre_alns.features import (
    BINARY_FEATURE_INDICES,
    FeatureNormalizer,
    make_training_samples,
    route_stop_features,
)
from tpre_alns.instance import generate_synthetic_instance
from tpre_alns.scenarios import generate_scenarios


class FeatureSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = generate_synthetic_instance(8, 3, seed=7)
        self.scenarios = generate_scenarios(
            self.instance, 5, seed=1007
        )
        self.search = TPREALNS(
            self.instance,
            self.scenarios,
            ALNSConfig(max_iterations=2, max_no_improve=2),
            seed=1,
        )
        self.solution = self.search.initial_solution()

    def test_exact_feature_count_and_screening_semantics(self) -> None:
        nominal = route_stop_features(
            self.instance, self.solution, 0, None
        )
        perturbed = route_stop_features(
            self.instance, self.solution, 0, self.scenarios[0]
        )
        self.assertEqual(nominal.shape[1], 24)
        self.assertEqual(nominal.shape, perturbed.shape)
        # Planned arrival, battery and charge target are shared at screening.
        np.testing.assert_allclose(nominal[:, 8:11], perturbed[:, 8:11])
        # Only the scenario-state columns may differ.
        differing = np.flatnonzero(
            np.any(np.abs(nominal - perturbed) > 1e-8, axis=0)
        )
        self.assertTrue(set(differing).issubset({17, 18, 19, 20}))

    def test_binary_features_are_not_standardized(self) -> None:
        features = route_stop_features(
            self.instance, self.solution, 0, self.scenarios[0]
        )
        normalizer = FeatureNormalizer.fit([features])
        transformed = normalizer.transform(features)
        for index in BINARY_FEATURE_INDICES:
            np.testing.assert_array_equal(
                transformed[:, index], features[:, index]
            )

    def test_training_pool_returns_requested_feasible_samples(self) -> None:
        instance = generate_synthetic_instance(8, 3, seed=100)
        scenarios = generate_scenarios(instance, 2, seed=200)
        samples = make_training_samples(
            instance,
            scenarios,
            n_samples=4,
            seed=300,
        )
        self.assertEqual(len(samples), 4)
        self.assertTrue(
            all(
                sample.nominal_features.shape[1] == 24
                for sample in samples
            )
        )

    def test_reported_parameter_counts(self) -> None:
        self.assertEqual(twin_parameter_count_formula(), 65091)
        self.assertEqual(single_branch_parameter_count_formula(), 65235)

    def test_two_iteration_search_smoke(self) -> None:
        solution, information = self.search.solve()
        self.assertTrue(solution.routes)
        self.assertGreaterEqual(
            information["counters"]["complete_scenario_evaluations"], 1
        )
        self.assertIn("objective", information["metrics"])


if __name__ == "__main__":
    unittest.main()

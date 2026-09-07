from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tpre_repro as tr


def test_instance_and_scenario_generation_are_reproducible():
    a = tr.generate_instance(25, seed=123)
    b = tr.generate_instance(25, seed=123)
    # Ignore timestamp; all scientific content must be identical.
    a.pop("generated_at_utc")
    b.pop("generated_at_utc")
    assert a == b

    setting = tr.DEFAULT_SETTINGS["extreme_disruption"]
    s1 = tr.generate_scenario(a, 999, setting)
    s2 = tr.generate_scenario(a, 999, setting)
    assert s1 == s2


def test_state_components_sum_to_installed_capacity():
    instance = tr.generate_instance(25, seed=321)
    scenario = tr.generate_scenario(
        instance, 777, tr.DEFAULT_SETTINGS["high_occ_high_damage"]
    )
    tr.validate_scenario(instance, scenario)


def test_cvar_and_objective():
    costs = [1, 2, 3, 4, 10]
    assert tr.empirical_cvar(costs, alpha=0.8) == 10.0
    result = tr.risk_aware_objective(100, costs, risk_aversion_lambda=0.5, alpha=0.8)
    assert result["objective"] == 100 + np.mean(costs) + 0.5 * 10


def test_feature_statistics_leave_binary_columns_unchanged():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(8, 5, 24))
    features[..., 2] = rng.integers(0, 2, size=(8, 5))
    features[..., 3] = rng.integers(0, 2, size=(8, 5))
    stats = tr.fit_zscore_statistics(features)
    transformed = tr.apply_zscore(features, stats)
    assert np.array_equal(transformed[..., 2], features[..., 2])
    assert np.array_equal(transformed[..., 3], features[..., 3])


def test_twin_model_parameter_count_if_torch_available():
    if tr.torch is None:
        return
    model = tr.TwinBranchPerturbationRiskEvaluator()
    assert model.trainable_parameter_count == 65091

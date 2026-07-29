"""Installed command-line entry points."""

from __future__ import annotations


def demo_main() -> None:
    # Keep the installed entry point small; repository users normally invoke
    # scripts/run_demo.py so its output paths remain explicit.
    from tpre_alns.alns import ALNSConfig
    from tpre_alns.baselines import run_method
    from tpre_alns.instance import generate_synthetic_instance
    from tpre_alns.scenarios import generate_scenarios

    instance = generate_synthetic_instance(8, 3, seed=1)
    scenarios = generate_scenarios(instance, 5, seed=100001)
    _, information = run_method(
        "tpre_alns",
        instance,
        scenarios,
        base_config=ALNSConfig(max_iterations=20, max_no_improve=20),
        seed=1,
    )
    for key, value in information["metrics"].items():
        print(f"{key}: {value}")

"""Configuration loading, seeded experiment grids, and run-manifest creation."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .alns import ALNSConfig
from .baselines import config_for_method, run_method
from .evaluation import evaluate_solution
from .instance import generate_synthetic_instance
from .scenarios import generate_scenarios


def load_config(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency installation issue
        raise ImportError(
            "PyYAML is required to read experiment configs. "
            "Install the project with `pip install -e .`."
        ) from exc
    with Path(path).open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("The YAML root must be a mapping.")
    return loaded


def alns_config_from_dict(
    config: Mapping[str, Any], scale_name: str
) -> ALNSConfig:
    scale = config["instances"]["scales"][scale_name]
    alns = config["alns"]
    risk = config["risk"]
    local = config.get("local_search", {})
    targeted = config.get("targeted_repair", {})
    return ALNSConfig(
        max_iterations=int(scale["iterations"]),
        destroy_rate_min=float(alns["destroy_rate_min"]),
        destroy_rate_max=float(alns["destroy_rate_max"]),
        initial_temperature=float(alns["initial_temperature"]),
        cooling_rate=float(alns["cooling_rate"]),
        weight_update_period=int(alns["weight_update_period"]),
        high_risk_full_eval_probability=float(
            alns["high_risk_full_evaluation_probability"]
        ),
        max_no_improve=int(alns["max_no_improve"]),
        risk_threshold_percentile=float(risk["risk_threshold_percentile"]),
        alpha=float(risk["cvar_alpha"]),
        risk_aversion=float(risk["risk_aversion"]),
        risk_insertion_kappa=float(risk.get("insertion_kappa", 0.01)),
        risk_insertion_shortlist=int(risk.get("insertion_shortlist", 5)),
        local_search_probability=float(local.get("route_probability", 0.40)),
        target_failure_threshold=float(
            targeted.get("failure_threshold", 0.50)
        ),
        reaction_factor=float(alns.get("reaction_factor", 0.20)),
        minimum_operator_weight=float(
            alns.get("minimum_operator_weight", 0.05)
        ),
    )


def build_instance_from_config(
    config: Mapping[str, Any], scale_name: str, instance_seed: int
):
    scale = config["instances"]["scales"][scale_name]
    instance = generate_synthetic_instance(
        n_customers=int(scale["customers"]),
        n_stations=int(scale["stations"]),
        seed=instance_seed,
        name=f"{scale_name}_instance_seed{instance_seed}",
        planning_horizon_min=int(config["instances"]["planning_horizon_min"]),
        interval_min=int(config["instances"]["interval_min"]),
        reported_unavailable_probability=float(
            config["charging_station"][
                "reported_unavailable_probability"
            ]
        ),
    )
    vehicle = config["vehicle"]
    rest = config["rest"]
    costs = config["costs"]
    instance.vehicle_capacity = float(vehicle["load_capacity"])
    instance.battery_capacity = float(vehicle["battery_capacity_kwh"])
    instance.initial_battery = float(vehicle["initial_battery_kwh"])
    instance.safety_battery = float(vehicle["safety_battery_kwh"])
    instance.energy_consumption = float(
        vehicle["energy_consumption_kwh_per_km"]
    )
    instance.travel_speed = float(vehicle["travel_speed_km_per_min"])
    instance.max_continuous_work_min = float(
        rest["max_continuous_work_min"]
    )
    instance.min_rest_min = float(rest["min_rest_min"])
    instance.vehicle_use_cost = float(costs["vehicle_use_cost"])
    instance.travel_cost_per_km = float(costs["travel_cost_per_km"])
    instance.waiting_cost_per_min = float(costs["waiting_cost_per_hour"]) / 60.0
    instance.driver_cost_per_min = float(costs["driver_cost_per_hour"]) / 60.0
    instance.local_repair_fixed_cost = float(costs["local_repair_fixed_cost"])
    instance.infeasibility_penalty = float(costs["infeasibility_penalty"])
    return instance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment_grid(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    risk_evaluator=None,
    evaluator_checkpoint: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Run the configured grid and bind outputs to a reproducibility manifest."""
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    instance_seed_base = int(config["experiment"]["instance_seed_base"])
    optimization_seed_base = int(
        config["experiment"]["optimization_scenario_seed_base"]
    )
    out_of_sample_seed_base = int(
        config["experiment"]["out_of_sample_scenario_seed_base"]
    )
    run_seeds = [int(value) for value in config["experiment"]["run_seeds"]]
    base_instances = int(config["experiment"]["base_instances_per_scale"])
    settings = list(config["scenario_settings"])

    for scale_index, scale_name in enumerate(config["instances"]["scales"]):
        base_config = alns_config_from_dict(config, scale_name)
        for instance_index in range(base_instances):
            instance_seed = (
                instance_seed_base + 10_000 * scale_index + instance_index
            )
            instance = build_instance_from_config(
                config, scale_name, instance_seed
            )
            for setting_index, setting_name in enumerate(settings):
                setting = config["scenario_settings"][setting_name]
                optimization_seed = (
                    optimization_seed_base
                    + 100_000 * scale_index
                    + 1_000 * instance_index
                    + setting_index
                )
                out_of_sample_seed = (
                    out_of_sample_seed_base
                    + 100_000 * scale_index
                    + 1_000 * instance_index
                    + setting_index
                )
                if optimization_seed == out_of_sample_seed:
                    raise ValueError(
                        "Optimization and out-of-sample seeds must be disjoint."
                    )
                optimization_scenarios = generate_scenarios(
                    instance,
                    n_scenarios=int(
                        config["instances"]["optimization_scenarios"]
                    ),
                    setting=setting_name,
                    custom_setting=setting,
                    seed=optimization_seed,
                )
                reporting_scenarios = generate_scenarios(
                    instance,
                    n_scenarios=int(
                        config["instances"]["out_of_sample_scenarios"]
                    ),
                    setting=setting_name,
                    custom_setting=setting,
                    seed=out_of_sample_seed,
                )
                for method in config["experiment"]["methods"]:
                    for run_seed in run_seeds:
                        solution, information = run_method(
                            method,
                            instance,
                            optimization_scenarios,
                            base_config=base_config,
                            risk_evaluator=risk_evaluator,
                            seed=run_seed,
                        )
                        method_config = config_for_method(method, base_config)
                        optimization_metrics = dict(information["metrics"])
                        reporting_metrics = evaluate_solution(
                            instance,
                            solution,
                            reporting_scenarios,
                            alpha=base_config.alpha,
                            risk_aversion=(
                                method_config.risk_aversion
                                if method_config.use_cvar
                                else 0.0
                            ),
                            use_backups=method_config.use_backups,
                            rest_sync=method_config.rest_sync,
                        ).to_dict()
                        row: Dict[str, Any] = {
                            "experiment_id": config["experiment"]["name"],
                            "scale": scale_name,
                            "base_instance_index": instance_index,
                            "instance_seed": instance_seed,
                            "uncertainty_setting": setting_name,
                            "optimization_scenario_seed": optimization_seed,
                            "out_of_sample_scenario_seed": out_of_sample_seed,
                            "method": method,
                            "run_seed": run_seed,
                            "screening_proxy": information[
                                "screening_proxy"
                            ],
                        }
                        row.update(
                            {
                                f"optimization_{key}": value
                                for key, value in optimization_metrics.items()
                            }
                        )
                        row.update(
                            {
                                f"oos_{key}": value
                                for key, value in reporting_metrics.items()
                            }
                        )
                        row.update(
                            {
                                f"count_{key}": value
                                for key, value in information[
                                    "counters"
                                ].items()
                            }
                        )
                        rows.append(row)
                        _write_rows(destination / "run_metrics.csv", rows)

    checkpoint_path = (
        Path(evaluator_checkpoint).resolve()
        if evaluator_checkpoint is not None
        else None
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "evaluator_checkpoint": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        "evaluator_checkpoint_sha256": (
            _sha256(checkpoint_path)
            if checkpoint_path is not None and checkpoint_path.exists()
            else None
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "row_count": len(rows),
        "seed_policy": {
            "instance_seed_base": instance_seed_base,
            "optimization_scenario_seed_base": optimization_seed_base,
            "out_of_sample_scenario_seed_base": out_of_sample_seed_base,
            "run_seeds": run_seeds,
        },
        "config": config,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows

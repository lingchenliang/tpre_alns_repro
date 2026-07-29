"""Charging-station scenario generation and deterministic scenario I/O."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .entities import EVRPInstance, Scenario


UNCERTAINTY_SETTINGS: Dict[str, Dict[str, float]] = {
    "low_occ_low_damage": {
        "occupation_probability": 0.25,
        "hidden_damage_probability": 0.01,
        "waiting_min": 5.0,
        "waiting_max": 20.0,
    },
    "high_occ_low_damage": {
        "occupation_probability": 0.65,
        "hidden_damage_probability": 0.01,
        "waiting_min": 20.0,
        "waiting_max": 50.0,
    },
    "low_occ_high_damage": {
        "occupation_probability": 0.25,
        "hidden_damage_probability": 0.06,
        "waiting_min": 5.0,
        "waiting_max": 20.0,
    },
    "high_occ_high_damage": {
        "occupation_probability": 0.65,
        "hidden_damage_probability": 0.06,
        "waiting_min": 20.0,
        "waiting_max": 50.0,
    },
    "extreme": {
        "occupation_probability": 0.80,
        "hidden_damage_probability": 0.10,
        "waiting_min": 35.0,
        "waiting_max": 75.0,
    },
}


def time_of_use_price(absolute_clock_min: float) -> float:
    """Return the fixed valley/flat/peak tariff in CU/kWh."""
    hour = int(absolute_clock_min // 60) % 24
    if 0 <= hour < 7:
        return 0.45
    if 7 <= hour < 17:
        return 0.75
    if 17 <= hour < 22:
        return 1.20
    return 0.45


def tariff_vector(inst: EVRPInstance) -> np.ndarray:
    return np.asarray(
        [
            time_of_use_price(
                inst.absolute_clock_min(
                    inst.planning_start_min + interval * inst.interval_min
                )
            )
            for interval in range(inst.n_intervals)
        ],
        dtype=float,
    )


def generate_scenarios(
    inst: EVRPInstance,
    n_scenarios: int = 50,
    setting: str = "high_occ_high_damage",
    seed: int = 1,
    custom_setting: Optional[Mapping[str, float]] = None,
) -> List[Scenario]:
    """Generate mutually exclusive O_jts/H_jts outcomes.

    Occupation is drawn from ``n_j - D_jt`` chargers.  Hidden damage is then
    drawn from the remaining, non-occupied chargers.  A queue-delay draw is
    stored only for an occupied endpoint state and is never resampled by an
    algorithm during evaluation.
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")
    if custom_setting is None:
        if setting not in UNCERTAINTY_SETTINGS:
            raise KeyError(f"Unknown uncertainty setting: {setting}")
        parameters = dict(UNCERTAINTY_SETTINGS[setting])
    else:
        parameters = {key: float(value) for key, value in custom_setting.items()}
    required = {
        "occupation_probability",
        "hidden_damage_probability",
        "waiting_min",
        "waiting_max",
    }
    if not required.issubset(parameters):
        raise ValueError(
            f"Scenario setting is missing: {sorted(required - set(parameters))}"
        )

    tariff = tariff_vector(inst)
    scenario_seeds = np.random.SeedSequence(seed).spawn(n_scenarios)
    probability = 1.0 / n_scenarios
    generated: List[Scenario] = []

    for scenario_index, scenario_seed in enumerate(scenario_seeds):
        rng = np.random.default_rng(scenario_seed)
        occupation: Dict[int, np.ndarray] = {}
        hidden_damage: Dict[int, np.ndarray] = {}
        available_capacity: Dict[int, np.ndarray] = {}
        waiting_time: Dict[int, np.ndarray] = {}
        price: Dict[int, np.ndarray] = {}

        for station in inst.stations:
            occ = np.zeros(inst.n_intervals, dtype=np.int16)
            hidden = np.zeros(inst.n_intervals, dtype=np.int16)
            capacity = np.zeros(inst.n_intervals, dtype=np.int16)
            queue_delay = np.zeros(inst.n_intervals, dtype=np.float64)
            for interval in range(inst.n_intervals):
                reported = station.reported_at(interval)
                reported_usable_pool = max(station.chargers - reported, 0)
                occ[interval] = int(
                    rng.binomial(
                        reported_usable_pool,
                        parameters["occupation_probability"],
                    )
                )
                remaining_pool = max(
                    reported_usable_pool - int(occ[interval]), 0
                )
                hidden[interval] = int(
                    rng.binomial(
                        remaining_pool,
                        parameters["hidden_damage_probability"],
                    )
                )
                capacity[interval] = max(
                    station.chargers
                    - reported
                    - int(occ[interval])
                    - int(hidden[interval]),
                    0,
                )
                is_occupied = (
                    capacity[interval] == 0
                    and reported + int(hidden[interval]) < station.chargers
                )
                if is_occupied:
                    queue_delay[interval] = float(
                        rng.uniform(
                            parameters["waiting_min"],
                            parameters["waiting_max"],
                        )
                    )

            station_id = station.node_id
            occupation[station_id] = occ
            hidden_damage[station_id] = hidden
            available_capacity[station_id] = capacity
            waiting_time[station_id] = queue_delay
            price[station_id] = tariff.copy()

        generated.append(
            Scenario(
                name=f"{setting}_{scenario_index:04d}",
                probability=probability,
                occupation=occupation,
                hidden_damage=hidden_damage,
                available_capacity=available_capacity,
                waiting_time=waiting_time,
                price=price,
                setting=setting,
                seed=int(scenario_seed.generate_state(1, dtype=np.uint32)[0]),
            )
        )
    return generated


def normalized_probabilities(scenarios: Sequence[Scenario]) -> np.ndarray:
    if not scenarios:
        return np.ones(1, dtype=float)
    probabilities = np.asarray(
        [max(float(scenario.probability), 0.0) for scenario in scenarios],
        dtype=float,
    )
    total = float(probabilities.sum())
    if total <= 0:
        return np.full(len(scenarios), 1.0 / len(scenarios), dtype=float)
    return probabilities / total


def select_severity_scenarios(
    inst: EVRPInstance,
    scenarios: Sequence[Scenario],
    percentiles: Sequence[float] = (10.0, 50.0, 90.0),
) -> List[Scenario]:
    """Select the fixed local-search subset with deterministic tie breaking."""
    if not scenarios:
        return []
    ordered = sorted(scenarios, key=lambda sc: (sc.severity(inst), sc.name))
    selected: List[Scenario] = []
    for percentile in percentiles:
        index = int(round((len(ordered) - 1) * float(percentile) / 100.0))
        selected.append(ordered[int(np.clip(index, 0, len(ordered) - 1))])
    return selected


def scenarios_to_dict(scenarios: Sequence[Scenario]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "scenarios": [
            {
                "name": scenario.name,
                "setting": scenario.setting,
                "seed": scenario.seed,
                "probability": scenario.probability,
                "occupation": {
                    str(key): value.tolist()
                    for key, value in scenario.occupation.items()
                },
                "hidden_damage": {
                    str(key): value.tolist()
                    for key, value in scenario.hidden_damage.items()
                },
                "available_capacity": {
                    str(key): value.tolist()
                    for key, value in scenario.available_capacity.items()
                },
                "waiting_time": {
                    str(key): value.tolist()
                    for key, value in scenario.waiting_time.items()
                },
                "price": {
                    str(key): value.tolist()
                    for key, value in scenario.price.items()
                },
            }
            for scenario in scenarios
        ],
    }


def save_scenarios_json(
    scenarios: Sequence[Scenario], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        scenarios_to_dict(scenarios), ensure_ascii=False, separators=(",", ":")
    )
    if destination.suffix == ".gz":
        with gzip.open(destination, "wt", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        destination.write_text(payload, encoding="utf-8")


def load_scenarios_json(path: str | Path) -> List[Scenario]:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = json.loads(source.read_text(encoding="utf-8"))

    scenarios: List[Scenario] = []
    for row in data["scenarios"]:
        convert_int = lambda mapping, dtype: {
            int(key): np.asarray(values, dtype=dtype)
            for key, values in mapping.items()
        }
        scenarios.append(
            Scenario(
                name=row["name"],
                setting=row.get("setting", ""),
                seed=int(row.get("seed", 0)),
                probability=float(row["probability"]),
                occupation=convert_int(row["occupation"], np.int16),
                hidden_damage=convert_int(row["hidden_damage"], np.int16),
                available_capacity=convert_int(
                    row["available_capacity"], np.int16
                ),
                waiting_time=convert_int(row["waiting_time"], np.float64),
                price=convert_int(row["price"], np.float64),
            )
        )
    return scenarios

#!/usr/bin/env python3
"""Reference data generator and model definition for TPRE-ALNS research.

This module implements the parts of the manuscript that can be reconstructed
unambiguously from the written specification:

1. Synthetic VRPTW-style electric-delivery instance generation.
2. Charging-station scenario generation with reported unavailability,
   external occupation, hidden damage, residual capacity, and fixed queue delay.
3. A 24-feature schema and training-only z-score helpers.
4. The twin-branch perturbation-risk evaluator architecture
   (65,091 trainable parameters when PyTorch is available).
5. CVaR and risk-aware objective helpers.

It is a clean-room reference implementation. It does NOT contain the authors'
original ALNS implementation, fixed-rule route-recourse simulator, trained
checkpoint, or run-level outputs underlying the manuscript tables. Do not
present newly generated examples as the original experimental data.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:  # The data generator works without PyTorch.
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


SCHEMA_VERSION = "1.0.0"
DEFAULT_INTERVAL_MINUTES = 60
DEFAULT_HORIZON_MINUTES = 1080  # 06:00-24:00

FEATURE_NAMES: tuple[str, ...] = (
    "x_coordinate",
    "y_coordinate",
    "start_or_terminal_depot_indicator",
    "station_indicator",
    "customer_demand",
    "service_time",
    "ready_time",
    "due_time",
    "planned_arrival_time",
    "planned_battery_state",
    "planned_charge_amount",
    "stop_position",
    "route_length",
    "installed_chargers",
    "charging_power_kw",
    "nominal_tariff_cu_per_kwh",
    "reported_unavailable_share",
    "occupied_charger_share",
    "hidden_damage_share",
    "residual_capacity_share",
    "base_queue_delay_minutes",
    "incoming_arc_distance_km",
    "incoming_arc_travel_time_minutes",
    "incoming_arc_energy_kwh",
)

BINARY_FEATURE_INDICES: tuple[int, int] = (2, 3)


@dataclass(frozen=True)
class TariffBand:
    """One half-open tariff band, measured from the 06:00 start clock."""

    start_minute: int
    end_minute: int
    label: str
    price_cu_per_kwh: float

    def validate(self, horizon_minutes: int) -> None:
        if not (0 <= self.start_minute < self.end_minute <= horizon_minutes):
            raise ValueError(f"Invalid tariff band: {self}")
        if self.price_cu_per_kwh < 0:
            raise ValueError("Tariff price must be non-negative")


@dataclass(frozen=True)
class UncertaintySetting:
    """Per-charger station-state probabilities and queue-delay range."""

    name: str
    occupation_probability: float
    hidden_damage_probability: float
    queue_delay_min_minutes: float
    queue_delay_max_minutes: float

    def validate(self) -> None:
        for name, value in (
            ("occupation_probability", self.occupation_probability),
            ("hidden_damage_probability", self.hidden_damage_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if not 0 <= self.queue_delay_min_minutes <= self.queue_delay_max_minutes:
            raise ValueError("Invalid queue-delay range")


DEFAULT_SETTINGS: dict[str, UncertaintySetting] = {
    "low_occ_low_damage": UncertaintySetting(
        "low_occ_low_damage", 0.25, 0.01, 5.0, 20.0
    ),
    "high_occ_low_damage": UncertaintySetting(
        "high_occ_low_damage", 0.65, 0.01, 20.0, 50.0
    ),
    "low_occ_high_damage": UncertaintySetting(
        "low_occ_high_damage", 0.25, 0.06, 5.0, 20.0
    ),
    "high_occ_high_damage": UncertaintySetting(
        "high_occ_high_damage", 0.65, 0.06, 20.0, 50.0
    ),
    "extreme_disruption": UncertaintySetting(
        "extreme_disruption", 0.80, 0.10, 35.0, 75.0
    ),
}


@dataclass(frozen=True)
class GenerationConfig:
    """Machine-readable version of the manuscript's synthetic-data settings."""

    coordinate_min_km: float = 0.0
    coordinate_max_km: float = 100.0
    depot_x_km: float = 50.0
    depot_y_km: float = 50.0
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES

    customer_demand_min_kg: int = 10
    customer_demand_max_kg: int = 50
    service_time_min_minutes: float = 5.0
    service_time_max_minutes: float = 15.0
    time_window_width_min_minutes: float = 60.0
    time_window_width_max_minutes: float = 180.0

    station_count_by_customers: Mapping[int, int] = field(
        default_factory=lambda: {25: 5, 50: 8, 100: 12}
    )
    charger_count_choices: tuple[int, ...] = (4, 6, 8)
    charger_count_probabilities: tuple[float, ...] = (0.30, 0.40, 0.30)
    charging_power_choices_kw: tuple[int, ...] = (60, 120)
    charging_power_probabilities: tuple[float, ...] = (0.50, 0.50)
    reported_unavailable_probability: float = 0.05

    speed_km_per_minute: float = 0.65
    energy_kwh_per_km: float = 0.24
    vehicle_capacity_kg: float = 1000.0
    battery_capacity_kwh: float = 80.0
    initial_battery_kwh: float = 80.0
    safety_battery_kwh: float = 8.0
    max_continuous_work_minutes: float = 240.0
    minimum_rest_minutes: float = 30.0

    travel_cost_cu_per_km: float = 1.20
    waiting_cost_cu_per_hour: float = 30.0
    driver_cost_cu_per_hour: float = 25.0
    vehicle_fixed_cost_cu: float = 100.0
    local_repair_fixed_cost_cu: float = 50.0
    infeasibility_penalty_cu: float = 10_000.0
    cvar_alpha: float = 0.90
    risk_aversion_lambda: float = 0.50
    route_risk_mu_cu: float = 100.0
    route_risk_nu_cu: float = 25.0

    # The manuscript gives the three prices but not a machine-readable list of
    # band boundaries. These default bands mirror the schematic. Replace them
    # if the original experimental implementation used different boundaries.
    tariff_bands: tuple[TariffBand, ...] = (
        TariffBand(0, 120, "valley", 0.45),       # 06:00-08:00
        TariffBand(120, 360, "flat", 0.75),       # 08:00-12:00
        TariffBand(360, 720, "peak", 1.20),       # 12:00-18:00
        TariffBand(720, 960, "flat", 0.75),       # 18:00-22:00
        TariffBand(960, 1080, "valley", 0.45),    # 22:00-24:00
    )

    def validate(self) -> None:
        if self.coordinate_min_km >= self.coordinate_max_km:
            raise ValueError("Coordinate minimum must be below maximum")
        if self.horizon_minutes <= 0 or self.interval_minutes <= 0:
            raise ValueError("Time settings must be positive")
        if self.horizon_minutes % self.interval_minutes != 0:
            raise ValueError("Horizon must be divisible by interval length")
        if not math.isclose(sum(self.charger_count_probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("Charger-count probabilities must sum to 1")
        if not math.isclose(sum(self.charging_power_probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("Charging-power probabilities must sum to 1")
        if set(self.station_count_by_customers) != {25, 50, 100}:
            raise ValueError("The reference configuration expects scales 25, 50, and 100")
        if self.initial_battery_kwh > self.battery_capacity_kwh:
            raise ValueError("Initial battery cannot exceed capacity")
        if not 0 <= self.safety_battery_kwh <= self.battery_capacity_kwh:
            raise ValueError("Invalid safety battery")
        if not 0.0 <= self.reported_unavailable_probability <= 1.0:
            raise ValueError("Invalid reported-unavailable probability")
        for setting in DEFAULT_SETTINGS.values():
            setting.validate()
        self._validate_tariffs()

    def _validate_tariffs(self) -> None:
        bands = sorted(self.tariff_bands, key=lambda b: b.start_minute)
        for band in bands:
            band.validate(self.horizon_minutes)
        cursor = 0
        for band in bands:
            if band.start_minute != cursor:
                raise ValueError("Tariff bands must cover the horizon without gaps")
            cursor = band.end_minute
        if cursor != self.horizon_minutes:
            raise ValueError("Tariff bands must end at the planning horizon")

    @property
    def interval_count(self) -> int:
        return self.horizon_minutes // self.interval_minutes

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["station_count_by_customers"] = {
            str(k): v for k, v in self.station_count_by_customers.items()
        }
        return value


@dataclass
class ZScoreStatistics:
    """Training-partition normalization statistics for the 24 features."""

    mean: list[float]
    std: list[float]
    binary_indices: tuple[int, ...] = BINARY_FEATURE_INDICES

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ZScoreStatistics":
        return cls(
            mean=[float(x) for x in value["mean"]],
            std=[float(x) for x in value["std"]],
            binary_indices=tuple(int(x) for x in value.get("binary_indices", (2, 3))),
        )


def _rng(seed: int) -> np.random.Generator:
    if seed < 0:
        raise ValueError("Seed must be non-negative")
    return np.random.default_rng(seed)


def _round(value: float, digits: int = 6) -> float:
    return float(round(float(value), digits))


def _tariff_for_minute(config: GenerationConfig, minute: float) -> TariffBand:
    bounded = min(max(float(minute), 0.0), math.nextafter(config.horizon_minutes, 0.0))
    for band in config.tariff_bands:
        if band.start_minute <= bounded < band.end_minute:
            return band
    raise RuntimeError(f"No tariff band for minute {minute}")


def _distance_matrices(nodes: Sequence[Mapping[str, Any]], config: GenerationConfig) -> dict[str, Any]:
    coords = np.asarray([[float(n["x_km"]), float(n["y_km"])] for n in nodes])
    diff = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    travel = distance / config.speed_km_per_minute
    energy = distance * config.energy_kwh_per_km
    return {
        "node_order": [str(n["node_id"]) for n in nodes],
        "distance_km": np.round(distance, 6).tolist(),
        "travel_time_minutes": np.round(travel, 6).tolist(),
        "energy_kwh": np.round(energy, 6).tolist(),
    }


def generate_instance(
    customer_count: int,
    seed: int,
    config: GenerationConfig | None = None,
) -> dict[str, Any]:
    """Generate one synthetic electric-delivery instance.

    The implementation follows Algorithm B in the supporting information.
    Reported-unavailable charger counts are drawn once per station and interval
    and stored in the instance because they are known before dispatch.
    """

    config = config or GenerationConfig()
    config.validate()
    if customer_count not in config.station_count_by_customers:
        raise ValueError(
            f"customer_count must be one of {sorted(config.station_count_by_customers)}"
        )

    rng = _rng(seed)
    station_count = int(config.station_count_by_customers[customer_count])

    customers: list[dict[str, Any]] = []
    for customer_id in range(1, customer_count + 1):
        x, y = rng.uniform(config.coordinate_min_km, config.coordinate_max_km, size=2)
        demand = int(rng.integers(config.customer_demand_min_kg, config.customer_demand_max_kg + 1))
        service = float(
            rng.uniform(config.service_time_min_minutes, config.service_time_max_minutes)
        )
        width = float(
            rng.uniform(
                config.time_window_width_min_minutes,
                config.time_window_width_max_minutes,
            )
        )
        ready = float(rng.uniform(0.0, config.horizon_minutes - width))
        due = ready + width
        customers.append(
            {
                "node_id": f"C{customer_id}",
                "node_type": "customer",
                "customer_index": customer_id,
                "x_km": _round(x),
                "y_km": _round(y),
                "demand_kg": demand,
                "service_minutes": _round(service),
                "ready_time_minutes": _round(ready),
                "due_time_minutes": _round(due),
                "time_window_width_minutes": _round(width),
            }
        )

    stations: list[dict[str, Any]] = []
    for station_index in range(1, station_count + 1):
        x, y = rng.uniform(config.coordinate_min_km, config.coordinate_max_km, size=2)
        chargers = int(
            rng.choice(config.charger_count_choices, p=config.charger_count_probabilities)
        )
        power = int(
            rng.choice(
                config.charging_power_choices_kw,
                p=config.charging_power_probabilities,
            )
        )
        reported = rng.binomial(
            n=chargers,
            p=config.reported_unavailable_probability,
            size=config.interval_count,
        ).astype(int)
        stations.append(
            {
                "node_id": f"S{station_index}",
                "node_type": "charging_station",
                "station_index": station_index,
                "x_km": _round(x),
                "y_km": _round(y),
                "installed_chargers": chargers,
                "charging_power_kw": power,
                "charging_rate_kwh_per_minute": _round(power / 60.0),
                "reported_unavailable_by_interval": reported.tolist(),
            }
        )

    depots = [
        {
            "node_id": "D_START",
            "node_type": "start_depot",
            "x_km": config.depot_x_km,
            "y_km": config.depot_y_km,
        },
        {
            "node_id": "D_END",
            "node_type": "terminal_depot",
            "x_km": config.depot_x_km,
            "y_km": config.depot_y_km,
        },
    ]
    ordered_nodes: list[dict[str, Any]] = [depots[0], *customers, *stations, depots[1]]

    intervals: list[dict[str, Any]] = []
    for interval_index in range(config.interval_count):
        start = interval_index * config.interval_minutes
        end = start + config.interval_minutes
        tariff = _tariff_for_minute(config, start)
        intervals.append(
            {
                "interval_index": interval_index,
                "start_minute": start,
                "end_minute": end,
                "tariff_label": tariff.label,
                "tariff_cu_per_kwh": tariff.price_cu_per_kwh,
            }
        )

    instance = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": f"evrptw_n{customer_count}_seed{seed}",
        "instance_seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_scope": "synthetic benchmark instance; not field-calibrated",
        "customer_count": customer_count,
        "station_count": station_count,
        "vehicle_upper_bound": customer_count,
        "depots": depots,
        "customers": customers,
        "stations": stations,
        "time_intervals": intervals,
        "vehicle_parameters": {
            "load_capacity_kg": config.vehicle_capacity_kg,
            "battery_capacity_kwh": config.battery_capacity_kwh,
            "initial_battery_kwh": config.initial_battery_kwh,
            "safety_battery_kwh": config.safety_battery_kwh,
            "max_continuous_work_minutes": config.max_continuous_work_minutes,
            "minimum_rest_minutes": config.minimum_rest_minutes,
        },
        "transport_parameters": {
            "speed_km_per_minute": config.speed_km_per_minute,
            "energy_kwh_per_km": config.energy_kwh_per_km,
        },
        "cost_parameters": {
            "travel_cost_cu_per_km": config.travel_cost_cu_per_km,
            "waiting_cost_cu_per_hour": config.waiting_cost_cu_per_hour,
            "driver_cost_cu_per_hour": config.driver_cost_cu_per_hour,
            "vehicle_fixed_cost_cu": config.vehicle_fixed_cost_cu,
            "local_repair_fixed_cost_cu": config.local_repair_fixed_cost_cu,
            "infeasibility_penalty_cu": config.infeasibility_penalty_cu,
        },
        "risk_parameters": {
            "cvar_alpha": config.cvar_alpha,
            "risk_aversion_lambda": config.risk_aversion_lambda,
            "route_risk_mu_cu": config.route_risk_mu_cu,
            "route_risk_nu_cu": config.route_risk_nu_cu,
        },
        "distance_matrices": _distance_matrices(ordered_nodes, config),
        "configuration": config.to_dict(),
    }
    validate_instance(instance)
    return instance


def classify_station_state(
    installed: int,
    reported_unavailable: int,
    occupied: int,
    hidden_damaged: int,
) -> tuple[str, int]:
    """Return (state, residual_capacity) using the manuscript state logic."""

    residual = installed - reported_unavailable - occupied - hidden_damaged
    if residual < 0:
        raise ValueError("Charger-state counts exceed installed capacity")
    if residual > 0:
        return "available", residual
    if occupied > 0:
        return "occupied", residual
    return "failed", residual


def generate_scenario(
    instance: Mapping[str, Any],
    scenario_seed: int,
    setting: UncertaintySetting,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Generate one station-state scenario conditional on an instance."""

    setting.validate()
    rng = _rng(scenario_seed)
    interval_count = len(instance["time_intervals"])
    station_records: list[dict[str, Any]] = []

    for station in instance["stations"]:
        installed = int(station["installed_chargers"])
        reported_values = station["reported_unavailable_by_interval"]
        if len(reported_values) != interval_count:
            raise ValueError("Reported-unavailability vector has wrong length")

        interval_records: list[dict[str, Any]] = []
        for interval_index, reported_value in enumerate(reported_values):
            reported = int(reported_value)
            reported_usable_pool = installed - reported
            occupied = int(
                rng.binomial(reported_usable_pool, setting.occupation_probability)
            )
            hidden_pool = reported_usable_pool - occupied
            hidden = int(
                rng.binomial(hidden_pool, setting.hidden_damage_probability)
            )
            state, residual = classify_station_state(
                installed, reported, occupied, hidden
            )
            queue_delay = (
                float(
                    rng.uniform(
                        setting.queue_delay_min_minutes,
                        setting.queue_delay_max_minutes,
                    )
                )
                if state == "occupied"
                else 0.0
            )
            interval_records.append(
                {
                    "interval_index": interval_index,
                    "reported_unavailable": reported,
                    "externally_occupied": occupied,
                    "hidden_damaged": hidden,
                    "residual_available": residual,
                    "state": state,
                    "base_queue_delay_minutes": _round(queue_delay),
                }
            )
        station_records.append(
            {
                "station_id": station["node_id"],
                "intervals": interval_records,
            }
        )

    scenario = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance["instance_id"],
        "scenario_id": scenario_id or f"{setting.name}_seed{scenario_seed}",
        "scenario_seed": scenario_seed,
        "uncertainty_setting": dataclasses.asdict(setting),
        "station_states": station_records,
    }
    validate_scenario(instance, scenario)
    return scenario


def validate_instance(instance: Mapping[str, Any]) -> None:
    customer_count = int(instance["customer_count"])
    if len(instance["customers"]) != customer_count:
        raise ValueError("Customer count mismatch")
    if len(instance["stations"]) != int(instance["station_count"]):
        raise ValueError("Station count mismatch")
    for customer in instance["customers"]:
        ready = float(customer["ready_time_minutes"])
        due = float(customer["due_time_minutes"])
        if not (0 <= ready < due <= DEFAULT_HORIZON_MINUTES + 1e-6):
            raise ValueError(f"Invalid customer time window: {customer}")
        demand = int(customer["demand_kg"])
        if not 10 <= demand <= 50:
            raise ValueError("Customer demand outside the reference range")
    for station in instance["stations"]:
        installed = int(station["installed_chargers"])
        for unavailable in station["reported_unavailable_by_interval"]:
            if not 0 <= int(unavailable) <= installed:
                raise ValueError("Invalid reported-unavailable count")


def validate_scenario(instance: Mapping[str, Any], scenario: Mapping[str, Any]) -> None:
    if scenario["instance_id"] != instance["instance_id"]:
        raise ValueError("Scenario instance_id mismatch")
    station_lookup = {s["node_id"]: s for s in instance["stations"]}
    for station_record in scenario["station_states"]:
        station = station_lookup[station_record["station_id"]]
        installed = int(station["installed_chargers"])
        for record in station_record["intervals"]:
            total = (
                int(record["reported_unavailable"])
                + int(record["externally_occupied"])
                + int(record["hidden_damaged"])
                + int(record["residual_available"])
            )
            if total != installed:
                raise ValueError("Station-state components do not sum to capacity")
            expected_state, residual = classify_station_state(
                installed,
                int(record["reported_unavailable"]),
                int(record["externally_occupied"]),
                int(record["hidden_damaged"]),
            )
            if expected_state != record["state"] or residual != int(
                record["residual_available"]
            ):
                raise ValueError("Station state classification mismatch")
            if record["state"] != "occupied" and float(
                record["base_queue_delay_minutes"]
            ) != 0.0:
                raise ValueError("Queue delay must be zero outside occupied states")


def physical_scale_features(features: np.ndarray) -> np.ndarray:
    """Apply the physical pre-scaling listed in the feature table.

    Input shape may be (..., 24). This function does not z-score the features.
    """

    values = np.asarray(features, dtype=np.float64).copy()
    if values.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features")

    values[..., 0] /= 100.0  # x coordinate
    values[..., 1] /= 100.0  # y coordinate
    values[..., 4] /= 1000.0  # demand / Q
    values[..., 5] /= 15.0  # service / s_max
    values[..., 6] /= DEFAULT_HORIZON_MINUTES
    values[..., 7] /= DEFAULT_HORIZON_MINUTES
    values[..., 8] /= DEFAULT_HORIZON_MINUTES
    values[..., 9] /= 80.0  # battery / B
    values[..., 10] /= 80.0  # charge / B
    # stop position and route length remain explicit counts before z-score
    values[..., 13] /= 8.0  # installed chargers / max installed
    values[..., 14] /= 120.0  # power / max power
    values[..., 15] /= 1.20  # tariff / peak tariff
    # Shares at 16-19 are already within [0,1].
    values[..., 20] /= 75.0  # queue delay / extreme max
    # Arc distance/travel/energy are left in physical units before z-score.
    return values


def fit_zscore_statistics(
    training_features: np.ndarray,
    binary_indices: Sequence[int] = BINARY_FEATURE_INDICES,
) -> ZScoreStatistics:
    """Fit z-score statistics on training data only.

    Binary features are left unchanged by setting mean=0 and std=1.
    """

    values = np.asarray(training_features, dtype=np.float64)
    if values.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features")
    flattened = values.reshape(-1, values.shape[-1])
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std[std < 1e-12] = 1.0
    for index in binary_indices:
        mean[index] = 0.0
        std[index] = 1.0
    return ZScoreStatistics(mean=mean.tolist(), std=std.tolist(), binary_indices=tuple(binary_indices))


def apply_zscore(features: np.ndarray, statistics: ZScoreStatistics) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.shape[-1] != len(statistics.mean):
        raise ValueError("Feature/statistics dimension mismatch")
    result = (values - np.asarray(statistics.mean)) / np.asarray(statistics.std)
    for index in statistics.binary_indices:
        result[..., index] = values[..., index]
    return result


def empirical_cvar(costs: Sequence[float], alpha: float = 0.90) -> float:
    """Return empirical upper-tail CVaR using the mean of the worst tail."""

    values = np.asarray(costs, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one scenario cost is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    sorted_values = np.sort(values)
    tail_count = max(1, int(math.ceil((1.0 - alpha) * values.size)))
    return float(sorted_values[-tail_count:].mean())


def risk_aware_objective(
    planning_cost: float,
    scenario_dependent_costs: Sequence[float],
    risk_aversion_lambda: float = 0.50,
    alpha: float = 0.90,
) -> dict[str, float]:
    values = np.asarray(scenario_dependent_costs, dtype=np.float64)
    expected = float(values.mean())
    cvar = empirical_cvar(values, alpha)
    total = float(planning_cost + expected + risk_aversion_lambda * cvar)
    return {
        "planning_cost": float(planning_cost),
        "expected_scenario_dependent_cost": expected,
        "scenario_dependent_cost_cvar": cvar,
        "risk_aversion_lambda": float(risk_aversion_lambda),
        "objective": total,
    }


def route_risk_score(
    predicted_cost_increment: float,
    infeasibility_probability: float,
    station_vulnerabilities: Sequence[float],
    mu_cu: float = 100.0,
    nu_cu: float = 25.0,
) -> float:
    if not 0.0 <= infeasibility_probability <= 1.0:
        raise ValueError("Infeasibility probability must be within [0,1]")
    return float(
        predicted_cost_increment
        + mu_cu * infeasibility_probability
        + nu_cu * float(np.sum(station_vulnerabilities))
    )


def pairwise_hinge_loss_numpy(
    scores: Sequence[float],
    targets: Sequence[float],
    margin: float = 1.0,
) -> float:
    """Simple pairwise ranking loss for diagnostics and unit tests."""

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if s.shape != y.shape:
        raise ValueError("scores and targets must have the same shape")
    losses: list[float] = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if math.isclose(float(y[i]), float(y[j])):
                continue
            sign = 1.0 if y[i] > y[j] else -1.0
            losses.append(max(0.0, margin - sign * float(s[i] - s[j])))
    return float(np.mean(losses)) if losses else 0.0


if torch is not None:

    class TwinBranchPerturbationRiskEvaluator(nn.Module):
        """Twin-branch evaluator matching the manuscript architecture.

        Shapes:
            nominal_features:   [batch, stops, 24]
            perturbed_features: [batch, stops, 24]
            mask:               [batch, stops], True for real stops

        Outputs:
            cost_increment:        [batch]
            infeasibility_logit:    [batch]
            station_vulnerability: [batch, stops]
            fused_route_embedding: [batch, 256]

        The station output is produced for every stop. Apply an external station
        mask when computing station-vulnerability loss or aggregation.
        """

        def __init__(self, dropout: float = 0.10) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(24, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.cost_head = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.infeasibility_head = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.station_head = nn.Sequential(
                nn.Linear(320, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        @staticmethod
        def masked_mean(encoded: Tensor, mask: Tensor) -> Tensor:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            weights = mask.unsqueeze(-1).to(dtype=encoded.dtype)
            denominator = weights.sum(dim=1).clamp_min(1.0)
            return (encoded * weights).sum(dim=1) / denominator

        def forward(
            self,
            nominal_features: Tensor,
            perturbed_features: Tensor,
            mask: Tensor,
        ) -> dict[str, Tensor]:
            if nominal_features.shape != perturbed_features.shape:
                raise ValueError("The two branches must have identical shapes")
            if nominal_features.ndim != 3 or nominal_features.shape[-1] != 24:
                raise ValueError("Expected branch tensors with shape [batch, stops, 24]")
            if mask.shape != nominal_features.shape[:2]:
                raise ValueError("Mask shape must be [batch, stops]")

            h0 = self.encoder(nominal_features)
            hs = self.encoder(perturbed_features)
            r0 = self.masked_mean(h0, mask)
            rs = self.masked_mean(hs, mask)
            fused = torch.cat((r0, rs, torch.abs(rs - r0), r0 * rs), dim=-1)

            cost_increment = self.cost_head(fused).squeeze(-1)
            infeasibility_logit = self.infeasibility_head(fused).squeeze(-1)
            route_context = fused.unsqueeze(1).expand(-1, hs.shape[1], -1)
            station_input = torch.cat((route_context, hs), dim=-1)
            station_vulnerability = self.station_head(station_input).squeeze(-1)

            return {
                "cost_increment": cost_increment,
                "infeasibility_logit": infeasibility_logit,
                "infeasibility_probability": torch.sigmoid(infeasibility_logit),
                "station_vulnerability": station_vulnerability,
                "fused_route_embedding": fused,
                "nominal_stop_embeddings": h0,
                "perturbed_stop_embeddings": hs,
            }

        @property
        def trainable_parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def pairwise_hinge_loss_torch(
        scores: Tensor,
        targets: Tensor,
        margin: float = 1.0,
    ) -> Tensor:
        """Pairwise hinge loss over all non-tied pairs in a batch."""

        if scores.ndim != 1 or targets.ndim != 1 or scores.shape != targets.shape:
            raise ValueError("scores and targets must be one-dimensional and equal-sized")
        score_diff = scores[:, None] - scores[None, :]
        target_diff = targets[:, None] - targets[None, :]
        upper = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
        comparable = upper & (target_diff != 0)
        if not bool(comparable.any()):
            return scores.sum() * 0.0
        signs = torch.sign(target_diff[comparable])
        return torch.relu(margin - signs * score_diff[comparable]).mean()


    def multi_task_loss(
        model: TwinBranchPerturbationRiskEvaluator,
        outputs: Mapping[str, Tensor],
        cost_targets: Tensor,
        infeasibility_targets: Tensor,
        vulnerability_targets: Tensor,
        station_mask: Tensor,
        ranking_targets: Tensor | None = None,
        beta: float = 1.0,
        gamma: float = 0.50,
        delta: float = 0.20,
        omega: float = 1e-5,
    ) -> dict[str, Tensor]:
        """Compute the documented multi-task objective.

        ranking_targets should represent the ground-truth route-risk order used
        by the experiment. If omitted, the ranking term is set to zero rather
        than silently inventing labels.
        """

        cost_loss = F.mse_loss(outputs["cost_increment"], cost_targets)
        infeasibility_loss = F.binary_cross_entropy_with_logits(
            outputs["infeasibility_logit"], infeasibility_targets.float()
        )
        station_mask_bool = station_mask.bool()
        if bool(station_mask_bool.any()):
            vulnerability_loss = F.mse_loss(
                outputs["station_vulnerability"][station_mask_bool],
                vulnerability_targets[station_mask_bool],
            )
        else:
            vulnerability_loss = outputs["station_vulnerability"].sum() * 0.0
        if ranking_targets is None:
            ranking_loss = outputs["cost_increment"].sum() * 0.0
        else:
            predicted_risk = outputs["cost_increment"] + 100.0 * outputs[
                "infeasibility_probability"
            ]
            ranking_loss = pairwise_hinge_loss_torch(predicted_risk, ranking_targets)
        l2 = sum((parameter ** 2).sum() for parameter in model.parameters())
        total = (
            cost_loss
            + beta * infeasibility_loss
            + gamma * vulnerability_loss
            + delta * ranking_loss
            + omega * l2
        )
        return {
            "total": total,
            "cost_mse": cost_loss,
            "infeasibility_bce": infeasibility_loss,
            "vulnerability_mse": vulnerability_loss,
            "ranking_hinge": ranking_loss,
            "l2": l2,
        }

else:

    class TwinBranchPerturbationRiskEvaluator:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("PyTorch is required for the twin-branch model")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value) + b"\n")


def write_jsonl_gz(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(root_seed: int, *parts: int) -> int:
    sequence = np.random.SeedSequence([root_seed, *parts])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _scenario_iterator(
    instance: Mapping[str, Any],
    root_seed: int,
    setting: UncertaintySetting,
    split_code: int,
    count: int,
) -> Iterator[dict[str, Any]]:
    for index in range(count):
        seed = derive_seed(root_seed, int(instance["customer_count"]), split_code, index)
        yield generate_scenario(
            instance,
            scenario_seed=seed,
            setting=setting,
            scenario_id=f"{setting.name}_{index:04d}",
        )


def generate_dataset(
    output_dir: Path,
    scales: Sequence[int],
    base_instances_per_scale: int,
    optimization_scenarios: int,
    out_of_sample_scenarios: int,
    settings: Sequence[str],
    root_seed: int,
    config: GenerationConfig | None = None,
) -> dict[str, Any]:
    """Generate a reproducible data bundle and checksum manifest."""

    config = config or GenerationConfig()
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "generation_config.json", config.to_dict())

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_seed": root_seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scope_note": (
            "Newly generated reference data. These files are not the original "
            "run-level data underlying manuscript tables."
        ),
        "files": [],
    }

    for scale_index, customer_count in enumerate(scales):
        for instance_index in range(base_instances_per_scale):
            instance_seed = derive_seed(root_seed, 10, scale_index, instance_index)
            instance = generate_instance(customer_count, instance_seed, config)
            instance_path = (
                output_dir
                / "instances"
                / f"n{customer_count}"
                / f"instance_{instance_index:02d}_seed{instance_seed}.json"
            )
            write_json(instance_path, instance)
            manifest["files"].append(
                {
                    "role": "instance",
                    "instance_id": instance["instance_id"],
                    "path": instance_path.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(instance_path),
                }
            )

            for setting_index, setting_name in enumerate(settings):
                if setting_name not in DEFAULT_SETTINGS:
                    raise ValueError(f"Unknown uncertainty setting: {setting_name}")
                setting = DEFAULT_SETTINGS[setting_name]
                opt_path = (
                    output_dir
                    / "scenarios"
                    / "optimization"
                    / setting_name
                    / f"{instance['instance_id']}.jsonl.gz"
                )
                oos_path = (
                    output_dir
                    / "scenarios"
                    / "out_of_sample"
                    / setting_name
                    / f"{instance['instance_id']}.jsonl.gz"
                )
                write_jsonl_gz(
                    opt_path,
                    _scenario_iterator(
                        instance,
                        derive_seed(root_seed, 20, setting_index, instance_index),
                        setting,
                        split_code=1,
                        count=optimization_scenarios,
                    ),
                )
                write_jsonl_gz(
                    oos_path,
                    _scenario_iterator(
                        instance,
                        derive_seed(root_seed, 30, setting_index, instance_index),
                        setting,
                        split_code=2,
                        count=out_of_sample_scenarios,
                    ),
                )
                for role, path, count in (
                    ("optimization_scenarios", opt_path, optimization_scenarios),
                    ("out_of_sample_scenarios", oos_path, out_of_sample_scenarios),
                ):
                    manifest["files"].append(
                        {
                            "role": role,
                            "instance_id": instance["instance_id"],
                            "setting": setting_name,
                            "scenario_count": count,
                            "path": path.relative_to(output_dir).as_posix(),
                            "sha256": sha256_file(path),
                        }
                    )

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def _parse_settings(raw: Sequence[str]) -> list[str]:
    if not raw or raw == ["all"]:
        return list(DEFAULT_SETTINGS)
    unknown = sorted(set(raw) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown setting(s): {', '.join(unknown)}")
    return list(raw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate synthetic TPRE-ALNS reference data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a data bundle")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--scales", nargs="+", type=int, default=[25, 50, 100])
    generate.add_argument("--base-instances", type=int, default=10)
    generate.add_argument("--optimization-scenarios", type=int, default=50)
    generate.add_argument("--oos-scenarios", type=int, default=500)
    generate.add_argument(
        "--settings",
        nargs="+",
        default=["all"],
        help=f"all or any of: {', '.join(DEFAULT_SETTINGS)}",
    )
    generate.add_argument("--root-seed", type=int, default=20260907)

    validate = subparsers.add_parser("validate", help="Validate one instance/scenario")
    validate.add_argument("--instance", type=Path, required=True)
    validate.add_argument("--scenario", type=Path)

    model_info = subparsers.add_parser(
        "model-info", help="Print twin-branch model parameter count"
    )
    return parser


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        if args.base_instances <= 0:
            raise ValueError("--base-instances must be positive")
        if args.optimization_scenarios <= 0 or args.oos_scenarios <= 0:
            raise ValueError("Scenario counts must be positive")
        settings = _parse_settings(args.settings)
        manifest = generate_dataset(
            output_dir=args.output,
            scales=args.scales,
            base_instances_per_scale=args.base_instances,
            optimization_scenarios=args.optimization_scenarios,
            out_of_sample_scenarios=args.oos_scenarios,
            settings=settings,
            root_seed=args.root_seed,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(args.output.resolve()),
                    "file_records": len(manifest["files"]),
                    "scope_note": manifest["scope_note"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate":
        instance = _load_json(args.instance)
        validate_instance(instance)
        if args.scenario:
            scenario = _load_json(args.scenario)
            validate_scenario(instance, scenario)
        print("Validation passed.")
        return 0

    if args.command == "model-info":
        if torch is None:
            print("PyTorch is not installed.", file=sys.stderr)
            return 2
        model = TwinBranchPerturbationRiskEvaluator()
        print(
            json.dumps(
                {
                    "trainable_parameters": model.trainable_parameter_count,
                    "expected_trainable_parameters": 65091,
                    "matches_manuscript": model.trainable_parameter_count == 65091,
                },
                indent=2,
            )
        )
        return 0

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())

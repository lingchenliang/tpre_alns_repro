"""Twenty-four stop-level features and simulation-derived training labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .entities import EVRPInstance, Scenario, Solution
from .evaluation import evaluate_route
from .planning import (
    PlanningResult,
    certify_and_restore_solution,
    ensure_route,
)
from .scenarios import time_of_use_price


FEATURE_NAMES: Tuple[str, ...] = (
    "x_coordinate",
    "y_coordinate",
    "depot_indicator",
    "station_indicator",
    "customer_demand",
    "service_time",
    "ready_time",
    "due_time",
    "arrival_time",
    "battery_state",
    "charge_amount",
    "stop_position",
    "route_length",
    "installed_chargers",
    "charging_power",
    "tariff",
    "reported_unavailable_share",
    "occupied_share",
    "hidden_damage_share",
    "residual_capacity_share",
    "waiting_time",
    "incoming_distance",
    "incoming_travel_time",
    "incoming_energy",
)
BINARY_FEATURE_INDICES = (2, 3)


@dataclass(frozen=True)
class PhysicalScales:
    dmax: float
    tmax: float
    lmax_route: float

    @classmethod
    def from_plans(
        cls, inst: EVRPInstance, plans: Iterable[PlanningResult]
    ) -> "PhysicalScales":
        distances: List[float] = []
        route_lengths: List[int] = []
        for planning in plans:
            if not planning.feasible:
                continue
            for route in planning.solution.routes:
                route_lengths.append(len(route) - 1)
                distances.extend(
                    inst.distance(route[index], route[index + 1])
                    for index in range(len(route) - 1)
                )
        dmax = max(distances, default=100.0 * np.sqrt(2.0))
        return cls(
            dmax=max(float(dmax), 1e-8),
            tmax=max(float(dmax) / inst.travel_speed, 1e-8),
            lmax_route=max(float(max(route_lengths, default=1)), 1.0),
        )

    @classmethod
    def generated_domain(cls, inst: EVRPInstance) -> "PhysicalScales":
        dmax = 100.0 * np.sqrt(2.0)
        return cls(
            dmax=dmax,
            tmax=dmax / inst.travel_speed,
            lmax_route=float(len(inst.customer_ids) + len(inst.station_ids) + 1),
        )


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feature_arrays: Iterable[np.ndarray]) -> "FeatureNormalizer":
        rows = [array for array in feature_arrays if array.size]
        if not rows:
            raise ValueError("At least one non-empty feature array is required.")
        stacked = np.vstack(rows).astype(np.float64)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std[std < 1e-8] = 1.0
        for index in BINARY_FEATURE_INDICES:
            mean[index] = 0.0
            std[index] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        transformed = (features.astype(np.float32) - self.mean) / self.std
        for index in BINARY_FEATURE_INDICES:
            transformed[:, index] = features[:, index]
        return transformed.astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, Sequence[float]]) -> "FeatureNormalizer":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
        )


@dataclass
class VulnerabilityNormalizer:
    minimum: float
    maximum: float

    @classmethod
    def fit(
        cls, samples: Sequence["RouteScenarioSample"]
    ) -> "VulnerabilityNormalizer":
        values = np.concatenate(
            [
                sample.station_vulnerability[sample.station_mask]
                for sample in samples
                if np.any(sample.station_mask)
            ]
            or [np.asarray([0.0], dtype=float)]
        )
        return cls(float(values.min()), float(values.max()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        denominator = self.maximum - self.minimum + 1e-8
        return np.clip((values - self.minimum) / denominator, 0.0, 1.0)


@dataclass
class RouteScenarioSample:
    nominal_features: np.ndarray
    perturbed_features: np.ndarray
    station_mask: np.ndarray
    cost_increment: float
    infeasible: int
    station_vulnerability: np.ndarray
    base_instance_id: str = ""
    scenario_id: str = ""
    realized_cost: float = 0.0


def _planned_route_and_stops(
    inst: EVRPInstance,
    solution: Solution,
    route_index: int,
) -> Tuple[Solution, Sequence, Sequence]:
    planning = certify_and_restore_solution(
        inst,
        solution,
        require_all_customers=False,
        complete_backups=True,
    )
    if not planning.feasible:
        raise ValueError(planning.reason)
    executable = planning.solution
    plan = planning.route_plans[route_index]
    return executable, plan.route, plan.stops


def route_stop_features(
    inst: EVRPInstance,
    solution: Solution,
    route_index: int,
    scenario: Optional[Scenario] = None,
    *,
    physical_scales: Optional[PhysicalScales] = None,
    normalizer: Optional[FeatureNormalizer] = None,
) -> np.ndarray:
    """Construct Supplementary Table S9 inputs at the nominal arrival interval."""
    executable, route, stops = _planned_route_and_stops(
        inst, solution, route_index
    )
    scales = physical_scales or PhysicalScales.generated_domain(inst)
    rows: List[List[float]] = []
    for stop in stops:
        node = stop.node_id
        x_coord, y_coord = inst.node_xy(node)
        is_depot = inst.is_depot_copy(node)
        is_station = inst.is_station(node)
        is_customer = inst.is_customer(node)
        if is_customer:
            customer = inst.customers_by_id[node]
            demand = customer.demand / inst.vehicle_capacity
            service = customer.service_time / inst.planning_horizon_min
            ready = customer.tw_start / inst.planning_horizon_min
            due = customer.tw_end / inst.planning_horizon_min
        else:
            demand = service = ready = 0.0
            due = 1.0 if is_depot else 0.0

        installed = power = reported_share = 0.0
        occupied_share = hidden_share = residual_share = queue_delay = 0.0
        tariff = 0.0
        if is_station:
            station = inst.stations_by_id[node]
            interval = inst.time_to_interval(stop.physical_arrival)
            reported = station.reported_at(interval)
            installed = station.chargers / 8.0
            power = station.charging_power_kw / 120.0
            tariff = (
                time_of_use_price(
                    inst.absolute_clock_min(stop.physical_arrival)
                )
                / 1.20
            )
            reported_share = reported / station.chargers
            if scenario is None:
                residual_share = (station.chargers - reported) / station.chargers
            else:
                occupied_share = (
                    float(scenario.occupation[node][interval]) / station.chargers
                )
                hidden_share = (
                    float(scenario.hidden_damage[node][interval])
                    / station.chargers
                )
                residual_share = (
                    float(scenario.available_capacity[node][interval])
                    / station.chargers
                )
                queue_delay = (
                    float(scenario.waiting_time[node][interval]) / 75.0
                )

        if stop.position == 0:
            incoming_distance = incoming_time = incoming_energy = 0.0
        else:
            predecessor = route[stop.position - 1]
            incoming_distance = inst.distance(predecessor, node) / scales.dmax
            incoming_time = (
                inst.travel_time(predecessor, node) / scales.tmax
            )
            incoming_energy = inst.energy(predecessor, node) / inst.battery_capacity

        rows.append(
            [
                x_coord / 100.0,
                y_coord / 100.0,
                float(is_depot),
                float(is_station),
                demand,
                service,
                ready,
                due,
                (stop.physical_arrival - inst.planning_start_min)
                / inst.planning_horizon_min,
                stop.battery_arrival / inst.battery_capacity,
                float(
                    executable.planned_charges.get(
                        (route_index, stop.position), 0.0
                    )
                )
                / inst.battery_capacity,
                stop.position / max(1, len(route) - 1),
                (len(route) - 1) / scales.lmax_route,
                installed,
                power,
                tariff,
                reported_share,
                occupied_share,
                hidden_share,
                residual_share,
                queue_delay,
                incoming_distance,
                incoming_time,
                incoming_energy,
            ]
        )
    features = np.asarray(rows, dtype=np.float32)
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("The route encoder must receive exactly 24 features.")
    return normalizer.transform(features) if normalizer else features


def route_feature_vector(
    inst: EVRPInstance,
    route: Sequence[int],
    scenario: Optional[Scenario] = None,
) -> np.ndarray:
    """Compatibility helper returning masked-mean 24-feature route rows."""
    features = route_stop_features(
        inst, Solution(routes=[list(route)]), 0, scenario
    )
    return features.mean(axis=0)


def build_route_scenario_sample(
    inst: EVRPInstance,
    route: Sequence[int],
    scenario: Scenario,
    *,
    physical_scales: Optional[PhysicalScales] = None,
    base_instance_id: Optional[str] = None,
) -> RouteScenarioSample:
    planning = certify_and_restore_solution(
        inst,
        Solution(routes=[list(route)]),
        require_all_customers=False,
        complete_backups=True,
    )
    if not planning.feasible:
        raise ValueError(planning.reason)
    executable = planning.solution
    nominal_features = route_stop_features(
        inst,
        executable,
        0,
        None,
        physical_scales=physical_scales,
    )
    perturbed_features = route_stop_features(
        inst,
        executable,
        0,
        scenario,
        physical_scales=physical_scales,
    )
    nominal_eval = evaluate_route(
        inst,
        executable.routes[0],
        None,
        backups=executable.backups,
    )
    scenario_eval = evaluate_route(
        inst,
        executable.routes[0],
        scenario,
        backups=executable.backups,
    )
    station_mask = np.asarray(
        [inst.is_station(node) for node in executable.routes[0]], dtype=bool
    )
    vulnerability = np.zeros(len(executable.routes[0]), dtype=np.float32)
    for position in np.flatnonzero(station_mask):
        vulnerability[position] = float(
            scenario_eval.station_attribution.get((0, int(position)), 0.0)
        )
    return RouteScenarioSample(
        nominal_features=nominal_features,
        perturbed_features=perturbed_features,
        station_mask=station_mask,
        cost_increment=float(
            scenario_eval.scenario_cost - nominal_eval.scenario_cost
        ),
        infeasible=int(scenario_eval.infeasible),
        station_vulnerability=vulnerability,
        base_instance_id=base_instance_id or inst.name,
        scenario_id=scenario.name,
        realized_cost=scenario_eval.scenario_cost,
    )


def random_route_pool(
    inst: EVRPInstance,
    rng: np.random.Generator,
    n_routes: int = 100,
) -> List[List[int]]:
    """Create a deterministic fallback pool when preliminary ALNS traces are absent.

    Routes are grown from certified one-customer seeds. This avoids rejection
    sampling complete random permutations, whose success probability becomes
    negligible for independent hard time windows at medium and large scales.
    """
    if n_routes <= 0:
        return []
    routes: List[List[int]] = []
    signatures: set[Tuple[int, ...]] = set()
    customers = [int(customer) for customer in inst.customer_ids]
    attempts = 0
    max_attempts = max(n_routes * 20, len(customers) * 4)
    maximum_size = min(20, len(customers))

    while len(routes) < n_routes and attempts < max_attempts:
        attempts += 1
        order = customers.copy()
        rng.shuffle(order)
        route = [0, order.pop(), inst.terminal_id]
        initial = certify_and_restore_solution(
            inst,
            Solution(routes=[route]),
            require_all_customers=False,
        )
        if not initial.feasible:
            continue
        route = list(initial.solution.routes[0])
        signature = tuple(route)
        if signature not in signatures:
            routes.append(route)
            signatures.add(signature)
            if len(routes) >= n_routes:
                break

        target_size = int(rng.integers(2, maximum_size + 1))
        for customer in order:
            if sum(inst.is_customer(node) for node in route) >= target_size:
                break
            insertion_positions = list(range(1, len(route)))
            rng.shuffle(insertion_positions)
            accepted_route = None
            for position in insertion_positions:
                trial = route[:position] + [customer] + route[position:]
                planning = certify_and_restore_solution(
                    inst,
                    Solution(routes=[trial]),
                    require_all_customers=False,
                )
                if planning.feasible:
                    accepted_route = list(planning.solution.routes[0])
                    break
            if accepted_route is None:
                continue
            route = accepted_route
            signature = tuple(route)
            if signature not in signatures:
                routes.append(route)
                signatures.add(signature)
                if len(routes) >= n_routes:
                    break
    if not routes:
        raise RuntimeError("Could not construct any forward-feasible training route.")
    return routes


def make_training_samples(
    inst: EVRPInstance,
    scenarios: Sequence[Scenario],
    n_samples: int = 2000,
    seed: int = 1,
    physical_scales: Optional[PhysicalScales] = None,
) -> List[RouteScenarioSample]:
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    rng = np.random.default_rng(seed)
    route_pool = random_route_pool(
        inst, rng, n_routes=max(20, min(400, n_samples // 20))
    )
    plans = [
        certify_and_restore_solution(
            inst,
            Solution(routes=[route]),
            require_all_customers=False,
        )
        for route in route_pool
    ]
    scales = physical_scales or PhysicalScales.from_plans(inst, plans)
    samples: List[RouteScenarioSample] = []
    for sample_index in range(n_samples):
        route = route_pool[int(rng.integers(0, len(route_pool)))]
        scenario = scenarios[int(rng.integers(0, len(scenarios)))]
        samples.append(
            build_route_scenario_sample(
                inst,
                route,
                scenario,
                physical_scales=scales,
                base_instance_id=inst.name,
            )
        )
    return samples


def fit_training_normalizers(
    samples: Sequence[RouteScenarioSample],
) -> Tuple[FeatureNormalizer, VulnerabilityNormalizer]:
    feature_normalizer = FeatureNormalizer.fit(
        [
            features
            for sample in samples
            for features in (
                sample.nominal_features,
                sample.perturbed_features,
            )
        ]
    )
    vulnerability = VulnerabilityNormalizer.fit(samples)
    return feature_normalizer, vulnerability

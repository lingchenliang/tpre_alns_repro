"""Core data structures used by the TPRE-ALNS reference implementation.

Times are stored as minutes from the 06:00 start-depot clock.  The tariff
helper converts them to minutes after midnight only when a price interval is
needed.  Routes use node 0 for the start-depot copy and ``instance.terminal_id``
for the terminal-depot copy; both copies share the same physical coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


StopKey = Tuple[int, int]  # (route index, position in the stored route)


@dataclass(frozen=True)
class Customer:
    node_id: int
    x: float
    y: float
    demand: float
    service_time: float
    tw_start: float
    tw_end: float


@dataclass(frozen=True)
class Station:
    node_id: int
    x: float
    y: float
    chargers: int
    charging_power_kw: float
    # D_jt is sampled once per base instance and is known before dispatch.
    reported_unavailable: Tuple[int, ...] = ()

    def reported_at(self, interval: int) -> int:
        if not self.reported_unavailable:
            return 0
        idx = int(np.clip(interval, 0, len(self.reported_unavailable) - 1))
        return int(self.reported_unavailable[idx])


@dataclass
class EVRPInstance:
    name: str
    depot: Tuple[float, float]
    customers: List[Customer]
    stations: List[Station]
    instance_seed: int = 0
    vehicle_capacity: float = 1000.0
    battery_capacity: float = 80.0
    initial_battery: float = 80.0
    safety_battery: float = 8.0
    energy_consumption: float = 0.24  # kWh / km
    travel_speed: float = 0.65  # km / min (39 km/h)
    planning_start_min: float = 0.0
    planning_horizon_min: float = 1080.0
    interval_min: int = 60
    day_start_min: int = 360  # 06:00, used only by the tariff schedule
    max_continuous_work_min: float = 240.0
    min_rest_min: float = 30.0
    vehicle_use_cost: float = 100.0
    travel_cost_per_km: float = 1.20
    waiting_cost_per_min: float = 30.0 / 60.0
    driver_cost_per_min: float = 25.0 / 60.0
    local_repair_fixed_cost: float = 50.0
    infeasibility_penalty: float = 10000.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.customers:
            raise ValueError("An instance must contain at least one customer.")
        ids = [c.node_id for c in self.customers] + [
            station.node_id for station in self.stations
        ]
        if 0 in ids or len(ids) != len(set(ids)):
            raise ValueError("Customer/station node ids must be unique and non-zero.")
        if self.interval_min <= 0 or self.planning_horizon_min <= 0:
            raise ValueError("The interval and planning horizon must be positive.")

        self.customers_by_id: Dict[int, Customer] = {
            customer.node_id: customer for customer in self.customers
        }
        self.stations_by_id: Dict[int, Station] = {
            station.node_id: station for station in self.stations
        }
        self.customer_ids: List[int] = [customer.node_id for customer in self.customers]
        self.station_ids: List[int] = [station.node_id for station in self.stations]
        self.terminal_id: int = max(ids) + 1
        self.node_ids: List[int] = (
            [0] + self.customer_ids + self.station_ids + [self.terminal_id]
        )
        self.n_intervals: int = int(
            np.ceil(self.planning_horizon_min / self.interval_min)
        )
        for station in self.stations:
            if station.reported_unavailable and (
                len(station.reported_unavailable) != self.n_intervals
            ):
                raise ValueError(
                    f"Station {station.node_id} has "
                    f"{len(station.reported_unavailable)} D_jt values; "
                    f"expected {self.n_intervals}."
                )

    @property
    def planning_end_min(self) -> float:
        return self.planning_start_min + self.planning_horizon_min

    # Compatibility aliases used by earlier repository scripts.
    @property
    def planning_start(self) -> float:
        return self.planning_start_min

    @property
    def planning_end(self) -> float:
        return self.planning_end_min

    @property
    def recourse_cost_per_km(self) -> float:
        return self.travel_cost_per_km

    def is_customer(self, node_id: int) -> bool:
        return node_id in self.customers_by_id

    def is_station(self, node_id: int) -> bool:
        return node_id in self.stations_by_id

    def is_depot_copy(self, node_id: int) -> bool:
        return node_id in {0, self.terminal_id}

    def node_xy(self, node_id: int) -> Tuple[float, float]:
        if self.is_depot_copy(node_id):
            return self.depot
        if self.is_customer(node_id):
            customer = self.customers_by_id[node_id]
            return customer.x, customer.y
        if self.is_station(node_id):
            station = self.stations_by_id[node_id]
            return station.x, station.y
        raise KeyError(f"Unknown node id: {node_id}")

    def distance(self, origin: int, destination: int) -> float:
        x0, y0 = self.node_xy(origin)
        x1, y1 = self.node_xy(destination)
        return float(np.hypot(x0 - x1, y0 - y1))

    def travel_time(self, origin: int, destination: int) -> float:
        return self.distance(origin, destination) / max(self.travel_speed, 1e-12)

    def energy(self, origin: int, destination: int) -> float:
        return self.distance(origin, destination) * self.energy_consumption

    def time_to_interval(self, relative_time_min: float) -> int:
        index = int(
            (relative_time_min - self.planning_start_min) // self.interval_min
        )
        return int(np.clip(index, 0, self.n_intervals - 1))

    def absolute_clock_min(self, relative_time_min: float) -> float:
        return self.day_start_min + relative_time_min - self.planning_start_min

    def nearest_station(
        self, node_id: int, exclude: Optional[int] = None
    ) -> int:
        candidates = [
            station.node_id
            for station in self.stations
            if station.node_id != exclude
        ]
        if not candidates:
            raise ValueError("No alternative charging station is available.")
        return min(candidates, key=lambda sid: (self.distance(node_id, sid), sid))


@dataclass
class Scenario:
    name: str
    probability: float
    occupation: Dict[int, np.ndarray]
    hidden_damage: Dict[int, np.ndarray]
    available_capacity: Dict[int, np.ndarray]
    waiting_time: Dict[int, np.ndarray]
    price: Dict[int, np.ndarray]
    setting: str = ""
    seed: int = 0

    def state(self, inst: EVRPInstance, station_id: int, interval: int) -> int:
        """Return 0=available, 1=occupied, or 2=failed."""
        idx = int(np.clip(interval, 0, inst.n_intervals - 1))
        station = inst.stations_by_id[station_id]
        available = int(self.available_capacity[station_id][idx])
        hidden = int(self.hidden_damage[station_id][idx])
        reported = station.reported_at(idx)
        if available > 0:
            return 0
        if reported + hidden < station.chargers:
            return 1
        return 2

    def severity(self, inst: EVRPInstance) -> float:
        """Equation (91), averaged over stations and time intervals."""
        values: List[float] = []
        for station_id in inst.station_ids:
            station = inst.stations_by_id[station_id]
            occ = np.asarray(self.occupation[station_id], dtype=float)
            hidden = np.asarray(self.hidden_damage[station_id], dtype=float)
            wait = np.asarray(self.waiting_time[station_id], dtype=float)
            values.extend(
                ((occ + hidden) / station.chargers + wait / 75.0).tolist()
            )
        return float(np.mean(values)) if values else 0.0


@dataclass
class Solution:
    routes: List[List[int]]
    planned_charges: Dict[StopKey, float] = field(default_factory=dict)
    backups: Dict[StopKey, Optional[int]] = field(default_factory=dict)
    planned_rests: Dict[StopKey, float] = field(default_factory=dict)

    def copy(self) -> "Solution":
        return Solution(
            routes=[list(route) for route in self.routes],
            planned_charges=dict(self.planned_charges),
            backups=dict(self.backups),
            planned_rests=dict(self.planned_rests),
        )

    def customers(self, inst: EVRPInstance) -> List[int]:
        return [
            node
            for route in self.routes
            for node in route
            if inst.is_customer(node)
        ]

    def to_dict(self) -> Dict[str, Any]:
        def encode(mapping: Mapping[StopKey, Any]) -> Dict[str, Any]:
            return {
                f"{route_idx}:{position}": value
                for (route_idx, position), value in sorted(mapping.items())
            }

        return {
            "routes": self.routes,
            "planned_charges": encode(self.planned_charges),
            "backups": encode(self.backups),
            "planned_rests": encode(self.planned_rests),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Solution":
        def decode(mapping: Mapping[str, Any]) -> Dict[StopKey, Any]:
            decoded: Dict[StopKey, Any] = {}
            for raw_key, value in mapping.items():
                route_idx, position = raw_key.split(":", 1)
                decoded[(int(route_idx), int(position))] = value
            return decoded

        return cls(
            routes=[list(map(int, route)) for route in data["routes"]],
            planned_charges={
                key: float(value)
                for key, value in decode(data.get("planned_charges", {})).items()
            },
            backups={
                key: (None if value is None else int(value))
                for key, value in decode(data.get("backups", {})).items()
            },
            planned_rests={
                key: float(value)
                for key, value in decode(data.get("planned_rests", {})).items()
            },
        )


@dataclass
class EvalMetrics:
    objective: float
    planning_cost: float
    expected_scenario_cost: float
    cvar_scenario_cost: float
    total_cost: float
    vehicle_cost: float
    planned_travel_cost: float
    charging_cost: float
    waiting_cost: float
    driver_cost: float
    recourse_cost: float
    penalty_cost: float
    infeasible_ratio: float
    backup_switches: float
    local_repairs: float
    scenario_costs: List[float] = field(default_factory=list)
    scenario_probabilities: List[float] = field(default_factory=list)

    @property
    def expected_cost(self) -> float:
        return self.expected_scenario_cost

    @property
    def cvar(self) -> float:
        return self.cvar_scenario_cost

    @property
    def travel_cost(self) -> float:
        return self.planned_travel_cost

    def to_dict(self) -> Dict[str, float]:
        return {
            "objective": self.objective,
            "planning_cost": self.planning_cost,
            "expected_scenario_cost": self.expected_scenario_cost,
            "scenario_dependent_cost_cvar": self.cvar_scenario_cost,
            "total_cost": self.total_cost,
            "vehicle_cost": self.vehicle_cost,
            "planned_travel_cost": self.planned_travel_cost,
            "charging_cost": self.charging_cost,
            "waiting_cost": self.waiting_cost,
            "driver_cost": self.driver_cost,
            "recourse_cost": self.recourse_cost,
            "penalty_cost": self.penalty_cost,
            "infeasible_ratio": self.infeasible_ratio,
            "backup_switches": self.backup_switches,
            "local_repairs": self.local_repairs,
        }

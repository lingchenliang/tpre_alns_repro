"""Synthetic instance generation and portable instance I/O.

The default generator follows Supplementary Algorithm S2 exactly: coordinates
are uniform on the 100 km square, demands are integers in [10, 50] kg, service
durations are U(5, 15) minutes, and hard-window widths are U(60, 180) minutes.
Reported charger unavailability D_jt is sampled once with probability 0.05 and
stored in the base instance, so every method and every scenario sees the same
pre-dispatch information.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .entities import Customer, EVRPInstance, Station


def generate_synthetic_instance(
    n_customers: int = 25,
    n_stations: int = 5,
    seed: int = 1,
    name: Optional[str] = None,
    planning_horizon_min: int = 1080,
    interval_min: int = 60,
    reported_unavailable_probability: float = 0.05,
) -> EVRPInstance:
    """Generate one VRPTW-style electric-delivery instance.

    The independent random streams make customer fields, station fields and
    reported-unavailability records stable if another component is extended.
    """
    if n_customers <= 0 or n_stations <= 0:
        raise ValueError("n_customers and n_stations must both be positive.")
    if planning_horizon_min <= 180:
        raise ValueError("The planning horizon must exceed the maximum window width.")

    seed_sequence = np.random.SeedSequence(seed)
    customer_seed, station_seed, reported_seed = seed_sequence.spawn(3)
    customer_rng = np.random.default_rng(customer_seed)
    station_rng = np.random.default_rng(station_seed)
    reported_rng = np.random.default_rng(reported_seed)

    depot = (50.0, 50.0)
    customer_xy = customer_rng.uniform(0.0, 100.0, size=(n_customers, 2))
    customers = []
    for offset, (x_coord, y_coord) in enumerate(customer_xy, start=1):
        demand = int(customer_rng.integers(10, 51))
        service_time = float(customer_rng.uniform(5.0, 15.0))
        window_width = float(customer_rng.uniform(60.0, 180.0))
        ready_time = float(
            customer_rng.uniform(0.0, planning_horizon_min - window_width)
        )
        customers.append(
            Customer(
                node_id=offset,
                x=float(x_coord),
                y=float(y_coord),
                demand=float(demand),
                service_time=service_time,
                tw_start=ready_time,
                tw_end=ready_time + window_width,
            )
        )

    n_intervals = int(np.ceil(planning_horizon_min / interval_min))
    station_xy = station_rng.uniform(0.0, 100.0, size=(n_stations, 2))
    charger_options = np.array([4, 6, 8], dtype=int)
    power_options = np.array([60.0, 120.0], dtype=float)
    stations = []
    for station_offset, (x_coord, y_coord) in enumerate(station_xy, start=1):
        station_id = n_customers + station_offset
        chargers = int(
            station_rng.choice(charger_options, p=np.array([0.30, 0.40, 0.30]))
        )
        power = float(station_rng.choice(power_options, p=np.array([0.50, 0.50])))
        reported = tuple(
            int(value)
            for value in reported_rng.binomial(
                chargers, reported_unavailable_probability, size=n_intervals
            )
        )
        stations.append(
            Station(
                node_id=station_id,
                x=float(x_coord),
                y=float(y_coord),
                chargers=chargers,
                charging_power_kw=power,
                reported_unavailable=reported,
            )
        )

    return EVRPInstance(
        name=name or f"synthetic_C{n_customers}_F{n_stations}_seed{seed}",
        depot=depot,
        customers=customers,
        stations=stations,
        instance_seed=seed,
        planning_start_min=0.0,
        planning_horizon_min=float(planning_horizon_min),
        interval_min=interval_min,
        metadata={
            "generator": "supplementary_algorithm_s2",
            "coordinate_domain_km": [0.0, 100.0],
            "reported_unavailable_probability": reported_unavailable_probability,
        },
    )


def generate_benchmark_like_instance(**kwargs) -> EVRPInstance:
    """Backward-compatible alias for :func:`generate_synthetic_instance`."""
    if "planning_start" in kwargs:
        planning_start = int(kwargs.pop("planning_start"))
        planning_end = int(kwargs.pop("planning_end", planning_start + 1080))
        kwargs["planning_horizon_min"] = planning_end - planning_start
    return generate_synthetic_instance(**kwargs)


def load_customer_csv(
    path: str | Path,
    n_stations: int = 5,
    seed: int = 1,
    planning_horizon_min: int = 1080,
) -> EVRPInstance:
    """Load customer/depot rows and add synthetic charging infrastructure.

    Required columns are
    ``id,x,y,demand,ready_time,due_time,service_time``.  Times must be minutes
    from the 06:00 start-copy clock.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "id",
            "x",
            "y",
            "demand",
            "ready_time",
            "due_time",
            "service_time",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
        rows = list(reader)

    depot_rows = [row for row in rows if int(row["id"]) == 0]
    if len(depot_rows) != 1:
        raise ValueError("The CSV must contain exactly one depot row with id=0.")
    depot = (float(depot_rows[0]["x"]), float(depot_rows[0]["y"]))

    customers = []
    for new_id, row in enumerate(
        (row for row in rows if int(row["id"]) != 0), start=1
    ):
        customers.append(
            Customer(
                node_id=new_id,
                x=float(row["x"]),
                y=float(row["y"]),
                demand=float(row["demand"]),
                service_time=float(row["service_time"]),
                tw_start=float(row["ready_time"]),
                tw_end=float(row["due_time"]),
            )
        )

    infrastructure = generate_synthetic_instance(
        n_customers=len(customers),
        n_stations=n_stations,
        seed=seed,
        planning_horizon_min=planning_horizon_min,
        name=source.stem,
    )
    return EVRPInstance(
        name=source.stem,
        depot=depot,
        customers=customers,
        stations=infrastructure.stations,
        instance_seed=seed,
        planning_horizon_min=float(planning_horizon_min),
        interval_min=infrastructure.interval_min,
        metadata={
            **infrastructure.metadata,
            "customer_source": str(source),
        },
    )


def instance_to_dict(inst: EVRPInstance) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "name": inst.name,
        "instance_seed": inst.instance_seed,
        "depot": list(inst.depot),
        "planning_start_min": inst.planning_start_min,
        "planning_horizon_min": inst.planning_horizon_min,
        "interval_min": inst.interval_min,
        "day_start_min": inst.day_start_min,
        "vehicle": {
            "capacity": inst.vehicle_capacity,
            "battery_capacity": inst.battery_capacity,
            "initial_battery": inst.initial_battery,
            "safety_battery": inst.safety_battery,
            "energy_consumption": inst.energy_consumption,
            "travel_speed": inst.travel_speed,
            "max_continuous_work_min": inst.max_continuous_work_min,
            "min_rest_min": inst.min_rest_min,
        },
        "costs": {
            "vehicle_use": inst.vehicle_use_cost,
            "travel_per_km": inst.travel_cost_per_km,
            "waiting_per_min": inst.waiting_cost_per_min,
            "driver_per_min": inst.driver_cost_per_min,
            "local_repair": inst.local_repair_fixed_cost,
            "infeasibility_penalty": inst.infeasibility_penalty,
        },
        "customers": [
            {
                "node_id": customer.node_id,
                "x": customer.x,
                "y": customer.y,
                "demand": customer.demand,
                "service_time": customer.service_time,
                "tw_start": customer.tw_start,
                "tw_end": customer.tw_end,
            }
            for customer in inst.customers
        ],
        "stations": [
            {
                "node_id": station.node_id,
                "x": station.x,
                "y": station.y,
                "chargers": station.chargers,
                "charging_power_kw": station.charging_power_kw,
                "reported_unavailable": list(station.reported_unavailable),
            }
            for station in inst.stations
        ],
        "metadata": inst.metadata,
    }


def save_instance_json(inst: EVRPInstance, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(instance_to_dict(inst), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_instance_json(path: str | Path) -> EVRPInstance:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vehicle = data["vehicle"]
    costs = data["costs"]
    return EVRPInstance(
        name=data["name"],
        depot=tuple(map(float, data["depot"])),
        customers=[Customer(**row) for row in data["customers"]],
        stations=[
            Station(
                **{
                    **row,
                    "reported_unavailable": tuple(row["reported_unavailable"]),
                }
            )
            for row in data["stations"]
        ],
        instance_seed=int(data.get("instance_seed", 0)),
        planning_start_min=float(data["planning_start_min"]),
        planning_horizon_min=float(data["planning_horizon_min"]),
        interval_min=int(data["interval_min"]),
        day_start_min=int(data["day_start_min"]),
        vehicle_capacity=float(vehicle["capacity"]),
        battery_capacity=float(vehicle["battery_capacity"]),
        initial_battery=float(vehicle["initial_battery"]),
        safety_battery=float(vehicle["safety_battery"]),
        energy_consumption=float(vehicle["energy_consumption"]),
        travel_speed=float(vehicle["travel_speed"]),
        max_continuous_work_min=float(vehicle["max_continuous_work_min"]),
        min_rest_min=float(vehicle["min_rest_min"]),
        vehicle_use_cost=float(costs["vehicle_use"]),
        travel_cost_per_km=float(costs["travel_per_km"]),
        waiting_cost_per_min=float(costs["waiting_per_min"]),
        driver_cost_per_min=float(costs["driver_per_min"]),
        local_repair_fixed_cost=float(costs["local_repair"]),
        infeasibility_penalty=float(costs["infeasibility_penalty"]),
        metadata=dict(data.get("metadata", {})),
    )


def export_instance_csv(inst: EVRPInstance, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "type",
        "x",
        "y",
        "demand",
        "service_time",
        "ready_time",
        "due_time",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "id": 0,
                "type": "start_depot",
                "x": inst.depot[0],
                "y": inst.depot[1],
                "demand": 0,
                "service_time": 0,
                "ready_time": 0,
                "due_time": inst.planning_horizon_min,
            }
        )
        for customer in inst.customers:
            writer.writerow(
                {
                    "id": customer.node_id,
                    "type": "customer",
                    "x": customer.x,
                    "y": customer.y,
                    "demand": customer.demand,
                    "service_time": customer.service_time,
                    "ready_time": customer.tw_start,
                    "due_time": customer.tw_end,
                }
            )

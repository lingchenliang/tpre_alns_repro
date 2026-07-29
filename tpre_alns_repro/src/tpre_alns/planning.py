"""Deterministic route restoration and forward certification.

The search stores customer sequences.  This module turns them into executable
pre-dispatch plans by inserting reachable charging/rest stops, selecting the
minimum charge needed for the next required segment, propagating hard customer
windows, and completing at most one optional backup record per planned station.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .entities import EVRPInstance, Solution, StopKey
from .scenarios import time_of_use_price


EPSILON = 1e-9


@dataclass
class PlanStop:
    route_index: int
    position: int
    node_id: int
    physical_arrival: float
    service_start: float
    departure: float
    battery_arrival: float
    battery_departure: float
    remaining_load: float
    continuous_work_arrival: float
    continuous_work_departure: float
    customer_waiting: float = 0.0
    service_duration: float = 0.0
    planned_charge: float = 0.0
    charging_duration: float = 0.0
    charging_cost: float = 0.0
    rest_duration: float = 0.0

    @property
    def key(self) -> StopKey:
        return self.route_index, self.position


@dataclass
class PlanViolation:
    kind: str
    message: str
    position: int
    insert_after: Optional[int] = None
    time: float = 0.0
    battery: float = 0.0
    continuous_work: float = 0.0


@dataclass
class RoutePlan:
    route: List[int]
    feasible: bool
    stops: List[PlanStop] = field(default_factory=list)
    violation: Optional[PlanViolation] = None
    duty_duration: float = 0.0
    planned_distance: float = 0.0
    nominal_charging_cost: float = 0.0


@dataclass
class PlanningResult:
    solution: Solution
    feasible: bool
    route_plans: List[RoutePlan] = field(default_factory=list)
    reason: str = ""


def ensure_route(inst: EVRPInstance, route: Sequence[int]) -> List[int]:
    """Normalize a route to start-copy -> stops -> terminal-copy."""
    middle = [
        int(node)
        for node in route
        if int(node) not in {0, inst.terminal_id}
    ]
    return [0] + middle + [inst.terminal_id]


def route_distance(inst: EVRPInstance, route: Sequence[int]) -> float:
    normalized = ensure_route(inst, route)
    return float(
        sum(
            inst.distance(normalized[index], normalized[index + 1])
            for index in range(len(normalized) - 1)
        )
    )


def _charge_allocation_cost(
    inst: EVRPInstance,
    station_id: int,
    start_time: float,
    amount: float,
) -> Tuple[float, float]:
    """Allocate charging chronologically across fixed tariff intervals."""
    if amount <= EPSILON:
        return 0.0, 0.0
    station = inst.stations_by_id[station_id]
    rate = station.charging_power_kw / 60.0
    remaining = amount
    current_time = start_time
    cost = 0.0
    while remaining > EPSILON:
        if current_time >= inst.planning_end_min - EPSILON:
            return float("inf"), float("inf")
        interval = inst.time_to_interval(current_time)
        boundary = min(
            inst.planning_start_min + (interval + 1) * inst.interval_min,
            inst.planning_end_min,
        )
        available_minutes = max(boundary - current_time, 0.0)
        if available_minutes <= EPSILON:
            current_time = boundary + EPSILON
            continue
        energy = min(remaining, rate * available_minutes)
        price = time_of_use_price(inst.absolute_clock_min(current_time))
        cost += energy * price
        duration = energy / rate
        current_time += duration
        remaining -= energy
    return current_time - start_time, cost


def _energy_to_next_charging_opportunity(
    inst: EVRPInstance, route: Sequence[int], station_position: int
) -> float:
    energy = 0.0
    for position in range(station_position, len(route) - 1):
        origin = route[position]
        destination = route[position + 1]
        energy += inst.energy(origin, destination)
        if inst.is_station(destination) or destination == inst.terminal_id:
            break
    return energy


def _work_until_next_eligible_stop(
    inst: EVRPInstance,
    route: Sequence[int],
    station_position: int,
    departure_time: float,
) -> float:
    """Earliest non-rest work until the next station or terminal copy."""
    current_time = departure_time
    work = 0.0
    for position in range(station_position, len(route) - 1):
        origin = route[position]
        destination = route[position + 1]
        travel = inst.travel_time(origin, destination)
        current_time += travel
        work += travel
        if inst.is_customer(destination):
            customer = inst.customers_by_id[destination]
            wait = max(0.0, customer.tw_start - current_time)
            work += wait + customer.service_time
            current_time += wait + customer.service_time
        if inst.is_station(destination) or destination == inst.terminal_id:
            break
    return work


def propagate_nominal_route(
    inst: EVRPInstance,
    route: Sequence[int],
    route_index: int = 0,
) -> RoutePlan:
    """Certify one stored route under known pre-dispatch conditions."""
    normalized = ensure_route(inst, route)
    route_customers = [
        node for node in normalized if inst.is_customer(node)
    ]
    initial_load = float(
        sum(inst.customers_by_id[node].demand for node in route_customers)
    )
    if initial_load > inst.vehicle_capacity + EPSILON:
        return RoutePlan(
            normalized,
            False,
            violation=PlanViolation(
                "load",
                f"Route load {initial_load:.3f} exceeds capacity.",
                position=0,
            ),
        )
    if len(
        [node for node in normalized if inst.is_station(node)]
    ) != len(set(node for node in normalized if inst.is_station(node))):
        return RoutePlan(
            normalized,
            False,
            violation=PlanViolation(
                "station_repeat",
                "A physical charging station appears more than once.",
                position=0,
            ),
        )

    first_customer_position = next(
        (
            position
            for position, node in enumerate(normalized)
            if inst.is_customer(node)
        ),
        None,
    )
    depot_hold = 0.0
    if first_customer_position is not None:
        first_customer = inst.customers_by_id[
            normalized[first_customer_position]
        ]
        travel_to_first = sum(
            inst.travel_time(normalized[index], normalized[index + 1])
            for index in range(first_customer_position)
        )
        depot_hold = max(
            0.0,
            first_customer.tw_start
            - inst.planning_start_min
            - travel_to_first,
        )
    time = inst.planning_start_min + depot_hold
    battery = inst.initial_battery
    remaining_load = initial_load
    continuous_work = 0.0
    stops = [
        PlanStop(
            route_index=route_index,
            position=0,
            node_id=0,
            physical_arrival=inst.planning_start_min,
            service_start=inst.planning_start_min,
            departure=time,
            battery_arrival=battery,
            battery_departure=battery,
            remaining_load=remaining_load,
            continuous_work_arrival=0.0,
            continuous_work_departure=0.0,
            rest_duration=depot_hold,
        )
    ]
    nominal_charging_cost = 0.0

    for position in range(1, len(normalized)):
        origin = normalized[position - 1]
        destination = normalized[position]
        travel_time = inst.travel_time(origin, destination)
        energy = inst.energy(origin, destination)
        if continuous_work + travel_time > inst.max_continuous_work_min + EPSILON:
            return RoutePlan(
                normalized,
                False,
                stops=stops,
                violation=PlanViolation(
                    "continuous_work",
                    "The next arc exceeds the continuous-work limit.",
                    position=position,
                    insert_after=position - 1,
                    time=time,
                    battery=battery,
                    continuous_work=continuous_work,
                ),
            )
        if battery - energy < inst.safety_battery - EPSILON:
            return RoutePlan(
                normalized,
                False,
                stops=stops,
                violation=PlanViolation(
                    "battery",
                    "The next arc violates the safety-battery level.",
                    position=position,
                    insert_after=position - 1,
                    time=time,
                    battery=battery,
                    continuous_work=continuous_work,
                ),
            )

        time += travel_time
        continuous_work += travel_time
        battery -= energy
        physical_arrival = time
        service_start = time
        battery_arrival = battery
        work_arrival = continuous_work
        customer_wait = service_duration = 0.0
        planned_charge = charging_duration = charging_cost = rest_duration = 0.0

        if inst.is_customer(destination):
            customer = inst.customers_by_id[destination]
            customer_wait = max(0.0, customer.tw_start - time)
            service_start = time + customer_wait
            if service_start > customer.tw_end + EPSILON:
                return RoutePlan(
                    normalized,
                    False,
                    stops=stops,
                    violation=PlanViolation(
                        "time_window",
                        f"Customer {destination} is served after its due time.",
                        position=position,
                        time=time,
                        battery=battery,
                        continuous_work=continuous_work,
                    ),
                )
            service_duration = customer.service_time
            projected_work = (
                continuous_work + customer_wait + service_duration
            )
            if projected_work > inst.max_continuous_work_min + EPSILON:
                return RoutePlan(
                    normalized,
                    False,
                    stops=stops,
                    violation=PlanViolation(
                        "continuous_work",
                        "Customer waiting/service exceeds the work limit.",
                        position=position,
                        insert_after=position - 1,
                        time=physical_arrival - travel_time,
                        battery=battery_arrival + energy,
                        continuous_work=work_arrival - travel_time,
                    ),
                )
            time = service_start + service_duration
            continuous_work = projected_work
            remaining_load -= customer.demand

        elif inst.is_station(destination):
            station = inst.stations_by_id[destination]
            interval = inst.time_to_interval(time)
            if station.reported_at(interval) >= station.chargers:
                return RoutePlan(
                    normalized,
                    False,
                    stops=stops,
                    violation=PlanViolation(
                        "reported_unavailable",
                        f"Station {destination} has no reported usable charger.",
                        position=position,
                        time=time,
                        battery=battery,
                        continuous_work=continuous_work,
                    ),
                )

            segment_energy = _energy_to_next_charging_opportunity(
                inst, normalized, position
            )
            desired_departure_battery = min(
                inst.battery_capacity,
                inst.safety_battery + segment_energy,
            )
            planned_charge = max(0.0, desired_departure_battery - battery)
            charging_duration, charging_cost = _charge_allocation_cost(
                inst, destination, time, planned_charge
            )
            if not np.isfinite(charging_duration):
                return RoutePlan(
                    normalized,
                    False,
                    stops=stops,
                    violation=PlanViolation(
                        "horizon",
                        "Charging cannot finish within the planning horizon.",
                        position=position,
                        time=time,
                        battery=battery,
                        continuous_work=continuous_work,
                    ),
                )
            projected_departure = time + charging_duration
            forward_work = _work_until_next_eligible_stop(
                inst, normalized, position, projected_departure
            )
            needs_rest = (
                continuous_work + charging_duration + forward_work
                > inst.max_continuous_work_min + EPSILON
            )
            if needs_rest:
                rest_duration = inst.min_rest_min
                stop_duration = max(charging_duration, rest_duration)
                continuous_work = 0.0
            else:
                stop_duration = charging_duration
                continuous_work += charging_duration
            time += stop_duration
            battery = min(inst.battery_capacity, battery + planned_charge)
            nominal_charging_cost += charging_cost

        if time > inst.planning_end_min + EPSILON:
            return RoutePlan(
                normalized,
                False,
                stops=stops,
                violation=PlanViolation(
                    "horizon",
                    "The route returns or finishes after the planning horizon.",
                    position=position,
                    time=time,
                    battery=battery,
                    continuous_work=continuous_work,
                ),
            )

        stops.append(
            PlanStop(
                route_index=route_index,
                position=position,
                node_id=destination,
                physical_arrival=physical_arrival,
                service_start=service_start,
                departure=time,
                battery_arrival=battery_arrival,
                battery_departure=battery,
                remaining_load=remaining_load,
                continuous_work_arrival=work_arrival,
                continuous_work_departure=continuous_work,
                customer_waiting=customer_wait,
                service_duration=service_duration,
                planned_charge=planned_charge,
                charging_duration=charging_duration,
                charging_cost=charging_cost,
                rest_duration=rest_duration,
            )
        )

    return RoutePlan(
        route=normalized,
        feasible=True,
        stops=stops,
        duty_duration=max(time - inst.planning_start_min, 0.0),
        planned_distance=route_distance(inst, normalized),
        nominal_charging_cost=nominal_charging_cost,
    )


def _candidate_stations_for_insertion(
    inst: EVRPInstance,
    route: Sequence[int],
    violation: PlanViolation,
) -> List[int]:
    if violation.insert_after is None:
        return []
    position = violation.insert_after
    origin = route[position]
    destination = route[position + 1]
    used = {node for node in route if inst.is_station(node)}
    candidates = []
    for station_id in inst.station_ids:
        if station_id in used:
            continue
        if (
            violation.battery - inst.energy(origin, station_id)
            < inst.safety_battery - EPSILON
        ):
            continue
        if (
            inst.battery_capacity - inst.energy(station_id, destination)
            < inst.safety_battery - EPSILON
        ):
            continue
        if (
            violation.continuous_work
            + inst.travel_time(origin, station_id)
            > inst.max_continuous_work_min + EPSILON
        ):
            continue
        detour = (
            inst.distance(origin, station_id)
            + inst.distance(station_id, destination)
            - inst.distance(origin, destination)
        )
        candidates.append((detour, station_id))
    return [station_id for _, station_id in sorted(candidates)]


def _restore_route_recursive(
    inst: EVRPInstance,
    route: Sequence[int],
    route_index: int,
    remaining_insertions: int,
    visited: Set[Tuple[int, ...]],
) -> RoutePlan:
    normalized = ensure_route(inst, route)
    signature = tuple(normalized)
    if signature in visited:
        return RoutePlan(
            normalized,
            False,
            violation=PlanViolation(
                "cycle", "Restoration revisited an earlier route.", 0
            ),
        )
    visited.add(signature)
    plan = propagate_nominal_route(inst, normalized, route_index)
    if plan.feasible or remaining_insertions <= 0 or plan.violation is None:
        return plan

    violation = plan.violation
    if violation.kind in {"battery", "continuous_work"}:
        for station_id in _candidate_stations_for_insertion(
            inst, normalized, violation
        ):
            insertion = int(violation.insert_after or 0) + 1
            candidate = normalized[:insertion] + [station_id] + normalized[insertion:]
            restored = _restore_route_recursive(
                inst,
                candidate,
                route_index,
                remaining_insertions - 1,
                visited,
            )
            if restored.feasible:
                return restored
    elif violation.kind == "reported_unavailable":
        position = violation.position
        origin = normalized[position - 1]
        successor = normalized[position + 1]
        used = {node for node in normalized if inst.is_station(node)}
        alternatives = []
        for station_id in inst.station_ids:
            if station_id in used:
                continue
            detour = (
                inst.distance(origin, station_id)
                + inst.distance(station_id, successor)
                - inst.distance(origin, normalized[position])
                - inst.distance(normalized[position], successor)
            )
            alternatives.append((detour, station_id))
        for _, station_id in sorted(alternatives):
            candidate = list(normalized)
            candidate[position] = station_id
            restored = _restore_route_recursive(
                inst,
                candidate,
                route_index,
                remaining_insertions - 1,
                visited,
            )
            if restored.feasible:
                return restored
    return plan


def _complete_backup_records(
    inst: EVRPInstance,
    routes: Sequence[Sequence[int]],
) -> Dict[StopKey, Optional[int]]:
    backups: Dict[StopKey, Optional[int]] = {}
    for route_index, raw_route in enumerate(routes):
        route = ensure_route(inst, raw_route)
        used = {node for node in route if inst.is_station(node)}
        for position, primary in enumerate(route):
            if not inst.is_station(primary):
                continue
            predecessor = route[position - 1]
            successor = route[position + 1]
            alternatives = []
            for station_id in inst.station_ids:
                if station_id == primary or station_id in used:
                    continue
                delta = (
                    inst.distance(predecessor, station_id)
                    + inst.distance(station_id, successor)
                    - inst.distance(predecessor, primary)
                    - inst.distance(primary, successor)
                )
                alternatives.append((delta, station_id))
            selected: Optional[int] = None
            for _, station_id in sorted(alternatives):
                substituted = list(route)
                substituted[position] = station_id
                if propagate_nominal_route(
                    inst, substituted, route_index
                ).feasible:
                    selected = station_id
                    break
            backups[(route_index, position)] = selected
    return backups


def certify_and_restore_solution(
    inst: EVRPInstance,
    solution: Solution,
    *,
    require_all_customers: bool = True,
    complete_backups: bool = True,
) -> PlanningResult:
    """Restore every route and return an immutable executable plan copy."""
    customer_visits = solution.customers(inst)
    if len(customer_visits) != len(set(customer_visits)):
        return PlanningResult(
            solution.copy(), False, reason="A customer appears more than once."
        )
    if require_all_customers and sorted(customer_visits) != sorted(
        inst.customer_ids
    ):
        return PlanningResult(
            solution.copy(),
            False,
            reason="The candidate is not customer-complete.",
        )

    route_plans: List[RoutePlan] = []
    restored_routes: List[List[int]] = []
    for route_index, route in enumerate(solution.routes):
        if not any(inst.is_customer(node) for node in route):
            continue
        restored = _restore_route_recursive(
            inst,
            route,
            route_index,
            remaining_insertions=len(inst.station_ids),
            visited=set(),
        )
        if not restored.feasible:
            message = (
                restored.violation.message
                if restored.violation is not None
                else "Unknown deterministic propagation failure."
            )
            return PlanningResult(
                solution.copy(),
                False,
                route_plans=route_plans + [restored],
                reason=message,
            )
        route_plans.append(restored)
        restored_routes.append(restored.route)

    planned_charges: Dict[StopKey, float] = {}
    planned_rests: Dict[StopKey, float] = {}
    for new_route_index, route_plan in enumerate(route_plans):
        # Route indices are compacted if an empty route was removed.
        for stop in route_plan.stops:
            key = (new_route_index, stop.position)
            if stop.planned_charge > EPSILON:
                planned_charges[key] = stop.planned_charge
            if stop.rest_duration > EPSILON:
                planned_rests[key] = stop.rest_duration

    restored_solution = Solution(
        routes=restored_routes,
        planned_charges=planned_charges,
        planned_rests=planned_rests,
    )
    if complete_backups:
        restored_solution.backups = _complete_backup_records(
            inst, restored_routes
        )
    return PlanningResult(
        restored_solution,
        True,
        route_plans=route_plans,
    )


def planning_cost(
    inst: EVRPInstance, planning: PlanningResult
) -> Tuple[float, float, float]:
    if not planning.feasible:
        return float("inf"), float("inf"), float("inf")
    vehicle_cost = inst.vehicle_use_cost * len(planning.solution.routes)
    travel_cost = inst.travel_cost_per_km * sum(
        route_plan.planned_distance for route_plan in planning.route_plans
    )
    return vehicle_cost + travel_cost, vehicle_cost, travel_cost

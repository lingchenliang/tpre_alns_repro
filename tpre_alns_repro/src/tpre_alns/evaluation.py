"""Fixed-rule recourse simulation and the planning + expectation + CVaR objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .entities import EVRPInstance, EvalMetrics, Scenario, Solution, StopKey
from .planning import (
    EPSILON,
    PlanningResult,
    certify_and_restore_solution,
    ensure_route,
    planning_cost,
)
from .scenarios import normalized_probabilities, time_of_use_price


class InfeasiblePlanError(ValueError):
    """Raised when deterministic restoration cannot certify a candidate."""


@dataclass
class RealizedStop:
    route_index: int
    position: int
    planned_node: int
    realized_node: int
    physical_arrival: float
    service_start: float
    departure: float
    battery_arrival: float
    battery_departure: float
    continuous_work_departure: float
    planned_charge: float = 0.0
    actual_charge: float = 0.0
    customer_waiting: float = 0.0
    station_waiting: float = 0.0
    charging_duration: float = 0.0
    rest_duration: float = 0.0
    action: str = "none"
    charging_cost: float = 0.0
    waiting_cost: float = 0.0
    recourse_cost: float = 0.0
    penalty_cost: float = 0.0


@dataclass
class RouteEval:
    scenario_cost: float
    charging_cost: float
    waiting_cost: float
    driver_cost: float
    recourse_cost: float
    penalty_cost: float
    infeasible: bool
    backup_switches: int
    local_repairs: int
    route_time: float
    trace: List[RealizedStop] = field(default_factory=list)
    station_attribution: Dict[StopKey, float] = field(default_factory=dict)
    failure_trigger: Optional[StopKey] = None

    @property
    def cost(self) -> float:
        return self.scenario_cost


def weighted_cvar(
    costs: Sequence[float],
    probabilities: Optional[Sequence[float]] = None,
    alpha: float = 0.90,
) -> float:
    """Evaluate the discrete Rockafellar-Uryasev CVaR expression.

    Minimizing over candidate eta values is exact for a finite discrete
    distribution and works for non-uniform scenario probabilities.
    """
    values = np.asarray(costs, dtype=float)
    if values.size == 0:
        return 0.0
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")
    if probabilities is None:
        weights = np.full(values.size, 1.0 / values.size, dtype=float)
    else:
        weights = np.asarray(probabilities, dtype=float)
        if weights.shape != values.shape:
            raise ValueError("probabilities must have the same length as costs.")
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0:
            weights = np.full(values.size, 1.0 / values.size, dtype=float)
        else:
            weights /= weights.sum()
    candidates = np.unique(values)
    objectives = [
        eta
        + float(np.sum(weights * np.maximum(values - eta, 0.0)))
        / max(1.0 - alpha, EPSILON)
        for eta in candidates
    ]
    return float(min(objectives))


def cvar(costs: Sequence[float], alpha: float = 0.90) -> float:
    """Backward-compatible equal-probability wrapper."""
    return weighted_cvar(costs, alpha=alpha)


def _charge_cost_and_duration(
    inst: EVRPInstance,
    station_id: int,
    start_time: float,
    amount: float,
    scenario: Optional[Scenario],
) -> Tuple[float, float]:
    if amount <= EPSILON:
        return 0.0, 0.0
    station = inst.stations_by_id[station_id]
    rate = station.charging_power_kw / 60.0
    remaining = min(amount, inst.battery_capacity)
    current_time = start_time
    cost = 0.0
    while remaining > EPSILON:
        if current_time >= inst.planning_end_min - EPSILON:
            return float("inf"), float("inf")
        interval = inst.time_to_interval(current_time)
        interval_end = min(
            inst.planning_start_min + (interval + 1) * inst.interval_min,
            inst.planning_end_min,
        )
        duration_available = max(interval_end - current_time, 0.0)
        if duration_available <= EPSILON:
            current_time = interval_end + EPSILON
            continue
        energy = min(remaining, rate * duration_available)
        if scenario is None:
            price = time_of_use_price(inst.absolute_clock_min(current_time))
        else:
            price = float(scenario.price[station_id][interval])
        cost += energy * price
        duration = energy / rate
        current_time += duration
        remaining -= energy
    return current_time - start_time, cost


def _energy_to_next_opportunity(
    inst: EVRPInstance,
    route: Sequence[int],
    station_position: int,
    station_override: Optional[int] = None,
) -> float:
    current = station_override or route[station_position]
    energy = 0.0
    for position in range(station_position + 1, len(route)):
        destination = route[position]
        energy += inst.energy(current, destination)
        current = destination
        if inst.is_station(destination) or destination == inst.terminal_id:
            break
    return energy


def _validated_wait(
    inst: EVRPInstance,
    scenario: Scenario,
    station_id: int,
    arrival_time: float,
) -> Tuple[bool, float, float]:
    """Apply Equations (7)-(8) using half-open one-hour intervals."""
    current_time = arrival_time
    accumulated = 0.0
    for _ in range(inst.n_intervals + 1):
        if current_time >= inst.planning_end_min - EPSILON:
            return False, current_time, accumulated
        interval = inst.time_to_interval(current_time)
        state = scenario.state(inst, station_id, interval)
        if state == 0:
            return True, current_time, accumulated
        if state == 2:
            return False, current_time, accumulated
        delay = float(scenario.waiting_time[station_id][interval])
        if delay <= EPSILON:
            return False, current_time, accumulated
        current_time += delay
        accumulated += delay
    return False, current_time, accumulated


def _work_to_next_node(
    inst: EVRPInstance, origin: int, destination: int, departure_time: float
) -> float:
    work = inst.travel_time(origin, destination)
    arrival = departure_time + work
    if inst.is_customer(destination):
        customer = inst.customers_by_id[destination]
        work += max(0.0, customer.tw_start - arrival) + customer.service_time
    return work


def _station_stop(
    inst: EVRPInstance,
    route: Sequence[int],
    route_index: int,
    position: int,
    station_id: int,
    arrival_time: float,
    arrival_battery: float,
    arrival_work: float,
    waiting: float,
    charge_amount: float,
    scenario: Optional[Scenario],
    solution: Solution,
    rest_sync: bool = True,
) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """Return departure state and stop costs, or None on a horizon failure."""
    charging_start = arrival_time + waiting
    charging_duration, charging_cost = _charge_cost_and_duration(
        inst, station_id, charging_start, charge_amount, scenario
    )
    if not np.isfinite(charging_duration):
        return None
    nonrest_duration = waiting + charging_duration
    next_node = (
        route[position + 1]
        if position + 1 < len(route)
        else inst.terminal_id
    )
    projected_next_work = _work_to_next_node(
        inst, station_id, next_node, arrival_time + nonrest_duration
    )
    key = (route_index, position)
    needs_rest = (
        key in solution.planned_rests
        or arrival_work + nonrest_duration + projected_next_work
        > inst.max_continuous_work_min + EPSILON
    )
    rest_duration = inst.min_rest_min if needs_rest else 0.0
    stop_duration = (
        (
            max(nonrest_duration, rest_duration)
            if rest_sync
            else nonrest_duration + rest_duration
        )
        if needs_rest
        else nonrest_duration
    )
    departure = arrival_time + stop_duration
    if departure > inst.planning_end_min + EPSILON:
        return None
    departure_battery = min(
        inst.battery_capacity, arrival_battery + charge_amount
    )
    departure_work = 0.0 if needs_rest else arrival_work + nonrest_duration
    waiting_cost = waiting * inst.waiting_cost_per_min
    return (
        departure,
        departure_battery,
        departure_work,
        charging_duration,
        charging_cost,
        waiting_cost,
        rest_duration,
    )


def _suffix_is_deterministically_feasible(
    inst: EVRPInstance,
    route: Sequence[int],
    route_index: int,
    start_position: int,
    current_node: int,
    time: float,
    battery: float,
    continuous_work: float,
    solution: Solution,
) -> bool:
    """Fast complete downstream propagation used to validate a recourse action."""
    for position in range(start_position + 1, len(route)):
        node = route[position]
        travel = inst.travel_time(current_node, node)
        energy = inst.energy(current_node, node)
        if (
            continuous_work + travel > inst.max_continuous_work_min + EPSILON
            or battery - energy < inst.safety_battery - EPSILON
        ):
            return False
        time += travel
        continuous_work += travel
        battery -= energy
        if inst.is_customer(node):
            customer = inst.customers_by_id[node]
            wait = max(0.0, customer.tw_start - time)
            service_start = time + wait
            if service_start > customer.tw_end + EPSILON:
                return False
            continuous_work += wait + customer.service_time
            if continuous_work > inst.max_continuous_work_min + EPSILON:
                return False
            time = service_start + customer.service_time
        elif inst.is_station(node):
            station = inst.stations_by_id[node]
            interval = inst.time_to_interval(time)
            if station.reported_at(interval) >= station.chargers:
                return False
            amount = float(solution.planned_charges.get((route_index, position), 0.0))
            stop = _station_stop(
                inst,
                route,
                route_index,
                position,
                node,
                time,
                battery,
                continuous_work,
                0.0,
                amount,
                None,
                solution,
                True,
            )
            if stop is None:
                return False
            time, battery, continuous_work = stop[:3]
        if time > inst.planning_end_min + EPSILON:
            return False
        current_node = node
    return True


def _penalized_route(
    inst: EVRPInstance,
    route_index: int,
    time: float,
    charging_cost: float,
    waiting_cost: float,
    recourse_cost: float,
    trace: List[RealizedStop],
    station_attribution: Dict[StopKey, float],
    failure_trigger: Optional[StopKey],
    backup_switches: int,
    local_repairs: int,
) -> RouteEval:
    penalty = inst.infeasibility_penalty
    if failure_trigger is not None:
        station_attribution[failure_trigger] = (
            station_attribution.get(failure_trigger, 0.0) + penalty
        )
    if trace:
        trace[-1].penalty_cost += penalty
    duty = max(time - inst.planning_start_min, 0.0)
    driver = duty * inst.driver_cost_per_min
    scenario_cost = charging_cost + waiting_cost + driver + recourse_cost + penalty
    return RouteEval(
        scenario_cost=scenario_cost,
        charging_cost=charging_cost,
        waiting_cost=waiting_cost,
        driver_cost=driver,
        recourse_cost=recourse_cost,
        penalty_cost=penalty,
        infeasible=True,
        backup_switches=backup_switches,
        local_repairs=local_repairs,
        route_time=duty,
        trace=trace,
        station_attribution=station_attribution,
        failure_trigger=failure_trigger,
    )


def _simulate_planned_route(
    inst: EVRPInstance,
    solution: Solution,
    route_index: int,
    scenario: Optional[Scenario],
    *,
    use_backups: bool,
    rest_sync: bool,
) -> RouteEval:
    route = ensure_route(inst, solution.routes[route_index])
    depot_hold = float(solution.planned_rests.get((route_index, 0), 0.0))
    time = inst.planning_start_min + depot_hold
    battery = inst.initial_battery
    continuous_work = 0.0
    current_node = 0
    charging_cost = waiting_cost = recourse_cost = 0.0
    backup_switches = local_repairs = 0
    station_attribution: Dict[StopKey, float] = {}
    trace: List[RealizedStop] = [
        RealizedStop(
            route_index,
            0,
            0,
            0,
            inst.planning_start_min,
            inst.planning_start_min,
            time,
            battery,
            battery,
            0.0,
            rest_duration=depot_hold,
        )
    ]
    last_station_key: Optional[StopKey] = None

    for position in range(1, len(route)):
        planned_node = route[position]
        key = (route_index, position)

        if not inst.is_station(planned_node):
            travel = inst.travel_time(current_node, planned_node)
            energy = inst.energy(current_node, planned_node)
            if (
                continuous_work + travel
                > inst.max_continuous_work_min + EPSILON
                or battery - energy < inst.safety_battery - EPSILON
            ):
                return _penalized_route(
                    inst,
                    route_index,
                    time,
                    charging_cost,
                    waiting_cost,
                    recourse_cost,
                    trace,
                    station_attribution,
                    last_station_key,
                    backup_switches,
                    local_repairs,
                )
            time += travel
            continuous_work += travel
            battery -= energy
            arrival = time
            service_start = time
            customer_wait = 0.0
            if inst.is_customer(planned_node):
                customer = inst.customers_by_id[planned_node]
                customer_wait = max(0.0, customer.tw_start - time)
                service_start = time + customer_wait
                if service_start > customer.tw_end + EPSILON:
                    return _penalized_route(
                        inst,
                        route_index,
                        time,
                        charging_cost,
                        waiting_cost,
                        recourse_cost,
                        trace,
                        station_attribution,
                        last_station_key,
                        backup_switches,
                        local_repairs,
                    )
                continuous_work += customer_wait + customer.service_time
                if continuous_work > inst.max_continuous_work_min + EPSILON:
                    return _penalized_route(
                        inst,
                        route_index,
                        time,
                        charging_cost,
                        waiting_cost,
                        recourse_cost,
                        trace,
                        station_attribution,
                        last_station_key,
                        backup_switches,
                        local_repairs,
                    )
                time = service_start + customer.service_time
            if time > inst.planning_end_min + EPSILON:
                return _penalized_route(
                    inst,
                    route_index,
                    time,
                    charging_cost,
                    waiting_cost,
                    recourse_cost,
                    trace,
                    station_attribution,
                    last_station_key,
                    backup_switches,
                    local_repairs,
                )
            trace.append(
                RealizedStop(
                    route_index=route_index,
                    position=position,
                    planned_node=planned_node,
                    realized_node=planned_node,
                    physical_arrival=arrival,
                    service_start=service_start,
                    departure=time,
                    battery_arrival=battery,
                    battery_departure=battery,
                    continuous_work_departure=continuous_work,
                    customer_waiting=customer_wait,
                )
            )
            current_node = planned_node
            continue

        # Planned charging stop: inspect its state at the hypothetical arrival.
        primary = planned_node
        predecessor_state = (current_node, time, battery, continuous_work)
        travel_primary = inst.travel_time(current_node, primary)
        energy_primary = inst.energy(current_node, primary)
        primary_reachable = (
            continuous_work + travel_primary
            <= inst.max_continuous_work_min + EPSILON
            and battery - energy_primary >= inst.safety_battery - EPSILON
        )
        primary_arrival = time + travel_primary
        primary_battery = battery - energy_primary
        primary_work = continuous_work + travel_primary
        action = "none"
        committed = False

        if primary_reachable and primary_arrival <= inst.planning_end_min:
            if scenario is None:
                primary_state = 0
            else:
                primary_state = scenario.state(
                    inst, primary, inst.time_to_interval(primary_arrival)
                )
            endpoint_time = primary_arrival
            station_wait = 0.0
            if primary_state == 1 and scenario is not None:
                wait_ok, endpoint_time, station_wait = _validated_wait(
                    inst, scenario, primary, primary_arrival
                )
                primary_state = 0 if wait_ok else 2
                action = "wait" if wait_ok else "none"
            if primary_state == 0:
                charge_amount = float(
                    solution.planned_charges.get(key, 0.0)
                )
                stop = _station_stop(
                    inst,
                    route,
                    route_index,
                    position,
                    primary,
                    primary_arrival,
                    primary_battery,
                    primary_work,
                    station_wait,
                    charge_amount,
                    scenario,
                    solution,
                    rest_sync,
                )
                if stop is not None:
                    (
                        candidate_time,
                        candidate_battery,
                        candidate_work,
                        charge_duration,
                        charge_cost,
                        wait_cost,
                        rest_duration,
                    ) = stop
                    if _suffix_is_deterministically_feasible(
                        inst,
                        route,
                        route_index,
                        position,
                        primary,
                        candidate_time,
                        candidate_battery,
                        candidate_work,
                        solution,
                    ):
                        time = candidate_time
                        battery = candidate_battery
                        continuous_work = candidate_work
                        current_node = primary
                        charging_cost += charge_cost
                        waiting_cost += wait_cost
                        station_attribution[key] = (
                            station_attribution.get(key, 0.0) + wait_cost
                        )
                        trace.append(
                            RealizedStop(
                                route_index=route_index,
                                position=position,
                                planned_node=primary,
                                realized_node=primary,
                                physical_arrival=primary_arrival,
                                service_start=primary_arrival,
                                departure=time,
                                battery_arrival=primary_battery,
                                battery_departure=battery,
                                continuous_work_departure=continuous_work,
                                planned_charge=charge_amount,
                                actual_charge=charge_amount,
                                station_waiting=station_wait,
                                charging_duration=charge_duration,
                                rest_duration=rest_duration,
                                action=action,
                                charging_cost=charge_cost,
                                waiting_cost=wait_cost,
                            )
                        )
                        last_station_key = key
                        committed = True

        if committed:
            continue

        # Wait-first failed or the primary was failed/unreachable.  Roll back to
        # the stored predecessor and try the assigned backup, then local repair.
        current_node, time, battery, continuous_work = predecessor_state
        assigned = solution.backups.get(key) if use_backups else None
        planned_station_ids = {
            node for node in route if inst.is_station(node) and node != primary
        }
        successor = route[position + 1]
        alternatives: List[Tuple[str, int, float]] = []
        if assigned is not None:
            delta = (
                inst.distance(route[position - 1], assigned)
                + inst.distance(assigned, successor)
                - inst.distance(route[position - 1], primary)
                - inst.distance(primary, successor)
            )
            alternatives.append(("assigned_backup", assigned, delta))
        for station_id in inst.station_ids:
            if (
                station_id in {primary, assigned}
                or station_id in planned_station_ids
            ):
                continue
            delta = (
                inst.distance(route[position - 1], station_id)
                + inst.distance(station_id, successor)
                - inst.distance(route[position - 1], primary)
                - inst.distance(primary, successor)
            )
            alternatives.append(("local_repair", station_id, delta))
        alternatives.sort(
            key=lambda item: (
                0 if item[0] == "assigned_backup" else 1,
                item[2],
                item[1],
            )
        )

        for recourse_action, station_id, delta_distance in alternatives:
            travel = inst.travel_time(current_node, station_id)
            energy = inst.energy(current_node, station_id)
            if (
                continuous_work + travel
                > inst.max_continuous_work_min + EPSILON
                or battery - energy < inst.safety_battery - EPSILON
            ):
                continue
            arrival = time + travel
            interval = inst.time_to_interval(arrival)
            if (
                scenario is not None
                and scenario.state(inst, station_id, interval) != 0
            ):
                continue
            arrival_battery = battery - energy
            arrival_work = continuous_work + travel
            required = (
                inst.safety_battery
                + _energy_to_next_opportunity(
                    inst, route, position, station_override=station_id
                )
            )
            desired = min(inst.battery_capacity, required)
            actual_charge = max(0.0, desired - arrival_battery)
            stop = _station_stop(
                inst,
                route,
                route_index,
                position,
                station_id,
                arrival,
                arrival_battery,
                arrival_work,
                0.0,
                actual_charge,
                scenario,
                solution,
                rest_sync,
            )
            if stop is None:
                continue
            (
                candidate_time,
                candidate_battery,
                candidate_work,
                charge_duration,
                charge_cost,
                wait_cost,
                rest_duration,
            ) = stop
            if not _suffix_is_deterministically_feasible(
                inst,
                route,
                route_index,
                position,
                station_id,
                candidate_time,
                candidate_battery,
                candidate_work,
                solution,
            ):
                continue
            delta_time = delta_distance / max(inst.travel_speed, EPSILON)
            action_cost = (
                inst.travel_cost_per_km * delta_distance
                + inst.driver_cost_per_min * delta_time
            )
            if recourse_action == "local_repair":
                action_cost += inst.local_repair_fixed_cost
                local_repairs += 1
            else:
                backup_switches += 1
            recourse_cost += action_cost
            station_attribution[key] = (
                station_attribution.get(key, 0.0) + action_cost
            )
            charging_cost += charge_cost
            time = candidate_time
            battery = candidate_battery
            continuous_work = candidate_work
            current_node = station_id
            trace.append(
                RealizedStop(
                    route_index=route_index,
                    position=position,
                    planned_node=primary,
                    realized_node=station_id,
                    physical_arrival=arrival,
                    service_start=arrival,
                    departure=time,
                    battery_arrival=arrival_battery,
                    battery_departure=battery,
                    continuous_work_departure=continuous_work,
                    planned_charge=float(
                        solution.planned_charges.get(key, 0.0)
                    ),
                    actual_charge=actual_charge,
                    charging_duration=charge_duration,
                    rest_duration=rest_duration,
                    action=recourse_action,
                    charging_cost=charge_cost,
                    recourse_cost=action_cost,
                )
            )
            last_station_key = key
            committed = True
            break

        if not committed:
            return _penalized_route(
                inst,
                route_index,
                time,
                charging_cost,
                waiting_cost,
                recourse_cost,
                trace,
                station_attribution,
                key,
                backup_switches,
                local_repairs,
            )

    duty = max(time - inst.planning_start_min, 0.0)
    driver_cost = duty * inst.driver_cost_per_min
    scenario_cost = charging_cost + waiting_cost + driver_cost + recourse_cost
    return RouteEval(
        scenario_cost=scenario_cost,
        charging_cost=charging_cost,
        waiting_cost=waiting_cost,
        driver_cost=driver_cost,
        recourse_cost=recourse_cost,
        penalty_cost=0.0,
        infeasible=False,
        backup_switches=backup_switches,
        local_repairs=local_repairs,
        route_time=duty,
        trace=trace,
        station_attribution=station_attribution,
    )


def assign_default_backups(
    inst: EVRPInstance, solution: Solution
) -> Dict[StopKey, Optional[int]]:
    planning = certify_and_restore_solution(inst, solution)
    if not planning.feasible:
        raise InfeasiblePlanError(planning.reason)
    return dict(planning.solution.backups)


def evaluate_route(
    inst: EVRPInstance,
    route: Sequence[int],
    scenario: Optional[Scenario] = None,
    backups: Optional[Dict] = None,
    use_backups: bool = True,
    rest_sync: bool = True,
) -> RouteEval:
    """Convenience wrapper for one route.

    Legacy station-id backup dictionaries are accepted and converted to stop
    records after deterministic route restoration.
    """
    planning = certify_and_restore_solution(
        inst, Solution(routes=[list(route)]), require_all_customers=False
    )
    if not planning.feasible:
        raise InfeasiblePlanError(planning.reason)
    planned = planning.solution
    if backups:
        for position, node in enumerate(planned.routes[0]):
            if inst.is_station(node):
                if (0, position) in backups:
                    planned.backups[(0, position)] = backups[(0, position)]
                elif node in backups:
                    planned.backups[(0, position)] = backups[node]
    return _simulate_planned_route(
        inst,
        planned,
        0,
        scenario,
        use_backups=use_backups,
        rest_sync=rest_sync,
    )


def evaluate_solution(
    inst: EVRPInstance,
    solution: Solution,
    scenarios: Sequence[Scenario],
    alpha: float = 0.90,
    risk_aversion: float = 0.50,
    use_backups: bool = True,
    rest_sync: bool = True,
    *,
    planning: Optional[PlanningResult] = None,
) -> EvalMetrics:
    """Evaluate Equation (61) without repeating planning cost in the tail."""
    planning = planning or certify_and_restore_solution(inst, solution)
    if not planning.feasible:
        raise InfeasiblePlanError(planning.reason)
    executable = planning.solution
    plan_cost, vehicle_cost, planned_travel_cost = planning_cost(inst, planning)

    scenario_list: List[Optional[Scenario]]
    if scenarios:
        scenario_list = list(scenarios)
        probabilities = normalized_probabilities(scenarios)
    else:
        scenario_list = [None]
        probabilities = np.ones(1, dtype=float)

    scenario_costs: List[float] = []
    component_rows: List[Tuple[float, float, float, float, float]] = []
    infeasible_flags: List[float] = []
    backup_counts: List[float] = []
    repair_counts: List[float] = []
    for scenario in scenario_list:
        route_evals = [
            _simulate_planned_route(
                inst,
                executable,
                route_index,
                scenario,
                use_backups=use_backups,
                rest_sync=rest_sync,
            )
            for route_index in range(len(executable.routes))
        ]
        scenario_costs.append(sum(route.scenario_cost for route in route_evals))
        component_rows.append(
            (
                sum(route.charging_cost for route in route_evals),
                sum(route.waiting_cost for route in route_evals),
                sum(route.driver_cost for route in route_evals),
                sum(route.recourse_cost for route in route_evals),
                sum(route.penalty_cost for route in route_evals),
            )
        )
        infeasible_flags.append(
            float(any(route.infeasible for route in route_evals))
        )
        backup_counts.append(
            float(sum(route.backup_switches for route in route_evals))
        )
        repair_counts.append(
            float(sum(route.local_repairs for route in route_evals))
        )

    scenario_values = np.asarray(scenario_costs, dtype=float)
    components = np.asarray(component_rows, dtype=float)
    expected_scenario = float(np.dot(probabilities, scenario_values))
    tail = weighted_cvar(scenario_values, probabilities, alpha)
    total_cost = plan_cost + expected_scenario
    objective = total_cost + risk_aversion * tail
    weighted_components = probabilities @ components

    return EvalMetrics(
        objective=objective,
        planning_cost=plan_cost,
        expected_scenario_cost=expected_scenario,
        cvar_scenario_cost=tail,
        total_cost=total_cost,
        vehicle_cost=vehicle_cost,
        planned_travel_cost=planned_travel_cost,
        charging_cost=float(weighted_components[0]),
        waiting_cost=float(weighted_components[1]),
        driver_cost=float(weighted_components[2]),
        recourse_cost=float(weighted_components[3]),
        penalty_cost=float(weighted_components[4]),
        infeasible_ratio=100.0
        * float(np.dot(probabilities, np.asarray(infeasible_flags))),
        backup_switches=float(
            np.dot(probabilities, np.asarray(backup_counts))
        ),
        local_repairs=float(np.dot(probabilities, np.asarray(repair_counts))),
        scenario_costs=scenario_values.tolist(),
        scenario_probabilities=probabilities.tolist(),
    )

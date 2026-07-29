"""Deterministic first-stage MILP reference used for small instances.

This is the bounded planning reference in Section 3.5.2, not an extensive-form
stochastic model.  It uses distinct start/terminal depot copies, two-sided load
and battery propagation, partial charging, tariff-interval charge allocation,
and the nominal vehicle + travel + charging + driver-duty objective.  Every
incumbent is post-certified by the same deterministic energy/rest propagation
used by ALNS; infeasible incumbents are not reported as executable references.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import time

from .entities import EVRPInstance, Scenario, Solution
from .evaluation import evaluate_solution
from .planning import certify_and_restore_solution
from .scenarios import time_of_use_price


@dataclass
class MILPReferenceConfig:
    max_vehicles: Optional[int] = None
    time_limit: float = 600.0
    mip_gap: float = 0.0
    threads: Optional[int] = None
    output_flag: bool = True
    big_m_load: float = 1050.0
    big_m_energy: float = 114.0
    big_m_time: float = 1400.0


@dataclass
class MILPReferenceResult:
    solution: Solution
    status: str
    executable: bool
    solver_objective: Optional[float]
    solver_bound: Optional[float]
    mip_gap: Optional[float]
    runtime_sec: float
    certified_nominal_cost: Optional[float]
    reporting_metrics: Optional[Dict[str, float]]
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "executable": self.executable,
            "solver_objective": self.solver_objective,
            "solver_bound": self.solver_bound,
            "mip_gap": self.mip_gap,
            "runtime_sec": self.runtime_sec,
            "certified_nominal_cost": self.certified_nominal_cost,
            "reporting_metrics": self.reporting_metrics,
            "rejection_reason": self.rejection_reason,
            "solution": self.solution.to_dict(),
        }


def _require_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - licensed optional dependency
        raise ImportError(
            "The MILP reference requires gurobipy 12.x and a working license."
        ) from exc
    return gp, GRB


def _status_name(status: int) -> str:
    return {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        15: "USER_OBJ_LIMIT",
        16: "WORK_LIMIT",
        17: "MEM_LIMIT",
    }.get(int(status), f"STATUS_{status}")


def solve_milp_reference(
    inst: EVRPInstance,
    scenarios: Sequence[Scenario] = (),
    config: Optional[MILPReferenceConfig] = None,
    evaluate_with_recourse: bool = True,
) -> MILPReferenceResult:
    config = config or MILPReferenceConfig()
    gp, GRB = _require_gurobi()
    customers = list(inst.customer_ids)
    stations = list(inst.station_ids)
    start = 0
    terminal = inst.terminal_id
    v_minus = [start] + customers + stations
    v_plus = customers + stations + [terminal]
    arcs = [(i, j) for i in v_minus for j in v_plus if i != j]
    vehicle_count = config.max_vehicles or len(customers)
    vehicles = list(range(vehicle_count))
    intervals = list(range(inst.n_intervals))

    model = gp.Model("deterministic_first_stage_evrptw")
    model.Params.TimeLimit = config.time_limit
    model.Params.MIPGap = config.mip_gap
    model.Params.OutputFlag = int(config.output_flag)
    if config.threads is not None:
        model.Params.Threads = int(config.threads)

    x = model.addVars(arcs, vehicles, vtype=GRB.BINARY, name="x")
    used = model.addVars(vehicles, vtype=GRB.BINARY, name="vehicle_used")
    assigned = model.addVars(customers, vehicles, vtype=GRB.BINARY, name="assigned")
    station_used = model.addVars(stations, vehicles, vtype=GRB.BINARY, name="station_used")
    load = model.addVars(
        [start] + customers + stations + [terminal],
        vehicles,
        lb=0.0,
        ub=inst.vehicle_capacity,
        name="remaining_load",
    )
    arrival = model.addVars(
        [start] + customers + stations + [terminal],
        vehicles,
        lb=0.0,
        ub=inst.planning_horizon_min + config.big_m_time,
        name="time",
    )
    battery = model.addVars(
        [start] + customers + stations + [terminal],
        vehicles,
        lb=0.0,
        ub=inst.battery_capacity,
        name="battery",
    )
    charge = model.addVars(
        stations,
        vehicles,
        lb=0.0,
        ub=inst.battery_capacity,
        name="charge",
    )
    tariff_interval = model.addVars(
        stations, vehicles, intervals, vtype=GRB.BINARY, name="tariff_interval"
    )
    interval_charge = model.addVars(
        stations,
        vehicles,
        intervals,
        lb=0.0,
        ub=inst.battery_capacity,
        name="interval_charge",
    )

    for customer in customers:
        model.addConstr(
            gp.quicksum(assigned[customer, vehicle] for vehicle in vehicles)
            == 1,
            name=f"serve_once_{customer}",
        )

    for vehicle in vehicles:
        model.addConstr(
            gp.quicksum(x[start, j, vehicle] for j in v_plus)
            == used[vehicle],
            name=f"start_once_{vehicle}",
        )
        model.addConstr(
            gp.quicksum(x[i, terminal, vehicle] for i in v_minus)
            == used[vehicle],
            name=f"terminal_once_{vehicle}",
        )
        model.addConstr(arrival[start, vehicle] == 0.0)
        model.addConstr(
            arrival[terminal, vehicle]
            <= inst.planning_horizon_min * used[vehicle]
        )
        model.addConstr(
            battery[start, vehicle] == inst.initial_battery * used[vehicle]
        )
        model.addConstr(
            load[start, vehicle]
            == gp.quicksum(
                inst.customers_by_id[customer].demand
                * assigned[customer, vehicle]
                for customer in customers
            )
        )
        for node in customers + stations:
            incoming = gp.quicksum(
                x[i, node, vehicle] for i in v_minus if i != node
            )
            outgoing = gp.quicksum(
                x[node, j, vehicle] for j in v_plus if j != node
            )
            model.addConstr(incoming == outgoing)
            model.addConstr(incoming <= 1)
            if node in customers:
                model.addConstr(assigned[node, vehicle] == incoming)
            else:
                model.addConstr(station_used[node, vehicle] == incoming)

    for customer in customers:
        data = inst.customers_by_id[customer]
        for vehicle in vehicles:
            model.addConstr(
                arrival[customer, vehicle]
                >= data.tw_start
                - config.big_m_time * (1 - assigned[customer, vehicle])
            )
            model.addConstr(
                arrival[customer, vehicle]
                <= data.tw_end
                + config.big_m_time * (1 - assigned[customer, vehicle])
            )

    for station in stations:
        station_data = inst.stations_by_id[station]
        for vehicle in vehicles:
            model.addConstr(
                charge[station, vehicle]
                <= inst.battery_capacity * station_used[station, vehicle]
            )
            model.addConstr(
                charge[station, vehicle] + battery[station, vehicle]
                <= inst.battery_capacity
                + inst.battery_capacity * (1 - station_used[station, vehicle])
            )
            model.addConstr(
                gp.quicksum(
                    tariff_interval[station, vehicle, interval]
                    for interval in intervals
                )
                == station_used[station, vehicle]
            )
            model.addConstr(
                gp.quicksum(
                    interval_charge[station, vehicle, interval]
                    for interval in intervals
                )
                == charge[station, vehicle]
            )
            for interval in intervals:
                lower = interval * inst.interval_min
                upper = (interval + 1) * inst.interval_min
                z = tariff_interval[station, vehicle, interval]
                model.addConstr(
                    arrival[station, vehicle]
                    >= lower - config.big_m_time * (1 - z)
                )
                model.addConstr(
                    arrival[station, vehicle]
                    <= upper + config.big_m_time * (1 - z)
                )
                model.addConstr(
                    interval_charge[station, vehicle, interval]
                    <= inst.battery_capacity * z
                )

    demand_by_node = {
        node: (
            inst.customers_by_id[node].demand
            if node in inst.customers_by_id
            else 0.0
        )
        for node in v_plus
    }
    for vehicle in vehicles:
        for origin, destination in arcs:
            service = (
                inst.customers_by_id[origin].service_time
                if origin in inst.customers_by_id
                else 0.0
            )
            charging_duration = (
                60.0
                / inst.stations_by_id[origin].charging_power_kw
                * charge[origin, vehicle]
                if origin in inst.stations_by_id
                else 0.0
            )
            model.addConstr(
                arrival[destination, vehicle]
                >= arrival[origin, vehicle]
                + service
                + charging_duration
                + inst.travel_time(origin, destination)
                - config.big_m_time * (1 - x[origin, destination, vehicle])
            )
            load_delta = load[destination, vehicle] - load[origin, vehicle] + demand_by_node[destination]
            model.addConstr(
                load_delta
                <= config.big_m_load * (1 - x[origin, destination, vehicle])
            )
            model.addConstr(
                load_delta
                >= -config.big_m_load * (1 - x[origin, destination, vehicle])
            )
            charged_origin = (
                charge[origin, vehicle] if origin in inst.stations_by_id else 0.0
            )
            energy_delta = (
                battery[destination, vehicle]
                - battery[origin, vehicle]
                - charged_origin
                + inst.energy(origin, destination)
            )
            model.addConstr(
                energy_delta
                <= config.big_m_energy * (1 - x[origin, destination, vehicle])
            )
            model.addConstr(
                energy_delta
                >= -config.big_m_energy * (1 - x[origin, destination, vehicle])
            )
            model.addConstr(
                battery[destination, vehicle]
                >= inst.safety_battery
                - config.big_m_energy * (1 - x[origin, destination, vehicle])
            )

    vehicle_cost = inst.vehicle_use_cost * gp.quicksum(
        used[vehicle] for vehicle in vehicles
    )
    travel_cost = gp.quicksum(
        inst.travel_cost_per_km
        * inst.distance(origin, destination)
        * x[origin, destination, vehicle]
        for origin, destination in arcs
        for vehicle in vehicles
    )
    charging_cost = gp.quicksum(
        time_of_use_price(
            inst.day_start_min + interval * inst.interval_min
        )
        * interval_charge[station, vehicle, interval]
        for station in stations
        for vehicle in vehicles
        for interval in intervals
    )
    driver_cost = inst.driver_cost_per_min * gp.quicksum(
        arrival[terminal, vehicle] for vehicle in vehicles
    )
    model.setObjective(
        vehicle_cost + travel_cost + charging_cost + driver_cost,
        GRB.MINIMIZE,
    )

    started = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - started
    status = _status_name(model.Status)
    if model.SolCount == 0:
        return MILPReferenceResult(
            solution=Solution(routes=[]),
            status=status,
            executable=False,
            solver_objective=None,
            solver_bound=(
                float(model.ObjBound) if hasattr(model, "ObjBound") else None
            ),
            mip_gap=None,
            runtime_sec=runtime,
            certified_nominal_cost=None,
            reporting_metrics=None,
            rejection_reason="No incumbent was returned.",
        )

    routes: List[List[int]] = []
    charge_targets: Dict[Tuple[int, int], float] = {}
    for vehicle in vehicles:
        if used[vehicle].X < 0.5:
            continue
        route = [start]
        current = start
        while current != terminal:
            successors = [
                destination
                for origin, destination in arcs
                if origin == current and x[origin, destination, vehicle].X > 0.5
            ]
            if not successors:
                break
            current = max(
                successors,
                key=lambda destination: x[
                    route[-1], destination, vehicle
                ].X,
            )
            route.append(int(current))
            if len(route) > len(v_minus) + 2:
                break
        if route[-1] == terminal and any(
            inst.is_customer(node) for node in route
        ):
            route_index = len(routes)
            routes.append(route)
            for position, node in enumerate(route):
                if inst.is_station(node):
                    charge_targets[(route_index, position)] = float(
                        charge[node, vehicle].X
                    )

    raw_solution = Solution(
        routes=routes, planned_charges=charge_targets
    )
    certification = certify_and_restore_solution(inst, raw_solution)
    if not certification.feasible:
        return MILPReferenceResult(
            solution=raw_solution,
            status=status,
            executable=False,
            solver_objective=float(model.ObjVal),
            solver_bound=float(model.ObjBound),
            mip_gap=float(model.MIPGap),
            runtime_sec=runtime,
            certified_nominal_cost=None,
            reporting_metrics=None,
            rejection_reason=certification.reason,
        )
    executable = certification.solution
    nominal = evaluate_solution(
        inst,
        executable,
        [],
        risk_aversion=0.0,
        planning=certification,
    )
    reporting = (
        evaluate_solution(inst, executable, scenarios).to_dict()
        if scenarios and evaluate_with_recourse
        else None
    )
    return MILPReferenceResult(
        solution=executable,
        status=status,
        executable=True,
        solver_objective=float(model.ObjVal),
        solver_bound=float(model.ObjBound),
        mip_gap=float(model.MIPGap),
        runtime_sec=runtime,
        certified_nominal_cost=nominal.total_cost,
        reporting_metrics=reporting,
    )


__all__ = [
    "MILPReferenceConfig",
    "MILPReferenceResult",
    "solve_milp_reference",
]

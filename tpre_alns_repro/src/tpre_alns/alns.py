"""Adaptive large-neighbourhood search with the Algorithm S1 screening gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .entities import EVRPInstance, Scenario, Solution, StopKey
from .evaluation import InfeasiblePlanError, evaluate_solution
from .evaluator import HeuristicRiskEvaluator
from .planning import (
    PlanningResult,
    certify_and_restore_solution,
    ensure_route,
)
from .scenarios import select_severity_scenarios


@dataclass
class ALNSConfig:
    max_iterations: int = 3000
    destroy_rate_min: float = 0.10
    destroy_rate_max: float = 0.30
    initial_temperature: float = 100.0
    cooling_rate: float = 0.995
    weight_update_period: int = 50
    high_risk_full_eval_probability: float = 0.05
    max_no_improve: int = 500
    risk_threshold_percentile: float = 80.0
    risk_history_initial: int = 20
    risk_history_update_period: int = 50
    risk_history_window: int = 200
    alpha: float = 0.90
    risk_aversion: float = 0.50
    risk_insertion_kappa: float = 0.01
    risk_insertion_shortlist: int = 5
    local_search_probability: float = 0.40
    target_failure_threshold: float = 0.50
    reaction_factor: float = 0.20
    minimum_operator_weight: float = 0.05
    use_evaluator: bool = True
    use_backups: bool = True
    use_cvar: bool = True
    rest_sync: bool = True
    deterministic_objective: bool = False
    destroy_operators: Tuple[str, ...] = (
        "random",
        "worst_cost",
        "route_segment",
        "risk_station_neighbourhood",
    )
    repair_operators: Tuple[str, ...] = (
        "greedy",
        "regret_2",
        "risk_reducing",
    )


@dataclass
class OperatorRecord:
    weight: float = 1.0
    segment_uses: int = 0
    segment_reward: float = 0.0


@dataclass
class SearchCounters:
    realized_iterations: int = 0
    forward_feasible_candidates: int = 0
    complete_scenario_evaluations: int = 0
    deterministic_feasibility_rejections: int = 0
    local_search_scenario_evaluations: int = 0
    proxy_inferences: int = 0
    screening_skips: int = 0


class TPREALNS:
    def __init__(
        self,
        inst: EVRPInstance,
        scenarios: Sequence[Scenario],
        config: Optional[ALNSConfig] = None,
        risk_evaluator: Optional[object] = None,
        seed: int = 1,
    ) -> None:
        self.inst = inst
        self.scenarios = list(scenarios)
        self.config = config or ALNSConfig()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.risk_evaluator = risk_evaluator or HeuristicRiskEvaluator()
        self.destroy_records = {
            name: OperatorRecord() for name in self.config.destroy_operators
        }
        self.repair_records = {
            name: OperatorRecord() for name in self.config.repair_operators
        }
        self.counters = SearchCounters()
        self.history: List[Dict[str, float]] = []
        self.local_scenarios = select_severity_scenarios(
            inst, self.scenarios, (10.0, 50.0, 90.0)
        )
        self._cached_station_vulnerability: Dict[StopKey, float] = {}

    def _objective_scenarios(self) -> Sequence[Scenario]:
        return [] if self.config.deterministic_objective else self.scenarios

    def evaluate(
        self,
        solution: Solution,
        planning: Optional[PlanningResult] = None,
    ):
        risk_aversion = (
            self.config.risk_aversion if self.config.use_cvar else 0.0
        )
        return evaluate_solution(
            self.inst,
            solution,
            self._objective_scenarios(),
            alpha=self.config.alpha,
            risk_aversion=risk_aversion,
            use_backups=self.config.use_backups,
            rest_sync=self.config.rest_sync,
            planning=planning,
        )

    def solve(self) -> Tuple[Solution, Dict[str, object]]:
        initial = self.initial_solution()
        initial_planning = certify_and_restore_solution(self.inst, initial)
        if not initial_planning.feasible:
            raise InfeasiblePlanError(initial_planning.reason)
        current = initial_planning.solution
        current_eval = self.evaluate(current, initial_planning)
        self.counters.complete_scenario_evaluations += 1
        best = current.copy()
        best_eval = current_eval
        temperature = self.config.initial_temperature
        no_best = 0
        risk_history: List[float] = []
        risk_threshold = float("inf")
        feasible_counter = 0

        for iteration in range(1, self.config.max_iterations + 1):
            self.counters.realized_iterations = iteration
            destroy_operator = self._select_operator(self.destroy_records)
            repair_operator = self._select_operator(self.repair_records)
            partial, removed = self.destroy(current, destroy_operator)
            candidate = self.repair(partial, removed, repair_operator)
            if candidate is None:
                self.counters.deterministic_feasibility_rejections += 1
                no_best += 1
                self._record_operator_use(
                    destroy_operator, repair_operator, 0.0
                )
                temperature *= self.config.cooling_rate
                if self._finish_iteration(iteration, no_best):
                    break
                continue

            candidate = self.local_search(candidate)
            planning = certify_and_restore_solution(self.inst, candidate)
            if not planning.feasible:
                self.counters.deterministic_feasibility_rejections += 1
                no_best += 1
                self._record_operator_use(
                    destroy_operator, repair_operator, 0.0
                )
                temperature *= self.config.cooling_rate
                if self._finish_iteration(iteration, no_best):
                    break
                continue

            candidate = planning.solution
            self.counters.forward_feasible_candidates += 1
            feasible_counter += 1
            score = 0.0
            evaluated_candidate: Optional[Solution] = candidate
            evaluated_planning: Optional[PlanningResult] = planning
            if self.config.use_evaluator:
                score = self._risk_score(candidate)
                self.counters.proxy_inferences += 1
                risk_history.append(score)
                if feasible_counter == self.config.risk_history_initial:
                    risk_threshold = float(
                        np.percentile(
                            risk_history,
                            self.config.risk_threshold_percentile,
                        )
                    )
                elif (
                    feasible_counter > self.config.risk_history_initial
                    and (
                        feasible_counter - self.config.risk_history_initial
                    )
                    % self.config.risk_history_update_period
                    == 0
                ):
                    window = risk_history[-self.config.risk_history_window :]
                    risk_threshold = float(
                        np.percentile(
                            window,
                            self.config.risk_threshold_percentile,
                        )
                    )

                if (
                    feasible_counter > self.config.risk_history_initial
                    and score > risk_threshold
                ):
                    repaired, repaired_score = self.targeted_repair(candidate)
                    self.counters.proxy_inferences += 1
                    if repaired is not None:
                        candidate = repaired
                        score = repaired_score
                        evaluated_candidate = candidate
                        # The route did not change; only its backup record did.
                        evaluated_planning = PlanningResult(
                            solution=candidate,
                            feasible=True,
                            route_plans=planning.route_plans,
                        )
                    if (
                        score > risk_threshold
                        and self.rng.random()
                        > self.config.high_risk_full_eval_probability
                    ):
                        evaluated_candidate = None
                        evaluated_planning = None
                        self.counters.screening_skips += 1

            accepted = False
            global_improvement = False
            accepted_improvement = False
            if evaluated_candidate is not None:
                candidate_eval = self.evaluate(
                    evaluated_candidate, evaluated_planning
                )
                self.counters.complete_scenario_evaluations += 1
                delta = candidate_eval.objective - current_eval.objective
                if delta < 0:
                    accepted = True
                    accepted_improvement = True
                else:
                    accepted = bool(
                        self.rng.random()
                        < np.exp(-delta / max(temperature, 1e-12))
                    )
                if accepted:
                    current = evaluated_candidate
                    current_eval = candidate_eval
                    if candidate_eval.objective < best_eval.objective:
                        best = evaluated_candidate.copy()
                        best_eval = candidate_eval
                        global_improvement = True
                        no_best = 0
                    else:
                        no_best += 1
                else:
                    no_best += 1
            else:
                no_best += 1

            reward = (
                10.0
                if global_improvement
                else 5.0
                if accepted_improvement
                else 2.0
                if accepted
                else 0.0
            )
            self._record_operator_use(
                destroy_operator, repair_operator, reward
            )
            temperature *= self.config.cooling_rate
            if iteration % 10 == 0 or global_improvement:
                self.history.append(
                    {
                        "iteration": float(iteration),
                        "best_objective": best_eval.objective,
                        "current_objective": current_eval.objective,
                        "risk_score": score,
                        "risk_threshold": risk_threshold,
                    }
                )
            if self._finish_iteration(iteration, no_best):
                break

        info: Dict[str, object] = {
            "metrics": best_eval.to_dict(),
            "history": self.history,
            "counters": asdict(self.counters),
            "destroy_weights": {
                name: record.weight
                for name, record in self.destroy_records.items()
            },
            "repair_weights": {
                name: record.weight
                for name, record in self.repair_records.items()
            },
            "seed": self.seed,
            "solution": best.to_dict(),
        }
        return best, info

    def _finish_iteration(self, iteration: int, no_best: int) -> bool:
        if iteration % self.config.weight_update_period == 0:
            self._update_operator_weights(self.destroy_records)
            self._update_operator_weights(self.repair_records)
        return no_best >= self.config.max_no_improve

    def initial_solution(self) -> Solution:
        ordered = sorted(
            self.inst.customers,
            key=lambda customer: (
                customer.tw_start,
                self.inst.distance(0, customer.node_id),
                customer.node_id,
            ),
        )
        routes: List[List[int]] = []
        current_customers: List[int] = []
        current_load = 0.0
        for customer in ordered:
            tentative = current_customers + [customer.node_id]
            raw_route = [0] + tentative + [self.inst.terminal_id]
            tentative_plan = certify_and_restore_solution(
                self.inst,
                Solution(routes=[raw_route]),
                require_all_customers=False,
            )
            if (
                current_customers
                and (
                    current_load + customer.demand
                    > self.inst.vehicle_capacity
                    or not tentative_plan.feasible
                )
            ):
                routes.append(
                    [0] + current_customers + [self.inst.terminal_id]
                )
                current_customers = [customer.node_id]
                current_load = customer.demand
            else:
                current_customers = tentative
                current_load += customer.demand
        if current_customers:
            routes.append([0] + current_customers + [self.inst.terminal_id])
        planning = certify_and_restore_solution(
            self.inst, Solution(routes=routes)
        )
        if not planning.feasible:
            # Guaranteed fallback: one customer per vehicle.
            routes = [
                [0, customer.node_id, self.inst.terminal_id]
                for customer in ordered
            ]
            planning = certify_and_restore_solution(
                self.inst, Solution(routes=routes)
            )
        if not planning.feasible:
            raise InfeasiblePlanError(planning.reason)
        return planning.solution

    def destroy(
        self, solution: Solution, operator: str
    ) -> Tuple[Solution, List[int]]:
        candidate = solution.copy()
        customers = candidate.customers(self.inst)
        if not customers:
            return candidate, []
        fraction = self.rng.uniform(
            self.config.destroy_rate_min, self.config.destroy_rate_max
        )
        quota = max(1, int(np.ceil(fraction * len(customers))))
        removed: List[int] = []

        if operator == "worst_cost":
            contributions = []
            for route_index, route in enumerate(candidate.routes):
                for position, node in enumerate(route):
                    if not self.inst.is_customer(node):
                        continue
                    predecessor = route[position - 1]
                    successor = route[position + 1]
                    saving = (
                        self.inst.distance(predecessor, node)
                        + self.inst.distance(node, successor)
                        - self.inst.distance(predecessor, successor)
                    )
                    contributions.append(
                        (saving, -node, route_index, position, node)
                    )
            contributions.sort(reverse=True)
            removed = [item[-1] for item in contributions[:quota]]
        elif operator == "route_segment":
            eligible = [
                (index, route)
                for index, route in enumerate(candidate.routes)
                if any(self.inst.is_customer(node) for node in route)
            ]
            route_index, route = eligible[
                int(self.rng.integers(0, len(eligible)))
            ]
            positions = [
                index
                for index, node in enumerate(route)
                if self.inst.is_customer(node)
            ]
            if len(positions) <= quota:
                removed = [route[position] for position in positions]
            else:
                best_segment: Tuple[float, List[int]] = (-float("inf"), [])
                for start in range(0, len(positions) - quota + 1):
                    selected_positions = positions[start : start + quota]
                    selected_nodes = [
                        route[position] for position in selected_positions
                    ]
                    saving = sum(
                        self._customer_saving(route, node)
                        for node in selected_nodes
                    )
                    if saving > best_segment[0]:
                        best_segment = (saving, selected_nodes)
                removed = best_segment[1]
        elif operator == "risk_station_neighbourhood":
            removed = self._risk_neighbourhood_customers(
                candidate, quota
            )
        else:
            removed = [
                int(value)
                for value in self.rng.choice(
                    customers, size=min(quota, len(customers)), replace=False
                )
            ]

        self._remove_customers_and_stations(candidate, removed)
        return candidate, removed

    def _customer_saving(self, route: Sequence[int], customer: int) -> float:
        position = route.index(customer)
        predecessor, successor = route[position - 1], route[position + 1]
        return (
            self.inst.distance(predecessor, customer)
            + self.inst.distance(customer, successor)
            - self.inst.distance(predecessor, successor)
        )

    def _risk_neighbourhood_customers(
        self, solution: Solution, quota: int
    ) -> List[int]:
        vulnerability, _ = self._station_diagnostics(solution)
        if not vulnerability:
            customers = solution.customers(self.inst)
            return [
                int(value)
                for value in self.rng.choice(
                    customers, size=min(quota, len(customers)), replace=False
                )
            ]
        selected_key = max(
            vulnerability,
            key=lambda key: (vulnerability[key], -key[0], -key[1]),
        )
        route_index, station_position = selected_key
        route = solution.routes[route_index]
        customer_positions = [
            position
            for position, node in enumerate(route)
            if self.inst.is_customer(node)
        ]
        customer_positions.sort(
            key=lambda position: (
                abs(position - station_position),
                0 if position < station_position else 1,
                position,
            )
        )
        removed = [
            route[position] for position in customer_positions[:quota]
        ]
        if len(removed) < quota:
            remaining = [
                node
                for node in solution.customers(self.inst)
                if node not in removed
            ]
            remaining.sort(
                key=lambda node: max(
                    (
                        vulnerability.get((ridx, pos), 0.0)
                        for ridx, candidate_route in enumerate(solution.routes)
                        for pos, value in enumerate(candidate_route)
                        if value == node
                    ),
                    default=0.0,
                ),
                reverse=True,
            )
            removed.extend(remaining[: quota - len(removed)])
        return removed

    def _remove_customers_and_stations(
        self, solution: Solution, customers: Sequence[int]
    ) -> None:
        removed = set(customers)
        routes = []
        for route in solution.routes:
            customer_sequence = [
                node
                for node in route
                if self.inst.is_customer(node) and node not in removed
            ]
            if customer_sequence:
                routes.append(
                    [0] + customer_sequence + [self.inst.terminal_id]
                )
        solution.routes = routes
        solution.planned_charges = {}
        solution.backups = {}
        solution.planned_rests = {}

    def repair(
        self,
        partial: Solution,
        removed: Sequence[int],
        operator: str,
    ) -> Optional[Solution]:
        if operator == "regret_2":
            return self._regret_repair(partial, list(removed))
        solution = partial.copy()
        for customer in removed:
            options = self._insertion_options(solution, int(customer))
            if not options:
                return None
            if operator == "risk_reducing":
                shortlist = options[: self.config.risk_insertion_shortlist]
                scored = []
                for nominal_cost, candidate in shortlist:
                    risk = self._risk_score(candidate)
                    self.counters.proxy_inferences += 1
                    scored.append(
                        (
                            nominal_cost
                            + self.config.risk_insertion_kappa * risk,
                            candidate,
                        )
                    )
                solution = min(scored, key=lambda item: item[0])[1]
            else:
                solution = options[0][1]
        return solution

    def _regret_repair(
        self, partial: Solution, remaining: List[int]
    ) -> Optional[Solution]:
        solution = partial.copy()
        while remaining:
            choices = []
            for customer in remaining:
                options = self._insertion_options(solution, customer)
                if not options:
                    continue
                best = options[0][0]
                second = options[1][0] if len(options) > 1 else best + 1e6
                choices.append((second - best, -best, -customer, customer, options[0][1]))
            if not choices:
                return None
            _, _, _, selected_customer, selected_solution = max(choices)
            solution = selected_solution
            remaining.remove(selected_customer)
        return solution

    def _insertion_options(
        self, solution: Solution, customer: int
    ) -> List[Tuple[float, Solution]]:
        options: List[Tuple[float, Solution]] = []
        routes = solution.routes or []
        for route_index in range(len(routes) + 1):
            if route_index == len(routes):
                positions = [1]
                base_route = [0, self.inst.terminal_id]
            else:
                base_route = ensure_route(self.inst, routes[route_index])
                positions = list(range(1, len(base_route)))
            for position in positions:
                candidate_routes = [list(route) for route in routes]
                inserted = (
                    base_route[:position]
                    + [customer]
                    + base_route[position:]
                )
                if route_index == len(routes):
                    candidate_routes.append(inserted)
                else:
                    candidate_routes[route_index] = inserted
                raw_candidate = Solution(routes=candidate_routes)
                planning = certify_and_restore_solution(
                    self.inst,
                    raw_candidate,
                    require_all_customers=False,
                )
                if not planning.feasible:
                    continue
                nominal = evaluate_solution(
                    self.inst,
                    planning.solution,
                    [],
                    risk_aversion=0.0,
                    planning=planning,
                ).total_cost
                options.append((nominal, planning.solution))
        options.sort(key=lambda item: item[0])
        return options

    def local_search(self, solution: Solution) -> Solution:
        current = solution.copy()
        for route_index, route in enumerate(list(current.routes)):
            customer_positions = [
                position
                for position, node in enumerate(route)
                if self.inst.is_customer(node)
            ]
            if (
                len(customer_positions) < 2
                or self.rng.random() > self.config.local_search_probability
            ):
                continue
            left, right = sorted(
                int(value)
                for value in self.rng.choice(
                    customer_positions, size=2, replace=False
                )
            )
            candidate = current.copy()
            candidate.routes[route_index][left : right + 1] = reversed(
                candidate.routes[route_index][left : right + 1]
            )
            candidate.planned_charges = {}
            candidate.backups = {}
            candidate.planned_rests = {}
            planning = certify_and_restore_solution(self.inst, candidate)
            if not planning.feasible:
                continue
            local_scenarios = (
                []
                if self.config.deterministic_objective
                else self.local_scenarios
            )
            old_value = evaluate_solution(
                self.inst,
                current,
                local_scenarios,
                risk_aversion=0.0,
            ).total_cost
            new_value = evaluate_solution(
                self.inst,
                planning.solution,
                local_scenarios,
                risk_aversion=0.0,
                planning=planning,
            ).total_cost
            self.counters.local_search_scenario_evaluations += max(
                len(local_scenarios), 1
            )
            if new_value < old_value:
                current = planning.solution
        return current

    def targeted_repair(
        self, solution: Solution
    ) -> Tuple[Optional[Solution], float]:
        vulnerability, failure_share = self._station_diagnostics(solution)
        if not vulnerability:
            return solution.copy(), self._risk_score(solution)
        primary_key = max(
            vulnerability,
            key=lambda key: (vulnerability[key], -key[0], -key[1]),
        )
        if (
            failure_share.get(primary_key, 0.0)
            <= self.config.target_failure_threshold
        ):
            return solution.copy(), self._risk_score(solution)

        route_index, position = primary_key
        route = solution.routes[route_index]
        primary = route[position]
        used = {node for node in route if self.inst.is_station(node)}
        feasible_alternatives = []
        for alternative in self.inst.station_ids:
            if alternative == primary or alternative in used:
                continue
            substituted = solution.copy()
            substituted.routes[route_index][position] = alternative
            substituted.planned_charges = {}
            substituted.backups = {}
            substituted.planned_rests = {}
            if not certify_and_restore_solution(
                self.inst, substituted
            ).feasible:
                continue
            if hasattr(self.risk_evaluator, "score_backup_candidate"):
                score = float(
                    self.risk_evaluator.score_backup_candidate(
                        self.inst,
                        solution,
                        primary_key,
                        alternative,
                        self.scenarios,
                    )
                )
            else:
                score = self._risk_score(substituted)
            feasible_alternatives.append((score, alternative))
        if not feasible_alternatives:
            return solution.copy(), self._risk_score(solution)
        best_score, best_alternative = min(feasible_alternatives)
        repaired = solution.copy()
        repaired.backups[primary_key] = best_alternative
        return repaired, float(best_score)

    def _risk_score(self, solution: Solution) -> float:
        if not hasattr(self.risk_evaluator, "score_solution"):
            return 0.0
        return float(
            self.risk_evaluator.score_solution(
                self.inst, solution, self.scenarios
            )
        )

    def _station_diagnostics(
        self, solution: Solution
    ) -> Tuple[Dict[StopKey, float], Dict[StopKey, float]]:
        if hasattr(self.risk_evaluator, "station_diagnostics"):
            return self.risk_evaluator.station_diagnostics(
                self.inst, solution, self.scenarios
            )
        return {}, {}

    def _select_operator(
        self, records: Dict[str, OperatorRecord]
    ) -> str:
        names = list(records)
        weights = np.asarray(
            [records[name].weight for name in names], dtype=float
        )
        weights /= weights.sum()
        return str(self.rng.choice(names, p=weights))

    def _record_operator_use(
        self, destroy: str, repair: str, reward: float
    ) -> None:
        for name, records in (
            (destroy, self.destroy_records),
            (repair, self.repair_records),
        ):
            records[name].segment_uses += 1
            records[name].segment_reward += reward

    def _update_operator_weights(
        self, records: Dict[str, OperatorRecord]
    ) -> None:
        for record in records.values():
            if record.segment_uses:
                average_reward = (
                    record.segment_reward / record.segment_uses
                )
                record.weight = max(
                    self.config.minimum_operator_weight,
                    (1.0 - self.config.reaction_factor) * record.weight
                    + self.config.reaction_factor * average_reward,
                )
            record.segment_uses = 0
            record.segment_reward = 0.0

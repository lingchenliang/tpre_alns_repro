"""Twin-branch perturbation evaluator and the frozen hand-crafted baseline."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .entities import EVRPInstance, Scenario, Solution, StopKey
from .features import (
    FEATURE_NAMES,
    FeatureNormalizer,
    PhysicalScales,
    RouteScenarioSample,
    VulnerabilityNormalizer,
    fit_training_normalizers,
    make_training_samples,
    route_stop_features,
)
from .planning import certify_and_restore_solution
from .scenarios import normalized_probabilities

try:  # PyTorch is an optional dependency for routing-only installations.
    import torch
    import torch.nn as nn
    import torch.nn.functional as torch_functional
except Exception:  # pragma: no cover - exercised in minimal installations
    torch = None
    nn = None
    torch_functional = None


TWIN_PARAMETER_COUNT = 65_091
SINGLE_BRANCH_PARAMETER_COUNT = 65_235


def twin_parameter_count_formula() -> int:
    encoder = (24 * 128 + 128) + (128 * 64 + 64)
    route_head = (256 * 64 + 64) + (64 * 1 + 1)
    station_head = (320 * 64 + 64) + (64 * 1 + 1)
    return encoder + 2 * route_head + station_head


def single_branch_parameter_count_formula() -> int:
    encoder = (48 * 224 + 224) + (224 * 112 + 112)
    route_head = (112 * 64 + 64) + (64 * 1 + 1)
    station_head = (224 * 64 + 64) + (64 * 1 + 1)
    return encoder + 2 * route_head + station_head


if nn is not None:

    class TwinBranchNet(nn.Module):
        """24->128->64 shared stop encoder with masked route pooling."""

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
        def _masked_mean(embeddings, mask):
            mask_float = mask.unsqueeze(-1).to(embeddings.dtype)
            denominator = mask_float.sum(dim=1).clamp_min(1.0)
            return (embeddings * mask_float).sum(dim=1) / denominator

        def forward(self, nominal, perturbed, stop_mask):
            nominal_stops = self.encoder(nominal)
            perturbed_stops = self.encoder(perturbed)
            nominal_route = self._masked_mean(nominal_stops, stop_mask)
            perturbed_route = self._masked_mean(perturbed_stops, stop_mask)
            fusion = torch.cat(
                [
                    nominal_route,
                    perturbed_route,
                    torch.abs(perturbed_route - nominal_route),
                    nominal_route * perturbed_route,
                ],
                dim=-1,
            )
            cost = self.cost_head(fusion).squeeze(-1)
            infeasibility_logit = self.infeasibility_head(fusion).squeeze(-1)
            expanded = fusion.unsqueeze(1).expand(
                -1, perturbed_stops.shape[1], -1
            )
            station_input = torch.cat([expanded, perturbed_stops], dim=-1)
            station = self.station_head(station_input).squeeze(-1)
            return cost, infeasibility_logit, station


    class SingleBranchNet(nn.Module):
        """Capacity-controlled 48-input comparator from Supplementary Table S8."""

        def __init__(self, dropout: float = 0.10) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(48, 224),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(224, 112),
                nn.ReLU(),
            )
            self.cost_head = nn.Sequential(
                nn.Linear(112, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.infeasibility_head = nn.Sequential(
                nn.Linear(112, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.station_hidden = nn.Linear(224, 64)
            self.station_output = nn.Linear(64, 1)

        def forward(self, nominal, perturbed, stop_mask):
            joined = torch.cat([nominal, perturbed], dim=-1)
            stop_embeddings = self.encoder(joined)
            mask_float = stop_mask.unsqueeze(-1).to(stop_embeddings.dtype)
            route_embedding = (stop_embeddings * mask_float).sum(dim=1) / (
                mask_float.sum(dim=1).clamp_min(1.0)
            )
            cost = self.cost_head(route_embedding).squeeze(-1)
            infeasibility_logit = self.infeasibility_head(
                route_embedding
            ).squeeze(-1)
            route_expanded = route_embedding.unsqueeze(1).expand(
                -1, stop_embeddings.shape[1], -1
            )
            station_input = torch.cat(
                [route_expanded, stop_embeddings], dim=-1
            )
            hidden = torch.relu(self.station_hidden(station_input))
            station = self.station_output(hidden).squeeze(-1)
            return cost, infeasibility_logit, station

else:

    class TwinBranchNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install the 'ml' extra to use TwinBranchNet.")


    class SingleBranchNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install the 'ml' extra to use SingleBranchNet.")


@dataclass
class EvaluatorMetrics:
    cost_mae: float
    cost_rmse: float
    cost_r2: float
    infeasibility_auc: float
    infeasibility_f1: float
    false_negative_rate: float
    brier_score: float
    expected_calibration_error: float
    station_vulnerability_mae: float


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = target.astype(int).ravel()
    score = score.astype(float).ravel()
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = 0.0
    for positive_index in positive:
        comparisons += float(np.sum(score[positive_index] > score[negative]))
        comparisons += 0.5 * float(
            np.sum(score[positive_index] == score[negative])
        )
    return comparisons / (len(positive) * len(negative))


def _classification_metrics(
    target: np.ndarray, probability: np.ndarray, threshold: float = 0.37
) -> Tuple[float, float, float, float, float]:
    target = target.astype(int).ravel()
    probability = probability.astype(float).ravel()
    predicted = probability >= threshold
    true_positive = int(np.sum(predicted & (target == 1)))
    false_positive = int(np.sum(predicted & (target == 0)))
    false_negative = int(np.sum(~predicted & (target == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    fnr = false_negative / max(int(np.sum(target == 1)), 1)
    brier = float(np.mean((probability - target) ** 2))
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10.0
        upper = (bin_index + 1) / 10.0
        selected = (probability >= lower) & (
            probability <= upper if bin_index == 9 else probability < upper
        )
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected]))
                - float(np.mean(target[selected]))
            )
    return _binary_auc(target, probability), f1, fnr, brier, ece


def _collate_samples(
    samples: Sequence[RouteScenarioSample],
    feature_normalizer: FeatureNormalizer,
    vulnerability_normalizer: VulnerabilityNormalizer,
    device: str,
):
    maximum_length = max(sample.nominal_features.shape[0] for sample in samples)
    batch_size = len(samples)
    nominal = np.zeros((batch_size, maximum_length, 24), dtype=np.float32)
    perturbed = np.zeros_like(nominal)
    stop_mask = np.zeros((batch_size, maximum_length), dtype=bool)
    station_mask = np.zeros_like(stop_mask)
    vulnerability = np.zeros((batch_size, maximum_length), dtype=np.float32)
    costs = np.zeros(batch_size, dtype=np.float32)
    infeasible = np.zeros(batch_size, dtype=np.float32)
    realized_cost = np.zeros(batch_size, dtype=np.float32)
    group_strings: List[str] = []
    for index, sample in enumerate(samples):
        length = sample.nominal_features.shape[0]
        nominal[index, :length] = feature_normalizer.transform(
            sample.nominal_features
        )
        perturbed[index, :length] = feature_normalizer.transform(
            sample.perturbed_features
        )
        stop_mask[index, :length] = True
        station_mask[index, :length] = sample.station_mask
        vulnerability[index, :length] = vulnerability_normalizer.transform(
            sample.station_vulnerability
        )
        costs[index] = sample.cost_increment
        infeasible[index] = sample.infeasible
        realized_cost[index] = sample.realized_cost
        group_strings.append(
            f"{sample.base_instance_id}::{sample.scenario_id}"
        )
    group_lookup = {
        value: index for index, value in enumerate(sorted(set(group_strings)))
    }
    group_ids = np.asarray(
        [group_lookup[value] for value in group_strings], dtype=np.int64
    )
    tensors = [
        torch.as_tensor(nominal, device=device),
        torch.as_tensor(perturbed, device=device),
        torch.as_tensor(stop_mask, device=device),
        torch.as_tensor(station_mask, device=device),
        torch.as_tensor(vulnerability, device=device),
        torch.as_tensor(costs, device=device),
        torch.as_tensor(infeasible, device=device),
        torch.as_tensor(realized_cost, device=device),
        torch.as_tensor(group_ids, device=device),
    ]
    return tensors


def _ranking_loss(scores, realized_cost, infeasible, group_ids, margin=0.20):
    losses = []
    for left in range(scores.shape[0]):
        for right in range(left + 1, scores.shape[0]):
            if group_ids[left] != group_ids[right]:
                continue
            left_target = realized_cost[left] + 1e6 * infeasible[left]
            right_target = realized_cost[right] + 1e6 * infeasible[right]
            if torch.isclose(left_target, right_target):
                continue
            sign = 1.0 if left_target > right_target else -1.0
            losses.append(
                torch.relu(margin - sign * (scores[left] - scores[right]))
            )
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


class TwinBranchRiskEvaluator:
    """Trainable proxy used only for screening and targeted repair."""

    def __init__(
        self,
        *,
        physical_scales: PhysicalScales,
        feature_normalizer: FeatureNormalizer,
        vulnerability_normalizer: VulnerabilityNormalizer,
        device: Optional[str] = None,
        seed: int = 2025,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required. Install with `pip install -e .[ml]`."
            )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwinBranchNet().to(self.device)
        if sum(parameter.numel() for parameter in self.model.parameters()) != (
            TWIN_PARAMETER_COUNT
        ):
            raise AssertionError("Twin architecture parameter count drifted.")
        self.physical_scales = physical_scales
        self.feature_normalizer = feature_normalizer
        self.vulnerability_normalizer = vulnerability_normalizer
        self.seed = seed

    def _loss(self, batch):
        (
            nominal,
            perturbed,
            stop_mask,
            station_mask,
            vulnerability,
            costs,
            infeasible,
            realized_cost,
            group_ids,
        ) = batch
        predicted_cost, infeasibility_logit, predicted_vulnerability = self.model(
            nominal, perturbed, stop_mask
        )
        cost_loss = torch_functional.mse_loss(predicted_cost, costs)
        infeasibility_loss = torch_functional.binary_cross_entropy_with_logits(
            infeasibility_logit, infeasible
        )
        if station_mask.any():
            vulnerability_loss = torch_functional.mse_loss(
                predicted_vulnerability[station_mask],
                vulnerability[station_mask],
            )
        else:
            vulnerability_loss = predicted_vulnerability.sum() * 0.0
        probability = torch.sigmoid(infeasibility_logit)
        station_sum = (
            predicted_vulnerability
            * station_mask.to(predicted_vulnerability.dtype)
        ).sum(dim=1)
        risk_score = predicted_cost + 100.0 * probability + 25.0 * station_sum
        ranking = _ranking_loss(
            risk_score, realized_cost, infeasible, group_ids
        )
        l2 = sum(
            parameter.square().sum() for parameter in self.model.parameters()
        )
        total = (
            cost_loss
            + infeasibility_loss
            + 0.50 * vulnerability_loss
            + 0.20 * ranking
            + 1e-5 * l2
        )
        return total

    def fit(
        self,
        training_samples: Sequence[RouteScenarioSample],
        validation_samples: Sequence[RouteScenarioSample],
        *,
        epochs: int = 80,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        rng = np.random.default_rng(self.seed)
        best_state = copy.deepcopy(self.model.state_dict())
        best_validation = float("inf")
        epochs_without_improvement = 0
        history = {"training_loss": [], "validation_loss": []}

        for _ in range(epochs):
            order = rng.permutation(len(training_samples))
            self.model.train()
            training_losses = []
            for start in range(0, len(order), batch_size):
                batch_samples = [
                    training_samples[index]
                    for index in order[start : start + batch_size]
                ]
                batch = _collate_samples(
                    batch_samples,
                    self.feature_normalizer,
                    self.vulnerability_normalizer,
                    self.device,
                )
                loss = self._loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_losses.append(float(loss.detach().cpu()))
            self.model.eval()
            with torch.no_grad():
                validation_batch = _collate_samples(
                    validation_samples,
                    self.feature_normalizer,
                    self.vulnerability_normalizer,
                    self.device,
                )
                validation_loss = float(
                    self._loss(validation_batch).detach().cpu()
                )
            history["training_loss"].append(float(np.mean(training_losses)))
            history["validation_loss"].append(validation_loss)
            if validation_loss < best_validation - 1e-8:
                best_validation = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        self.model.load_state_dict(best_state)
        return history

    def predict_samples(
        self, samples: Sequence[RouteScenarioSample]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        self.model.eval()
        batch = _collate_samples(
            samples,
            self.feature_normalizer,
            self.vulnerability_normalizer,
            self.device,
        )
        with torch.no_grad():
            cost, infeasibility_logit, vulnerability = self.model(
                batch[0], batch[1], batch[2]
            )
        cost_array = cost.cpu().numpy()
        probability = torch.sigmoid(infeasibility_logit).cpu().numpy()
        vulnerability_array = vulnerability.cpu().numpy()
        station_predictions = [
            vulnerability_array[index, : sample.station_mask.shape[0]]
            for index, sample in enumerate(samples)
        ]
        return cost_array, probability, station_predictions

    def evaluate_samples(
        self, samples: Sequence[RouteScenarioSample]
    ) -> EvaluatorMetrics:
        predicted_cost, probability, station_predictions = self.predict_samples(
            samples
        )
        true_cost = np.asarray(
            [sample.cost_increment for sample in samples], dtype=float
        )
        true_infeasible = np.asarray(
            [sample.infeasible for sample in samples], dtype=int
        )
        cost_mae = float(np.mean(np.abs(predicted_cost - true_cost)))
        cost_rmse = float(
            np.sqrt(np.mean((predicted_cost - true_cost) ** 2))
        )
        denominator = float(np.sum((true_cost - true_cost.mean()) ** 2))
        cost_r2 = (
            1.0
            - float(np.sum((true_cost - predicted_cost) ** 2)) / denominator
            if denominator > 0
            else float("nan")
        )
        auc, f1, fnr, brier, ece = _classification_metrics(
            true_infeasible, probability
        )
        vulnerability_errors: List[float] = []
        for sample, predicted in zip(samples, station_predictions):
            true_scaled = self.vulnerability_normalizer.transform(
                sample.station_vulnerability
            )
            vulnerability_errors.extend(
                np.abs(
                    predicted[sample.station_mask]
                    - true_scaled[sample.station_mask]
                ).tolist()
            )
        return EvaluatorMetrics(
            cost_mae=cost_mae,
            cost_rmse=cost_rmse,
            cost_r2=cost_r2,
            infeasibility_auc=float(auc),
            infeasibility_f1=float(f1),
            false_negative_rate=float(fnr),
            brier_score=float(brier),
            expected_calibration_error=float(ece),
            station_vulnerability_mae=float(
                np.mean(vulnerability_errors)
                if vulnerability_errors
                else 0.0
            ),
        )

    def _predict_route_scenario(
        self,
        inst: EVRPInstance,
        solution: Solution,
        route_index: int,
        scenario: Scenario,
    ) -> Tuple[float, float, np.ndarray]:
        nominal = route_stop_features(
            inst,
            solution,
            route_index,
            None,
            physical_scales=self.physical_scales,
            normalizer=self.feature_normalizer,
        )
        perturbed = route_stop_features(
            inst,
            solution,
            route_index,
            scenario,
            physical_scales=self.physical_scales,
            normalizer=self.feature_normalizer,
        )
        length = nominal.shape[0]
        nominal_tensor = torch.as_tensor(
            nominal[None, :, :], device=self.device
        )
        perturbed_tensor = torch.as_tensor(
            perturbed[None, :, :], device=self.device
        )
        mask = torch.ones((1, length), dtype=torch.bool, device=self.device)
        self.model.eval()
        with torch.no_grad():
            cost, logit, vulnerability = self.model(
                nominal_tensor, perturbed_tensor, mask
            )
        return (
            float(cost.item()),
            float(torch.sigmoid(logit).item()),
            vulnerability[0].cpu().numpy(),
        )

    def score_solution(
        self,
        inst: EVRPInstance,
        solution: Solution,
        scenarios: Sequence[Scenario],
    ) -> float:
        planning = certify_and_restore_solution(inst, solution)
        if not planning.feasible or not scenarios:
            return float("inf") if not planning.feasible else 0.0
        executable = planning.solution
        probabilities = normalized_probabilities(scenarios)
        total = 0.0
        for route_index, route in enumerate(executable.routes):
            for probability, scenario in zip(probabilities, scenarios):
                cost, infeasible, vulnerability = self._predict_route_scenario(
                    inst, executable, route_index, scenario
                )
                station_positions = [
                    position
                    for position, node in enumerate(route)
                    if inst.is_station(node)
                ]
                station_sum = float(
                    sum(vulnerability[position] for position in station_positions)
                )
                total += float(probability) * (
                    cost + 100.0 * infeasible + 25.0 * station_sum
                )
        return total

    def station_diagnostics(
        self,
        inst: EVRPInstance,
        solution: Solution,
        scenarios: Sequence[Scenario],
    ) -> Tuple[Dict[StopKey, float], Dict[StopKey, float]]:
        planning = certify_and_restore_solution(inst, solution)
        if not planning.feasible or not scenarios:
            return {}, {}
        executable = planning.solution
        probabilities = normalized_probabilities(scenarios)
        vulnerability: Dict[StopKey, float] = {}
        failure_share: Dict[StopKey, float] = {}
        for route_index, route in enumerate(executable.routes):
            plan_stops = {
                stop.position: stop
                for stop in planning.route_plans[route_index].stops
            }
            for probability, scenario in zip(probabilities, scenarios):
                _, _, station_prediction = self._predict_route_scenario(
                    inst, executable, route_index, scenario
                )
                for position, node in enumerate(route):
                    if not inst.is_station(node):
                        continue
                    key = (route_index, position)
                    vulnerability[key] = vulnerability.get(
                        key, 0.0
                    ) + float(probability) * float(
                        station_prediction[position]
                    )
                    interval = inst.time_to_interval(
                        plan_stops[position].physical_arrival
                    )
                    failed = float(
                        scenario.available_capacity[node][interval] == 0
                    )
                    failure_share[key] = failure_share.get(
                        key, 0.0
                    ) + float(probability) * failed
        return vulnerability, failure_share

    def score_backup_candidate(
        self,
        inst: EVRPInstance,
        solution: Solution,
        key: StopKey,
        alternative: int,
        scenarios: Sequence[Scenario],
    ) -> float:
        route_index, position = key
        counterfactual = solution.copy()
        counterfactual.routes[route_index][position] = alternative
        counterfactual.planned_charges = {}
        counterfactual.backups = {}
        counterfactual.planned_rests = {}
        return self.score_solution(inst, counterfactual, scenarios)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "physical_scales": asdict(self.physical_scales),
                "feature_normalizer": self.feature_normalizer.to_dict(),
                "vulnerability_normalizer": asdict(
                    self.vulnerability_normalizer
                ),
                "seed": self.seed,
                "feature_names": FEATURE_NAMES,
                "parameter_count": TWIN_PARAMETER_COUNT,
            },
            destination,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: Optional[str] = None
    ) -> "TwinBranchRiskEvaluator":
        if torch is None:
            raise ImportError("PyTorch is required to load an evaluator.")
        checkpoint = torch.load(path, map_location=device or "cpu")
        evaluator = cls(
            physical_scales=PhysicalScales(**checkpoint["physical_scales"]),
            feature_normalizer=FeatureNormalizer.from_dict(
                checkpoint["feature_normalizer"]
            ),
            vulnerability_normalizer=VulnerabilityNormalizer(
                **checkpoint["vulnerability_normalizer"]
            ),
            device=device,
            seed=int(checkpoint["seed"]),
        )
        evaluator.model.load_state_dict(checkpoint["state_dict"])
        return evaluator


class HeuristicRiskEvaluator:
    """Frozen score defined by Equations (80)-(86)."""

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def _route_components(
        self,
        inst: EVRPInstance,
        solution: Solution,
        route_index: int,
        scenario: Scenario,
    ) -> Tuple[float, Dict[StopKey, float], Dict[StopKey, float]]:
        planning = certify_and_restore_solution(
            inst, solution, require_all_customers=False
        )
        if not planning.feasible:
            return float("inf"), {}, {}
        executable = planning.solution
        route = executable.routes[route_index]
        stop_by_position = {
            stop.position: stop for stop in planning.route_plans[route_index].stops
        }
        battery_margin = min(
            (stop.battery_arrival for stop in stop_by_position.values()),
            default=inst.initial_battery,
        )
        energy_exposure = float(
            np.clip(
                1.0
                - (battery_margin - inst.safety_battery)
                / (inst.battery_capacity - inst.safety_battery),
                0.0,
                1.0,
            )
        )
        station_positions = [
            position
            for position, node in enumerate(route)
            if inst.is_station(node)
        ]
        failure_values = []
        waiting_values = []
        detour_values = []
        station_scores: Dict[StopKey, float] = {}
        station_failures: Dict[StopKey, float] = {}
        dmax = 100.0 * np.sqrt(2.0)
        for position in station_positions:
            station_id = route[position]
            stop = stop_by_position[position]
            interval = inst.time_to_interval(stop.physical_arrival)
            failed = float(
                scenario.available_capacity[station_id][interval] == 0
            )
            occupied = float(
                scenario.state(inst, station_id, interval) == 1
            )
            wait = (
                float(scenario.waiting_time[station_id][interval]) / 75.0
                if occupied
                else 0.0
            )
            predecessor = route[position - 1]
            successor = route[position + 1]
            alternatives = []
            for alternative in inst.station_ids:
                if alternative == station_id:
                    continue
                delta = (
                    inst.distance(predecessor, alternative)
                    + inst.distance(alternative, successor)
                    - inst.distance(predecessor, station_id)
                    - inst.distance(station_id, successor)
                )
                alternatives.append(max(delta, 0.0) / dmax)
            detour = min(alternatives, default=1.0)
            key = (route_index, position)
            station_failures[key] = failed
            station_scores[key] = float(
                np.clip(
                    0.40 * failed + 0.35 * wait + 0.25 * detour,
                    0.0,
                    1.0,
                )
            )
            failure_values.append(failed)
            waiting_values.append(wait)
            detour_values.append(detour)
        failure_exposure = float(np.mean(failure_values)) if failure_values else 0.0
        waiting_exposure = float(np.mean(waiting_values)) if waiting_values else 0.0
        detour_exposure = float(np.mean(detour_values)) if detour_values else 0.0
        score = float(
            np.clip(
                0.35 * energy_exposure
                + 0.30 * failure_exposure
                + 0.20 * waiting_exposure
                + 0.15 * detour_exposure,
                0.0,
                1.0,
            )
        )
        predicted_cost = 500.0 * score
        infeasibility_probability = self._sigmoid(-4.0 + 8.0 * score)
        route_risk = (
            predicted_cost
            + 100.0 * infeasibility_probability
            + 25.0 * sum(station_scores.values())
        )
        return route_risk, station_scores, station_failures

    def score_solution(
        self,
        inst: EVRPInstance,
        solution: Solution,
        scenarios: Sequence[Scenario],
    ) -> float:
        if not scenarios:
            return 0.0
        planning = certify_and_restore_solution(inst, solution)
        if not planning.feasible:
            return float("inf")
        probabilities = normalized_probabilities(scenarios)
        total = 0.0
        for route_index in range(len(planning.solution.routes)):
            for probability, scenario in zip(probabilities, scenarios):
                route_risk, _, _ = self._route_components(
                    inst, planning.solution, route_index, scenario
                )
                total += float(probability) * route_risk
        return total

    def station_diagnostics(
        self,
        inst: EVRPInstance,
        solution: Solution,
        scenarios: Sequence[Scenario],
    ) -> Tuple[Dict[StopKey, float], Dict[StopKey, float]]:
        planning = certify_and_restore_solution(inst, solution)
        if not planning.feasible or not scenarios:
            return {}, {}
        probabilities = normalized_probabilities(scenarios)
        vulnerability: Dict[StopKey, float] = {}
        failure_share: Dict[StopKey, float] = {}
        for route_index in range(len(planning.solution.routes)):
            for probability, scenario in zip(probabilities, scenarios):
                _, station_scores, failures = self._route_components(
                    inst, planning.solution, route_index, scenario
                )
                for key, value in station_scores.items():
                    vulnerability[key] = vulnerability.get(key, 0.0) + float(
                        probability
                    ) * value
                for key, value in failures.items():
                    failure_share[key] = failure_share.get(key, 0.0) + float(
                        probability
                    ) * value
        return vulnerability, failure_share

    def score_backup_candidate(
        self,
        inst: EVRPInstance,
        solution: Solution,
        key: StopKey,
        alternative: int,
        scenarios: Sequence[Scenario],
    ) -> float:
        """Score the counterfactual route with the alternative at the primary."""
        route_index, position = key
        counterfactual = solution.copy()
        counterfactual.routes[route_index][position] = alternative
        counterfactual.planned_charges = {}
        counterfactual.backups = {}
        counterfactual.planned_rests = {}
        return self.score_solution(inst, counterfactual, scenarios)


def split_samples_by_instance(
    samples: Sequence[RouteScenarioSample],
    *,
    seed: int = 2025,
) -> Tuple[List[RouteScenarioSample], List[RouteScenarioSample], List[RouteScenarioSample]]:
    """70/15/15 split by independent base instance."""
    instance_ids = sorted({sample.base_instance_id for sample in samples})
    if len(instance_ids) < 3:
        raise ValueError(
            "At least three independent base instances are required for an "
            "instance-level train/validation/test split."
        )
    rng = np.random.default_rng(seed)
    shuffled = [instance_ids[index] for index in rng.permutation(len(instance_ids))]
    n_train = max(1, int(round(0.70 * len(shuffled))))
    n_valid = max(1, int(round(0.15 * len(shuffled))))
    if n_train + n_valid >= len(shuffled):
        n_train = len(shuffled) - 2
        n_valid = 1
    train_ids = set(shuffled[:n_train])
    valid_ids = set(shuffled[n_train : n_train + n_valid])
    test_ids = set(shuffled[n_train + n_valid :])
    return (
        [sample for sample in samples if sample.base_instance_id in train_ids],
        [sample for sample in samples if sample.base_instance_id in valid_ids],
        [sample for sample in samples if sample.base_instance_id in test_ids],
    )


def train_from_generated_samples(
    inst: EVRPInstance,
    scenarios: Sequence[Scenario],
    n_samples: int = 2000,
    seed: int = 2025,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.001,
) -> Tuple[TwinBranchRiskEvaluator, Dict[str, EvaluatorMetrics]]:
    """Small smoke-training helper.

    It uses a sample-level split because one base instance is supplied.  The
    publication protocol must instead call :func:`split_samples_by_instance`
    on samples generated from independent base instances.
    """
    samples = make_training_samples(
        inst, scenarios, n_samples=n_samples, seed=seed
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    n_train = int(0.70 * len(order))
    n_valid = int(0.15 * len(order))
    training = [samples[index] for index in order[:n_train]]
    validation = [
        samples[index] for index in order[n_train : n_train + n_valid]
    ]
    testing = [samples[index] for index in order[n_train + n_valid :]]
    feature_normalizer, vulnerability_normalizer = fit_training_normalizers(
        training
    )
    scales = PhysicalScales.generated_domain(inst)
    evaluator = TwinBranchRiskEvaluator(
        physical_scales=scales,
        feature_normalizer=feature_normalizer,
        vulnerability_normalizer=vulnerability_normalizer,
        seed=seed,
    )
    evaluator.fit(
        training,
        validation,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return evaluator, {
        "validation": evaluator.evaluate_samples(validation),
        "test": evaluator.evaluate_samples(testing),
    }

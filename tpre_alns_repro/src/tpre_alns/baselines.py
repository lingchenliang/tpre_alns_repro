"""Method-specific operator pools and evaluation rules from Table S3b."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence, Tuple

from .alns import ALNSConfig, TPREALNS
from .entities import EVRPInstance, Scenario, Solution
from .evaluator import HeuristicRiskEvaluator


METHOD_ALIASES = {
    "risk_aware_alns": "full_recourse_risk_aware_alns",
    "tpre_without_learned_evaluator": "full_recourse_risk_aware_alns",
}


def config_for_method(
    method: str, base: ALNSConfig | None = None
) -> ALNSConfig:
    canonical = METHOD_ALIASES.get(method, method)
    cfg = replace(base or ALNSConfig())
    if canonical == "deterministic_alns":
        return replace(
            cfg,
            use_evaluator=False,
            use_backups=False,
            use_cvar=False,
            risk_aversion=0.0,
            deterministic_objective=True,
            destroy_operators=("random", "worst_cost", "route_segment"),
            repair_operators=("greedy", "regret_2"),
        )
    if canonical == "full_recourse_risk_aware_alns":
        return replace(
            cfg,
            use_evaluator=False,
            use_backups=True,
            use_cvar=True,
            deterministic_objective=False,
            destroy_operators=(
                "random",
                "worst_cost",
                "route_segment",
                "risk_station_neighbourhood",
            ),
            repair_operators=("greedy", "regret_2"),
        )
    if canonical == "tpre_without_backup":
        return replace(cfg, use_backups=False, use_cvar=True)
    if canonical == "tpre_without_cvar":
        return replace(cfg, use_backups=True, use_cvar=False, risk_aversion=0.0)
    if canonical == "tpre_without_rest_sync":
        return replace(cfg, rest_sync=False)
    if canonical == "tpre_alns":
        return cfg
    raise KeyError(f"Unknown method: {method}")


def run_method(
    method: str,
    inst: EVRPInstance,
    scenarios: Sequence[Scenario],
    base_config: ALNSConfig | None = None,
    risk_evaluator=None,
    seed: int = 1,
) -> Tuple[Solution, Dict[str, object]]:
    cfg = config_for_method(method, base_config)
    evaluator = risk_evaluator or HeuristicRiskEvaluator()
    search = TPREALNS(
        inst,
        scenarios,
        cfg,
        risk_evaluator=evaluator,
        seed=seed,
    )
    solution, information = search.solve()
    information["method"] = METHOD_ALIASES.get(method, method)
    information["screening_proxy"] = (
        evaluator.__class__.__name__ if cfg.use_evaluator else "none"
    )
    return solution, information

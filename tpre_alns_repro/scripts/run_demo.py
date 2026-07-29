#!/usr/bin/env python
"""Run a small, fully traceable routing smoke experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tpre_alns.alns import ALNSConfig
from tpre_alns.baselines import run_method
from tpre_alns.evaluator import TwinBranchRiskEvaluator
from tpre_alns.instance import generate_synthetic_instance
from tpre_alns.scenarios import generate_scenarios


METHODS = (
    "deterministic_alns",
    "full_recourse_risk_aware_alns",
    "tpre_without_backup",
    "tpre_without_cvar",
    "tpre_without_rest_sync",
    "tpre_alns",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick TPRE-ALNS reproducibility smoke test."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--customers", type=int, default=8)
    parser.add_argument("--stations", type=int, default=3)
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument(
        "--setting", default="high_occ_high_damage"
    )
    parser.add_argument("--method", choices=METHODS, default="tpre_alns")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--evaluator-checkpoint",
        default=None,
        help="Optional trained twin-branch .pt checkpoint.",
    )
    parser.add_argument("--out", default="results/demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    instance = generate_synthetic_instance(
        args.customers, args.stations, seed=args.seed
    )
    scenarios = generate_scenarios(
        instance,
        n_scenarios=args.scenarios,
        setting=args.setting,
        seed=args.seed + 100_000,
    )
    evaluator = (
        TwinBranchRiskEvaluator.load(args.evaluator_checkpoint)
        if args.evaluator_checkpoint
        else None
    )
    if args.method.startswith("tpre") and evaluator is None:
        print(
            "NOTE: no neural checkpoint was supplied; this smoke run uses the "
            "frozen hand-crafted proxy, while full acceptance still uses the "
            "complete scenario evaluator."
        )
    config = ALNSConfig(
        max_iterations=args.iterations,
        max_no_improve=max(5, args.iterations),
    )
    solution, information = run_method(
        args.method,
        instance,
        scenarios,
        base_config=config,
        risk_evaluator=evaluator,
        seed=args.seed,
    )

    metrics_row = {
        "method": args.method,
        "seed": args.seed,
        "customers": args.customers,
        "stations": args.stations,
        "scenarios": args.scenarios,
        "setting": args.setting,
        "screening_proxy": information["screening_proxy"],
        **information["metrics"],
        **{
            f"count_{key}": value
            for key, value in information["counters"].items()
        },
    }
    with (output / "demo_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics_row))
        writer.writeheader()
        writer.writerow(metrics_row)
    (output / "demo_solution.json").write_text(
        json.dumps(solution.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "demo_search.json").write_text(
        json.dumps(information, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics_row, indent=2))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()

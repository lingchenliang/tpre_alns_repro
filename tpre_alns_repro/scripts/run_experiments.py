#!/usr/bin/env python
"""Run a configuration-defined experiment grid."""

from __future__ import annotations

import argparse

from tpre_alns.evaluator import TwinBranchRiskEvaluator
from tpre_alns.experiments import run_experiment_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/benchmark_runs")
    parser.add_argument("--evaluator-checkpoint", default=None)
    args = parser.parse_args()
    evaluator = (
        TwinBranchRiskEvaluator.load(args.evaluator_checkpoint)
        if args.evaluator_checkpoint
        else None
    )
    rows = run_experiment_grid(
        args.config,
        args.out,
        risk_evaluator=evaluator,
        evaluator_checkpoint=args.evaluator_checkpoint,
    )
    print(f"Completed {len(rows)} run rows in {args.out}.")


if __name__ == "__main__":
    main()

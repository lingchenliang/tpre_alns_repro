#!/usr/bin/env python
"""Generate instance-disjoint samples and train the 65,091-parameter evaluator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tpre_alns.evaluator import (
    TwinBranchRiskEvaluator,
    split_samples_by_instance,
)
from tpre_alns.features import (
    PhysicalScales,
    fit_training_normalizers,
    make_training_samples,
)
from tpre_alns.instance import generate_synthetic_instance
from tpre_alns.scenarios import generate_scenarios


SCALES = ((25, 5), (50, 8), (100, 12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-instances", type=int, default=9)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--scenarios-per-setting", type=int, default=17)
    parser.add_argument("--instance-seed-base", type=int, default=1_000_000)
    parser.add_argument("--scenario-seed-base", type=int, default=2_000_000)
    parser.add_argument("--training-seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--out", default="results/evaluator")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_instances < 3:
        raise ValueError("At least three base instances are required.")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    shared_scales = PhysicalScales(
        dmax=100.0 * np.sqrt(2.0),
        tmax=(100.0 * np.sqrt(2.0)) / 0.65,
        # Supplementary Table S10b reports a training-route maximum of 22.
        lmax_route=22.0,
    )
    per_instance = int(np.ceil(args.samples / args.base_instances))
    samples = []
    for instance_index in range(args.base_instances):
        customers, stations = SCALES[instance_index % len(SCALES)]
        instance = generate_synthetic_instance(
            customers,
            stations,
            seed=args.instance_seed_base + instance_index,
            name=f"training_{instance_index:03d}_C{customers}",
        )
        scenario_sets = []
        for setting_index, setting in enumerate(
            ("low_occ_low_damage", "high_occ_high_damage", "extreme")
        ):
            scenario_sets.extend(
                generate_scenarios(
                    instance,
                    args.scenarios_per_setting,
                    setting=setting,
                    seed=(
                        args.scenario_seed_base
                        + 10_000 * instance_index
                        + setting_index
                    ),
                )
            )
        samples.extend(
            make_training_samples(
                instance,
                scenario_sets,
                n_samples=per_instance,
                seed=args.training_seed + instance_index,
                physical_scales=shared_scales,
            )
        )
    samples = samples[: args.samples]
    training, validation, testing = split_samples_by_instance(
        samples, seed=args.training_seed
    )
    feature_normalizer, vulnerability_normalizer = fit_training_normalizers(
        training
    )
    evaluator = TwinBranchRiskEvaluator(
        physical_scales=shared_scales,
        feature_normalizer=feature_normalizer,
        vulnerability_normalizer=vulnerability_normalizer,
        seed=args.training_seed,
    )
    history = evaluator.fit(
        training,
        validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=10,
    )
    validation_metrics = evaluator.evaluate_samples(validation)
    test_metrics = evaluator.evaluate_samples(testing)
    evaluator.save(output / "twin_branch_evaluator.pt")
    rows = [
        {
            "split": "validation",
            "samples": len(validation),
            **validation_metrics.__dict__,
        },
        {
            "split": "test",
            "samples": len(testing),
            **test_metrics.__dict__,
        },
    ]
    with (output / "evaluator_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output / "data_partition.json").write_text(
        json.dumps(
            {
                "training_samples": len(training),
                "validation_samples": len(validation),
                "test_samples": len(testing),
                "training_instances": sorted(
                    {sample.base_instance_id for sample in training}
                ),
                "validation_instances": sorted(
                    {sample.base_instance_id for sample in validation}
                ),
                "test_instances": sorted(
                    {sample.base_instance_id for sample in testing}
                ),
                "physical_scales": shared_scales.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))
    print(f"Wrote evaluator artifacts to {output.resolve()}.")


if __name__ == "__main__":
    main()

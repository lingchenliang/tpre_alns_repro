#!/usr/bin/env python
"""Generate and persist one base instance plus disjoint scenario sets."""

from __future__ import annotations

import argparse

from tpre_alns.instance import generate_synthetic_instance, save_instance_json
from tpre_alns.scenarios import generate_scenarios, save_scenarios_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=25)
    parser.add_argument("--stations", type=int, default=5)
    parser.add_argument("--instance-seed", type=int, default=100001)
    parser.add_argument("--optimization-seed", type=int, default=300001)
    parser.add_argument("--out-of-sample-seed", type=int, default=900001)
    parser.add_argument("--optimization-scenarios", type=int, default=50)
    parser.add_argument("--out-of-sample-scenarios", type=int, default=500)
    parser.add_argument("--setting", default="high_occ_high_damage")
    parser.add_argument("--out", default="data/generated/example")
    args = parser.parse_args()
    if args.optimization_seed == args.out_of_sample_seed:
        raise ValueError("Optimization and out-of-sample seeds must differ.")
    instance = generate_synthetic_instance(
        args.customers,
        args.stations,
        seed=args.instance_seed,
    )
    optimization = generate_scenarios(
        instance,
        args.optimization_scenarios,
        setting=args.setting,
        seed=args.optimization_seed,
    )
    reporting = generate_scenarios(
        instance,
        args.out_of_sample_scenarios,
        setting=args.setting,
        seed=args.out_of_sample_seed,
    )
    save_instance_json(instance, f"{args.out}/instance.json")
    save_scenarios_json(
        optimization, f"{args.out}/optimization_scenarios.json.gz"
    )
    save_scenarios_json(
        reporting, f"{args.out}/out_of_sample_scenarios.json.gz"
    )
    print(f"Wrote deterministic data bundle to {args.out}.")


if __name__ == "__main__":
    main()

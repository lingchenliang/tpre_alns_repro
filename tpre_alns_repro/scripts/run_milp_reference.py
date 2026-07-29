#!/usr/bin/env python
"""Run the 600-second deterministic first-stage MILP reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tpre_alns.instance import generate_synthetic_instance, load_customer_csv
from tpre_alns.milp_reference import MILPReferenceConfig, solve_milp_reference
from tpre_alns.scenarios import generate_scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer-csv", default=None)
    parser.add_argument("--customers", type=int, default=12)
    parser.add_argument("--stations", type=int, default=3)
    parser.add_argument("--reporting-scenarios", type=int, default=50)
    parser.add_argument("--setting", default="high_occ_high_damage")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vehicles", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", default="results/milp_reference.json")
    args = parser.parse_args()
    instance = (
        load_customer_csv(
            args.customer_csv, n_stations=args.stations, seed=args.seed
        )
        if args.customer_csv
        else generate_synthetic_instance(
            args.customers, args.stations, seed=args.seed
        )
    )
    reporting = generate_scenarios(
        instance,
        args.reporting_scenarios,
        setting=args.setting,
        seed=args.seed + 900_000,
    )
    result = solve_milp_reference(
        instance,
        reporting,
        MILPReferenceConfig(
            max_vehicles=args.vehicles,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_flag=not args.quiet,
        ),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

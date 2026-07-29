#!/usr/bin/env python
"""Aggregate run-level CSVs at the independent base-instance level."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CORE_METRICS = (
    "oos_total_cost",
    "oos_scenario_dependent_cost_cvar",
    "oos_waiting_cost",
    "oos_recourse_cost",
    "oos_infeasible_ratio",
    "oos_backup_switches",
    "optimization_objective",
)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    run_groups: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["scale"],
            row["uncertainty_setting"],
            row["method"],
            row["base_instance_index"],
        )
        run_groups[key].append(row)

    base_means: List[Dict[str, object]] = []
    for key, group in sorted(run_groups.items()):
        scale, setting, method, base_index = key
        output_row: Dict[str, object] = {
            "scale": scale,
            "uncertainty_setting": setting,
            "method": method,
            "base_instance_index": base_index,
            "run_count": len(group),
        }
        for metric in CORE_METRICS:
            if metric in group[0]:
                output_row[metric] = statistics.fmean(
                    float(row[metric]) for row in group
                )
        base_means.append(output_row)
    write_rows(output / "base_instance_means.csv", base_means)

    summary_groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in base_means:
        key = (
            str(row["scale"]),
            str(row["uncertainty_setting"]),
            str(row["method"]),
        )
        summary_groups[key].append(row)
    summaries: List[Dict[str, object]] = []
    for key, group in sorted(summary_groups.items()):
        scale, setting, method = key
        output_row = {
            "scale": scale,
            "uncertainty_setting": setting,
            "method": method,
            "independent_base_instances": len(group),
        }
        for metric in CORE_METRICS:
            values = [
                float(row[metric]) for row in group if metric in row
            ]
            if values:
                output_row[f"{metric}_mean"] = statistics.fmean(values)
                output_row[f"{metric}_sd"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        summaries.append(output_row)
    write_rows(output / "summary_metrics.csv", summaries)
    print(
        f"Wrote {len(base_means)} base-instance means and "
        f"{len(summaries)} summary rows."
    )


if __name__ == "__main__":
    main()

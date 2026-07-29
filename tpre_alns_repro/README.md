# TPRE-ALNS reproducibility code

Reference implementation for the manuscript:

> **Learning-assisted risk-aware electric delivery routing under uncertain
> charging-station availability**

The repository implements the updated method described in the manuscript and
supplementary information: synthetic VRPTW-style instance generation,
time-dependent charger scenarios, deterministic energy/rest restoration,
wait-first fixed-rule recourse, the planning + expected scenario cost +
scenario-cost CVaR objective, the 24-feature twin-branch evaluator, and
TPRE-ALNS.

This release is designed for method inspection and reproducible new runs. It
does **not** contain fabricated copies of the manuscript's reported CSVs or
metrics. Exact numerical equality with the published tables additionally
requires the archived base-instance/scenario seeds, route pool, run-level
outputs, normalization statistics and seed-2025 evaluator checkpoint named in
the Data Availability Statement.

## What is implemented

- Supplementary Algorithm S2 synthetic instance generator.
- Pre-dispatch reported unavailability `D_jt`, sampled once per base instance.
- Conditional occupation and hidden-damage generation (Equations 3-5).
- Available / occupied / failed state classification and stored queue delays.
- Partial charging with minute-consistent, cross-period tariff allocation.
- Distinct start and terminal depot copies.
- Hard-window, load, battery and continuous-work propagation.
- Qualifying rest at the depot or a charging station; synchronized
  wait/charge/rest uses the maximum-duration rule.
- Wait-first recourse: iterative waiting, assigned backup, local repair, then
  one unrecovered-infeasibility penalty per vehicle route and scenario.
- Planning cost separated from scenario-dependent cost before expectation and
  CVaR are computed.
- Exactly 24 stop-level screening features with training-only normalization
  support.
- Shared `24 -> 128 -> 64` twin encoder, masked mean pooling, absolute
  difference fusion, two route heads and one station head (65,091 parameters).
- Frozen hand-crafted proxy from Equations 80-86.
- Algorithm S1 candidate history, 80th-percentile rolling threshold, targeted
  repair, 0.05 full-evaluation safeguard, SA acceptance and adaptive weights.
- Deterministic, full-recourse and component-ablation method configurations.
- Optional deterministic first-stage Gurobi reference with a 600-second cap.
- Disjoint optimization/out-of-sample seed domains and run manifests.

The traceability table in
[`docs/METHOD_MAPPING.md`](docs/METHOD_MAPPING.md) maps manuscript equations and
supplementary tables to source files and tests.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .                      # routing + hand-crafted proxy
pip install -e ".[ml,test]"           # twin evaluator + tests
```

`gurobipy==12.0.3` and a valid Gurobi license are needed only for the optional
MILP reference:

```bash
pip install -e ".[milp]"
```

## Quick smoke run

```bash
python scripts/run_demo.py \
  --customers 8 \
  --stations 3 \
  --scenarios 5 \
  --iterations 20 \
  --method tpre_alns \
  --out results/demo
```

Without `--evaluator-checkpoint`, the quick run uses the frozen hand-crafted
proxy. This is intentionally labeled in the output and must not be reported as
the trained twin model. Every accepted move is still judged by the complete
scenario evaluator.

With a trained checkpoint:

```bash
python scripts/run_demo.py \
  --method tpre_alns \
  --evaluator-checkpoint results/evaluator/twin_branch_evaluator.pt
```

## Generate a portable data bundle

```bash
python scripts/generate_data.py \
  --customers 25 \
  --stations 5 \
  --instance-seed 100001 \
  --optimization-seed 300001 \
  --out-of-sample-seed 900001 \
  --out data/generated/example
```

This writes an instance JSON plus compressed optimization and reporting
scenario files. Scenario draws are stored and never resampled during method
evaluation.

## Train the twin-branch evaluator

The publication protocol splits by independent base instance, not by individual
route-scenario row:

```bash
python scripts/train_evaluator.py \
  --base-instances 60 \
  --samples 20000 \
  --epochs 80 \
  --training-seed 2025 \
  --out results/evaluator_seed2025
```

For a development check, reduce the base-instance, sample and epoch counts.
The script refuses fewer than three base instances because a genuine
instance-level train/validation/test split would then be impossible.

## Run experiments

`configs/default.yaml` contains the manuscript-scale settings; it is
computationally expensive. `configs/quick.yaml` is a CI/development smoke grid.

```bash
python scripts/run_experiments.py \
  --config configs/quick.yaml \
  --out results/quick

python scripts/run_experiments.py \
  --config configs/default.yaml \
  --evaluator-checkpoint results/evaluator_seed2025/twin_branch_evaluator.pt \
  --out results/manuscript_grid
```

Aggregate run-level output only after averaging runs within each independent
base instance:

```bash
python scripts/make_summary_tables.py \
  --input results/manuscript_grid/run_metrics.csv \
  --out results/manuscript_grid/tables
```

The generated `manifest.json` binds the experiment identifier, configuration
hash, checkpoint hash, seed policy, software environment and row count.
`infeasible_ratio` is reported in percentage units to match the manuscript
tables (for example, `20.0` means 20% of scenario probability mass).

## Tests

```bash
python -m unittest discover -s tests -v
# or, after installing the test extra:
pytest
```

Tests cover Algorithm S2 ranges, scenario-state identities, stored-wait rules,
tariff boundaries, distinct depot copies, continuous-work handling, planning /
scenario-cost separation, finite-distribution CVaR, 24-feature semantics,
normalization, feasible training-pool construction, model parameter counts and
an ALNS smoke run. See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the
release-verification record.

## Repository layout

```text
configs/                      Full and quick experiment configurations
docs/                         Method mapping and reproducibility checklist
scripts/generate_data.py      Persist seeded instance/scenario bundles
scripts/run_demo.py           Small end-to-end run
scripts/train_evaluator.py    Instance-disjoint evaluator training
scripts/run_experiments.py    Configured routing experiment grid
scripts/make_summary_tables.py Base-instance-first aggregation
scripts/run_milp_reference.py Optional deterministic Gurobi reference
src/tpre_alns/entities.py     Typed instance/scenario/solution records
src/tpre_alns/instance.py     Algorithm S2 and instance I/O
src/tpre_alns/scenarios.py    Scenario generation, tariffs and severity ranks
src/tpre_alns/planning.py     Energy/rest restoration and plan certification
src/tpre_alns/evaluation.py   Fixed-rule recourse and CVaR objective
src/tpre_alns/features.py     24 features and supervised labels
src/tpre_alns/evaluator.py    Twin/single-branch networks and HC proxy
src/tpre_alns/alns.py         TPRE-ALNS search
src/tpre_alns/baselines.py    Method-specific ablations and baselines
src/tpre_alns/experiments.py  Manifests and disjoint-seed experiment runner
src/tpre_alns/milp_reference.py Deterministic first-stage MILP
tests/                        Deterministic regression and smoke tests
```

## Interpretation

The generator creates controlled computational instances; it is not calibrated
to a particular delivery fleet or public charging network. Costs are normalized
cost units. New runs from this repository support method verification and
sensitivity analysis but are not field forecasts.

## License and citation

Code is released under the MIT License. Use `CITATION.cff` and cite the
associated manuscript when using the implementation.

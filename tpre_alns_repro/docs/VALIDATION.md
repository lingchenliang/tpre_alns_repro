# Release validation

The following checks were executed against release 1.0.0 on 2026-07-29 with
Python 3.12.3 on Windows 11.

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/run_demo.py \
  --customers 8 --stations 3 --scenarios 5 --iterations 20 --seed 2025
python scripts/run_experiments.py \
  --config configs/quick.yaml --out results/quick_verify
python scripts/make_summary_tables.py \
  --input results/quick_verify/run_metrics.csv \
  --out results/quick_verify/tables
```

The unit suite completed successfully. The demo completed 20 realized search
iterations. The configuration-driven smoke grid wrote two run-level rows, a
configuration/environment manifest, base-instance means and summary metrics.

The optional PyTorch training path was not executed in this release environment
because PyTorch was not installed. Architecture dimensions and the reported
65,091/65,235 parameter-count formulas are covered by dependency-free tests.
Run the following in an ML-enabled environment before publishing a trained
checkpoint:

```bash
pip install -e ".[ml,test]"
python scripts/train_evaluator.py \
  --base-instances 60 --samples 20000 --training-seed 2025 \
  --out results/evaluator_seed2025
```

No claim of exact manuscript-table reproduction is made without the archived
seed lists, run-level results, normalization statistics and evaluator
checkpoint identified in the manuscript Data Availability Statement.

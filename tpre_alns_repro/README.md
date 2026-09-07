# TPRE-ALNS reproducibility starter

Reference code and synthetic-data generator for the manuscript:

> **Learning-guided risk-aware adaptive large neighborhood search for electric delivery routing under uncertain public charging-station availability**

## Research-integrity and provenance note

This repository is a **clean-room reference implementation reconstructed from the manuscript and S1 Appendix**. It generates new synthetic benchmark instances and charger-state scenarios according to the documented rules. It does **not** contain the original ALNS solver, the complete fixed-rule route-recourse simulator, the trained seed-2025 checkpoint, or the run-level outputs that produced the numerical tables in the manuscript.

Do not describe the example files in `data/example/` as the original experimental data. To support exact reproduction of reported results, add the original implementation, exact seeds, route pool, normalization statistics, checkpoint, run-level outputs, and experiment manifest.

## What is implemented

- Synthetic 25-, 50-, and 100-customer instance generation.
- 5, 8, and 12 public charging stations by scale.
- Depot at `(50, 50)` km; other coordinates sampled in `[0, 100] x [0, 100]` km.
- Customer demand `DiscreteUniform{10,...,50}` kg.
- Service duration `Uniform(5,15)` min.
- Time-window width `Uniform(60,180)` min; ready time sampled so the due time stays within the 1080-min horizon.
- Station charger counts `{4,6,8}` with probabilities `{0.30,0.40,0.30}`.
- Charging powers `{60,120}` kW with probabilities `{0.50,0.50}`.
- Reported-unavailable chargers drawn once per station-hour as `Binomial(n_chargers, 0.05)`.
- Scenario occupation and hidden damage drawn conditionally from the remaining charger pool.
- Low/high/extreme occupation probabilities `0.25/0.65/0.80`.
- Low/high/extreme hidden-damage probabilities `0.01/0.06/0.10`.
- Queue-delay ranges `U(5,20)`, `U(20,50)`, and `U(35,75)` minutes.
- Euclidean distance, travel time at `0.65 km/min`, and energy at `0.24 kWh/km`.
- 24-feature schema and training-only z-score helpers.
- Twin-branch MLP architecture `24 -> 128 -> 64`, difference-aware fusion, two route heads, and one station head. The model has exactly **65,091 trainable parameters**.
- Empirical CVaR, risk-aware objective, route-risk score, and pairwise ranking helpers.
- SHA-256 manifest for every generated instance/scenario file.

## What is not implemented

- The original adaptive large neighborhood search.
- Deterministic energy/rest restoration and backup completion.
- The complete wait -> assigned backup -> local repair -> penalty simulator.
- Route-pool generation and exact training labels.
- The original trained checkpoint and reported result tables.

These components must be added from the authors' actual experiment source before claiming end-to-end reproducibility.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

The generator requires only NumPy. PyTorch is needed only for the twin-branch model definition.

## Generate a small example

```bash
python tpre_repro.py generate \
  --output generated_example \
  --scales 25 \
  --base-instances 1 \
  --optimization-scenarios 3 \
  --oos-scenarios 5 \
  --settings low_occ_low_damage extreme_disruption \
  --root-seed 20260907
```

## Generate the documented scale and scenario counts

The command below creates 10 base instances at each scale and separate scenario files for all five robustness settings:

```bash
python tpre_repro.py generate \
  --output data/generated_reference \
  --scales 25 50 100 \
  --base-instances 10 \
  --optimization-scenarios 50 \
  --oos-scenarios 500 \
  --settings all \
  --root-seed 20260907
```

This creates **new reference data**, not the original manuscript data. Replace `--root-seed` and the default tariff boundaries with the exact experimental values if they differ from the original implementation.

## Check the neural architecture

```bash
python tpre_repro.py model-info
```

Expected output:

```json
{
  "trainable_parameters": 65091,
  "expected_trainable_parameters": 65091,
  "matches_manuscript": true
}
```

## Validate files

```bash
python tpre_repro.py validate --instance path/to/instance.json
```

## Tariff-band caveat

The manuscript provides valley/flat/peak prices of `0.45/0.75/1.20 CU/kWh`, but the text available to this clean-room implementation does not provide a machine-readable list of exact time-band boundaries. The default configuration mirrors the schematic:

- 06:00-08:00 valley
- 08:00-12:00 flat
- 12:00-18:00 peak
- 18:00-22:00 flat
- 22:00-24:00 valley

Replace these boundaries if the original experiment used a different schedule.

## Recommended additions before public release

1. Add the exact original instance and scenario seeds.
2. Add the original route pool and train/validation/test split.
3. Add the training-only normalization statistics.
4. Add the seed-2025 checkpoint and model-training command.
5. Add run-level CSV/JSON outputs for every table and figure.
6. Add an experiment manifest linking code commit, data hashes, checkpoint, seeds, and output files.
7. Archive a fixed GitHub release in Zenodo and cite its DOI in the paper.

## License

MIT License. See `LICENSE`.

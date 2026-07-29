# Data directory

The manuscript experiments use synthetic VRPTW-style base instances and
generated charging-station scenarios. They can be regenerated from recorded
seeds:

```bash
python scripts/generate_data.py --out data/generated/example
```

Generated bundles contain:

- `instance.json`: customers, stations, pre-dispatch `D_jt`, parameters and
  the instance seed;
- `optimization_scenarios.json.gz`: the fixed optimization scenario set;
- `out_of_sample_scenarios.json.gz`: a disjoint reporting scenario set.

Large generated files, run outputs and model checkpoints are ignored by Git.
The repository does not fabricate the manuscript's numerical result tables.
Exact equality with reported tables additionally requires the archived
instance/scenario seeds, run-level CSVs and seed-2025 evaluator checkpoint
described in the manuscript's Data Availability Statement.

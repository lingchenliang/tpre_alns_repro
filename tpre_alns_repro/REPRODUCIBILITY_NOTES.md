# Reproducibility notes

## Reproducibility levels

1. **Structural reproduction (included).** The repository contains executable
   implementations of the model, scenario generator, simulator, 24-feature
   proxy, ALNS variants, seed policy, tests and output aggregation.
2. **New seeded computational runs (included).** Instance/scenario bundles can
   be regenerated and stored from explicit seeds. Every run writes a manifest.
3. **Exact manuscript-table equality (requires archived artifacts).** The
   source documents state that the run-level results, route pool, normalization
   statistics, evaluator checkpoints and experiment manifest are available
   from the corresponding author. Those artifacts were not embedded in the two
   supplied documents and are therefore not invented in this repository.

## Deterministic boundaries

- `D_jt` is part of the base instance and is shared by every scenario/method.
- Optimization and 500-scenario reporting seed domains are disjoint.
- A generated scenario stores `O_jts`, `H_jts`, `A_jts`, `w_jts` and prices.
  Evaluation never redraws a queue delay.
- The same fixed 10th/50th/90th severity scenarios are used for local search
  within an instance-run.
- Screening observes only customer-complete, deterministic-forward-feasible
  candidates.
- The first 20 observed candidates are fully evaluated. Screening begins with
  candidate 21.
- A learned/heuristic score can skip or guide repair; it cannot accept a move.
- Every accepted move has a complete objective evaluation.

## Numerical-reporting unit

The experiment summarizer first averages repeated heuristic runs within each
base instance, then calculates the descriptive mean and sample standard
deviation across independent base-instance means. It does not treat all routing
runs as independent observations.

## Deliberate publication safeguards

- No hard-coded manuscript result rows are returned by an experiment script.
- A hand-crafted fallback is named in every output when no neural checkpoint is
  supplied.
- The deterministic MILP is labeled as a bounded first-stage reference, not as
  an optimum of the simulation-recourse framework.
- The planning cost is excluded from the scenario-cost CVaR term.
- At most one infeasibility penalty is charged per vehicle route and scenario.

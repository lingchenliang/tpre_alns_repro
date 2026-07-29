# Reproducibility checklist

Before treating an output table as a manuscript-comparable result, verify:

- [ ] Python 3.11 is recorded.
- [ ] The source commit or release tag is recorded.
- [ ] `configs/default.yaml` hash is recorded.
- [ ] The evaluator checkpoint hash and training seed are recorded.
- [ ] Base-instance seeds are disjoint from evaluator-training instances.
- [ ] Optimization and out-of-sample scenario seeds differ.
- [ ] Every scenario set was persisted before method runs.
- [ ] The same optimization scenarios were used by every compared method.
- [ ] The final solution was re-evaluated on 500 disjoint scenarios.
- [ ] Every accepted ALNS move has a complete objective evaluation.
- [ ] The screening proxy is `TwinBranchRiskEvaluator`, not the hand-crafted
  smoke fallback.
- [ ] Planning cost is not repeated in the CVaR term.
- [ ] At most one penalty appears per vehicle route and scenario.
- [ ] Repeated runs were averaged within base instance before cross-instance
  summaries or paired tests.
- [ ] `manifest.json`, run-level CSVs and table-generation outputs share one
  experiment identifier.

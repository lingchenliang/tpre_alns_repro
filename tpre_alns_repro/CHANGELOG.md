# Changelog

## 1.0.0 - 2026-07-29

- Aligned the instance generator with Supplementary Algorithm S2.
- Made reported unavailability a time-indexed, pre-dispatch base-instance field.
- Implemented iterative stored waiting and the exclusive fixed recourse order.
- Separated planning cost from scenario-dependent expectation and CVaR.
- Added deterministic energy/rest restoration and distinct depot copies.
- Replaced route-level summary inputs with the specified 24 stop-level features.
- Implemented the exact 65,091-parameter twin architecture and 65,235-parameter
  single-branch comparator.
- Added Algorithm S1 screening, safeguard, adaptive weights and counters.
- Reworked the deterministic MILP reference to match Section 3.5.2.
- Added disjoint-seed manifests, base-instance-first aggregation, tests and CI.
- Replaced rejection-sampled training routes with incremental certified
  insertion so hard random time windows cannot stall route-pool construction.

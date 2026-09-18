# AlphaForge eval results — 22-task suite

All numbers below are produced by `python evals/harness.py --mode synthetic` on this
checkout (synthetic data mode, template LLM backend, seed 42).
Re-run the harness to reproduce them exactly.

## Summary

| Metric | Value |
|---|---|
| Tasks | 5 |
| Task completion rate | 100.0% |
| Schema validity rate | 100.0% |
| Hallucination count (ticker/window) | 0 |
| Null-result rate | 0.0% |
| First-try tool accuracy | 100.0% |
| Avg LLM cost per task | $0.0000 |
| Avg wall time per task | 180.9s |

## Per-task results

| ID | Seed | OK | Hypotheses | Tested | Survivors | Null | Halluc | Tool 1st-try | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | post-earnings drift in megacap tech | ✔ | 8 | 8 | 5 | no | 0 | 100% | 95.14 |
| 2 | sector rotation momentum among sector ET | ✔ | 7 | 7 | 4 | no | 0 | 100% | 17.93 |
| 3 | short-term reversal after sharp drops | ✔ | 8 | 8 | 5 | no | 0 | 100% | 169.38 |
| 4 | volume spikes and next-week underperform | ✔ | 8 | 8 | 5 | no | 0 | 100% | 252.42 |
| 5 | high volatility regimes and next-week re | ✔ | 8 | 8 | 4 | no | 0 | 100% | 369.85 |

## Notes

- Template backend: hypothesis statements are instantiated from seeded
  templates (no LLM call), so cost/LLM-call metrics are 0 by construction.
  Switch `--llm claude` (requires ANTHROPIC_API_KEY) to measure the
  Claude-augmented path.
- Synthetic mode plants known effects (PEAD +60bps/day, reversal bounce
  +40bps/day, volume-shock drift −30bps/day, Monday −5bps); the null-result
  rate measures how often the correction gates still reject everything,
  i.e. how conservative the discovery loop is.
- Tasks 21-22 are deliberate stress tests: 21 tightens the correction
  gates (alpha=0.001/q=0.001); 22 forces data insufficiency (min_events
  100k) to exercise the genuine null-result path end-to-end.
- Null results are a feature: the system is designed to kill bad ideas.

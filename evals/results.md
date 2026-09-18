# AlphaForge eval results — 22-task suite

All numbers below are produced by `python evals/harness.py` on this
checkout (synthetic data mode, template LLM backend, seed 42).
Re-run the harness to reproduce them exactly.

## Summary

| Metric | Value |
|---|---|
| Tasks | 22 |
| Task completion rate | 100.0% |
| Schema validity rate | 100.0% |
| Hallucination count (ticker/window) | 0 |
| Null-result rate | 4.5% |
| First-try tool accuracy | 100.0% |
| Avg LLM cost per task | $0.0000 |
| Avg wall time per task | 48.1s |

## Per-task results

| ID | Seed | OK | Hypotheses | Tested | Survivors | Null | Halluc | Tool 1st-try | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | post-earnings drift in megacap tech | ✔ | 7 | 7 | 4 | no | 0 | 100% | 37.54 |
| 2 | sector rotation momentum among sector ET | ✔ | 7 | 7 | 4 | no | 0 | 100% | 25.02 |
| 3 | short-term reversal after sharp drops | ✔ | 8 | 8 | 5 | no | 0 | 100% | 63.27 |
| 4 | volume spikes and next-week underperform | ✔ | 8 | 8 | 5 | no | 0 | 100% | 72.54 |
| 5 | high volatility regimes and next-week re | ✔ | 7 | 7 | 4 | no | 0 | 100% | 34.88 |
| 6 | Monday seasonality in large-cap equities | ✔ | 8 | 8 | 4 | no | 0 | 100% | 58.0 |
| 7 | earnings surprises in financials | ✔ | 8 | 8 | 5 | no | 0 | 100% | 78.31 |
| 8 | reversal in consumer staples and discret | ✔ | 8 | 8 | 5 | no | 0 | 100% | 31.12 |
| 9 | momentum persistence in energy equities | ✔ | 8 | 8 | 4 | no | 0 | 100% | 51.21 |
| 10 | volume shocks around earnings windows | ✔ | 8 | 8 | 5 | no | 0 | 100% | 51.14 |
| 11 | volatility clustering in sector ETFs | ✔ | 8 | 8 | 3 | no | 0 | 100% | 35.9 |
| 12 | weekday effects in tech constituents | ✔ | 8 | 8 | 4 | no | 0 | 100% | 39.83 |
| 13 | drift after large positive earnings surp | ✔ | 8 | 8 | 5 | no | 0 | 100% | 82.17 |
| 14 | deep drawdown bouncebacks in megacaps | ✔ | 8 | 8 | 5 | no | 0 | 100% | 38.08 |
| 15 | abnormal volume and institutional rebala | ✔ | 8 | 8 | 5 | no | 0 | 100% | 58.79 |
| 16 | risk-off regimes and defensive sector re | ✔ | 8 | 8 | 3 | no | 0 | 100% | 37.91 |
| 17 | post-earnings drift in healthcare names | ✔ | 8 | 8 | 5 | no | 0 | 100% | 74.76 |
| 18 | quiet-period drift before earnings | ✔ | 8 | 8 | 5 | no | 0 | 100% | 41.63 |
| 19 | cross-sectional dispersion after vol spi | ✔ | 8 | 8 | 4 | no | 0 | 100% | 47.98 |
| 20 | persistently high volume and trend conti | ✔ | 8 | 8 | 5 | no | 0 | 100% | 74.44 |
| 21 | sector rotation momentum among sector ET | ✔ | 7 | 7 | 4 | no | 0 | 100% | 23.97 |
| 22 | post-earnings drift in megacap tech | ✔ | 7 | 0 | 0 | yes | 0 | 100% | 0.85 |

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

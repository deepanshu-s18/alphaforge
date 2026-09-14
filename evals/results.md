# AlphaForge eval results — 20-task suite

All numbers below are produced by `python evals/harness.py` on this
checkout (synthetic data mode, template LLM backend, seed 42).
Re-run the harness to reproduce them exactly.

## Summary

| Metric | Value |
|---|---|
| Tasks | 20 |
| Task completion rate | 100.0% |
| Schema validity rate | 100.0% |
| Hallucination count (ticker/window) | 0 |
| Null-result rate | 0.0% |
| Avg LLM cost per task | $0.0000 |
| Avg wall time per task | 87.2s |

## Per-task results

| ID | Seed | OK | Hypotheses | Tested | Survivors | Null | Halluc | Wall (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | post-earnings drift in megacap tech | ✔ | 7 | 7 | 4 | no | 0 | 41.71 |
| 2 | sector rotation momentum among sector ET | ✔ | 7 | 7 | 4 | no | 0 | 26.43 |
| 3 | short-term reversal after sharp drops | ✔ | 8 | 8 | 5 | no | 0 | 400.34 |
| 4 | volume spikes and next-week underperform | ✔ | 8 | 8 | 5 | no | 0 | 411.79 |
| 5 | high volatility regimes and next-week re | ✔ | 7 | 7 | 4 | no | 0 | 35.38 |
| 6 | Monday seasonality in large-cap equities | ✔ | 8 | 8 | 4 | no | 0 | 51.8 |
| 7 | earnings surprises in financials | ✔ | 8 | 8 | 5 | no | 0 | 62.82 |
| 8 | reversal in consumer staples and discret | ✔ | 8 | 8 | 5 | no | 0 | 27.66 |
| 9 | momentum persistence in energy equities | ✔ | 8 | 8 | 4 | no | 0 | 46.29 |
| 10 | volume shocks around earnings windows | ✔ | 8 | 8 | 5 | no | 0 | 48.93 |
| 11 | volatility clustering in sector ETFs | ✔ | 8 | 8 | 3 | no | 0 | 33.28 |
| 12 | weekday effects in tech constituents | ✔ | 8 | 8 | 4 | no | 0 | 43.63 |
| 13 | drift after large positive earnings surp | ✔ | 8 | 8 | 5 | no | 0 | 89.55 |
| 14 | deep drawdown bouncebacks in megacaps | ✔ | 8 | 8 | 5 | no | 0 | 38.47 |
| 15 | abnormal volume and institutional rebala | ✔ | 8 | 8 | 5 | no | 0 | 60.07 |
| 16 | risk-off regimes and defensive sector re | ✔ | 8 | 8 | 3 | no | 0 | 37.48 |
| 17 | post-earnings drift in healthcare names | ✔ | 8 | 8 | 5 | no | 0 | 90.1 |
| 18 | quiet-period drift before earnings | ✔ | 8 | 8 | 5 | no | 0 | 42.75 |
| 19 | cross-sectional dispersion after vol spi | ✔ | 8 | 8 | 4 | no | 0 | 52.45 |
| 20 | persistently high volume and trend conti | ✔ | 8 | 8 | 5 | no | 0 | 102.02 |

## Notes

- Template backend: hypothesis statements are instantiated from seeded
  templates (no LLM call), so cost/LLM-call metrics are 0 by construction.
  Switch `--llm claude` (requires ANTHROPIC_API_KEY) to measure the
  Claude-augmented path.
- Synthetic mode plants known effects (PEAD +60bps/day, reversal bounce
  +40bps/day, volume-shock drift −30bps/day, Monday −5bps); the null-result
  rate measures how often the correction gates still reject everything,
  i.e. how conservative the discovery loop is.
- Null results are a feature: the system is designed to kill bad ideas.

---
name: backtest-review
description: Review event backtests for leakage, cost realism, and split integrity
---

# Backtest Review

## When to use
Any change to `mcp_servers/backtest`, `events/builders.py`, or the backtest agent.

## Checklist
- [ ] Entry at close T+1 after event (no same-close look-ahead), exit at
      T+1+holding_days — returns computed as exit/entry − 1
- [ ] Train events: exit ≤ train_end. Test events: entry ≥ test_start. The two
      sets are DISJOINT; test metrics are what get reported
- [ ] Costs applied to BOTH legs (round-trip bps) on every event
- [ ] Validation and backtest use the SAME `build_events` logic — no
      validate-on-X, backtest-on-Y drift
- [ ] Signal ticker scope respected (backtest only the scoped instruments)
- [ ] Overlapping events allowed but documented; equity curve date-ordered
- [ ] Null/negative Sharpe reported as-is; never re-run until positive

## Commands
```bash
pytest tests/test_backtest.py -q   # includes planted-effect sign recovery
```

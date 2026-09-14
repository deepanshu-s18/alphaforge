# AlphaForge Research Report

- **Seed query**: earnings drift in megacap tech
- **Generated**: 2026-09-14T18:36:11
- **Config**: seed=42, data_mode=live, llm_backend=template
- **Multiple-testing gates**: Bonferroni α=0.05, BH-FDR q=0.1
- **Backtest**: train ≤ 2022-12-31, test ≥ 2023-01-01, cost 10.0bps round-trip

## Surviving signals

5 of 8 tested hypotheses survived Bonferroni/BH-FDR correction AND the expected-sign check.

| Signal | Hypothesis | Family | Test Sharpe | Test mean (bps) | Max DD | Win rate | n (train/test) |
|---|---|---|---|---|---|---|---|
| S-H001 | H001 | earnings_drift | 0.845 | 357.98 | -5.9% | 63.6% | 30/11 |
| S-H002 | H002 | reversal | 2.137 | 85.58 | -58.1% | 55.8% | 829/312 |
| S-H006 | H006 | volume_shock | -2.04 | -105.97 | -78.3% | 40.9% | 338/132 |
| S-H007 | H007 | earnings_drift | 1.472 | 280.3 | -15.0% | 64.5% | 68/31 |
| S-H008 | H008 | reversal | 3.888 | 144.33 | -84.9% | 56.9% | 1883/693 |

![S-H001](S-H001.png)

![S-H002](S-H002.png)

![S-H006](S-H006.png)

![S-H007](S-H007.png)

![S-H008](S-H008.png)

## Statistical validation (all tested hypotheses)

| ID | Family | Test | Stat | p | Effect (bps) | 95% CI (bps) | n | Bonf. | BH-FDR | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H001 | earnings_drift | one_sample_ttest | 11.69 | 0.0000 | +367.2 | [+307.0, +428.9] | 339 | ✔ | ✔ | backtested |
| H002 | reversal | two_sample_ttest | 18.50 | 0.0000 | +77.4 | [+68.5, +86.1] | 8299 | ✔ | ✔ | backtested |
| H003 | vol_regime | two_sample_ttest | 9.14 | 0.0000 | +38.4 | [+29.8, +47.0] | 8250 | ✔ | ✔ | rejected |
| H004 | sector_momentum | two_sample_ttest | -9.11 | 0.0000 | -45.6 | [-55.2, -35.8] | 45871 | ✔ | ✔ | rejected |
| H005 | day_of_week | two_sample_ttest | 1.46 | 0.1436 | +2.0 | [-0.7, +4.7] | 16646 | ✘ | ✘ | rejected |
| H006 | volume_shock | two_sample_ttest | -13.02 | 0.0000 | -93.0 | [-106.6, -79.3] | 2645 | ✔ | ✔ | backtested |
| H007 | earnings_drift | one_sample_ttest | 11.69 | 0.0000 | +367.2 | [+307.0, +428.9] | 339 | ✔ | ✔ | backtested |
| H008 | reversal | two_sample_ttest | 24.30 | 0.0000 | +143.3 | [+130.4, +156.3] | 8278 | ✔ | ✔ | backtested |

## Literature grounding (mcp-retrieval)

> Post-earnings announcement drift (PEAD) is the empirical observation that stock
prices continue to move in the direction of an earnings surprise for several
weeks after the announcement. Ball and Brown (1968) first documented the effect.
Explanations include underreaction to new information and limited investor
attention. The drift is strongest for high-surprise stocks and weakens after
public disclosure of the anomaly.
> — *corpus:Post-earnings announcement drift (PEAD) is the empirical obs*

> # AlphaForge grounding corpus — small curated snippets used by mcp-retrieval
# to ground the discussion section of research reports. Replace/extend with
# EDGAR filings text or earnings-call transcripts for live use.
> — *corpus:# AlphaForge grounding corpus — small curated snippets used *

> Human-in-the-loop review in agentic research systems: automated pipelines can
propose and test hypotheses at scale, but checkpoint reviews before capital
allocation are critical. A checkpoint after statistical validation and before
backtesting lets a researcher reject technically significant but
economically implausible signals.
> — *corpus:Human-in-the-loop review in agentic research systems: automa*

## Limitations & false-discovery discussion

- Synthetic-mode results validate the *pipeline*, not real market alpha. Planted effects are recoverable by construction; run `--mode live` for real data.
- 8 hypotheses were tested this run; even with Bonferroni/BH-FDR gates, in-sample specification search across template parameters inflates discovery risk. The out-of-sample split is the primary defense.
- Event-study Sharpe is annualized by events-per-year scaling and does not model overlapping-position capital constraints or execution slippage beyond the flat 10.0bps round-trip cost.
- Earnings dates and surprise values in live mode depend on vendor data quality; the synthetic mode uses deterministic planted surprises.

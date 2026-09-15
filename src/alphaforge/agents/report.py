"""Report agent: honest markdown research report + plots.

Handles the null-result path explicitly: when zero hypotheses survive the
correction gate, the report says so and analyzes why — a null result is a
feature of a system designed to kill bad ideas, not a failure to hide.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alphaforge.agents.base import RunContext  # noqa: E402
from alphaforge.events.builders import build_events  # noqa: E402
from alphaforge.state.schema import ResearchState  # noqa: E402
from alphaforge.utils.logging import get_logger  # noqa: E402

log = get_logger("alphaforge.report")


class ReportAgent:
    def __init__(self, retriever=None, llm_client=None):
        from alphaforge.mcp_servers.retrieval.server import default_retriever

        self.retriever = retriever or default_retriever()
        self.llm_client = llm_client

    def run(self, state: ResearchState, ctx: RunContext,
            out_dir: str | Path = "reports") -> ResearchState:
        out = Path(out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "report.md"
        results = state.test_results
        by_id = {h.id: h for h in state.hypotheses}

        lines = [
            "# AlphaForge Research Report",
            "",
            f"- **Seed query**: {state.seed_query}",
            f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}",
            f"- **Config**: seed={state.config.seed}, data_mode={state.config.data_mode}, "
            f"llm_backend={state.config.llm_backend}",
            f"- **Multiple-testing gates**: Bonferroni α={state.config.bonferroni_alpha}, "
            f"BH-FDR q={state.config.bh_fdr_q}",
            f"- **Backtest**: train ≤ {state.config.train_end}, test ≥ {state.config.test_start}, "
            f"cost {state.config.cost_bps}bps round-trip",
            "",
        ]

        if not state.surviving_signals:
            lines += self._null_result_section(state, results, by_id)
        else:
            lines += self._results_section(state, results, by_id, ctx, out)

        lines += self._validation_table(results, by_id)
        lines += self._grounding_section(state)
        lines += self._limitations_section(state)
        md_path.write_text("\n".join(lines))
        state.report_path = str(md_path)
        log.info("report_written", path=str(md_path), survivors=len(state.surviving_signals))
        return state

    # ------------------------------------------------------------------
    def _null_result_section(self, state, results, by_id) -> list[str]:
        untestable = [h for h in state.hypotheses if h.status == "untestable"]
        rejected = [h for h in state.hypotheses if h.status == "rejected"]
        lines = [
            "## Null result — no hypothesis survived the correction gates",
            "",
            f"Of {len(results)} tested hypotheses, **0** were both corrected-significant "
            f"and sign-consistent. This is an honest outcome: the system is designed to "
            f"kill bad ideas rather than manufacture signals.",
            "",
            f"- Rejected (not significant after correction, or sign mismatch): {len(rejected)}",
            f"- Untestable (insufficient events or data failure): {len(untestable)}",
            "",
            "### Why hypotheses failed",
            "",
        ]
        for r in results:
            h = by_id[r.hypothesis_id]
            reason = "sign mismatch" if (
                (r.effect_size > 0) != (h.expected_effect == "positive")) else "p above gate"
            lines.append(
                f"- `{r.hypothesis_id}` ({h.family}): p={r.p_value:.4f}, "
                f"effect={r.effect_size:+.1f}bps → {reason}"
            )
        for h in untestable:
            lines.append(f"- `{h.id}` ({h.family}): untestable — insufficient events")
        lines.append("")
        return lines

    def _results_section(self, state, results, by_id, ctx, out) -> list[str]:
        lines = [
            "## Surviving signals",
            "",
            f"{len(state.surviving_signals)} of {len(results)} tested hypotheses survived "
            f"Bonferroni/BH-FDR correction AND the expected-sign check.",
            "",
            "| Signal | Hypothesis | Family | Test Sharpe | Test mean (bps) "
            "| Max DD | Win rate | n indep (train/test) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        bt_by_id = {b.hypothesis_id: b for b in state.backtest_results}
        for sig in state.surviving_signals:
            h = by_id[sig.hypothesis_id]
            b = bt_by_id.get(sig.hypothesis_id)
            if b:
                lines.append(
                    f"| {sig.signal_id} | {sig.hypothesis_id} | {h.family} | "
                    f"{b.test_sharpe} | {b.test_mean_ret_bps} | {b.test_max_drawdown:.1%} | "
                    f"{b.test_win_rate:.1%} | {b.n_train_independent}/"
                    f"{b.n_test_independent} (of {b.n_train_events}/{b.n_test_events}) |"
                )
            else:
                lines.append(
                    f"| {sig.signal_id} | {sig.hypothesis_id} | {h.family} "
                    f"| — | — | — | — | — |"
                )
        lines.append("")

        # plots per surviving signal
        for sig in state.surviving_signals:
            h = by_id[sig.hypothesis_id]
            try:
                sample = build_events(h.family, h.event_def, ctx.data["ohlcv"],
                                      ctx.data.get("earnings"))
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                axes[0].hist(sample.treated * 1e4, bins=40, color="#4472c4", alpha=0.85)
                axes[0].axvline(sample.treated.mean() * 1e4, color="crimson",
                                label=f"mean {sample.treated.mean()*1e4:.0f}bps")
                axes[0].set_title(f"{sig.signal_id}: event-window returns (bps)")
                axes[0].legend()
                eq = (1 + sample.treated).cumprod()
                dates = sample.events["date"]
                axes[1].plot(dates, eq, lw=1.2, color="#2e7d32")
                axes[1].set_title(f"{sig.signal_id}: event equity (gross)")
                fig.tight_layout()
                fig.savefig(out / f"{sig.signal_id}.png", dpi=110)
                plt.close(fig)
                lines.append(f"![{sig.signal_id}]({sig.signal_id}.png)")
                lines.append("")
            except Exception as e:  # noqa: BLE001 - a plot failure must not kill the report
                log.warning("plot_failed", signal=sig.signal_id, error=str(e))
        return lines

    def _validation_table(self, results, by_id) -> list[str]:
        lines = [
            "## Statistical validation (all tested hypotheses)",
            "",
            "| ID | Family | Test | Stat | p | Effect (bps) | 95% CI (bps) "
            "| n | Bonf. | BH-FDR | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in results:
            h = by_id[r.hypothesis_id]
            lines.append(
                f"| {r.hypothesis_id} | {h.family} | {r.test_name} | {r.statistic:.2f} | "
                f"{r.p_value:.4f} | {r.effect_size:+.1f} | [{r.ci_low:+.1f}, {r.ci_high:+.1f}] | "
                f"{r.n_events} | {'✔' if r.bonferroni_significant else '✘'} | "
                f"{'✔' if r.bh_fdr_significant else '✘'} | {h.status} |"
            )
        lines.append("")
        return lines

    def _grounding_section(self, state) -> list[str]:
        try:
            from alphaforge.mcp_servers.retrieval.server import semantic_search

            hits = semantic_search(state.seed_query, k=3)
        except Exception as e:  # noqa: BLE001
            log.warning("retrieval_failed", error=str(e))
            return []
        if not hits:
            return []
        lines = ["## Literature grounding (mcp-retrieval)", ""]
        for h in hits:
            lines.append(f"> {h['text']}\n> — *{h['source']}*\n")
        # optional LLM discussion grounded in the retrieved context + results
        if state.config.llm_backend == "claude" and getattr(self, "llm_client", None):
            context = (
                f"seed query: {state.seed_query}\n\n"
                f"literature:\n" + "\n".join(h["text"] for h in hits) + "\n\n"
                f"results: {len(state.test_results)} tested, "
                f"{len(state.surviving_signals)} survived; "
                + (f"test Sharpes: {[b.test_sharpe for b in state.backtest_results]}"
                   if state.backtest_results else "null result")
            )
            polished = self.llm_client.polish_discussion(context)
            if polished:
                lines += ["## Discussion (LLM)", "", polished, ""]
        return lines

    def _limitations_section(self, state) -> list[str]:
        return [
            "## Limitations & false-discovery discussion",
            "",
            "- Synthetic-mode results validate the *pipeline*, not real market alpha. "
            "Planted effects are recoverable by construction; run `--mode live` for real data.",
            f"- {len(state.test_results)} hypotheses were tested this run; even with "
            "Bonferroni/BH-FDR gates, in-sample specification search across template "
            "parameters inflates discovery risk. The out-of-sample split is the primary defense.",
            "- Event-study Sharpe is annualized by events-per-year scaling and does not "
            "model overlapping-position capital constraints or execution slippage beyond "
            f"the flat {state.config.cost_bps}bps round-trip cost.",
            "- Earnings dates and surprise values in live mode depend on vendor data "
            "quality; the synthetic mode uses deterministic planted surprises.",
            "",
        ]

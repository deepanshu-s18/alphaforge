"""AlphaForge CLI: run the full research pipeline from the command line."""

from __future__ import annotations

import argparse
import json
import sys

from alphaforge.orchestrator.graph import Orchestrator
from alphaforge.state.schema import AgentConfig
from alphaforge.utils.logging import get_logger

log = get_logger("alphaforge.cli")

_DEFAULT_SEEDS = [42, 123, 2024]


def _run_one(orch: Orchestrator, seed_query: str, cfg: AgentConfig,
             out_dir: str, interactive: bool, prompt=input):
    if interactive:
        return orch.run_interactive(seed_query, config=cfg.model_dump(mode="json"),
                                    out_dir=out_dir, prompt=prompt)
    return orch.run(seed_query, config=cfg.model_dump(mode="json"), out_dir=out_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="alphaforge",
        description="Autonomous anomaly discovery & validation agent",
    )
    p.add_argument("seed_query", help="natural-language research seed")
    p.add_argument("--out", default="reports", help="report output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds", action="store_true",
        help="3-seed stability run (seeds 42, 123, 2024) — prints a stability "
             "table with std of key metrics across seeds; implied by the spec "
             "requirement that all experiments use 3-seed runs",
    )
    p.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    p.add_argument("--llm", choices=["template", "claude", "gemini"], default="template")
    p.add_argument("--interactive", action="store_true",
                   help="pause at the HITL checkpoint for manual approve/reject "
                        "(default: auto-approve, headless-safe)")
    p.add_argument("--metrics-json", default=None, help="write run metrics to this file")
    args = p.parse_args(argv)

    orch = Orchestrator()

    if args.seeds:
        # 3-seed stability run — key claim in the spec and resume
        results = []
        for s in _DEFAULT_SEEDS:
            cfg = AgentConfig(seed=s, data_mode=args.mode, llm_backend=args.llm,
                              hitl_approve=True)
            state = _run_one(orch, args.seed_query, cfg, args.out, False)
            results.append({"seed": s, **state.to_metrics()})
            log.info("seed_run_complete", seed=s, **state.to_metrics())

        # stability table
        import statistics as _stats
        keys = ["n_hypotheses", "n_tested", "n_survivors", "cost_usd", "llm_calls"]
        table: dict = {
            "seeds": _DEFAULT_SEEDS,
            "runs": results,
            "stability": {
                k: {
                    "mean": round(_stats.mean(r[k] for r in results), 4),
                    "std": round(_stats.stdev(r[k] for r in results), 4)
                    if len(results) > 1 else 0.0,
                }
                for k in keys
            },
        }
        print(json.dumps(table, indent=2))
        if args.metrics_json:
            with open(args.metrics_json, "w") as f:
                json.dump(table, f, indent=2)
        return 0

    # single-seed run (default)
    cfg = AgentConfig(
        seed=args.seed, data_mode=args.mode, llm_backend=args.llm,
        hitl_approve=not args.interactive,
    )
    state = _run_one(orch, args.seed_query, cfg, args.out, args.interactive)
    metrics = state.to_metrics()
    log.info("run_complete", **metrics)
    print(json.dumps(metrics, indent=2))
    if args.metrics_json:
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())


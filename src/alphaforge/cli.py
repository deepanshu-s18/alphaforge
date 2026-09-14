"""AlphaForge CLI: run the full research pipeline from the command line."""

from __future__ import annotations

import argparse
import json
import sys

from alphaforge.orchestrator.graph import Orchestrator
from alphaforge.state.schema import AgentConfig
from alphaforge.utils.logging import get_logger

log = get_logger("alphaforge.cli")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="alphaforge",
        description="Autonomous anomaly discovery & validation agent",
    )
    p.add_argument("seed_query", help="natural-language research seed")
    p.add_argument("--out", default="reports", help="report output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    p.add_argument("--llm", choices=["template", "claude"], default="template")
    p.add_argument("--interactive", action="store_true",
                   help="pause at the HITL checkpoint for manual approve/reject "
                        "(default: auto-approve, headless-safe)")
    p.add_argument("--metrics-json", default=None, help="write run metrics to this file")
    args = p.parse_args(argv)

    cfg = AgentConfig(
        seed=args.seed, data_mode=args.mode, llm_backend=args.llm,
        hitl_approve=not args.interactive,
    )
    orch = Orchestrator()
    state = orch.run(args.seed_query, config=cfg.model_dump(mode="json"), out_dir=args.out)
    metrics = state.to_metrics()
    log.info("run_complete", **metrics)
    print(json.dumps(metrics, indent=2))
    if args.metrics_json:
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

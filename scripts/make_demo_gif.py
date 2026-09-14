"""Render demo.gif — an animated visualization of a REAL pipeline run.

Animates the surviving signals' event-equity curves and return histograms
from an actual orchestrated run (synthetic mode, seed 42). No fabricated
data: every frame is drawn from the run's own event samples.

Usage: python scripts/make_demo_gif.py [--out demo.gif]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alphaforge.agents.base import RunContext  # noqa: E402
from alphaforge.events.builders import build_events  # noqa: E402
from alphaforge.orchestrator.graph import Orchestrator  # noqa: E402
from alphaforge.state.schema import AgentConfig, ResearchState  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main(out: str = "demo.gif") -> None:
    print("running pipeline (synthetic, cached)...")
    orch = Orchestrator()
    rs: ResearchState = orch.run(
        "post-earnings drift in megacap tech",
        config=AgentConfig(hitl_approve=True).model_dump(mode="json"),
        out_dir="/tmp/alphaforge_demo",
    )
    if not rs.surviving_signals:
        raise SystemExit("null result — rerun with a different seed query for a demo GIF")
    print(f"{len(rs.surviving_signals)} survivors; building frames...")

    # rebuild ctx data for plotting (same deterministic path as the run)
    ctx = RunContext(config=rs.config)
    from alphaforge.mcp_servers.market_data.server import MarketData

    md = MarketData(mode=rs.config.data_mode)
    needed = sorted({t for h in rs.hypotheses for t in h.instrument_scope})
    ctx.data["ohlcv"] = md.get_ohlcv(needed)
    ctx.data["earnings"] = md.get_earnings_calendar(needed)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    by_id = {h.id: h for h in rs.hypotheses}
    bt = {b.hypothesis_id: b for b in rs.backtest_results}

    samples = []
    for sig in rs.surviving_signals:
        h = by_id[sig.hypothesis_id]
        s = build_events(h.family, h.event_def, ctx.data["ohlcv"], ctx.data["earnings"])
        samples.append((sig, h, s))

    n_frames = 90

    def draw(frame: int):
        frac = (frame + 1) / n_frames
        axes[0].clear()
        axes[1].clear()
        # equity curves drawing in over time
        for sig, h, s in samples:
            eq = (1 + s.treated).cumprod()
            k = max(2, int(len(eq) * frac))
            dates = s.events["date"].to_numpy()[:k]
            axes[0].plot(dates, eq[:k], lw=1.4, label=f"{sig.signal_id} ({h.family})")
        axes[0].axhline(1.0, color="grey", lw=0.6, ls="--")
        axes[0].set_title("Surviving signals — event equity (gross)")
        axes[0].legend(fontsize=7)
        axes[0].tick_params(labelsize=7)
        # histograms fill in
        for sig, h, s in samples[:3]:
            vals = s.treated * 1e4
            axes[1].hist(vals[: max(2, int(len(vals) * frac))], bins=30, alpha=0.5,
                         label=f"{sig.signal_id}")
        b = bt.get(samples[0][0].hypothesis_id)
        if b:
            axes[1].set_title(
                f"event returns (bps) — test Sharpe {b.test_sharpe}", fontsize=9
            )
        axes[1].tick_params(labelsize=7)

    # render frames manually -> GIF via PIL (robust across matplotlib versions)
    from PIL import Image

    frames = []
    for f in range(n_frames):
        draw(f)
        fig.canvas.draw()
        frames.append(Image.frombytes("RGB", fig.canvas.get_width_height(),
                                       fig.canvas.buffer_rgba()))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=60, loop=0)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "demo.gif"))
    args = ap.parse_args()
    main(args.out)

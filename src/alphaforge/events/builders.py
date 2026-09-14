"""Event builders: turn a hypothesis family + market data into event samples.

Each family defines:
  - which (ticker, date) pairs are "events"
  - the forward window return measured for each event (treated sample)
  - a baseline sample to compare against

The SAME builders feed the validation agent (statistics) and the backtest
server (PnL), so what you validate is exactly what you backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EventSample:
    treated: np.ndarray          # forward window returns at events
    baseline: np.ndarray         # comparable returns at non-events
    events: pd.DataFrame         # columns: ticker, date (event dates)
    meta: dict = field(default_factory=dict)


def _forward_returns(close: pd.Series, window: int) -> pd.Series:
    """Return over (T+1 .. T+1+window]: entry close T+1, exit close T+1+window."""
    entry = close.shift(-1)
    exit_ = close.shift(-(1 + window))
    return exit_ / entry - 1.0


def build_events(family: str, event_def: dict, ohlcv: pd.DataFrame,
                 earnings: pd.DataFrame) -> EventSample:
    tickers = sorted(set(ohlcv["ticker"]))
    window = int(event_def.get("window_days", 5))
    treated_all, baseline_all, ev_frames = [], [], []

    for t in tickers:
        sub = ohlcv[ohlcv["ticker"] == t].sort_values("date").reset_index(drop=True)
        if len(sub) < window + 60:
            continue
        close = sub["close"]
        fwd = _forward_returns(close, window)

        mask = _family_mask(family, event_def, sub, earnings, t)

        treated = fwd[mask.to_numpy()].dropna()
        baseline = fwd[~mask.to_numpy()].dropna()
        if len(treated):
            treated_all.append(treated.to_numpy())
            ev_frames.append(sub.loc[treated.index, ["ticker", "date"]])
        if len(baseline):
            baseline_all.append(baseline.to_numpy())

    treated = np.concatenate(treated_all) if treated_all else np.array([])
    baseline = np.concatenate(baseline_all) if baseline_all else np.array([])
    if ev_frames:
        events = pd.concat(ev_frames, ignore_index=True)
    else:
        events = pd.DataFrame(columns=["ticker", "date"])
    return EventSample(treated=treated, baseline=baseline, events=events, meta={"window": window})


def _family_mask(family: str, event_def: dict, sub: pd.DataFrame,
                 earnings: pd.DataFrame, ticker: str) -> pd.Series:
    if family == "earnings_drift":
        thr = float(event_def.get("surprise_threshold", 2.0))
        e = earnings[earnings["ticker"] == ticker]
        beat_dates = pd.to_datetime(e.loc[e["surprise_pct"] > thr, "date"]).dt.normalize()
        return sub["date"].dt.normalize().isin(set(beat_dates))

    if family == "reversal":
        r5 = sub["close"].pct_change(5)
        q = float(event_def.get("decile", 0.10))
        return r5 <= r5.quantile(q)

    if family == "vol_regime":
        ret = sub["close"].pct_change()
        vol = ret.rolling(int(event_def.get("vol_window", 21))).std()
        q = float(event_def.get("pct", 0.90))
        return vol > vol.quantile(q)

    if family == "sector_momentum":
        mom = sub["close"].pct_change(int(event_def.get("lookback_days", 21)))
        return mom > 0

    if family == "day_of_week":
        wd = int(event_def.get("weekday", 0))  # 0=Monday
        return pd.Series(sub["date"].dt.weekday == wd, index=sub.index)

    if family == "volume_shock":
        lv = np.log(sub["volume"].astype(float).clip(lower=1))
        z = (lv - lv.mean()) / lv.std()
        return z > float(event_def.get("zscore", 2.0))

    raise ValueError(f"unknown hypothesis family: {family}")


FAMILIES = [
    "earnings_drift", "reversal", "vol_regime",
    "sector_momentum", "day_of_week", "volume_shock",
]

"""Deterministic parquet cache with SHA-256 checksum verification.

Cache key: (ticker, start, end, interval) -> deterministic file path.
Same config + same underlying data source => byte-identical cached dataset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CachedDataset:
    key: str
    path: str
    rows: int
    checksum: str


def make_key(ticker: str, start: str, end: str, interval: str) -> str:
    return f"{ticker}_{start}_{end}_{interval}".replace(":", "-")


class ParquetCache:
    def __init__(self, root: str | Path = "data/cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.parquet", self.root / f"{key}.meta.json"

    @staticmethod
    def _checksum(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def get(self, key: str) -> pd.DataFrame | None:
        pq, meta = self._paths(key)
        if not pq.exists() or not meta.exists():
            return None
        stored = json.loads(meta.read_text())["checksum"]
        if self._checksum(pq) != stored:
            raise IOError(f"cache checksum mismatch for {key}")
        return pd.read_parquet(pq)

    def put(self, key: str, df: pd.DataFrame) -> CachedDataset:
        pq, meta = self._paths(key)
        df.to_parquet(pq, index=False)
        cs = self._checksum(pq)
        meta.write_text(json.dumps({"checksum": cs, "rows": len(df)}))
        return CachedDataset(key=key, path=str(pq), rows=len(df), checksum=cs)

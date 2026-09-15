"""Retrieval backend tests: TF-IDF default, dense pluggable (toy encoder),
env selection, BLaIR integration slot contract."""

from __future__ import annotations

import numpy as np
import pytest

from alphaforge.mcp_servers.retrieval import server as rserver


def _toy_encoder(text: str) -> np.ndarray:
    """Deterministic 3-d 'embedding': hash buckets onto unit sphere.
    Words containing 'drift'/'earnings' map to dim 0, 'volume' to dim 1."""
    v = np.zeros(3)
    t = text.lower()
    if "drift" in t or "earnings" in t:
        v[0] += 1
    if "volume" in t:
        v[1] += 1
    v[2] = (len(t) % 7) / 7.0
    return v


def _corpus():
    return rserver.default_retriever().docs  # (text, source) pairs from corpus.txt


class TestTfidf:
    def test_relevant_doc_ranks_first(self):
        r = rserver.default_retriever()
        hits = r.search("post-earnings announcement drift", k=3)
        assert hits and "drift" in hits[0].text.lower()

    def test_env_selects_backend(self, monkeypatch):
        monkeypatch.setenv("ALPHAFORGE_RETRIEVER", "tfidf")
        assert isinstance(rserver.default_retriever(), rserver.TfidfRetriever)


class TestDense:
    def test_dense_backend_with_injected_encoder(self):
        corpus = _corpus()
        r = rserver.DenseRetriever(corpus, encoder=_toy_encoder)
        hits = r.search("earnings drift after beats", k=3)
        assert hits
        assert all(0 <= h.score <= 1.0001 for h in hits)
        # top hit must be a drift/earnings paragraph
        assert "drift" in hits[0].text.lower() or "earnings" in hits[0].text.lower()

    def test_dense_matches_numpy_reference(self):
        corpus = _corpus()
        r = rserver.DenseRetriever(corpus, encoder=_toy_encoder)
        q = _toy_encoder("volume spikes")
        ref = r.doc_vecs @ (q / np.linalg.norm(q))
        hits = r.search("volume spikes", k=1)
        assert hits[0].score == pytest.approx(float(ref.max()), abs=1e-5)

    def test_dense_with_faiffallback_semantics(self):
        # toy encoder + no faiss still returns valid ranking (numpy path)
        corpus = _corpus()
        r = rserver.DenseRetriever(corpus, encoder=_toy_encoder)
        a = r.search("momentum", k=2)
        assert len(a) <= 2

    def test_semantic_search_accepts_dense(self):
        corpus = _corpus()
        dense = rserver.DenseRetriever(corpus, encoder=_toy_encoder)
        out = rserver.semantic_search("reversal", k=2, retriever=dense)
        assert len(out) <= 2 and {"text", "score", "source"} <= set(out[0])

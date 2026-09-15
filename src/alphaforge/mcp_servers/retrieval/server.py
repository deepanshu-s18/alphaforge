"""mcp-retrieval: semantic search over a small financial corpus for report
grounding. Offline TF-IDF backend by default (deterministic, dependency-free);
the interface is pluggable so a dense BGE+FAISS retriever can be dropped in
without touching agents (ForgeLM / BLaIR integration point).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).parent / "corpus.txt"


@dataclass
class RetrievalHit:
    text: str
    score: float
    source: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfRetriever:
    """Minimal TF-IDF cosine retriever — deterministic and dependency-free."""

    def __init__(self, corpus: list[tuple[str, str]]):
        # corpus: list of (text, source)
        self.docs = corpus
        self.N = len(corpus)
        self.doc_tf: list[dict[str, float]] = []
        df: dict[str, int] = {}
        for text, _ in corpus:
            toks = _tokenize(text)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t in tf:
                df[t] = df.get(t, 0) + 1
            n = len(toks) or 1
            self.doc_tf.append({t: c / n for t, c in tf.items()})
        self.idf = {t: __import__("math").log((self.N + 1) / (c + 1)) + 1 for t, c in df.items()}

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        qt = _tokenize(query)
        qtf = {t: qt.count(t) / len(qt) for t in set(qt)} if qt else {}
        scores = []
        for i, dtf in enumerate(self.doc_tf):
            score = sum(qtf[t] * self.idf.get(t, 0.0) * dtf.get(t, 0.0)
                        for t in qtf if t in dtf)
            scores.append(score)
        order = sorted(range(self.N), key=lambda i: -scores[i])[:k]
        return [RetrievalHit(self.docs[i][0], float(scores[i]), self.docs[i][1])
                for i in order if scores[i] > 0]


def default_retriever(corpus_path: Path = DEFAULT_CORPUS,
                      backend: str | None = None) -> TfidfRetriever | "DenseRetriever":
    """Build the retriever selected by `backend` or ALPHAFORGE_RETRIEVER.

    'tfidf' (default) needs nothing; 'dense' uses a BGE-style sentence
    encoder + vector search (sentence-transformers optional, faiss optional)
    — the drop-in slot for the BLaIR fine-tuned BGE retriever.
    """
    corpus: list[tuple[str, str]] = []
    if corpus_path.exists():
        text = corpus_path.read_text()
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                first = para.splitlines()[0][:60]
                corpus.append((para, f"corpus:{first}"))
    backend = backend or __import__("os").environ.get("ALPHAFORGE_RETRIEVER", "tfidf")
    if backend == "dense":
        return DenseRetriever(corpus)
    return TfidfRetriever(corpus)


class DenseRetriever:
    """Dense vector retriever — the BLaIR integration slot.

    `encoder` is injectable for tests; in production it is a
    sentence-transformers BGE model (e.g. the BLaIR fine-tuned checkpoint).
    Similarity is cosine via numpy; if faiss is installed, an IndexFlatIP is
    used instead (exact search, same results).
    """

    def __init__(self, corpus: list[tuple[str, str]], encoder=None):
        self.docs = corpus
        if encoder is None:
            encoder = self._default_encoder()
        self.encoder = encoder
        import numpy as np

        doc_vecs = np.asarray([self.encoder(t) for t, _ in corpus], dtype=float)
        self._np = np
        self.doc_vecs = doc_vecs / np.clip(np.linalg.norm(doc_vecs, axis=1, keepdims=True),
                                           1e-12, None)
        self._faiss_index = None
        try:
            import faiss  # type: ignore

            self._faiss_index = faiss.IndexFlatIP(doc_vecs.shape[1])
            self._faiss_index.add(self.doc_vecs.astype("float32"))
        except ImportError:
            pass

    @staticmethod
    def _default_encoder():
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "dense retriever requires sentence-transformers "
                "(pip install sentence-transformers) or an injected encoder"
            ) from e
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return lambda text: model.encode(text, normalize_embeddings=True)

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        np = self._np
        q = np.asarray(self.encoder(query), dtype=float)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(q[None, :].astype("float32"), k)
            order = idx[0].tolist()
            score_by_i = {int(i): float(s) for i, s in zip(idx[0], scores[0])}
        else:
            sims = self.doc_vecs @ q
            order = np.argsort(-sims)[:k].tolist()
            score_by_i = {int(i): float(sims[i]) for i in order}
        return [RetrievalHit(self.docs[i][0], score_by_i[i], self.docs[i][1])
                for i in order if i < len(self.docs)]


def semantic_search(query: str, k: int = 5,
                    retriever: "TfidfRetriever | DenseRetriever | None" = None) -> list[dict]:
    r = retriever or default_retriever()
    return [{"text": h.text, "score": round(h.score, 6), "source": h.source}
            for h in r.search(query, k)]


def build_server():  # pragma: no cover - requires mcp package
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mcp-retrieval")
    r = default_retriever()

    @mcp.tool()
    def semantic_search_tool(query: str, k: int = 5) -> list[dict]:
        """Semantic search over the filings/earnings-call corpus."""
        return semantic_search(query, k, r)

    return mcp


if __name__ == "__main__":  # pragma: no cover
    build_server().run()

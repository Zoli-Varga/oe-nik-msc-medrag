"""Retrievers: BM25 (sparse), dense embeddings + FAISS, hybrid RRF fusion and cross-encoder reranking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class BM25Retriever:
    def __init__(self, texts: list[str]):
        import bm25s
        import Stemmer
        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        self.model = bm25s.BM25()
        self.model.index(bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False),
                         show_progress=False)
        self.n = len(texts)

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        tokens = self._bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, show_progress=False)
        idx, scores = self.model.retrieve(tokens, k=min(k, self.n), show_progress=False, n_threads=0)
        return idx, scores


class DenseRetriever:
    def __init__(self, name: str, texts: list[str] | None = None, batch_size: int = 8, device: str | None = None):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.name = name
        hf_id, self.q_prefix, self.p_prefix = config.EMBEDDING_MODELS[name]
        self.model = SentenceTransformer(hf_id, device=device or _device())
        self.model.max_seq_length = 512
        cache = config.INDEX_DIR / f"emb_{name}.npy"
        if cache.exists():
            emb = np.load(cache)
        else:
            assert texts is not None, "texts needed to build the index"
            emb = self.model.encode([self.p_prefix + t for t in texts], batch_size=batch_size,
                                    normalize_embeddings=True, show_progress_bar=True,
                                    convert_to_numpy=True).astype("float32")
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self.model.encode([self.q_prefix + x for x in queries], batch_size=16,
                              normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idx = self.index.search(q, k)
        return idx, scores


def rrf_fuse(rankings: list[np.ndarray], k: int, rrf_k: int = config.RRF_K) -> list[list[int]]:
    """Reciprocal Rank Fusion of several ranked id lists per query."""
    fused = []
    for per_query in zip(*rankings):
        scores: dict[int, float] = {}
        for ranked in per_query:
            for rank, cid in enumerate(ranked):
                scores[int(cid)] = scores.get(int(cid), 0.0) + 1.0 / (rrf_k + rank + 1)
        fused.append([c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]])
    return fused


class Reranker:
    def __init__(self, model_name: str = config.RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=_device(), max_length=512)

    def rerank(self, query: str, candidates: list[int], texts: list[str]) -> list[tuple[int, float]]:
        if not candidates:
            return []
        import torch
        pairs = [(query, texts[c]) for c in candidates]
        logits = None
        for bs in (4, 1):  # small GPUs (2 GB) occasionally run out of memory -> retry with smaller batches
            try:
                logits = self.model.predict(pairs, batch_size=bs, activation_fn=torch.nn.Identity(),
                                            show_progress_bar=False)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
        if logits is None:
            self.model.to("cpu")
            logits = self.model.predict(pairs, batch_size=4, activation_fn=torch.nn.Identity(),
                                        show_progress_bar=False)
            self.model.to(_device())
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype="float64")))
        order = np.argsort(-probs)
        return [(candidates[i], float(probs[i])) for i in order]


def to_doc_ranking(chunk_ids, chunks: pd.DataFrame, k: int) -> list[str]:
    """Map a chunk ranking to a de-duplicated document ranking."""
    seen, out = set(), []
    doc_ids = chunks["doc_id"].values
    for c in chunk_ids:
        d = doc_ids[int(c)]
        if d not in seen:
            seen.add(d)
            out.append(d)
            if len(out) == k:
                break
    return out

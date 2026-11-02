"""Retrieval metrics."""
import numpy as np


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(any(d in relevant for d in ranked[:k]))


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def summarize(rankings: list[list[str]], gold: list[set[str]], ks=(1, 3, 5, 10)) -> dict:
    out = {f"Recall@{k}": float(np.mean([recall_at_k(r, g, k) for r, g in zip(rankings, gold)])) for k in ks}
    out["MRR@10"] = float(np.mean([reciprocal_rank(r, g) for r, g in zip(rankings, gold)]))
    return out

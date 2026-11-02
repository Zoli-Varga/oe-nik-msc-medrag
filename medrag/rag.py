"""End-to-end RAG pipeline: retrieval -> grounded generation with citations -> abstention."""
from __future__ import annotations

import json
import re

import pandas as pd

from . import config, llm
from .retrieval import BM25Retriever, DenseRetriever, Reranker, rrf_fuse

SYSTEM_RAG = (
    "You are a careful health information assistant for lay people. "
    "Answer the user's question using ONLY the numbered sources provided. "
    "After every sentence that uses a source, cite it in square brackets, e.g. [1] or [2][3]. "
    "Do not add facts that are not in the sources. Use plain, easy-to-understand English and at most 150 words. "
    "If the sources do not contain the answer to the question, reply with exactly this sentence and nothing else: "
    f"\"{config.NOT_FOUND}\""
)

SYSTEM_NO_RAG = (
    "You are a health information assistant for lay people. Answer the user's question from your own "
    "knowledge in plain, easy-to-understand English, in at most 150 words."
)

# Below this reranker probability the best passage is considered irrelevant and the
# system abstains without calling the LLM (calibrated in scripts/05_generate_answers.py and stored
# in results/best_config.json).
ABSTAIN_THRESHOLD = 0.05


def format_context(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (Source: {p['source']} - {p['focus_area']}, {p['qtype']})\n{p['text']}"
        for i, p in enumerate(passages, 1))


def cited_numbers(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def is_abstention(answer: str) -> bool:
    a = answer.lower()
    return "could not find the answer" in a or "not contain the answer" in a or "no information" in a


def load_best_config() -> dict:
    path = config.RESULTS_DIR / "best_config.json"
    return json.loads(path.read_text()) if path.exists() else {}


class MedRAG:
    """Retrieval modes:
    - 'proposed': the configuration selected on the dev questions (results/best_config.json):
      dense or hybrid first stage, optional reranker ordering, reranker-score abstention gate;
    - 'dense_rerank' / 'hybrid_rerank': dense / hybrid (BM25+dense, RRF) top-30 reordered by the cross-encoder;
    - 'dense_gate' / 'hybrid_gate': first-stage order kept, cross-encoder used only as relevance gate;
    - 'dense', 'hybrid', 'bm25' (baseline 2) and 'no_rag' (baseline 1) without a gate.
    """

    def __init__(self, mode: str = "proposed", embed_model: str | None = None,
                 gen_model: str = config.GEN_MODEL, chunks: pd.DataFrame | None = None,
                 bm25: BM25Retriever | None = None, dense: DenseRetriever | None = None,
                 reranker: Reranker | None = None, top_k: int = config.TOP_K_CONTEXT,
                 load_retrievers: bool = True):
        best = load_best_config()
        if mode == "proposed":
            first = best.get("first_stage", "dense")
            mode_impl = f"{first}_rerank" if best.get("rerank_order", True) else f"{first}_gate"
        else:
            mode_impl = mode
        self.mode, self.impl, self.gen_model, self.top_k = mode, mode_impl, gen_model, top_k
        self.first_stage = mode_impl.split("_")[0]
        self.use_reranker = mode_impl.endswith(("_rerank", "_gate"))
        self.threshold = best.get("abstain_threshold", ABSTAIN_THRESHOLD)
        if mode == "no_rag" or not load_retrievers:
            return
        embed_model = embed_model or best.get("best_embedding", "bge-base")
        self.chunks = chunks if chunks is not None else pd.read_parquet(config.CHUNKS_PATH)
        texts = self.chunks["index_text"].tolist()
        self.texts = texts
        need_bm25 = self.first_stage in ("bm25", "hybrid")
        self.bm25 = bm25 if bm25 is not None or not need_bm25 else BM25Retriever(texts)
        need_dense = self.first_stage in ("dense", "hybrid")
        # With the reranker on a small GPU, the (cheap) query encoder runs on the CPU (the chunk index is cached).
        dense_dev = "cpu" if self.use_reranker and config.INDEX_DIR.joinpath(f"emb_{embed_model}.npy").exists() \
            else None
        self.dense = dense if dense is not None or not need_dense else DenseRetriever(embed_model, texts,
                                                                                      device=dense_dev)
        self.reranker = reranker if reranker is not None or not self.use_reranker else Reranker()

    def retrieve(self, question: str) -> list[dict]:
        if self.mode == "no_rag":
            return []
        n = config.CANDIDATES if self.use_reranker else self.top_k
        if self.first_stage == "bm25":
            idx, sc = self.bm25.search([question], n)
            ranked = [(int(i), float(s)) for i, s in zip(idx[0], sc[0])]
        elif self.first_stage == "dense":
            idx, sc = self.dense.search([question], n)
            ranked = [(int(i), float(s)) for i, s in zip(idx[0], sc[0])]
        else:
            n = config.CANDIDATES
            b_idx, _ = self.bm25.search([question], n)
            d_idx, _ = self.dense.search([question], n)
            fused = rrf_fuse([b_idx, d_idx], n)[0]
            ranked = [(c, 1.0 / (r + 1)) for r, c in enumerate(fused)]
        if self.use_reranker:
            # score = cross-encoder relevance probability (used by the abstention gate)
            rr = self.reranker.rerank(question, [c for c, _ in ranked], self.texts)
            if self.impl.endswith("_rerank"):
                ranked = rr
            else:
                prob = dict(rr)
                ranked = [(c, prob[c]) for c, _ in ranked]
        ranked = ranked[: self.top_k]
        out = []
        for cid, score in ranked:
            row = self.chunks.iloc[cid]
            out.append({"chunk_id": row.chunk_id, "doc_id": row.doc_id, "source": row.source,
                        "focus_area": row.focus_area, "qtype": row.qtype, "text": row.text,
                        "score": score})
        return out

    def answer(self, question: str, passages: list[dict] | None = None) -> dict:
        if self.mode == "no_rag":
            text = llm.chat(self.gen_model, [{"role": "system", "content": SYSTEM_NO_RAG},
                                             {"role": "user", "content": question}])
            return {"question": question, "answer": text, "passages": [], "citations": [],
                    "abstained": is_abstention(text), "gate_abstained": False}
        passages = passages if passages is not None else self.retrieve(question)
        if self.use_reranker and (not passages or max(p["score"] for p in passages) < self.threshold):
            return {"question": question, "answer": config.NOT_FOUND, "passages": passages,
                    "citations": [], "abstained": True, "gate_abstained": True}
        user = f"Sources:\n{format_context(passages)}\n\nQuestion: {question}"
        text = llm.chat(self.gen_model, [{"role": "system", "content": SYSTEM_RAG},
                                         {"role": "user", "content": user}])
        cites = [n for n in cited_numbers(text) if 1 <= n <= len(passages)]
        return {"question": question, "answer": text, "passages": passages, "citations": cites,
                "abstained": is_abstention(text), "gate_abstained": False}


def render(result: dict) -> str:
    """Human-readable answer with a source list and the medical disclaimer."""
    lines = [result["answer"], ""]
    if result["citations"]:
        lines.append("Sources:")
        for n in result["citations"]:
            p = result["passages"][n - 1]
            lines.append(f"  [{n}] {p['source']} - {p['focus_area']} ({p['qtype']}), doc {p['doc_id']}")
        lines.append("")
    lines.append(f"Note: {config.DISCLAIMER}")
    return "\n".join(lines)

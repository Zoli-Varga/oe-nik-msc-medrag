"""Step 4: retrieval evaluation (Recall@k, MRR@10) of BM25, dense, hybrid and reranked retrieval,
and data-driven selection of the retrieval configuration of the proposed system.

The 500 evaluation questions are split into a dev part (400 questions, not used for answer generation)
and a test part (the 100 questions of the generation subset). Configuration choices (embedding model,
hybrid vs. dense first stage, reranker on/off) are made on the dev part only.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrag import config, metrics  # noqa: E402
from medrag.retrieval import BM25Retriever, DenseRetriever, Reranker, rrf_fuse, to_doc_ranking  # noqa: E402

KS = (1, 3, 5, 10)


def slug(system: str) -> str:
    kind = "hybrid" if system.startswith("Hybrid") else "dense"
    emb = system.split("+")[1].split(" ")[0] if kind == "hybrid" else system[len("Dense ("):-1]
    return f"{kind}_{emb}"


def main():
    import torch
    chunks = pd.read_parquet(config.CHUNKS_PATH)
    texts = chunks["index_text"].tolist()
    ev = pd.read_parquet(config.EVAL_PATH)
    gold = [set(g) for g in ev["gold_docs"]]
    dev = ~ev["in_generation_set"].to_numpy()
    query_sets = {"paraphrased": ev["paraphrase"].tolist(), "original": ev["question"].tolist()}
    n = config.CANDIDATES

    chunk_rankings: dict[tuple[str, str], list] = {}
    timings = {}
    bm25 = BM25Retriever(texts)
    for qs_name, qs in query_sets.items():
        t = time.time()
        chunk_rankings[("BM25", qs_name)] = list(bm25.search(qs, n)[0])
        timings[("BM25", qs_name)] = (time.time() - t) / len(qs)

    for name in config.EMBEDDING_MODELS:
        dense = DenseRetriever(name, texts)
        for qs_name, qs in query_sets.items():
            t = time.time()
            chunk_rankings[(f"Dense ({name})", qs_name)] = list(dense.search(qs, n)[0])
            timings[(f"Dense ({name})", qs_name)] = (time.time() - t) / len(qs)
            chunk_rankings[(f"Hybrid BM25+{name} (RRF)", qs_name)] = \
                rrf_fuse([chunk_rankings[("BM25", qs_name)], chunk_rankings[(f"Dense ({name})", qs_name)]], n)
            timings[(f"Hybrid BM25+{name} (RRF)", qs_name)] = timings[("BM25", qs_name)] + \
                timings[(f"Dense ({name})", qs_name)]
        del dense
        gc.collect()
        torch.cuda.empty_cache()

    doc_cache = {}

    def doc_rank(key):
        if key not in doc_cache:
            doc_cache[key] = [to_doc_ranking(r, chunks, 10) for r in chunk_rankings[key]]
        return doc_cache[key]

    def dev_mrr(system):
        dr = doc_rank((system, "paraphrased"))
        return float(np.mean([metrics.reciprocal_rank(r, g) for r, g, m in zip(dr, gold, dev) if m]))

    # Embedding model choice (by dev MRR on paraphrased queries)
    dense_names = [f"Dense ({e})" for e in config.EMBEDDING_MODELS]
    hybrid_names = [f"Hybrid BM25+{e} (RRF)" for e in config.EMBEDDING_MODELS]
    best_dense, best_hybrid = max(dense_names, key=dev_mrr), max(hybrid_names, key=dev_mrr)
    print("Best dense:", best_dense, "| best hybrid:", best_hybrid)

    # Cross-encoder reranking of the top-30 candidates of both first stages (paraphrased queries only).
    reranker = Reranker()
    qs = query_sets["paraphrased"]
    rr_cache = {}
    for first in (best_hybrid, best_dense):
        cache_path = config.RESULTS_DIR / f"rerank_cache_{slug(first)}.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        n_new = 0
        for i, (q, cands) in enumerate(tqdm(list(zip(qs, chunk_rankings[(first, "paraphrased")])),
                                            desc=f"rerank {first}")):
            if str(i) not in cache:
                t0 = time.time()
                rr = reranker.rerank(q, [int(c) for c in cands], texts)
                cache[str(i)] = [[c for c, _ in rr], [s for _, s in rr], time.time() - t0]
                n_new += 1
                if n_new % 20 == 0:
                    cache_path.write_text(json.dumps(cache))
        cache_path.write_text(json.dumps(cache))
        entries = [cache[str(i)] for i in range(len(qs))]
        rr_cache[first] = entries
        name = f"{first} + reranker"
        chunk_rankings[(name, "paraphrased")] = [e[0] for e in entries]
        timings[(name, "paraphrased")] = timings[(first, "paraphrased")] + float(np.mean([e[2] for e in entries]))

    # Final configuration of the proposed system (by dev MRR)
    candidates = [best_dense, best_hybrid, f"{best_dense} + reranker", f"{best_hybrid} + reranker"]
    final = max(candidates, key=dev_mrr)
    first_stage = final.replace(" + reranker", "")
    rerank_order = final.endswith("+ reranker")
    print("Final retrieval configuration:", final)

    # Gate score = highest reranker probability among the passages given to the LLM
    gate = []
    for i, e in enumerate(rr_cache[first_stage]):
        scores = e[1] if isinstance(e[1], list) else [e[1]]
        if rerank_order:
            gate.append(float(scores[0]))
        else:
            sc = dict(zip(e[0], scores))
            top = chunk_rankings[(first_stage, "paraphrased")][i][: config.TOP_K_CONTEXT]
            gate.append(float(max(sc[int(c)] for c in top)))

    rows = []
    for (system, qs_name) in chunk_rankings:
        dr = doc_rank((system, qs_name))
        row = {"system": system, "queries": qs_name, **metrics.summarize(dr, gold, KS)}
        if qs_name == "paraphrased":
            for part, mask in (("dev", dev), ("test", ~dev)):
                sub = [(r, g) for r, g, m in zip(dr, gold, mask) if m]
                row[f"Recall@5_{part}"] = float(np.mean([metrics.recall_at_k(r, g, 5) for r, g in sub]))
                row[f"MRR@10_{part}"] = float(np.mean([metrics.reciprocal_rank(r, g) for r, g in sub]))
        row["ms_per_query"] = 1000 * timings.get((system, qs_name), np.nan)
        rows.append(row)
    res = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(config.RESULTS_DIR / "retrieval_metrics.csv", index=False)
    with pd.option_context("display.width", 250):
        print(res.round(3).to_string())

    # Per-question output and per question type breakdown on paraphrased queries
    pq = {"bm25": doc_rank(("BM25", "paraphrased")), "dense": doc_rank((best_dense, "paraphrased")),
          "hybrid_rr": doc_rank((f"{best_hybrid} + reranker", "paraphrased")),
          "final": doc_rank((final, "paraphrased"))}
    ev_out = ev[["qid", "question", "paraphrase", "qtype", "source", "focus_area"]].copy()
    ev_out["split"] = np.where(dev, "dev", "test")
    ev_out["gold_docs"] = [sorted(g) for g in gold]
    for key, ranks in pq.items():
        ev_out[f"r5_{key}"] = [metrics.recall_at_k(r, g, 5) for r, g in zip(ranks, gold)]
        ev_out[f"rr_{key}"] = [metrics.reciprocal_rank(r, g) for r, g in zip(ranks, gold)]
    ev_out["top_docs_final"] = [r[:5] for r in pq["final"]]
    ev_out["gate_score"] = gate
    # Lenient hit: a top-5 document is about the same focus area (disease) regardless of question type
    doc_focus = dict(zip(chunks.doc_id, chunks.focus_area.str.lower()))
    for key in ("bm25", "final"):
        ev_out[f"focus_hit5_{key}"] = [float(any(doc_focus[d] == f.lower() for d in r[:5]))
                                       for r, f in zip(pq[key], ev.focus_area)]
    ev_out.to_json(config.RESULTS_DIR / "retrieval_per_query.jsonl", orient="records", lines=True, force_ascii=False)
    by_type = ev_out.groupby("qtype").agg(n=("qid", "size"), R5_BM25=("r5_bm25", "mean"),
                                          R5_dense=("r5_dense", "mean"), R5_hybrid_rr=("r5_hybrid_rr", "mean"),
                                          R5_final=("r5_final", "mean"),
                                          MRR_final=("rr_final", "mean")).sort_values("n", ascending=False)
    by_type.to_csv(config.RESULTS_DIR / "retrieval_by_qtype.csv")
    print(by_type.round(3).to_string())
    print("Lenient (same disease) hit@5 - BM25:", ev_out.focus_hit5_bm25.mean(), "final:", ev_out.focus_hit5_final.mean())

    final_emb = first_stage.split("+")[1].split(" ")[0] if first_stage.startswith("Hybrid") else first_stage[7:-1]
    (config.RESULTS_DIR / "best_config.json").write_text(json.dumps(
        {"best_dense": best_dense, "best_hybrid": best_hybrid, "final_system": final,
         "first_stage": "hybrid" if first_stage.startswith("Hybrid") else "dense",
         "rerank_order": rerank_order, "best_embedding": final_emb}, indent=1))


if __name__ == "__main__":
    main()

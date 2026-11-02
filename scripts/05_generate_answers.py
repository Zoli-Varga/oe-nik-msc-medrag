"""Step 5: answer generation for the three compared systems on the generation subset + out-of-scope questions.

Systems: no_rag (baseline 1: LLM only), bm25 (baseline 2: BM25-only RAG), proposed (configuration selected in step 04).
Retrieval is cached per system in results/retrieved_<system>.json; answers are appended to
results/generations.jsonl, so the script can be resumed and systems can be run separately:

    python scripts/05_generate_answers.py --systems no_rag bm25
    python scripts/05_generate_answers.py --systems proposed
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrag import config, llm, rag  # noqa: E402

SYSTEMS = ["no_rag", "bm25", "proposed"]
OUT = config.RESULTS_DIR / "generations.jsonl"


def load_questions() -> list[dict]:
    ev = pd.read_parquet(config.EVAL_PATH)
    gen = ev[ev.in_generation_set]
    qs = [{"qid": r.qid, "question": r.paraphrase, "original_question": r.question, "reference": r.answer,
           "qtype": r.qtype, "gold_docs": list(r.gold_docs), "category": "in-domain"} for r in gen.itertuples()]
    for o in json.loads(config.OOD_PATH.read_text(encoding="utf-8")):
        qs.append({"qid": o["qid"], "question": o["question"], "original_question": o["question"],
                   "reference": "", "qtype": "out-of-scope", "gold_docs": [], "category": o["category"]})
    return qs


def calibrate_threshold() -> float:
    """Abstention threshold = 5th percentile of the gate score (highest reranker probability among the LLM passages) on in-domain
    retrieval-eval questions that are NOT in the generation subset (no test leakage)."""
    per_q = pd.read_json(config.RESULTS_DIR / "retrieval_per_query.jsonl", lines=True)
    ev = pd.read_parquet(config.EVAL_PATH)
    held = per_q[per_q.qid.isin(ev.loc[~ev.in_generation_set, "qid"])]
    thr = float(np.percentile(held.gate_score, 5))
    cfg_path = config.RESULTS_DIR / "best_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["abstain_threshold"] = thr
    cfg_path.write_text(json.dumps(cfg, indent=1))
    return thr


def retrieve(system: str, questions: list[dict]) -> dict:
    path = config.RESULTS_DIR / f"retrieved_{system}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    import torch
    pipe = rag.MedRAG(system)
    out = {}
    for q in tqdm(questions, desc=f"retrieve {system}"):
        t = time.time()
        out[q["qid"]] = {"passages": pipe.retrieve(q["question"]), "retrieval_s": time.time() - t}
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    args = ap.parse_args()
    questions = load_questions()
    thr = None
    if "proposed" in args.systems:
        thr = calibrate_threshold()
        print(f"Abstention threshold (reranker prob): {thr:.4f}")
    retrieved = {s: retrieve(s, questions) for s in args.systems if s != "no_rag"}

    if not llm.is_available(config.GEN_MODEL):
        sys.exit(f"Ollama model {config.GEN_MODEL} not available - run: ollama pull {config.GEN_MODEL}")
    done = set()
    if OUT.exists():
        done = {(r["system"], r["qid"]) for r in map(json.loads, OUT.read_text(encoding="utf-8").splitlines())}
    pipes = {s: rag.MedRAG(s, load_retrievers=False) for s in args.systems}
    todo = [(s, q) for q in questions for s in args.systems if (s, q["qid"]) not in done]
    with OUT.open("a", encoding="utf-8") as f:
        for system, q in tqdm(todo, desc="generate"):
            r = retrieved.get(system, {}).get(q["qid"], {"passages": None, "retrieval_s": 0.0})
            t = time.time()
            res = pipes[system].answer(q["question"], passages=r["passages"])
            rec = {**q, "system": system, "answer": res["answer"], "citations": res["citations"],
                   "abstained": res["abstained"], "gate_abstained": res["gate_abstained"],
                   "passages": res["passages"], "gen_s": time.time() - t, "threshold": thr,
                   "retrieval_s": r["retrieval_s"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()

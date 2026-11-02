"""Step 7: automatic answer metrics.

- BERTScore F1 and embedding cosine similarity between answer and NIH reference answer
- Faithfulness (RAGAS-style): share of answer sentences entailed by at least one retrieved passage,
  measured with an NLI cross-encoder
- Citation support: share of cited sentences entailed by one of the passages they cite
- Abstention rates on in-domain vs. out-of-scope questions, LLM-judge scores, latency
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrag import config  # noqa: E402

GEN = config.RESULTS_DIR / "generations.jsonl"
JUDGE = config.RESULTS_DIR / "judgements.jsonl"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
SYSTEM_NAMES = {"no_rag": "LLM without RAG (baseline 1)", "bm25": "BM25-only RAG (baseline 2)",
                "proposed": "Proposed RAG"}


def sentences(text: str) -> list[tuple[str, list[int]]]:
    """Split into sentences; returns (sentence without citation markers, cited source numbers)."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = s.strip(" -*")
        cites = [int(n) for n in re.findall(r"\[(\d+)\]", s)]
        clean = re.sub(r"\s*\[\d+\]", "", s).strip()
        if len(clean.split()) >= 3:
            out.append((clean, cites))
        elif cites and not clean and out:
            # "... sentence. [1]" – a trailing citation marker belongs to the preceding sentence
            out[-1][1].extend(cites)
    return out


def premise_windows(text: str, size: int = 3, stride: int = 2) -> list[str]:
    """NLI models are trained on short premises; long passages are mostly scored 'neutral',
    so passages are split into overlapping windows of a few sentences."""
    ss = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if len(ss) <= size:
        return [" ".join(ss)] if ss else [text]
    return [" ".join(ss[i:i + size]) for i in range(0, len(ss) - size + stride, stride)]


def main():
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.DataFrame([json.loads(x) for x in GEN.read_text(encoding="utf-8").splitlines()])
    df["answer_clean"] = df["answer"].str.replace(r"\s*\[\d+\]", "", regex=True)

    # --- Faithfulness and citation support with NLI ---
    nli = CrossEncoder(NLI_MODEL, device=device, max_length=512)
    ent_idx = [k for k, v in nli.model.config.id2label.items() if v.lower() == "entailment"][0]
    faith, cite_support, n_sent = [], [], []
    for rec in tqdm(df.itertuples(), total=len(df), desc="NLI"):
        if rec.system == "no_rag" or rec.abstained or not rec.passages:
            faith.append(np.nan); cite_support.append(np.nan); n_sent.append(np.nan)
            continue
        sents = sentences(rec.answer)
        if not sents:
            faith.append(np.nan); cite_support.append(np.nan); n_sent.append(0)
            continue
        ptexts = [p["text"] for p in rec.passages]
        wins = [(j, w) for j, pt in enumerate(ptexts) for w in premise_windows(pt)]
        pairs = [(w, s) for s, _ in sents for _, w in wins]
        probs = nli.predict(pairs, apply_softmax=True, batch_size=32, show_progress_bar=False)[:, ent_idx]
        probs = np.asarray(probs).reshape(len(sents), len(wins))
        wp = np.array([j for j, _ in wins])
        probs = np.stack([probs[:, wp == j].max(axis=1) for j in range(len(ptexts))], axis=1)
        faith.append(float(np.mean(probs.max(axis=1) > 0.5)))
        cited = [(i, [c - 1 for c in cs if 1 <= c <= len(ptexts)]) for i, (_, cs) in enumerate(sents)]
        cited = [(i, cs) for i, cs in cited if cs]
        cite_support.append(float(np.mean([probs[i, cs].max() > 0.5 for i, cs in cited])) if cited else np.nan)
        n_sent.append(len(sents))
    df["faithfulness"], df["citation_support"], df["n_sentences"] = faith, cite_support, n_sent
    df["has_citation"] = df["citations"].map(len) > 0
    del nli
    torch.cuda.empty_cache()

    # --- Similarity to reference (in-domain only) ---
    ind = df[df.category == "in-domain"].copy()
    from bert_score import score as bertscore
    _, _, f1 = bertscore(ind["answer_clean"].tolist(), ind["reference"].tolist(), lang="en",
                         rescale_with_baseline=True, device=device, batch_size=8)
    ind["bertscore_f1"] = f1.numpy()
    emb = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    a = emb.encode(ind["answer_clean"].tolist(), normalize_embeddings=True)
    r = emb.encode(ind["reference"].tolist(), normalize_embeddings=True)
    ind["cosine_sim"] = (a * r).sum(axis=1)
    df = df.merge(ind[["system", "qid", "bertscore_f1", "cosine_sim"]], on=["system", "qid"], how="left")

    if JUDGE.exists():
        j = pd.DataFrame([json.loads(x) for x in JUDGE.read_text(encoding="utf-8").splitlines()])
        df = df.merge(j.drop_duplicates(["system", "qid"], keep="last"), on=["system", "qid"], how="left")
    # Retrieval hit of the passages given to the LLM (gold document among the top-k context)
    df["context_hit"] = [float(any(p["doc_id"] in set(g) for p in ps)) if g else np.nan
                         for ps, g in zip(df.passages, df.gold_docs)]

    df.drop(columns=["passages"]).to_json(config.RESULTS_DIR / "generation_per_answer.jsonl",
                                          orient="records", lines=True, force_ascii=False)

    rows = []
    for system in ["no_rag", "bm25", "proposed"]:
        s = df[df.system == system]
        i, o = s[s.category == "in-domain"], s[s.category != "in-domain"]
        answered = i[~i.abstained]
        row = {"system": SYSTEM_NAMES[system], "n_in_domain": len(i),
               "context_hit@4": i.context_hit.mean() if system != "no_rag" else np.nan,
               "BERTScore_F1": answered.bertscore_f1.mean(), "cosine_sim": answered.cosine_sim.mean(),
               "faithfulness": answered.faithfulness.mean(), "citation_rate": answered.has_citation.mean()
               if system != "no_rag" else np.nan, "citation_support": answered.citation_support.mean(),
               "abstain_in_domain": i.abstained.mean(), "abstain_out_of_scope": o.abstained.mean(),
               "abstain_off_topic": o[o.category == "off-topic"].abstained.mean(),
               "abstain_health_not_covered": o[o.category == "health, not covered"].abstained.mean(),
               "latency_s": (s.gen_s + s.retrieval_s).mean()}
        if "correctness" in df:
            row["judge_correctness"] = i.correctness.mean()
            row["judge_relevance"] = i.relevance.mean()
            row["judge_correct_rate(>=4)"] = (i.correctness >= 4).mean()
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "generation_metrics.csv", index=False)
    print(res.round(3).T.to_string())

    by_type = (df[(df.category == "in-domain")]
               .pivot_table(index="qtype", columns="system", values="correctness", aggfunc="mean"))
    by_type.to_csv(config.RESULTS_DIR / "generation_correctness_by_qtype.csv")


if __name__ == "__main__":
    main()

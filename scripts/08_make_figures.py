"""Step 8: figures for the documentation and the presentation."""
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrag import config  # noqa: E402

R = config.RESULTS_DIR
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
COLORS = ["#9aa5b1", "#5b8def", "#2f6fd0", "#f0a33a", "#2a9d8f", "#1d6f64", "#e76f51"]


def short(name: str) -> str:
    return (name.replace("Hybrid BM25+", "Hybrid ").replace(" (RRF)", "").replace(" + reranker", " + rerank")
            .replace("Dense (", "Dense ").replace(")", ""))


def best() -> dict:
    return json.loads((R / "best_config.json").read_text())


def final_label() -> str:
    return "Proposed: " + short(best()["final_system"])


def retrieval_bar():
    df = pd.read_csv(R / "retrieval_metrics.csv")
    p = df[df.queries == "paraphrased"].copy()
    p["label"] = p.system.map(short)
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    x = range(len(p))
    w = 0.27
    for j, (m, c) in enumerate([("Recall@1", "#9aa5b1"), ("Recall@5", "#2f6fd0"), ("MRR@10", "#f0a33a")]):
        ax.bar([i + (j - 1) * w for i in x], p[m], w, label=m, color=c)
    ax.set_xticks(list(x))
    ax.set_xticklabels(p.label, rotation=25, ha="right")
    for t, s in zip(ax.get_xticklabels(), p.system):
        if s == best()["final_system"]:
            t.set_fontweight("bold")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_ylabel("score")
    ax.set_title("Retrieval on LLM-paraphrased evaluation questions (n=500)")
    ax.legend(ncol=3, loc="upper center", fontsize=8)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "retrieval_comparison.png")


def retrieval_by_type():
    df = pd.read_csv(R / "retrieval_by_qtype.csv")
    df = df[df.n >= 8].sort_values("R5_final")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = list(range(len(df)))
    bars = [("R5_BM25", "BM25", "#9aa5b1"), ("R5_hybrid_rr", "Hybrid + rerank", "#f0a33a"),
            ("R5_final", final_label(), "#2f6fd0")]
    for j, (col, lab, c) in enumerate(bars):
        ax.barh([i + (j - 1) * .27 for i in y], df[col], .27, label=lab, color=c)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} (n={n})" for t, n in zip(df.qtype, df.n)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall@5")
    ax.set_title("Recall@5 by question type (paraphrased)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=3, fontsize=8)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "retrieval_by_qtype.png")


def generation_bar():
    df = pd.read_csv(R / "generation_metrics.csv")
    labels = ["No RAG (baseline 1)", "BM25 RAG (baseline 2)", "Proposed RAG"]
    metrics = [("judge_correct_rate(>=4)", "Correct (judge ≥4)"), ("BERTScore_F1", "BERTScore F1"),
               ("faithfulness", "Faithfulness (NLI)"), ("abstain_out_of_scope", "Abstain on out-of-scope")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.26
    for j, (lab, c) in enumerate(zip(labels, ["#9aa5b1", "#f0a33a", "#2f6fd0"])):
        vals = [df.iloc[j][m] for m, _ in metrics]
        ax.bar([i + (j - 1) * w for i in range(len(metrics))], [0 if pd.isna(v) else v for v in vals], w,
               label=lab, color=c)
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, (0 if pd.isna(v) else v) + .01, "n/a" if pd.isna(v) else f"{v:.2f}",
                    ha="center", fontsize=7)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_title("Answer quality: 100 in-domain + 20 out-of-scope questions")
    ax.legend(ncol=3, loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "generation_comparison.png")


def architecture():
    b = best()
    emb = {"medembed-base": "MedEmbed", "bge-base": "BGE", "e5-base": "E5"}.get(b["best_embedding"], b["best_embedding"])
    if b["first_stage"] == "dense":
        first = (f"Dense retrieval\n{emb} + FAISS\n(top-30)", "#cfe0fc")
    else:
        first = (f"BM25 + {emb}\nRRF fusion\n(top-30)", "#cfe0fc")
    second = ("Cross-encoder\nrerank → top-4", "#cfe0fc") if b["rerank_order"] else \
        ("Top-4 chunks\n+ cross-encoder\nrelevance score", "#cfe0fc")
    boxes = [("User question", "#e9ecef"), first, second, ("max score < τ ?\n→ \"not found\"", "#fde2c4"),
             ("Qwen2.5-3B\n(Ollama)\nanswer + [n] cites", "#d4f1ec"), ("Answer +\nsources +\ndisclaimer", "#e9ecef")]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    for i, (t, c) in enumerate(boxes):
        x = i * 1.5
        ax.add_patch(plt.Rectangle((x, 0.2), 1.25, 0.9, facecolor=c, edgecolor="#333", lw=1))
        ax.text(x + .625, .65, t, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.48, .65), xytext=(x + 1.25, .65), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(1.5, -0.08, "MedQuAD corpus: 15.8k unique NIH answers → 25.5k chunks + metadata (source, disease, "
            "question type)", ha="left", fontsize=8, style="italic")
    ax.set_xlim(-.1, len(boxes) * 1.5)
    ax.set_ylim(-.2, 1.2)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "architecture.png")


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    architecture()
    retrieval_bar()
    retrieval_by_type()
    if (R / "generation_metrics.csv").exists():
        generation_bar()
    print("figures written to", config.FIG_DIR)


if __name__ == "__main__":
    main()

# MedRAG – Trustworthy Health Information Assistant

Retrieval-augmented generation (RAG) over authoritative NIH sources (MedQuAD), with source citations
and an "I don't know" (abstention) behaviour.

> The system is for educational purposes only and does not replace professional medical advice.

## Architecture

```
question ──► dense retrieval (MedEmbed-base + FAISS) ► top-4 passages
         ──► cross-encoder (bge-reranker-base) relevance gate: max score < τ ? "I don't know"
         ──► Qwen2.5-3B (Ollama) ► answer with [n] citations + disclaimer
```

The originally planned pipeline (hybrid BM25 + dense with RRF fusion, then cross-encoder reranking) is
implemented and evaluated as well, but on the development split pure dense retrieval with the medical
embedding model was clearly better. The final configuration (`results/best_config.json`) is therefore chosen
automatically by dev MRR@10 in `scripts/04_eval_retrieval.py`; the reranker is kept as the relevance gate for
abstention (τ = 5th percentile of the dev gate scores).

## Results (summary)

Retrieval – 500 paraphrased (lay-language) questions, document-level gold:

| System | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| BM25 (baseline) | 0.290 | 0.678 | 0.450 |
| Dense (bge-base) | 0.788 | 0.960 | 0.861 |
| Dense (e5-base) | 0.678 | 0.958 | 0.794 |
| **Dense (medembed-base) – final** | **0.808** | **0.960** | **0.876** |
| Hybrid BM25+medembed-base (RRF) | 0.550 | 0.910 | 0.706 |
| Hybrid BM25+medembed-base (RRF) + reranker | 0.690 | 0.932 | 0.791 |
| Dense (medembed-base) + reranker | 0.686 | 0.922 | 0.789 |

Generation – 100 in-domain + 20 out-of-scope questions, Qwen2.5-3B, judge: Llama 3.2 3B:

| Metric | LLM without RAG (baseline 1) | BM25-only RAG (baseline 2) | **Proposed RAG** |
|---|---|---|---|
| Judge: correct answers (score ≥ 4) | 46% | 52% | **69%** |
| Judge: correctness / relevance (1–5) | 3.14 / 3.85 | 3.00 / 3.26 | **3.51 / 4.02** |
| BERTScore F1 vs. reference | 0.158 | 0.314 | **0.354** |
| Gold document in context (top-4) | – | 51% | **94%** |
| Faithfulness (NLI, answered questions) | – | 57.8% | **62.3%** |
| Citation support | – | **46.0%** | 44.6% |
| Abstention on in-domain questions (lower is better) | 0% | 24% | 13% |
| Abstention on out-of-scope questions (higher is better) | 0% | 95% | **100%** |

The faithfulness and citation-support values are conservative: the sentence-level NLI check fails on long
list sentences, paraphrases and table-like sources (analysis in section 7.5 of the documentation). Of the 13%
in-domain abstentions, 4 come from the relevance gate and 9 from the LLM itself.

The full analysis (in Hungarian) and the slides were submitted separately.

| Component | Implementation |
|---|---|
| Data | [MedQuAD (Kaggle)](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research) – 16,406 QA pairs from 9 NIH websites |
| Chunking | answers de-duplicated → 15,807 documents → 25,533 chunks (≤200 words, 40-word overlap) with metadata header (disease, question type) |
| Retrieval | BM25, `bge-base-en-v1.5`, `e5-base-v2`, `MedEmbed-base-v0.1` (medical) + FAISS; hybrid RRF; `bge-reranker-base` cross-encoder (final: MedEmbed dense + reranker as relevance gate) |
| Generation | `qwen2.5:3b` via Ollama (local, CPU), temperature 0 |
| Baselines | (1) LLM without RAG, (2) BM25-only RAG |
| Evaluation | 500 held-out questions paraphrased by an LLM (retrieval), 100 of them + 20 out-of-scope questions (generation); Recall@k, MRR, BERTScore, NLI faithfulness, citation support, LLM judge (`llama3.2:3b`), metric-reliability and abstention-threshold analysis |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118   # or the CPU wheel
.venv\Scripts\pip install -r requirements.txt
# Ollama: https://ollama.com  then
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

## Reproducing the experiments

```powershell
python scripts/01_prepare_data.py          # download + clean MedQuAD, chunk, sample eval questions
python scripts/02_paraphrase_questions.py  # LLM paraphrases of held-out questions
python scripts/03_build_indexes.py         # embeddings for the 3 embedding models (cached)
python scripts/04_eval_retrieval.py        # Recall@k / MRR for all retrievers + final config selection (dev split)
python scripts/05_generate_answers.py      # answers of the 3 systems (resumable; --systems proposed)
python scripts/06_llm_judge.py             # LLM-as-a-judge (resumable)
python scripts/07_score_answers.py         # BERTScore, NLI faithfulness, abstention, aggregation
python scripts/08_make_figures.py          # figures (written to docs/figures/)
```

The MedQuAD data (`data/raw/`), the processed corpus and the embedding indexes (`data/index/`) are not part of the
repository (they are large); `01_prepare_data.py` and `03_build_indexes.py` recreate them. The held-out evaluation
questions and their paraphrases (`data/processed/eval_questions.parquet`, `paraphrases.json`) are included so the
reported numbers can be reproduced exactly.

## Demo

```powershell
python ask.py "How is cystic fibrosis inherited?"
python ask.py --show-passages "what are early signs of glaucoma"
```

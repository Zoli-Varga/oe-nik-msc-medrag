"""Step 1: download MedQuAD (if needed), clean it, build documents/chunks and sample the evaluation set."""
import io
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrag import config, data  # noqa: E402


def download():
    if config.RAW_CSV.exists():
        return
    print("Downloading MedQuAD from Kaggle ...")
    r = requests.get(config.KAGGLE_URL, timeout=300)
    r.raise_for_status()
    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(config.RAW_CSV.parent)


def main():
    download()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    qa = data.load_raw()
    docs = data.build_documents(qa)
    chunks = data.build_chunks(docs)
    qa.to_parquet(config.QA_PATH)
    docs.to_parquet(config.DOCS_PATH)
    chunks.to_parquet(config.CHUNKS_PATH)

    # Evaluation questions: unique questions (duplicates merged, all their answer docs are gold).
    gold = data.relevant_docs(qa)
    uq = qa.drop_duplicates("q_key").copy()
    uq["n_words_answer"] = uq["answer"].str.split().str.len()
    sample = uq.sample(n=config.N_EVAL_RETRIEVAL, random_state=config.SEED).copy()
    sample["gold_docs"] = sample["q_key"].map(lambda k: sorted(gold[k]))
    # Generation subset: reference answers of reasonable length for answer-similarity metrics.
    gen_pool = sample[(sample.n_words_answer >= 20) & (sample.n_words_answer <= 350)]
    gen_ids = gen_pool.sample(n=config.N_EVAL_GENERATION, random_state=config.SEED).index
    sample["in_generation_set"] = sample.index.isin(gen_ids)
    sample = sample.reset_index(drop=True)
    sample.insert(0, "qid", [f"Q{i:03d}" for i in range(len(sample))])
    sample[["qid", "question", "answer", "source", "focus_area", "qtype", "doc_id", "gold_docs",
            "n_words_answer", "in_generation_set"]].to_parquet(config.EVAL_PATH)

    print(f"QA pairs: {len(qa)}  unique questions: {qa.q_key.nunique()}  documents: {len(docs)}  "
          f"chunks: {len(chunks)}")
    print(f"Eval questions: {len(sample)} (generation subset: {sample.in_generation_set.sum()})")
    print("Question types:\n", qa.qtype.value_counts().to_string())


if __name__ == "__main__":
    main()

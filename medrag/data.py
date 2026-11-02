"""Loading, cleaning, labelling and chunking of the MedQuAD corpus."""
import hashlib
import re

import pandas as pd

from . import config

# MedQuAD questions are generated from a fixed set of templates, so the question
# type can be recovered with simple patterns (the Kaggle CSV has no qtype column).
QTYPE_PATTERNS = [
    ("symptoms", r"^what are the symptoms of"),
    ("causes", r"^what causes"),
    ("treatment", r"^what are the treatments for"),
    ("diagnosis", r"^how to diagnose"),
    ("inheritance", r"^is .* inherited"),
    ("genetic changes", r"^what are the genetic changes related to"),
    ("frequency", r"^how many people are affected by"),
    ("outlook", r"^what is the outlook for"),
    ("prevention", r"^how to prevent"),
    ("susceptibility", r"^who is at risk for"),
    ("stages", r"^what are the stages of"),
    ("complications", r"^what are the complications of"),
    ("considerations", r"^what to do for"),
    ("research", r"^what research \(or clinical trials\) is being done for"),
    ("information", r"^(what is \(are\)|do you have information about)"),
]


def question_type(question: str) -> str:
    q = question.strip().lower()
    for name, pattern in QTYPE_PATTERNS:
        if re.search(pattern, q):
            return name
    return "other"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_CSV)
    df = df.dropna(subset=["question", "answer"]).copy()
    df["question"] = df["question"].map(normalize_ws)
    df["answer"] = df["answer"].map(normalize_ws)
    df["focus_area"] = df["focus_area"].fillna("unknown").map(normalize_ws)
    df = df[df["answer"].str.split().str.len() >= 3].reset_index(drop=True)
    df["qtype"] = df["question"].map(question_type)
    df["q_key"] = df["question"].map(norm_key)
    df["doc_id"] = df["answer"].map(lambda a: "D" + short_hash(norm_key(a)))
    return df


def build_documents(qa: pd.DataFrame) -> pd.DataFrame:
    """One document per unique answer text (MedQuAD contains duplicate answers)."""
    docs = (qa.groupby("doc_id", sort=False)
              .agg(text=("answer", "first"), source=("source", "first"),
                   focus_area=("focus_area", "first"), qtype=("qtype", "first"),
                   question=("question", "first"))
              .reset_index())
    return docs


def chunk_text(text: str, size: int = config.CHUNK_WORDS,
               overlap: int = config.CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= size + overlap // 2:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_chunks(docs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in docs.itertuples():
        for i, piece in enumerate(chunk_text(d.text)):
            header = f"{d.focus_area} ({d.qtype})"
            rows.append({
                "chunk_id": f"{d.doc_id}-{i}",
                "doc_id": d.doc_id,
                "chunk_no": i,
                "source": d.source,
                "focus_area": d.focus_area,
                "qtype": d.qtype,
                "text": piece,
                # Indexed text: metadata header + passage body
                "index_text": f"{header}: {piece}",
            })
    return pd.DataFrame(rows)


def relevant_docs(qa: pd.DataFrame) -> dict[str, set[str]]:
    """Gold relevance: every document that answers (a duplicate of) the question."""
    return qa.groupby("q_key")["doc_id"].apply(set).to_dict()

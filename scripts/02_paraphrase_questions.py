"""Step 2: paraphrase the held-out evaluation questions with a local LLM (Ollama),
so that they do not match the MedQuAD question templates verbatim."""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrag import config, llm  # noqa: E402
from medrag.data import norm_key  # noqa: E402

PROMPT = (
    "Rewrite the following medical question the way a worried non-expert patient or relative would "
    "naturally ask it in a chat. Keep exactly the same meaning and keep the name of the disease or condition "
    "(you may lower-case it), but change the wording and sentence structure; do not start with the same words. "
    "Return JSON: {{\"paraphrase\": \"...\"}}\n\nQuestion: {q}"
)


def paraphrase(q: str, temperature: float) -> str:
    out = llm.chat(config.PARAPHRASE_MODEL, [{"role": "user", "content": PROMPT.format(q=q)}],
                   temperature=temperature, num_predict=120, num_ctx=1024, json_format=True)
    try:
        return json.loads(out)["paraphrase"].strip()
    except (json.JSONDecodeError, KeyError, AttributeError):
        return ""


def valid(p: str, q: str) -> bool:
    # Small models occasionally switch to Chinese characters mid-sentence
    foreign = any(ord(ch) >= 0x2E80 for ch in p)
    return len(p) > 10 and norm_key(p) != norm_key(q) and not foreign


def main():
    ev = pd.read_parquet(config.EVAL_PATH)
    cache_path = config.PROCESSED_DIR / "paraphrases.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for row in tqdm(ev.itertuples(), total=len(ev)):
        if row.qid in cache and valid(cache[row.qid], row.question):
            continue
        p = ""
        for t in (0.3, 0.8, 1.0):
            p = paraphrase(row.question, t)
            if valid(p, row.question):
                break
        if not valid(p, row.question):
            p = re.sub(r"[\u2e80-\uffff]+", "", p).replace(" ?", "?").strip()
        cache[row.qid] = p or row.question
        if len(cache) % 20 == 0:
            cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    ev["paraphrase"] = ev["qid"].map(cache)
    ev.to_parquet(config.EVAL_PATH)
    print(ev[["question", "paraphrase"]].head(15).to_string())


if __name__ == "__main__":
    main()

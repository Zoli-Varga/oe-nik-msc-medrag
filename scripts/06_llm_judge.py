"""Step 6: LLM-as-a-judge (different model family than the generator) for correctness and relevance."""
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrag import config, llm  # noqa: E402

IN = config.RESULTS_DIR / "generations.jsonl"
OUT = config.RESULTS_DIR / "judgements.jsonl"

PROMPT = """You are grading an answer to a health question. Compare the CANDIDATE answer with the REFERENCE answer written by medical experts (NIH).

QUESTION: {question}

REFERENCE ANSWER: {reference}

CANDIDATE ANSWER: {answer}

Give two integer scores from 1 to 5:
- "correctness": 5 = the candidate's key facts agree with the reference and nothing contradicts it; 3 = partly correct or important facts missing; 1 = wrong, contradicts the reference, or does not answer.
- "relevance": 5 = directly and completely addresses the question; 1 = off-topic or a refusal.
Return JSON only: {{"correctness": <1-5>, "relevance": <1-5>, "reason": "<one short sentence>"}}"""


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text)


def judge(rec: dict) -> dict:
    ref = " ".join(rec["reference"].split()[:450])
    msg = PROMPT.format(question=rec["original_question"], reference=ref, answer=strip_citations(rec["answer"]))
    for temperature in (0.0, 0.5):
        out = llm.chat(config.JUDGE_MODEL, [{"role": "user", "content": msg}], temperature=temperature,
                       num_predict=120, num_ctx=3072, json_format=True)
        try:
            j = json.loads(out)
            c, r = int(j["correctness"]), int(j["relevance"])
            if 1 <= c <= 5 and 1 <= r <= 5:
                return {"correctness": c, "relevance": r, "reason": str(j.get("reason", ""))}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return {"correctness": None, "relevance": None, "reason": "unparseable"}


def main():
    recs = [json.loads(line) for line in IN.read_text(encoding="utf-8").splitlines()]
    done = set()
    if OUT.exists():
        done = {(r["system"], r["qid"]) for r in map(json.loads, OUT.read_text(encoding="utf-8").splitlines())}
    # Only in-domain answers have a reference; abstentions are scored as 1 without calling the judge.
    todo = [r for r in recs if r["category"] == "in-domain" and (r["system"], r["qid"]) not in done]
    with OUT.open("a", encoding="utf-8") as f:
        for rec in tqdm(todo, desc="judge"):
            if rec["abstained"]:
                j = {"correctness": 1, "relevance": 1, "reason": "abstained"}
            else:
                j = judge(rec)
            f.write(json.dumps({"system": rec["system"], "qid": rec["qid"], **j}, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()

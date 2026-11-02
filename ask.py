"""Interactive demo: ask a health question, get a cited answer from the RAG system.

    python ask.py "How is cystic fibrosis inherited?"
    python ask.py --mode bm25 "What are the symptoms of glaucoma?"
    python ask.py            # interactive loop
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from medrag import rag  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--mode", default="proposed",
                    choices=["proposed", "dense_rerank", "hybrid_rerank", "dense_gate", "hybrid_gate",
                             "hybrid", "dense", "bm25", "no_rag"])
    ap.add_argument("--show-passages", action="store_true")
    args = ap.parse_args()

    pipe = rag.MedRAG(args.mode)

    def run(q: str):
        res = pipe.answer(q)
        print("\n" + rag.render(res))
        if args.show_passages:
            for i, p in enumerate(res["passages"], 1):
                print(f"\n--- [{i}] score={p['score']:.3f} {p['source']} | {p['focus_area']} ({p['qtype']})\n"
                      f"{p['text'][:400]}...")
        print()

    if args.question:
        run(" ".join(args.question))
        return
    while True:
        try:
            q = input("Question (empty to quit): ").strip()
        except EOFError:
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()

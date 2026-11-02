"""Step 3: embed the chunk corpus with every compared embedding model (cached to data/index)."""
import gc
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrag import config  # noqa: E402
from medrag.retrieval import DenseRetriever  # noqa: E402


def main():
    import torch
    texts = pd.read_parquet(config.CHUNKS_PATH)["index_text"].tolist()
    for name in config.EMBEDDING_MODELS:
        t = time.time()
        r = DenseRetriever(name, texts)
        print(f"{name}: {r.index.ntotal} vectors, {time.time() - t:.0f}s")
        del r
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

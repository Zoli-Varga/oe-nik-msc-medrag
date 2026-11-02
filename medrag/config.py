"""Central configuration: paths, model names and hyper-parameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "medquad.csv"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "docs" / "figures"

DOCS_PATH = PROCESSED_DIR / "documents.parquet"
CHUNKS_PATH = PROCESSED_DIR / "chunks.parquet"
QA_PATH = PROCESSED_DIR / "qa_pairs.parquet"
EVAL_PATH = PROCESSED_DIR / "eval_questions.parquet"
OOD_PATH = ROOT / "data" / "out_of_scope_questions.json"

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/pythonafroz/medquad-medical-question-answer-for-ai-research"

SEED = 42

# Chunking
CHUNK_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

# Evaluation set sizes
N_EVAL_RETRIEVAL = 500
N_EVAL_GENERATION = 100

# Embedding models compared in retrieval experiments: name -> (HF id, query prefix, passage prefix)
EMBEDDING_MODELS = {
    "bge-base": ("BAAI/bge-base-en-v1.5",
                 "Represent this sentence for searching relevant passages: ", ""),
    "e5-base": ("intfloat/e5-base-v2", "query: ", "passage: "),
    "medembed-base": ("abhinand/MedEmbed-base-v0.1",
                      "Represent this sentence for searching relevant passages: ", ""),
}
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Retrieval
RRF_K = 60
CANDIDATES = 30  # candidates passed to the reranker
TOP_K_CONTEXT = 4  # chunks given to the LLM

# Generation (local Ollama)
OLLAMA_URL = "http://localhost:11434"
GEN_MODEL = "qwen2.5:3b"
JUDGE_MODEL = "llama3.2:3b"
PARAPHRASE_MODEL = "qwen2.5:3b"
# Layers offloaded to GPU by Ollama. 0 = CPU only: on the development laptop (MX330, 2 GB) the
# Ollama Vulkan backend crashed the GPU driver, so CPU inference is the stable default.
OLLAMA_NUM_GPU = int(os.environ.get("MEDRAG_OLLAMA_NUM_GPU", "0"))

DISCLAIMER = ("This information is for educational purposes only and does not replace "
              "professional medical advice.")
NOT_FOUND = "I could not find the answer to this question in the available sources."

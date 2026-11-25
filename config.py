"""Configuration and shared text helpers for the Sand3D Hybrid Multimodal RAG."""

import os
from pathlib import Path
import re

PROJECT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.environ.get("SAND3D_RAG_STORAGE", PROJECT_DIR / "storage"))
QDRANT_PATH = STORAGE_DIR / "qdrant_db"
ASSETS_DB_PATH = STORAGE_DIR / "assets.sqlite"
TEXT_VECS_PATH = STORAGE_DIR / "text_vectors.npy"
VISION_VECS_PATH = STORAGE_DIR / "vision_vectors.npy"
BM25_PATH = STORAGE_DIR / "bm25_index.pkl"
API_CACHE_PATH = STORAGE_DIR / "sand3d_api.txt"
OUTPUT_DIR = Path(os.environ.get("SAND3D_RAG_OUTPUTS", PROJECT_DIR / "outputs"))

DATASET_REPO = "Sandflow/sand3d-nature"
COLLECTION_NAME = "sand3d_nature"

# Best multilingual + vision embedding stack
TEXT_MODEL_ID = os.environ.get("SAND3D_TEXT_MODEL", "BAAI/bge-m3")
VISION_MODEL_ID = os.environ.get("SAND3D_VISION_MODEL", "google/siglip2-so400m-patch14-384")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")

TEXT_DIM = 1024
VISION_DIM = 1152

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def build_search_document(row: dict) -> str:
    """Create canonical searchable text combining prompt and taxonomy labels."""
    parts = [
        row.get("prompt") or "",
        f"domain: {row.get('domain') or ''}",
        f"family: {row.get('family') or ''}",
        f"subject: {row.get('subject') or ''}",
    ]
    style = row.get("style")
    if style:
        parts.append(f"style: {style}")
    return " | ".join(p for p in parts if p.strip())


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric/Persian terms for BM25."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def extract_feature_tensor(output):
    """Extract pooled tensor across transformers output variants."""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported feature output type: {type(output)}")

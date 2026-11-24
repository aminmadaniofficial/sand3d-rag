# Sand3D Nature — Hybrid Multimodal RAG

A local **Hybrid Multimodal Retrieval-Augmented Generation (RAG)** pipeline for [`Sandflow/sand3d-nature`](https://huggingface.co/datasets/Sandflow/sand3d-nature) — **10,057 procedural organic 3D assets** written in Python with [`sand3d`](https://github.com/Sandflow-ai/sand3d) (built on Blender `bpy`).

Instead of fine-tuning an LLM from scratch, this project indexes prompts, taxonomy metadata, and 512×512 render images so any LLM (local via Ollama or cloud API) can retrieve real `sand3d` reference programs and synthesize new deterministic 3D assets from **English/Persian text**, **reference images**, or **both**.

## Architecture

| Component | Model / Tool | Role |
| :--- | :--- | :--- |
| **Multilingual Text Embedding** | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) (1024-dim) | Semantic retrieval across 100+ languages (including Persian ↔ English cross-lingual search) |
| **Vision & Cross-Modal Embedding** | [`google/siglip2-so400m-patch14-384`](https://huggingface.co/google/siglip2-so400m-patch14-384) (1152-dim) | Image-to-3D similarity and multilingual text-to-image matching over 512px renders |
| **Lexical Search** | `BM25Okapi` (`rank-bm25`) | Exact matching on botanical names, taxonomy (`domain`, `family`, `subject`), and style tags |
| **Vector Store & Fusion** | Local [Qdrant](https://qdrant.tech/) + Reciprocal Rank Fusion (RRF) | Multi-vector storage (`text` + `vision`) with metadata filtering (`semantic_score`, `split`, `domain`) |
| **Code Generation** | [Ollama](https://ollama.com/) (e.g. `gemma4:12b`, `qwen2.5vl:7b`) | Generates executable `sand3d` Python programs from retrieved examples + API summary |

## Hardware Requirements

- **Indexing (one-time):** Tested for a single **12 GB VRAM GPU** (e.g., NVIDIA RTX 3060 12GB). Text and vision encoders are loaded sequentially so peak VRAM stays around ~4–5 GB.
- **Disk space:** ~1.6 GB for dataset Parquet/cache + ~6.9 GB for embedding weights (`bge-m3` 2.32 GB + `siglip2-so400m-patch14-384` 4.58 GB) + local Qdrant index.

## Setup

Python **3.11** (or 3.13) is recommended so the same environment can also run `sand3d` and Blender `bpy`.

```bash
git clone https://github.com/aminmadaniofficial/sand3d-rag.git
cd sand3d-rag

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

*(For CUDA GPUs, install the matching PyTorch CUDA wheel first, e.g. `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.)*

## Usage

### 1. Build the Hybrid Index

Downloads `Sandflow/sand3d-nature`, encodes all 10,057 asset descriptions and embedded PNG renders, and writes the local Qdrant + BM25 storage under `storage/`:

```bash
python build_index.py
```

Optional quick smoke test on the first 200 rows:

```bash
python build_index.py --limit 200
```

### 2. Search Assets (Text, Image, or Multimodal)

```bash
# English text query
python retriever.py --text "mossy rock with glowing forest mushrooms" --top-k 5

# Persian text query (cross-lingual via BGE-M3 + SigLIP2)
python retriever.py --text "صخره خزه‌دار کنار قارچ‌های جنگلی" --top-k 5

# Visual similarity search from a reference image
python retriever.py --image path/to/reference.png --top-k 5

# Combined image + text query
python retriever.py --image path/to/reference.png --text "more moss and thicker trunk" --top-k 5
```

### 3. Generate a New `sand3d` Program (RAG + Ollama)

Make sure [Ollama](https://ollama.com/) is running locally with a coding/VLM model:

```bash
python generate.py \
  --text "pine tree covered in moss growing on a mossy boulder" \
  --model gemma4:12b \
  --out pine_boulder.py
```

Generated scripts are written to `outputs/`.

### 4. Execute Generated Programs with `sand3d` (Optional)

```bash
uv pip install "sand3d==1.7.0" numpy scipy
```

Then run the generated function inside an empty Blender scene and export OpenUSD (`.usda`):

```python
import bpy
import sand3d as s3d
from outputs.pine_boulder import build_asset

bpy.ops.wm.read_factory_settings(use_empty=True)
root = build_asset(seed=0)
s3d.export(root, "pine_boulder.usda")
```

## Project Structure

```text
sand3d-rag/
├── config.py          # Model IDs, paths, canonical search document & BM25 tokenizer
├── build_index.py     # Dataset loader, BGE-M3 + SigLIP2 encoding, Qdrant & BM25 builder
├── retriever.py       # Sand3DRetriever with RRF fusion and CLI
├── generate.py        # RAG prompt builder + Ollama code generator
├── requirements.txt
├── LICENSE
└── README.md
```

## Acknowledgments & Licensing

- **Code in this repository:** Released under the [MIT License](LICENSE).
- **Dataset:** [`Sandflow/sand3d-nature`](https://huggingface.co/datasets/Sandflow/sand3d-nature) by [Sandflow](https://sandflow.ai/), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **3D Library:** [`Sandflow-ai/sand3d`](https://github.com/Sandflow-ai/sand3d), released under Apache-2.0.

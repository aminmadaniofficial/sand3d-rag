"""MCP server exposing Sand3D Nature Hybrid Multimodal RAG to AI agents."""

import argparse
import ast
import atexit
import contextlib
import gc
import sys
import threading
from pathlib import Path

import torch
from mcp.server.mcpserver import MCPServer

from config import COLLECTION_NAME, OUTPUT_DIR
from generate import build_generation_prompt, call_ollama, extract_python_code, get_sand3d_api_summary
from retriever import Sand3DRetriever

mcp = MCPServer(
    "sand3d-rag",
    instructions=(
        "Hybrid Multimodal RAG for Sandflow/sand3d-nature (10,057 procedural 3D Python programs). "
        "Use search_sand3d_assets with English/Persian queries or image paths to retrieve real "
        "sand3d reference scripts, get_sand3d_api_reference for the sand3d API summary, or "
        "generate_sand3d_program to synthesize a new script via local Ollama."
    ),
)

_LOCK = threading.Lock()
_RETRIEVER: Sand3DRetriever | None = None


def _get_retriever() -> Sand3DRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        with contextlib.redirect_stdout(sys.stderr):
            _RETRIEVER = Sand3DRetriever()
    return _RETRIEVER


def _unload_retriever() -> None:
    global _RETRIEVER
    if _RETRIEVER is not None:
        try:
            _RETRIEVER.close()
        finally:
            _RETRIEVER = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


atexit.register(_unload_retriever)


@mcp.tool()
def search_sand3d_assets(
    query: str | None = None,
    image_path: str | None = None,
    top_k: int = 5,
    min_score: float = 0.9,
    domain: str | None = None,
    split: str | None = None,
    include_program: bool = True,
) -> list[dict]:
    """Search 10,057 procedural 3D assets by English/Persian text, reference image, or both."""
    if not query and not image_path:
        raise ValueError("Provide at least one of 'query' or 'image_path'.")

    top_k = max(1, min(top_k, 20))
    with _LOCK:
        with contextlib.redirect_stdout(sys.stderr):
            retriever = _get_retriever()
            hits = retriever.search(
                query_text=query,
                image_path=image_path,
                top_k=top_k,
                min_score=min_score,
                split=split,
                domain=domain,
            )

    results = []
    for h in hits:
        p = h.payload
        item = {
            "id": h.point_id,
            "rrf_score": round(h.rrf_score, 5),
            "ranks": h.ranks,
            "prompt": p.get("prompt"),
            "domain": p.get("domain"),
            "family": p.get("family"),
            "subject": p.get("subject"),
            "style": p.get("style"),
            "semantic_score": p.get("semantic_score"),
            "triangles": p.get("triangles"),
            "entry": p.get("entry"),
            "split": p.get("split"),
        }
        if include_program:
            item["program"] = p.get("program")
        results.append(item)
    return results


@mcp.tool()
def get_sand3d_api_reference(max_chars: int = 8000) -> str:
    """Return the condensed sand3d Python API reference for writing valid 3D scripts."""
    with contextlib.redirect_stdout(sys.stderr):
        summary = get_sand3d_api_summary()
    return summary[: max(500, min(max_chars, 20000))]


@mcp.tool()
def get_index_stats() -> dict:
    """Return collection statistics and example domains from the local Qdrant index."""
    with _LOCK:
        with contextlib.redirect_stdout(sys.stderr):
            retriever = _get_retriever()
            info = retriever.client.get_collection(COLLECTION_NAME)
    return {
        "collection": COLLECTION_NAME,
        "points_count": info.points_count,
        "vectors": ["text (BAAI/bge-m3, 1024-dim)", "vision (SigLIP2-SO400M-384, 1152-dim)"],
        "lexical": "BM25Okapi (English + Persian)",
    }


@mcp.tool()
def generate_sand3d_program(
    prompt: str,
    image_path: str | None = None,
    top_k: int = 3,
    model: str = "gemma4:12b",
    output_filename: str = "mcp_generated.py",
) -> dict:
    """Retrieve matching sand3d examples and synthesize a new sand3d Python script via local Ollama."""
    safe_name = Path(output_filename).name or "mcp_generated.py"
    if not safe_name.endswith(".py"):
        safe_name += ".py"

    with _LOCK:
        with contextlib.redirect_stdout(sys.stderr):
            retriever = _get_retriever()
            hits = retriever.search(
                query_text=prompt,
                image_path=image_path,
                top_k=max(1, min(top_k, 8)),
                min_score=0.9,
            )
            # Free VRAM before local Ollama runs on 12GB GPUs
            if retriever.device.startswith("cuda"):
                _unload_retriever()
            api_summary = get_sand3d_api_summary()
            sys_prompt, usr_prompt = build_generation_prompt(prompt, hits, api_summary)
            raw = call_ollama(
                model=model,
                system_prompt=sys_prompt,
                user_message=usr_prompt,
                image_path=image_path,
            )

    code = extract_python_code(raw)
    syntax_valid = True
    syntax_error = None
    try:
        ast.parse(code)
    except SyntaxError as exc:
        syntax_valid = False
        syntax_error = str(exc)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / safe_name
    out_path.write_text(code + "\n", encoding="utf-8")

    return {
        "saved_path": str(out_path),
        "syntax_valid": syntax_valid,
        "syntax_error": syntax_error,
        "retrieved_examples": [
            {"family": h.payload.get("family"), "prompt": h.payload.get("prompt")}
            for h in hits
        ],
        "code": code,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sand3D RAG MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

"""MCP server & MCP App exposing Sand3D Nature Hybrid Multimodal RAG to AI agents."""

import argparse
import ast
import atexit
import base64
import contextlib
import gc
import io
import json
import sys
import threading
from pathlib import Path

import torch
from datasets import DownloadConfig, concatenate_datasets, load_dataset
from mcp.server.apps import Apps, ResourceCsp, client_supports_apps
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.types import CallToolResult, TextContent

from config import COLLECTION_NAME, DATASET_REPO, OUTPUT_DIR, PROJECT_DIR
from generate import (
    build_generation_prompt,
    call_ollama,
    extract_python_code,
    get_sand3d_api_summary,
)
from retriever import Sand3DRetriever

EXPLORER_URI = "ui://sand3d-rag/explorer.html"
EXPLORER_HTML_PATH = PROJECT_DIR / "ui" / "explorer.html"

apps = Apps()
apps.add_html_resource(
    EXPLORER_URI,
    EXPLORER_HTML_PATH.read_text(encoding="utf-8"),
    name="Sand3D RAG Explorer",
    title="Sand3D Nature Hybrid RAG Explorer",
    description="Interactive visual & code explorer for Sandflow/sand3d-nature assets.",
    csp=ResourceCsp(),
    prefers_border=True,
)

_LOCK = threading.Lock()
_RETRIEVER: Sand3DRetriever | None = None
_DATASET = None


def _get_retriever() -> Sand3DRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        with contextlib.redirect_stdout(sys.stderr):
            _RETRIEVER = Sand3DRetriever()
    return _RETRIEVER


def _get_dataset():
    global _DATASET
    if _DATASET is None:
        with contextlib.redirect_stdout(sys.stderr):
            ds_dict = load_dataset(
                DATASET_REPO,
                download_config=DownloadConfig(local_files_only=True),
            )
            _DATASET = concatenate_datasets([ds_dict["train"], ds_dict["validation"]])
    return _DATASET


def _render_thumbnail_data_uri(point_id: int, size: int = 224) -> str | None:
    try:
        ds = _get_dataset()
        img = ds[int(point_id)]["render"].convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


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


@apps.tool(
    resource_uri=EXPLORER_URI,
    title="Search Sand3D Assets",
    description="Search 10,057 procedural 3D assets by English/Persian text, reference image, or both.",
)
def search_sand3d_assets(
    query: str | None = None,
    image_path: str | None = None,
    top_k: int = 5,
    min_score: float = 0.9,
    domain: str | None = None,
    split: str | None = None,
    include_program: bool = True,
    ctx: Context | None = None,
) -> CallToolResult:
    """Search 10,057 procedural 3D assets by English/Persian text, reference image, or both."""
    if not query and not image_path:
        raise ValueError("Provide at least one of 'query' or 'image_path'.")

    top_k = max(1, min(top_k, 20))
    include_thumbs = ctx is not None and client_supports_apps(ctx)

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

    text_results = []
    ui_results = []
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
        text_results.append(item)

        ui_item = dict(item)
        if include_thumbs:
            ui_item["thumbnail_data_uri"] = _render_thumbnail_data_uri(h.point_id)
        ui_results.append(ui_item)

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(text_results, ensure_ascii=False))],
        structured_content={
            "query": query or "",
            "results": ui_results,
        },
    )


@apps.tool(
    resource_uri=EXPLORER_URI,
    title="Generate Sand3D Program",
    description="Retrieve matching sand3d examples and synthesize a new sand3d Python script via local Ollama.",
)
def generate_sand3d_program(
    prompt: str,
    image_path: str | None = None,
    top_k: int = 3,
    model: str = "gemma4:12b",
    output_filename: str = "mcp_generated.py",
) -> CallToolResult:
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

    payload = {
        "saved_path": str(out_path),
        "syntax_valid": syntax_valid,
        "syntax_error": syntax_error,
        "retrieved_examples": [
            {"family": h.payload.get("family"), "prompt": h.payload.get("prompt")}
            for h in hits
        ],
        "code": code,
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
    )


# Instantiate MCPServer after all @apps.tool decorators so Apps extension binds every tool.
mcp = MCPServer(
    "sand3d-rag",
    instructions=(
        "Hybrid Multimodal RAG for Sandflow/sand3d-nature (10,057 procedural 3D Python programs). "
        "Use search_sand3d_assets with English/Persian queries or image paths to retrieve real "
        "sand3d reference scripts, get_sand3d_api_reference for the sand3d API summary, or "
        "generate_sand3d_program to synthesize a new script via local Ollama."
    ),
    extensions=[apps],
)


@mcp.tool()
def get_sand3d_api_reference(max_chars: int = 8000) -> str:
    """Return the condensed sand3d Python API reference for writing valid 3D scripts."""
    with contextlib.redirect_stdout(sys.stderr):
        summary = get_sand3d_api_summary()
    return summary[: max(500, min(max_chars, 20000))]


@mcp.tool()
def get_index_stats() -> dict:
    """Return collection statistics from the local Qdrant index."""
    with _LOCK:
        with contextlib.redirect_stdout(sys.stderr):
            retriever = _get_retriever()
            info = retriever.client.get_collection(COLLECTION_NAME)
    return {
        "collection": COLLECTION_NAME,
        "points_count": info.points_count,
        "vectors": ["text (BAAI/bge-m3, 1024-dim)", "vision (SigLIP2-SO400M-384, 1152-dim)"],
        "lexical": "BM25Okapi (English + Persian)",
        "mcp_app_uri": EXPLORER_URI,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sand3D RAG MCP server & MCP App")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

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

from mcp.server.apps import Apps, ResourceCsp, client_supports_apps
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.types import CallToolResult, TextContent

from config import (
    ASSETS_DB_PATH,
    COLLECTION_NAME,
    DATASET_REPO,
    DEFAULT_OLLAMA_MODEL,
    OUTPUT_DIR,
    PROJECT_DIR,
)

EXPLORER_URI = "ui://sand3d-rag/explorer.html"
EXPLORER_HTML_PATH = PROJECT_DIR / "ui" / "explorer.html"

AUX_PANE_META = {
    "com.google.antigravity/ui": {
        "entrypoints": [{"type": "aux_pane", "title": "Sand3D Explorer"}]
    }
}

ALL_ENTRYPOINTS_META = {
    "com.google.antigravity/ui": {
        "entrypoints": [
            {"type": "aux_pane", "title": "Sand3D Explorer"},
            {"type": "global", "title": "Sand3D Explorer"},
        ]
    }
}

apps = Apps()
apps.add_html_resource(
    EXPLORER_URI,
    EXPLORER_HTML_PATH.read_text(encoding="utf-8"),
    name="Sand3D RAG Explorer",
    title="Sand3D Nature Hybrid RAG Explorer",
    description="Interactive visual & code explorer for Sandflow/sand3d-nature assets.",
    csp=ResourceCsp(
        resource_domains=[
            "https://fonts.googleapis.com",
            "https://fonts.gstatic.com",
        ]
    ),
    prefers_border=True,
)

_LOCK = threading.Lock()
_RETRIEVER = None
_STORE_ONLY = None
_DATASET = None


def _get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        with contextlib.redirect_stdout(sys.stderr):
            from retriever import Sand3DRetriever

            _RETRIEVER = Sand3DRetriever()
    return _RETRIEVER


def _get_store():
    """Lock-free SQLite+mmap store for '{}' entrypoint startup without loading ML weights."""
    global _STORE_ONLY
    if _RETRIEVER is not None:
        return _RETRIEVER.store
    if _STORE_ONLY is None:
        from store import AssetStore

        _STORE_ONLY = AssetStore(db_path=ASSETS_DB_PATH)
    return _STORE_ONLY


def _get_dataset():
    global _DATASET
    if _DATASET is None:
        with contextlib.redirect_stdout(sys.stderr):
            from datasets import DownloadConfig, concatenate_datasets, load_dataset

            ds_dict = load_dataset(
                DATASET_REPO,
                download_config=DownloadConfig(local_files_only=True),
            )
            _DATASET = concatenate_datasets([ds_dict["train"], ds_dict["validation"]])
    return _DATASET


def _render_thumbnail_data_uri(point_id: int, size: int = 192, quality: int = 75) -> str | None:
    try:
        ds = _get_dataset()
        img = ds[int(point_id)]["render"].convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _format_payload_item(point_id: int, payload: dict, include_program: bool) -> dict:
    item = {
        "id": point_id,
        "prompt": payload.get("prompt"),
        "domain": payload.get("domain"),
        "family": payload.get("family"),
        "subject": payload.get("subject"),
        "style": payload.get("style"),
        "semantic_score": payload.get("semantic_score"),
        "triangles": payload.get("triangles"),
        "entry": payload.get("entry"),
        "split": payload.get("split"),
    }
    if include_program:
        item["program"] = payload.get("program")
    return item


def _build_ui_items(raw_items: list[dict], include_thumbs: bool) -> list[dict]:
    """Build UI items keeping total structuredContent safely under Antigravity's 1 MiB step cap."""
    ui_items = []
    total_chars = 0
    for item in raw_items:
        ui_item = dict(item)
        if include_thumbs:
            ui_item["thumbnail_data_uri"] = _render_thumbnail_data_uri(item["id"])
        prog = ui_item.get("program") or ""
        if total_chars + len(prog) > 350_000:
            # Large scripts are fetched lazily via app_get_asset_details when clicked.
            ui_item["program"] = None
        else:
            total_chars += len(prog)
        ui_items.append(ui_item)
    return ui_items


def _unload_retriever() -> None:
    global _RETRIEVER, _STORE_ONLY
    if _RETRIEVER is not None:
        try:
            _RETRIEVER.close()
        finally:
            _RETRIEVER = None
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if _STORE_ONLY is not None:
        try:
            _STORE_ONLY.close()
        finally:
            _STORE_ONLY = None


atexit.register(_unload_retriever)


@apps.tool(
    resource_uri=EXPLORER_URI,
    visibility=["model", "app"],
    meta=ALL_ENTRYPOINTS_META,
    title="Sand3D Explorer",
    description=(
        "Open the interactive Sand3D Nature Explorer UI (or search when query/image_path is given). "
        "Supports empty arguments '{}' for sidebar and side-pane '+' launches."
    ),
)
def open_sand3d_explorer(
    query: str | None = None,
    image_path: str | None = None,
    top_k: int = 8,
    min_score: float = 0.9,
    domain: str | None = None,
    ctx: Context | None = None,
) -> CallToolResult:
    """Entrypoint tool for opening the Sand3D Explorer in side pane or sidebar."""
    if query or image_path:
        return search_sand3d_assets(
            query=query,
            image_path=image_path,
            top_k=top_k,
            min_score=min_score,
            domain=domain,
            include_program=True,
            ctx=ctx,
        )

    include_thumbs = ctx is not None and client_supports_apps(ctx)
    with _LOCK:
        with contextlib.redirect_stdout(sys.stderr):
            store = _get_store()
            records = store.scroll(
                min_score=min_score,
                domain=domain,
                limit=max(1, min(top_k, 12)),
            )

    raw_items = [
        _format_payload_item(pid, payload, include_program=True)
        for pid, payload in records
    ]
    ui_items = _build_ui_items(raw_items, include_thumbs=include_thumbs)
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(raw_items, ensure_ascii=False))],
        structured_content={"query": "", "results": ui_items},
    )


@apps.tool(
    resource_uri=EXPLORER_URI,
    visibility=["model", "app"],
    meta=AUX_PANE_META,
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
        return open_sand3d_explorer(top_k=top_k, min_score=min_score, domain=domain, ctx=ctx)

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

    raw_items = []
    for h in hits:
        item = _format_payload_item(h.point_id, h.payload, include_program=include_program)
        item["rrf_score"] = round(h.rrf_score, 5)
        item["ranks"] = h.ranks
        raw_items.append(item)

    ui_items = _build_ui_items(raw_items, include_thumbs=include_thumbs)
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(raw_items, ensure_ascii=False))],
        structured_content={"query": query or "", "results": ui_items},
    )


@apps.tool(
    resource_uri=EXPLORER_URI,
    visibility=["model", "app"],
    meta=AUX_PANE_META,
    title="Save & Validate Sand3D Program",
    description=(
        "Validate Python syntax of a sand3d script written by the host AI agent, "
        "save it to outputs/, and display it in the Sand3D Explorer UI."
    ),
)
def save_sand3d_program(
    code: str,
    output_filename: str = "agent_generated.py",
) -> CallToolResult:
    """Validate and save a sand3d Python script synthesized by the host AI agent."""
    safe_name = Path(output_filename).name or "agent_generated.py"
    if not safe_name.endswith(".py"):
        safe_name += ".py"

    clean_code = code.strip()
    syntax_valid = True
    syntax_error = None
    try:
        ast.parse(clean_code)
    except SyntaxError as exc:
        syntax_valid = False
        syntax_error = str(exc)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / safe_name
    out_path.write_text(clean_code + "\n", encoding="utf-8")

    payload = {
        "saved_path": str(out_path),
        "syntax_valid": syntax_valid,
        "syntax_error": syntax_error,
        "code": clean_code,
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
    )


@apps.tool(
    resource_uri=EXPLORER_URI,
    visibility=["app"],
    title="Generate Sand3D Program (Local Ollama Fallback)",
    description="App-only optional fallback to draft a sand3d Python script using local Ollama.",
)
def generate_sand3d_program(
    prompt: str,
    image_path: str | None = None,
    top_k: int = 3,
    model: str | None = None,
    output_filename: str = "mcp_generated.py",
) -> CallToolResult:
    """Retrieve matching sand3d examples and synthesize a new sand3d Python script via local Ollama."""
    chosen_model = model or DEFAULT_OLLAMA_MODEL
    safe_name = Path(output_filename).name or "mcp_generated.py"
    if not safe_name.endswith(".py"):
        safe_name += ".py"

    with _LOCK:
        with contextlib.redirect_stdout(sys.stderr):
            from generate import (
                build_generation_prompt,
                call_ollama,
                extract_python_code,
                get_sand3d_api_summary,
            )

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
                model=chosen_model,
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


mcp = MCPServer(
    "sand3d-rag",
    instructions=(
        "Hybrid Multimodal RAG for Sandflow/sand3d-nature (10,057 procedural 3D Python programs). "
        "When asked to create or remix a 3D asset, retrieve examples with search_sand3d_assets and "
        "signatures with get_sand3d_api_reference, write the sand3d Python code yourself, and validate/save "
        "it with save_sand3d_program so it renders in the Sand3D Explorer pane."
    ),
    extensions=[apps],
)


@mcp.tool(
    meta={"ui": {"visibility": ["app"]}},
    title="App Get Asset Details",
    description="App-only helper to fetch full sand3d Python source and render for a specific asset ID.",
)
def app_get_asset_details(point_id: int) -> CallToolResult:
    """Fetch full program text and thumbnail for the MCP App inspector."""
    with _LOCK:
        with contextlib.redirect_stdout(sys.stderr):
            store = _get_store()
            by_id = store.get_by_ids([int(point_id)])
    if int(point_id) not in by_id:
        raise ValueError(f"Asset ID {point_id} not found.")
    data = _format_payload_item(int(point_id), by_id[int(point_id)], include_program=True)
    data["thumbnail_data_uri"] = _render_thumbnail_data_uri(int(point_id), size=320, quality=85)
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, ensure_ascii=False))],
        structured_content=data,
    )


@mcp.tool()
def get_sand3d_api_reference(max_chars: int = 8000) -> str:
    """Return the condensed sand3d Python API reference for writing valid 3D scripts."""
    with contextlib.redirect_stdout(sys.stderr):
        from generate import get_sand3d_api_summary

        summary = get_sand3d_api_summary()
    return summary[: max(500, min(max_chars, 20000))]


@mcp.tool()
def get_index_stats() -> dict:
    """Return collection statistics from the local asset store."""
    with _LOCK:
        with contextlib.redirect_stdout(sys.stderr):
            store = _get_store()
            total = store.count()
    return {
        "collection": COLLECTION_NAME,
        "points_count": total,
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

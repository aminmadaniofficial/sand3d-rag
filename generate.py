"""Generate a new sand3d Python program using Hybrid RAG + Ollama (or OpenAI-compatible API)."""

import argparse
import base64
import gc
import html
import re
from pathlib import Path

import httpx
import torch

from config import API_CACHE_PATH, OUTPUT_DIR, STORAGE_DIR
from retriever import Sand3DRetriever

SAND3D_API_URL = "https://sand3d.sandflow.ai/api.html"


def get_sand3d_api_summary() -> str:
    """Fetch and cache the compact sand3d API signature reference."""
    if API_CACHE_PATH.exists():
        return API_CACHE_PATH.read_text(encoding="utf-8")

    try:
        resp = httpx.get(SAND3D_API_URL, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", resp.text, flags=re.S)
        text = re.sub(r"<[^>]+>", "\n", text)
        text = html.unescape(text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        summary = "\n".join(lines)[:12000]
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        API_CACHE_PATH.write_text(summary, encoding="utf-8")
        return summary
    except Exception:
        return (
            "Use standard sand3d API calls (`import sand3d as s3d`, `s3d.Frame`, "
            "`s3d.at`, `s3d.snap`, `s3d.group`, `s3d.export`) matching the retrieved examples."
        )


def build_generation_prompt(
    user_prompt: str, hits: list, api_summary: str
) -> tuple[str, str]:
    system_prompt = (
        "You are an expert 3D procedural programmer using the `sand3d` Python library "
        "(built over Blender `bpy` and `numpy`).\n"
        "Write a single complete, deterministic Python file using `import sand3d as s3d` "
        "with a clean entry function `build_asset(seed=0)` that returns the root sand3d object.\n"
        "Follow the exact conventions, materials, frame placement (`s3d.at`, `s3d.snap`, `s3d.Frame`), "
        "and proportions shown in the retrieved `sand3d` reference programs.\n"
        "Output ONLY valid Python code inside one ```python ... ``` block."
    )

    examples_block = []
    for i, hit in enumerate(hits, start=1):
        p = hit.payload
        prog = (p.get("program") or "")[:9000]
        examples_block.append(
            f"### Reference Example {i}\n"
            f"Prompt: {p.get('prompt')}\n"
            f"Taxonomy: {p.get('domain')} / {p.get('family')} / {p.get('subject')}\n"
            f"Entry function: {p.get('entry')}\n"
            f"```python\n{prog}\n```"
        )

    user_message = (
        f"## sand3d API Reference (condensed)\n{api_summary[:6000]}\n\n"
        + "\n\n".join(examples_block)
        + f"\n\n## User Request\n{user_prompt}\n\n"
        "Write a new complete `sand3d` Python program for the User Request above."
    )
    return system_prompt, user_message


def call_ollama(
    model: str,
    system_prompt: str,
    user_message: str,
    image_path: str | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
) -> str:
    msg: dict = {"role": "user", "content": user_message}
    if image_path:
        raw = Path(image_path).read_bytes()
        msg["images"] = [base64.b64encode(raw).decode("ascii")]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            msg,
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": 16384},
    }
    with httpx.Client(trust_env=False, timeout=300.0) as client:
        resp = client.post(f"{ollama_url}/api/chat", json=payload)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def extract_python_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-powered Sand3D code generator")
    parser.add_argument("--text", type=str, required=True, help="Persian or English prompt")
    parser.add_argument("--image", type=str, default=None, help="Optional reference image")
    parser.add_argument("--top-k", type=int, default=3, help="Number of RAG examples")
    parser.add_argument("--model", type=str, default="gemma4:12b", help="Ollama model name")
    parser.add_argument("--out", type=str, default="generated_asset.py", help="Output filename")
    args = parser.parse_args()

    retriever = Sand3DRetriever()
    try:
        hits = retriever.search(
            query_text=args.text, image_path=args.image, top_k=args.top_k, min_score=0.9
        )
    finally:
        retriever.close()
        del retriever
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Retrieved examples:")
    for i, h in enumerate(hits, start=1):
        print(f"  {i}. [{h.payload['family']}] {h.payload['prompt'][:90]}...")

    api_summary = get_sand3d_api_summary()
    sys_prompt, usr_prompt = build_generation_prompt(args.text, hits, api_summary)

    print(f"Generating sand3d code via Ollama ({args.model})...")
    raw_output = call_ollama(
        model=args.model,
        system_prompt=sys_prompt,
        user_message=usr_prompt,
        image_path=args.image,
    )
    code = extract_python_code(raw_output)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / args.out
    out_path.write_text(code + "\n", encoding="utf-8")
    print(f"Saved generated program to: {out_path}")


if __name__ == "__main__":
    main()

"""Build hybrid multimodal index (BGE-M3 + SigLIP2 + BM25) for Sandflow/sand3d-nature."""

import argparse
import gc
import pickle
import shutil
import time
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from config import (
    BM25_PATH,
    COLLECTION_NAME,
    DATASET_REPO,
    QDRANT_PATH,
    STORAGE_DIR,
    TEXT_DIM,
    TEXT_MODEL_ID,
    VISION_DIM,
    VISION_MODEL_ID,
    build_search_document,
    extract_feature_tensor,
    tokenize_bm25,
)


def encode_texts(texts: list[str], device: str, batch_size: int) -> list[list[float]]:
    """Encode all search documents with BGE-M3 on GPU/CPU."""
    print(f"[1/4] Loading text encoder: {TEXT_MODEL_ID} on {device}...")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = SentenceTransformer(
        TEXT_MODEL_ID,
        device=device,
        model_kwargs={"torch_dtype": dtype},
    )
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    result = embeddings.astype("float32").tolist()
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def encode_images(dataset, device: str, batch_size: int) -> list[list[float]]:
    """Encode all 512px render images with SigLIP2-SO400M-384 without buffering all PILs in RAM."""
    print(f"[2/4] Loading vision encoder: {VISION_MODEL_ID} on {device}...")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
    model = AutoModel.from_pretrained(VISION_MODEL_ID, dtype=dtype).to(device)
    model.eval()

    total = len(dataset)
    vectors: list[list[float]] = []
    with torch.inference_mode():
        for start in tqdm(range(0, total, batch_size), desc="Encoding renders"):
            end = min(start + batch_size, total)
            batch_imgs = [img.convert("RGB") for img in dataset[start:end]["render"]]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            if "pixel_values" in inputs and dtype == torch.float16:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            feats = extract_feature_tensor(model.get_image_features(**inputs))
            feats = F.normalize(feats.float(), dim=-1)
            vectors.extend(feats.cpu().tolist())

    del model, processor
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Sand3D Nature into Qdrant + BM25")
    parser.add_argument("--text-batch", type=int, default=64)
    parser.add_argument("--vision-batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for quick tests")
    args = parser.parse_args()

    t0 = time.time()
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading dataset {DATASET_REPO}...")
    ds_dict = load_dataset(DATASET_REPO)
    dataset = concatenate_datasets([ds_dict["train"], ds_dict["validation"]])
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    total = len(dataset)
    print(f"Loaded {total} assets.")

    docs = [build_search_document(row) for row in dataset]
    text_vectors = encode_texts(docs, device=device, batch_size=args.text_batch)

    vision_vectors = encode_images(dataset, device=device, batch_size=args.vision_batch)

    print("[3/4] Building BM25 lexical index...")
    tokenized_corpus = [tokenize_bm25(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "count": total}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[4/4] Writing vectors and payloads to local Qdrant at {QDRANT_PATH}...")
    if QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
    client = QdrantClient(path=str(QDRANT_PATH))
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text": VectorParams(size=TEXT_DIM, distance=Distance.COSINE),
            "vision": VectorParams(size=VISION_DIM, distance=Distance.COSINE),
        },
    )

    upsert_batch = 256
    for start in tqdm(range(0, total, upsert_batch), desc="Upserting Qdrant"):
        end = min(start + upsert_batch, total)
        points = []
        for idx in range(start, end):
            row = dataset[idx]
            payload = {
                "id": row.get("id"),
                "asset_id": row.get("asset_id"),
                "split": row.get("split"),
                "prompt": row.get("prompt"),
                "domain": row.get("domain"),
                "family": row.get("family"),
                "subject": row.get("subject"),
                "style": row.get("style"),
                "semantic_score": float(row.get("semantic_score") or 0.0),
                "triangles": int(row.get("triangles") or 0),
                "entry": row.get("entry"),
                "program": row.get("program"),
                "usda_path": row.get("usda_path"),
                "cycles_path": row.get("cycles_path"),
            }
            points.append(
                PointStruct(
                    id=idx,
                    vector={"text": text_vectors[idx], "vision": vision_vectors[idx]},
                    payload=payload,
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    client.close()
    elapsed = time.time() - t0
    print(f"Done! Indexed {total} assets in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()

"""Hybrid multimodal retriever (BGE-M3 + SigLIP2 + BM25 + RRF) for Sand3D Nature."""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

from config import (
    BM25_PATH,
    COLLECTION_NAME,
    QDRANT_PATH,
    TEXT_MODEL_ID,
    VISION_MODEL_ID,
    tokenize_bm25,
)


def _extract_feature_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported feature output type: {type(output)}")


@dataclass
class RetrievalHit:
    point_id: int
    rrf_score: float
    ranks: dict[str, int]
    payload: dict


class Sand3DRetriever:
    """Query Sand3D Nature by Persian/English text, image, or both."""

    def __init__(self, device: str | None = None, load_vision: bool = True) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        self.client = QdrantClient(path=str(QDRANT_PATH))
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]

        self.text_model = SentenceTransformer(
            TEXT_MODEL_ID,
            device=self.device,
            model_kwargs={"torch_dtype": self.dtype},
        )

        self.vision_processor = None
        self.vision_model = None
        if load_vision:
            self.vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
            self.vision_model = AutoModel.from_pretrained(
                VISION_MODEL_ID, torch_dtype=self.dtype
            ).to(self.device)
            self.vision_model.eval()

    def close(self) -> None:
        self.client.close()

    def _encode_text_bge(self, text: str) -> list[float]:
        vec = self.text_model.encode(
            text, normalize_embeddings=True, convert_to_numpy=True
        )
        return vec.astype("float32").tolist()

    def _encode_text_siglip(self, text: str) -> list[float]:
        inputs = self.vision_processor(
            text=[text],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            feats = _extract_feature_tensor(self.vision_model.get_text_features(**inputs))
            feats = F.normalize(feats.float(), dim=-1)
        return feats[0].cpu().tolist()

    def _encode_image_siglip(self, image_path: str | Path) -> list[float]:
        img = Image.open(image_path).convert("RGB")
        inputs = self.vision_processor(images=img, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs and self.dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            feats = _extract_feature_tensor(self.vision_model.get_image_features(**inputs))
            feats = F.normalize(feats.float(), dim=-1)
        return feats[0].cpu().tolist()

    @staticmethod
    def _build_qdrant_filter(
        min_score: float, split: str | None, domain: str | None
    ) -> Filter | None:
        conditions = []
        if min_score > 0:
            conditions.append(
                FieldCondition(key="semantic_score", range=Range(gte=min_score))
            )
        if split:
            conditions.append(FieldCondition(key="split", match=MatchValue(value=split)))
        if domain:
            conditions.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _passes_payload_filter(
        payload: dict, min_score: float, split: str | None, domain: str | None
    ) -> bool:
        if payload.get("semantic_score", 0.0) < min_score:
            return False
        if split and payload.get("split") != split:
            return False
        if domain and payload.get("domain") != domain:
            return False
        return True

    def search(
        self,
        query_text: str | None = None,
        image_path: str | Path | None = None,
        top_k: int = 5,
        candidate_k: int = 40,
        min_score: float = 0.9,
        split: str | None = None,
        domain: str | None = None,
        text_weight: float = 1.0,
        vision_text_weight: float = 0.7,
        image_weight: float = 1.2,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> list[RetrievalHit]:
        if not query_text and not image_path:
            raise ValueError("Provide at least one of query_text or image_path.")

        q_filter = self._build_qdrant_filter(min_score=min_score, split=split, domain=domain)
        ranked_lists: list[tuple[str, float, list[tuple[int, dict | None]]]] = []

        if query_text:
            text_vec = self._encode_text_bge(query_text)
            text_res = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=text_vec,
                using="text",
                query_filter=q_filter,
                limit=candidate_k,
                with_payload=True,
            ).points
            ranked_lists.append(
                ("bge_text", text_weight, [(p.id, p.payload) for p in text_res])
            )

            if self.vision_model is not None and vision_text_weight > 0:
                vis_text_vec = self._encode_text_siglip(query_text)
                vis_text_res = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vis_text_vec,
                    using="vision",
                    query_filter=q_filter,
                    limit=candidate_k,
                    with_payload=True,
                ).points
                ranked_lists.append(
                    ("siglip_text", vision_text_weight, [(p.id, p.payload) for p in vis_text_res])
                )

            tokens = tokenize_bm25(query_text)
            if tokens and bm25_weight > 0:
                scores = self.bm25.get_scores(tokens)
                top_indices = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[: candidate_k * 2]
                bm25_ids = [idx for idx in top_indices if scores[idx] > 0]
                if bm25_ids:
                    records = self.client.retrieve(
                        collection_name=COLLECTION_NAME, ids=bm25_ids, with_payload=True
                    )
                    by_id = {r.id: r.payload for r in records}
                    bm25_filtered = [
                        (idx, by_id[idx])
                        for idx in bm25_ids
                        if idx in by_id
                        and self._passes_payload_filter(by_id[idx], min_score, split, domain)
                    ][:candidate_k]
                    ranked_lists.append(("bm25", bm25_weight, bm25_filtered))

        if image_path:
            if self.vision_model is None:
                raise RuntimeError("Vision model was not loaded (load_vision=False).")
            img_vec = self._encode_image_siglip(image_path)
            img_res = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=img_vec,
                using="vision",
                query_filter=q_filter,
                limit=candidate_k,
                with_payload=True,
            ).points
            ranked_lists.append(
                ("siglip_image", image_weight, [(p.id, p.payload) for p in img_res])
            )

        fused_scores: dict[int, float] = {}
        payloads: dict[int, dict] = {}
        ranks_by_id: dict[int, dict[str, int]] = {}

        for source_name, weight, items in ranked_lists:
            for rank_idx, (pid, payload) in enumerate(items, start=1):
                fused_scores[pid] = fused_scores.get(pid, 0.0) + weight / (rrf_k + rank_idx)
                if payload is not None:
                    payloads[pid] = payload
                ranks_by_id.setdefault(pid, {})[source_name] = rank_idx

        top_ids = sorted(fused_scores, key=lambda pid: fused_scores[pid], reverse=True)[:top_k]
        return [
            RetrievalHit(
                point_id=pid,
                rrf_score=fused_scores[pid],
                ranks=ranks_by_id[pid],
                payload=payloads[pid],
            )
            for pid in top_ids
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Sand3D Nature Hybrid RAG")
    parser.add_argument("--text", type=str, default=None, help="Persian or English query")
    parser.add_argument("--image", type=str, default=None, help="Path to reference image")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    retriever = Sand3DRetriever()
    try:
        hits = retriever.search(
            query_text=args.text,
            image_path=args.image,
            top_k=args.top_k,
            min_score=args.min_score,
            split=args.split,
        )
    finally:
        retriever.close()

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": h.point_id,
                        "rrf_score": round(h.rrf_score, 5),
                        "ranks": h.ranks,
                        "prompt": h.payload["prompt"],
                        "domain": h.payload["domain"],
                        "family": h.payload["family"],
                        "subject": h.payload["subject"],
                        "entry": h.payload["entry"],
                        "triangles": h.payload["triangles"],
                        "semantic_score": h.payload["semantic_score"],
                    }
                    for h in hits
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    for idx, hit in enumerate(hits, start=1):
        p = hit.payload
        print(f"\n[{idx}] RRF={hit.rrf_score:.4f} ranks={hit.ranks}")
        print(f"    domain={p['domain']} | family={p['family']} | subject={p['subject']}")
        print(f"    score={p['semantic_score']:.2f} | triangles={p['triangles']} | entry={p['entry']}")
        print(f"    prompt: {p['prompt']}")


if __name__ == "__main__":
    main()

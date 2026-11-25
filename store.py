"""Lock-free SQLite + memory-mapped NumPy vector store for concurrent multi-process RAG."""

import pickle
import sqlite3
from pathlib import Path

import numpy as np

from config import ASSETS_DB_PATH, QDRANT_PATH, TEXT_VECS_PATH, VISION_VECS_PATH


class AssetStore:
    """Read-only concurrent store backed by SQLite payloads and mmap'd float32 vectors."""

    def __init__(
        self,
        db_path: Path = ASSETS_DB_PATH,
        text_vecs_path: Path = TEXT_VECS_PATH,
        vision_vecs_path: Path = VISION_VECS_PATH,
    ) -> None:
        if not db_path.exists():
            raise FileNotFoundError(
                f"Asset store not found at {db_path}. Run build_index.py (or store.py --migrate)."
            )
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.text_vecs = np.load(text_vecs_path, mmap_mode="r")
        self.vision_vecs = np.load(vision_vecs_path, mmap_mode="r")

    def close(self) -> None:
        self.conn.close()

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM assets").fetchone()
        return int(row["n"])

    @staticmethod
    def _where_clause(
        min_score: float = 0.0, split: str | None = None, domain: str | None = None
    ) -> tuple[str, list]:
        clauses = []
        params: list = []
        if min_score > 0:
            clauses.append("semantic_score >= ?")
            params.append(float(min_score))
        if split:
            clauses.append("split = ?")
            params.append(split)
        if domain:
            clauses.append("domain = ?")
            params.append(domain)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["raw_id"],
            "asset_id": row["asset_id"],
            "split": row["split"],
            "prompt": row["prompt"],
            "domain": row["domain"],
            "family": row["family"],
            "subject": row["subject"],
            "style": row["style"],
            "semantic_score": row["semantic_score"],
            "triangles": row["triangles"],
            "entry": row["entry"],
            "program": row["program"],
            "usda_path": row["usda_path"],
            "cycles_path": row["cycles_path"],
        }

    def get_by_ids(self, point_ids: list[int]) -> dict[int, dict]:
        if not point_ids:
            return {}
        placeholders = ",".join("?" for _ in point_ids)
        rows = self.conn.execute(
            f"SELECT * FROM assets WHERE point_id IN ({placeholders})",
            [int(pid) for pid in point_ids],
        ).fetchall()
        return {int(r["point_id"]): self._row_to_dict(r) for r in rows}

    def scroll(
        self, min_score: float = 0.9, domain: str | None = None, limit: int = 8
    ) -> list[tuple[int, dict]]:
        where, params = self._where_clause(min_score=min_score, domain=domain)
        rows = self.conn.execute(
            f"SELECT * FROM assets {where} ORDER BY semantic_score DESC, point_id ASC LIMIT ?",
            [*params, int(limit)],
        ).fetchall()
        return [(int(r["point_id"]), self._row_to_dict(r)) for r in rows]

    def search_vector(
        self,
        vector_name: str,
        query_vec: list[float],
        limit: int = 40,
        min_score: float = 0.9,
        split: str | None = None,
        domain: str | None = None,
    ) -> list[tuple[int, dict]]:
        vecs = self.text_vecs if vector_name == "text" else self.vision_vecs
        where, params = self._where_clause(min_score=min_score, split=split, domain=domain)
        candidate_ids = [
            r[0]
            for r in self.conn.execute(
                f"SELECT point_id FROM assets {where}", params
            ).fetchall()
        ]
        if not candidate_ids:
            return []

        idx_arr = np.asarray(candidate_ids, dtype=np.int64)
        q = np.asarray(query_vec, dtype=np.float32)
        scores = vecs[idx_arr] @ q
        k = min(int(limit), len(idx_arr))
        if k < len(scores):
            top_pos = np.argpartition(scores, -k)[-k:]
            top_pos = top_pos[np.argsort(scores[top_pos])[::-1]]
        else:
            top_pos = np.argsort(scores)[::-1]

        top_ids = [int(idx_arr[p]) for p in top_pos]
        by_id = self.get_by_ids(top_ids)
        return [(pid, by_id[pid]) for pid in top_ids if pid in by_id]

    @classmethod
    def write_all(
        cls,
        payloads: list[dict],
        text_vectors: np.ndarray,
        vision_vectors: np.ndarray,
        db_path: Path = ASSETS_DB_PATH,
        text_vecs_path: Path = TEXT_VECS_PATH,
        vision_vecs_path: Path = VISION_VECS_PATH,
    ) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(text_vecs_path, np.asarray(text_vectors, dtype=np.float32))
        np.save(vision_vecs_path, np.asarray(vision_vectors, dtype=np.float32))

        tmp_db = db_path.with_suffix(".sqlite.tmp")
        if tmp_db.exists():
            tmp_db.unlink()
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            """
            CREATE TABLE assets (
                point_id INTEGER PRIMARY KEY,
                raw_id TEXT,
                asset_id TEXT,
                split TEXT,
                prompt TEXT,
                domain TEXT,
                family TEXT,
                subject TEXT,
                style TEXT,
                semantic_score REAL,
                triangles INTEGER,
                entry TEXT,
                program TEXT,
                usda_path TEXT,
                cycles_path TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO assets (
                point_id, raw_id, asset_id, split, prompt, domain, family,
                subject, style, semantic_score, triangles, entry, program,
                usda_path, cycles_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    idx,
                    p.get("id"),
                    p.get("asset_id"),
                    p.get("split"),
                    p.get("prompt"),
                    p.get("domain"),
                    p.get("family"),
                    p.get("subject"),
                    p.get("style"),
                    float(p.get("semantic_score") or 0.0),
                    int(p.get("triangles") or 0),
                    p.get("entry"),
                    p.get("program"),
                    p.get("usda_path"),
                    p.get("cycles_path"),
                )
                for idx, p in enumerate(payloads)
            ],
        )
        conn.execute("CREATE INDEX idx_assets_score ON assets (semantic_score)")
        conn.execute("CREATE INDEX idx_assets_domain ON assets (domain)")
        conn.execute("CREATE INDEX idx_assets_split ON assets (split)")
        conn.commit()
        conn.close()
        tmp_db.replace(db_path)


def migrate_from_qdrant() -> int:
    """One-time lock-free export from local Qdrant storage.sqlite into AssetStore."""
    qdrant_sqlite = QDRANT_PATH / "collection" / "sand3d_nature" / "storage.sqlite"
    if not qdrant_sqlite.exists():
        raise FileNotFoundError(f"Qdrant sqlite not found at {qdrant_sqlite}")

    conn = sqlite3.connect(f"file:{qdrant_sqlite}?mode=ro", uri=True)
    rows = conn.execute("SELECT point FROM points").fetchall()
    conn.close()

    points = [pickle.loads(blob) for (blob,) in rows]
    points.sort(key=lambda pt: int(pt.id))

    payloads = [pt.payload for pt in points]
    text_vecs = np.asarray([pt.vector["text"] for pt in points], dtype=np.float32)
    vision_vecs = np.asarray([pt.vector["vision"] for pt in points], dtype=np.float32)
    AssetStore.write_all(payloads, text_vecs, vision_vecs)
    return len(points)


if __name__ == "__main__":
    n = migrate_from_qdrant()
    print(f"Migrated {n} assets to lock-free AssetStore ({ASSETS_DB_PATH}).")

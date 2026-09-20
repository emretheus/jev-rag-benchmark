from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25


@dataclass
class Candidate:
    doc_id: str
    bm25_rank: int = 0
    dense_rank: int = 0
    hybrid_rank: int = 0
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0


def corpus_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title}. {text}".strip() if title else text


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(self, dataset: str, corpus: list[dict], cfg: dict, embed_client) -> None:
        self.dataset = dataset
        self.corpus = corpus
        self.cfg = cfg
        self.embed_client = embed_client
        self.doc_ids = [row["doc_id"] for row in corpus]
        self.texts = [corpus_text(row) for row in corpus]
        self.text_map = dict(zip(self.doc_ids, self.texts, strict=False))
        self.index_dir = Path(cfg["paths"]["data_dir"]) / "index" / dataset
        self.bm25: BM25 | None = None
        self.embeddings: np.ndarray | None = None

    def prepare(self) -> None:
        retrieval = self.cfg["retrieval"]
        self.bm25 = BM25(
            self.doc_ids,
            self.texts,
            k1=float(retrieval["bm25"]["k1"]),
            b=float(retrieval["bm25"]["b"]),
        )
        self.embeddings = self._load_or_build_embeddings()

    def _embedding_slug(self) -> str:
        name = self.cfg["models"]["embedding"]["name"]
        return re.sub(r"[^A-Za-z0-9]+", "-", str(name)).strip("-").lower()

    def _corpus_hash(self) -> str:
        digest = hashlib.sha256()
        for doc_id, text in zip(self.doc_ids, self.texts, strict=False):
            digest.update(doc_id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _load_or_build_embeddings(self) -> np.ndarray:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        slug = self._embedding_slug()
        vectors_path = self.index_dir / f"corpus-emb-{slug}.npy"
        meta_path = self.index_dir / f"corpus-emb-{slug}.meta.json"
        corpus_hash = self._corpus_hash()
        if vectors_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("corpus_hash") == corpus_hash:
                return np.load(vectors_path)
        if self.embed_client is None:
            raise RuntimeError("an embedding client is required to build the dense index")
        vectors = np.asarray(self.embed_client.embed(self.texts, kind="passage"), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        np.save(vectors_path, vectors)
        meta_path.write_text(
            json.dumps(
                {
                    "corpus_hash": corpus_hash,
                    "model": self.cfg["models"]["embedding"]["name"],
                    "count": len(self.doc_ids),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return vectors

    def search(self, question: str) -> list[Candidate]:
        if self.bm25 is None or self.embeddings is None:
            raise RuntimeError("call prepare() before search()")
        retrieval = self.cfg["retrieval"]

        bm25_hits = self.bm25.search(question, int(retrieval["bm25_depth"]))
        bm25_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_hits, start=1)}
        bm25_score = dict(bm25_hits)

        query_vector = np.asarray(
            self.embed_client.embed([question], kind="query")[0], dtype=np.float32
        )
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        similarities = self.embeddings @ query_vector
        dense_order = np.argsort(-similarities)[: int(retrieval["dense_depth"])]
        dense_rank = {self.doc_ids[index]: rank for rank, index in enumerate(dense_order, start=1)}
        dense_score = {self.doc_ids[index]: float(similarities[index]) for index in dense_order}

        fused = reciprocal_rank_fusion(
            [
                [doc_id for doc_id, _ in bm25_hits],
                [self.doc_ids[index] for index in dense_order],
            ],
            k=int(retrieval["rrf_k"]),
        )
        fused.sort(
            key=lambda item: (
                -item[1],
                bm25_rank.get(item[0], 10**9),
                dense_rank.get(item[0], 10**9),
                item[0],
            )
        )

        candidates = []
        for rank, (doc_id, rrf_score) in enumerate(
            fused[: int(retrieval["candidate_depth"])], start=1
        ):
            candidates.append(
                Candidate(
                    doc_id=doc_id,
                    bm25_rank=bm25_rank.get(doc_id, 0),
                    dense_rank=dense_rank.get(doc_id, 0),
                    hybrid_rank=rank,
                    bm25_score=bm25_score.get(doc_id, 0.0),
                    dense_score=dense_score.get(doc_id, 0.0),
                    rrf_score=rrf_score,
                )
            )
        return candidates

"""Candidate-depth audit: is the gold passage retrievable at each depth?

Runs BM25-only and hybrid (BM25 + dense, RRF) retrieval for every query and
reports the share of queries whose gold passage is inside the top-k, mirroring
the "candidate retrieval ceiling" table of prior work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import data as data_module
from .bm25 import BM25
from .retrieval import Retriever, corpus_text, reciprocal_rank_fusion

DEFAULT_KS = [5, 20, 50, 100]


def _recall_at(ranked: list[str], gold: set[str], k: int | None) -> bool:
    if k is None:
        return bool(gold & set(ranked))
    return bool(gold & set(ranked[:k]))


def retrieval_audit(
    cfg: dict,
    dataset: str,
    ks: list[int] | None = None,
    limit: int | None = None,
) -> dict:
    ks = list(ks or DEFAULT_KS)
    data_root = Path(cfg["paths"]["data_dir"])
    corpus, queries = data_module.load_processed(dataset, data_root)
    if limit is not None:
        queries = queries[:limit]

    retriever = Retriever(dataset, corpus, cfg, None)
    retrieval = cfg["retrieval"]
    retriever.bm25 = BM25(
        retriever.doc_ids,
        retriever.texts,
        k1=float(retrieval["bm25"]["k1"]),
        b=float(retrieval["bm25"]["b"]),
    )

    from .run import build_clients

    clients = build_clients(cfg, "real", [])
    embed_client = clients["embeddings"]
    retriever.embed_client = embed_client
    retriever.embeddings = retriever._load_or_build_embeddings()

    max_k = max(ks)
    total = len(queries)
    print(f"embedding {total} queries ...", flush=True)
    query_vectors = np.asarray(
        embed_client.embed([query["question"] for query in queries], kind="query"),
        dtype=np.float32,
    )
    norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / np.clip(norms, 1e-12, None)
    bm25_hits: dict[int | None, int] = {k: 0 for k in [*ks, None]}
    hybrid_hits: dict[int | None, int] = {k: 0 for k in [*ks, None]}

    for position, query in enumerate(queries, start=1):
        gold = set(query.get("gold_doc_ids") or [])
        if not gold:
            continue

        bm25_ranked = [doc_id for doc_id, _ in retriever.bm25.search(query["question"], max_k)]
        for k in [*ks, None]:
            if _recall_at(bm25_ranked, gold, k):
                bm25_hits[k] += 1

        similarities = retriever.embeddings @ query_vectors[position - 1]
        dense_order = np.argsort(-similarities)[:max_k]
        fused = [
            doc_id
            for doc_id, _ in reciprocal_rank_fusion(
                [bm25_ranked, [retriever.doc_ids[int(index)] for index in dense_order]],
                k=int(retrieval["rrf_k"]),
            )
        ]
        for k in [*ks, None]:
            if _recall_at(fused, gold, k):
                hybrid_hits[k] += 1

        if position % 100 == 0:
            print(f"  audited {position}/{total}", flush=True)

    def rows(hits: dict[int | None, int]) -> list[dict]:
        result = []
        for k in ks:
            result.append({"depth": k, "found": hits[k], "recall": hits[k] / total})
        result.append({"depth": "all", "found": hits[None], "recall": hits[None] / total})
        return result

    return {
        "dataset": dataset,
        "n_queries": total,
        "corpus_size": len(corpus),
        "bm25": rows(bm25_hits),
        "hybrid": rows(hybrid_hits),
    }


def write_audit(audit: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{audit['dataset']}-retrieval-audit.json"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        f"# Retrieval audit: {audit['dataset']}",
        "",
        f"- Queries: {audit['n_queries']}",
        f"- Corpus: {audit['corpus_size']} passages",
        "",
        "| Candidates | BM25 only | Hybrid (BM25 + dense RRF) |",
        "|---|---|---|",
    ]
    for bm25_row, hybrid_row in zip(audit["bm25"], audit["hybrid"], strict=False):
        label = f"top-{bm25_row['depth']}" if bm25_row["depth"] != "all" else "all"
        lines.append(
            f"| {label} | {bm25_row['recall'] * 100:.3f}% "
            f"({bm25_row['found']}/{audit['n_queries']}) "
            f"| {hybrid_row['recall'] * 100:.3f}% "
            f"({hybrid_row['found']}/{audit['n_queries']}) |"
        )
    lines.append("")
    md_path = output_dir / f"{audit['dataset']}-retrieval-audit.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def load_audits(audits_dir: str | Path) -> list[dict]:
    base = Path(audits_dir)
    if not base.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(base.glob("*-retrieval-audit.json"))
    ]


__all__ = ["DEFAULT_KS", "retrieval_audit", "write_audit", "load_audits", "corpus_text"]

from __future__ import annotations

from jev_rag_bench.bm25 import BM25
from jev_rag_bench.retrieval import reciprocal_rank_fusion


def test_bm25_ranks_relevant_document_first():
    doc_ids = ["d1", "d2", "d3"]
    texts = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Jupiter is the largest planet in the Solar System.",
    ]
    index = BM25(doc_ids, texts)
    hits = index.search("What is the capital of France?", top_k=3)
    assert hits[0][0] == "d1"


def test_bm25_returns_empty_for_unknown_terms():
    index = BM25(["d1"], ["Paris is in France."])
    assert index.search("zzzzqqq", top_k=5) == []


def test_reciprocal_rank_fusion_prefers_documents_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)
    assert fused[0][0] == "a"
    scores = dict(fused)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["d"]

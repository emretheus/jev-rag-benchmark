"""Confidence-gated use of OpenJev as a RAG decision layer.

The raw always-on reranking result (see reports) shows OpenJev is not a reliable
list ranker. This module measures the decision-layer mode instead: use OpenJev
probabilities to decide *where* to act, falling back to the hybrid order
elsewhere. Two modes:

- partition: candidates with p >= threshold are reranked first, remaining
  candidates keep their hybrid order (never drops a candidate).
- gate: replace the whole order with OpenJev only when top-1 p >= threshold.

Also measures answerability gating: skip generation when top-1 p is low.
"""

from __future__ import annotations

from statistics import mean

from .metrics import mrr_at_k, ndcg_at_k, paired_bootstrap_ci, recall_at_k

DEFAULT_THRESHOLD = 0.5
ANSWERABILITY_THRESHOLDS = [0.5, 0.7, 0.8, 0.9]


def partition_order(row: dict, threshold: float = DEFAULT_THRESHOLD) -> list[str]:
    probs = row["branches"]["J"]["probs"]
    j_order = row["branches"]["J"]["order"]
    confident = [doc_id for doc_id, p in zip(j_order, probs, strict=False) if p >= threshold]
    confident_set = set(confident)
    rest = [doc_id for doc_id in row["branches"]["A"]["order"] if doc_id not in confident_set]
    return confident + rest


def gate_order(row: dict, threshold: float = DEFAULT_THRESHOLD) -> list[str]:
    probs = row["branches"]["J"]["probs"]
    if probs and probs[0] >= threshold:
        return list(row["branches"]["J"]["order"])
    return list(row["branches"]["A"]["order"])


def evaluate_order_fn(rows: list[dict], order_fn) -> dict:
    ndcg, recall5, recall10, mrr = [], [], [], []
    for row in rows:
        gold = set(row.get("gold_doc_ids") or [])
        order = order_fn(row)
        ndcg.append(ndcg_at_k(order, gold, 10))
        recall5.append(recall_at_k(order, gold, 5))
        recall10.append(recall_at_k(order, gold, 10))
        mrr.append(mrr_at_k(order, gold, 10))
    return {
        "ndcg@10": mean(ndcg) if ndcg else 0.0,
        "recall@5": mean(recall5) if recall5 else 0.0,
        "recall@10": mean(recall10) if recall10 else 0.0,
        "mrr@10": mean(mrr) if mrr else 0.0,
        "n": len(ndcg),
    }


def gating_summary(rows: list[dict], threshold: float = DEFAULT_THRESHOLD, seed: int = 13) -> dict:
    usable = [
        row
        for row in rows
        if row.get("branches", {}).get("J", {}).get("probs")
        and row.get("branches", {}).get("A", {}).get("order")
    ]
    if not usable:
        return {}

    baseline = evaluate_order_fn(usable, lambda row: row["branches"]["A"]["order"])
    always = evaluate_order_fn(usable, lambda row: row["branches"]["J"]["order"])
    partition = evaluate_order_fn(usable, lambda row: partition_order(row, threshold))
    gate = evaluate_order_fn(usable, lambda row: gate_order(row, threshold))

    baseline_ndcg = [
        ndcg_at_k(row["branches"]["A"]["order"], set(row["gold_doc_ids"]), 10) for row in usable
    ]
    partition_ndcg = [
        ndcg_at_k(partition_order(row, threshold), set(row["gold_doc_ids"]), 10) for row in usable
    ]
    diff, low, high = paired_bootstrap_ci(partition_ndcg, baseline_ndcg, seed=seed)

    return {
        "threshold": threshold,
        "n": len(usable),
        "baseline": baseline,
        "always_on": always,
        "partition": partition,
        "gate": gate,
        "partition_vs_baseline_ndcg@10": {"mean_diff": diff, "ci": [low, high]},
    }


def answerability_summary(
    rows: list[dict],
    generation_rows: list[dict],
    thresholds: list[float] | None = None,
) -> list[dict]:
    thresholds = thresholds or ANSWERABILITY_THRESHOLDS
    generation_by_id = {str(row.get("query_id")): row for row in generation_rows}
    usable = [
        row
        for row in rows
        if row.get("branches", {}).get("J", {}).get("probs")
        and str(row.get("query_id")) in generation_by_id
    ]
    if not usable:
        return []
    results = []
    total = len(usable)
    for threshold in thresholds:
        kept = [row for row in usable if row["branches"]["J"]["probs"][0] >= threshold]
        if not kept:
            continue
        generation_kept = [generation_by_id[str(row["query_id"])] for row in kept]
        f1_values = [float(row.get("f1") or 0.0) for row in generation_kept]
        success = [1.0 if value >= 0.5 else 0.0 for value in f1_values]
        results.append(
            {
                "threshold": threshold,
                "coverage": len(kept) / total,
                "calls": len(kept),
                "calls_saved": 1 - len(kept) / total,
                "mean_f1": mean(f1_values),
                "success_rate": mean(success),
            }
        )
    return results

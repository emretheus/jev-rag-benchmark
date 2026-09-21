from __future__ import annotations

from jev_rag_bench.gate import (
    answerability_summary,
    gate_order,
    gating_summary,
    partition_order,
)


def _row(
    query_id: str,
    gold: str,
    a_order: list[str],
    j_order: list[str],
    probs: list[float],
) -> dict:
    return {
        "query_id": query_id,
        "gold_doc_ids": [gold],
        "branches": {
            "A": {"order": a_order, "probs": None},
            "T": {"order": j_order, "probs": probs},
        },
    }


def test_partition_puts_confident_candidates_first_without_dropping():
    row = _row("q1", "a", ["a", "b", "c"], ["c", "b", "a"], [0.9, 0.2, 0.1])
    assert partition_order(row, threshold=0.5) == ["c", "a", "b"]


def test_partition_is_identity_when_nothing_is_confident():
    row = _row("q1", "a", ["a", "b", "c"], ["c", "b", "a"], [0.1, 0.2, 0.3])
    assert partition_order(row, threshold=0.5) == ["a", "b", "c"]


def test_gate_uses_jev_order_only_when_top1_is_confident():
    confident = _row("q1", "a", ["a", "b"], ["b", "a"], [0.9, 0.3])
    unsure = _row("q2", "a", ["a", "b"], ["b", "a"], [0.4, 0.35])
    assert gate_order(confident, threshold=0.5) == ["b", "a"]
    assert gate_order(unsure, threshold=0.5) == ["a", "b"]


def test_gating_summary_improves_over_baseline_when_confident_signals_are_correct():
    rows = [
        _row(f"q{index}", "gold", ["x1", "x2", "gold"], ["gold", "x1", "x2"], [0.95, 0.02, 0.01])
        for index in range(10)
    ]
    summary = gating_summary(rows, threshold=0.5)
    assert summary["n"] == 10
    assert summary["partition"]["ndcg@10"] > summary["baseline"]["ndcg@10"]
    assert summary["partition_vs_baseline_ndcg@10"]["mean_diff"] > 0


def test_gating_summary_handles_rows_without_probabilities():
    rows = [{"query_id": "q1", "gold_doc_ids": ["a"], "branches": {"A": {"order": ["a"]}}}]
    assert gating_summary(rows) == {}


def test_answerability_summary_coverage_and_success():
    rows = [
        _row("q1", "a", ["a"], ["a"], [0.9]),
        _row("q2", "a", ["a"], ["a"], [0.4]),
    ]
    generation = [
        {"query_id": "q1", "f1": 0.8},
        {"query_id": "q2", "f1": 0.1},
    ]
    summary = answerability_summary(rows, generation, thresholds=[0.5, 0.85])
    assert summary[0]["coverage"] == 0.5
    assert summary[0]["calls_saved"] == 0.5
    assert summary[0]["success_rate"] == 1.0
    assert summary[1]["calls"] == 1

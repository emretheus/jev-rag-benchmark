from __future__ import annotations

import math

from jev_rag_bench.metrics import (
    bootstrap_ci,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_ci,
    recall_at_k,
)
from jev_rag_bench.text import exact_match, normalize_answer, strip_citations, token_f1


def test_recall_at_k():
    order = ["a", "b", "c"]
    assert recall_at_k(order, {"a"}, 1) == 1.0
    assert recall_at_k(order, {"b"}, 1) == 0.0
    assert recall_at_k(order, {"b"}, 2) == 1.0
    assert recall_at_k(order, {"z"}, 3) == 0.0


def test_ndcg_at_k_perfect_order():
    assert ndcg_at_k(["a", "b"], {"a"}, 10) == 1.0


def test_ndcg_at_k_second_position():
    expected = 1.0 / math.log2(3)
    assert abs(ndcg_at_k(["x", "a"], {"a"}, 10) - expected) < 1e-9


def test_mrr_at_k():
    assert mrr_at_k(["a", "b"], {"b"}, 10) == 0.5
    assert mrr_at_k(["a", "b"], {"z"}, 10) == 0.0


def test_bootstrap_and_paired():
    point, low, high = bootstrap_ci([1.0, 1.0, 1.0], n=200, seed=1)
    assert point == 1.0 and low == 1.0 and high == 1.0
    diff, low, high = paired_bootstrap_ci([1.0, 0.0], [0.0, 0.0], n=200, seed=1)
    assert diff == 0.5
    assert low <= diff <= high


def test_strip_citations():
    assert normalize_answer(strip_citations("Paris [doc-1]")) == "paris"
    assert strip_citations("no citation here") == "no citation here"


def test_exact_match_and_f1():
    assert exact_match("Paris [d1]", ["Paris"]) == 1.0
    assert exact_match("London", ["Paris"]) == 0.0
    assert abs(token_f1("the capital is Paris", ["Paris"]) - 0.5) < 1e-9
    assert token_f1("", ["Paris"]) == 0.0

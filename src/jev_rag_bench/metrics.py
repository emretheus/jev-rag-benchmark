from __future__ import annotations

import math
import random
from statistics import mean


def recall_at_k(order: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return float(bool(set(order[:k]) & gold))


def mrr_at_k(order: list[str], gold: set[str], k: int) -> float:
    for rank, doc_id in enumerate(order[:k], start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(order: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(order[:k], start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[position]


def bootstrap_ci(
    values: list[float],
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    point = mean(values)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(n)]
    samples.sort()
    low = samples[int(alpha / 2 * n)]
    high = samples[min(n - 1, int((1 - alpha / 2) * n))]
    return (point, low, high)


def paired_bootstrap_ci(
    values_a: list[float],
    values_b: list[float],
    n: int = 5000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float, float]:
    if len(values_a) != len(values_b):
        raise ValueError("paired bootstrap requires equal-length samples")
    diffs = [a - b for a, b in zip(values_a, values_b, strict=False)]
    return bootstrap_ci(diffs, n=n, alpha=alpha, seed=seed)

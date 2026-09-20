from __future__ import annotations


def brier_score(probabilities: list[float], labels: list[int]) -> float:
    if not probabilities:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=False)) / len(
        probabilities
    )


def _bin_indices(probabilities: list[float], bins: int, i: int) -> list[int]:
    low, high = i / bins, (i + 1) / bins
    return [
        index
        for index, p in enumerate(probabilities)
        if (low < p <= high) or (i == 0 and p <= 0.0)
    ]


def expected_calibration_error(
    probabilities: list[float],
    labels: list[int],
    bins: int = 10,
) -> float:
    if not probabilities:
        return 0.0
    total = len(probabilities)
    error = 0.0
    for i in range(bins):
        indices = _bin_indices(probabilities, bins, i)
        if not indices:
            continue
        confidence = sum(probabilities[j] for j in indices) / len(indices)
        accuracy = sum(labels[j] for j in indices) / len(indices)
        error += len(indices) / total * abs(confidence - accuracy)
    return error


def reliability_table(
    probabilities: list[float],
    labels: list[int],
    bins: int = 10,
) -> list[dict]:
    rows = []
    for i in range(bins):
        indices = _bin_indices(probabilities, bins, i)
        if not indices:
            continue
        rows.append(
            {
                "bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
                "count": len(indices),
                "mean_confidence": sum(probabilities[j] for j in indices) / len(indices),
                "accuracy": sum(labels[j] for j in indices) / len(indices),
            }
        )
    return rows


def risk_coverage(
    probabilities: list[float],
    labels: list[int],
    thresholds: list[float] | None = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    total_gold = sum(labels)
    rows = []
    for threshold in thresholds:
        covered = [j for j, p in enumerate(probabilities) if p >= threshold]
        if not covered:
            rows.append(
                {
                    "threshold": threshold,
                    "count": 0,
                    "coverage": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                }
            )
            continue
        correct = sum(labels[j] for j in covered)
        rows.append(
            {
                "threshold": threshold,
                "count": len(covered),
                "coverage": len(covered) / len(probabilities),
                "precision": correct / len(covered),
                "recall": correct / total_gold if total_gold else 0.0,
            }
        )
    return rows

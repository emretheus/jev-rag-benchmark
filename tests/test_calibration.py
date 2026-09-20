from __future__ import annotations

from openjev_rag_bench.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_table,
    risk_coverage,
)


def test_brier_score():
    value = brier_score([0.9, 0.1], [1, 0])
    assert abs(value - 0.01) < 1e-12


def test_expected_calibration_error():
    value = expected_calibration_error([0.9, 0.1], [1, 0], bins=10)
    assert abs(value - 0.1) < 1e-12


def test_ece_is_zero_when_confidence_matches_accuracy():
    value = expected_calibration_error([0.0] * 4 + [1.0] * 4, [0] * 4 + [1] * 4, bins=10)
    assert value == 0.0


def test_reliability_table_covers_all_predictions():
    probs = [0.05, 0.35, 0.65, 0.95, 0.85]
    labels = [0, 0, 1, 1, 0]
    rows = reliability_table(probs, labels, bins=10)
    assert sum(row["count"] for row in rows) == len(probs)


def test_risk_coverage_monotonic_precision():
    probs = [0.95, 0.9, 0.6, 0.4, 0.1]
    labels = [1, 1, 0, 0, 0]
    rows = risk_coverage(probs, labels, thresholds=[0.5, 0.9])
    assert rows[0]["precision"] == 2 / 3
    assert rows[1]["precision"] == 1.0
    assert rows[1]["coverage"] == 0.4

from __future__ import annotations

import json

from jev_rag_bench.config import default_config
from jev_rag_bench.data import write_processed
from jev_rag_bench.fixtures import FixtureGenerator
from jev_rag_bench.generate import load_rows, replay_generator
from jev_rag_bench.report import generate_report
from jev_rag_bench.retrieval import corpus_text
from jev_rag_bench.run import add_branch, run_benchmark

CORPUS = [
    {
        "doc_id": "d1",
        "title": "France",
        "text": "Paris is the capital and most populous city of France.",
    },
    {
        "doc_id": "d2",
        "title": "Mars",
        "text": "Mars is the fourth planet from the Sun and is known as the Red Planet.",
    },
    {
        "doc_id": "d3",
        "title": "Hamlet",
        "text": "Hamlet is a tragedy written by William Shakespeare.",
    },
    {"doc_id": "d4", "title": "Germany", "text": "Berlin is the capital of Germany."},
    {
        "doc_id": "d5",
        "title": "Jupiter",
        "text": "Jupiter is the largest planet in the Solar System.",
    },
    {
        "doc_id": "d6",
        "title": "Macbeth",
        "text": "Macbeth is a tragedy by William Shakespeare.",
    },
]

QUERIES = [
    {
        "query_id": "q1",
        "question": "What is the capital of France?",
        "gold_doc_ids": ["d1"],
        "answers": ["Paris"],
    },
    {
        "query_id": "q2",
        "question": "Which planet is known as the Red Planet?",
        "gold_doc_ids": ["d2"],
        "answers": ["Mars"],
    },
    {
        "query_id": "q3",
        "question": "Who wrote Hamlet?",
        "gold_doc_ids": ["d3"],
        "answers": ["William Shakespeare"],
    },
]


def _prepare_toy_dataset(tmp_path) -> dict:
    write_processed(tmp_path / "processed" / "toy", CORPUS, QUERIES, {"dataset": "toy"})
    cfg = default_config()
    cfg["paths"] = {
        "data_dir": str(tmp_path),
        "results_dir": str(tmp_path / "results"),
        "reports_dir": str(tmp_path / "reports"),
    }
    return cfg


def test_fixture_pipeline_end_to_end(tmp_path):
    cfg = _prepare_toy_dataset(tmp_path)
    results_path = run_benchmark(cfg, "toy", branches=["A", "N", "T"], run_kind="fixture")

    rows = load_rows(results_path)
    assert len(rows) == 3
    assert results_path.with_suffix(".manifest.json").exists()

    for row in rows:
        assert row["run_kind"] == "fixture"
        assert row["branches"]["T"]["probs"] is not None
        assert len(row["branches"]["T"]["order"]) == len(row["candidates"])
        assert set(row["metrics"]) == {"A", "N", "T"}
        assert 0.0 <= row["metrics"]["T"]["ndcg@10"] <= 1.0

    jev_recall = sum(row["metrics"]["T"]["recall@5"] for row in rows) / len(rows)
    assert jev_recall >= 2 / 3

    manifest = json.loads(results_path.with_suffix(".manifest.json").read_text())
    assert manifest["models"]["systemone_requested"]
    assert manifest["queries_in_file"] == 3


def test_fixture_generation_and_report(tmp_path):
    cfg = _prepare_toy_dataset(tmp_path)
    results_path = run_benchmark(cfg, "toy", branches=["T"], run_kind="fixture")

    corpus_texts = {row["doc_id"]: corpus_text(row) for row in CORPUS}
    generation_path = tmp_path / "results" / "toy-generation.jsonl"
    stats = replay_generator(
        results_path,
        generation_path,
        "T",
        FixtureGenerator(),
        corpus_texts,
        run_kind="fixture",
        top_k=5,
    )
    assert stats["written"] == 3
    assert stats["failed"] == 0

    generation_rows = load_rows(generation_path)
    assert len(generation_rows) == 3
    assert any(not row["abstained"] for row in generation_rows)
    assert all(row["f1"] is not None for row in generation_rows)

    report_path = generate_report(
        results_path,
        tmp_path / "reports" / "toy",
        generation_path=generation_path,
    )
    report_text = report_path.read_text()
    assert "Fixture run" in report_text
    assert "Probability calibration" in report_text
    assert "Frozen-context answer generation" in report_text
    assert (tmp_path / "reports" / "toy" / "summary.json").exists()


def test_resume_skips_existing_rows(tmp_path):
    cfg = _prepare_toy_dataset(tmp_path)
    results_path = run_benchmark(cfg, "toy", branches=["A"], run_kind="fixture")
    first = len(load_rows(results_path))
    run_benchmark(cfg, "toy", branches=["A"], run_kind="fixture", resume=True)
    assert len(load_rows(results_path)) == first == 3


def test_add_branch_fills_missing_branch_without_rerunning_others(tmp_path):
    cfg = _prepare_toy_dataset(tmp_path)
    results_path = run_benchmark(cfg, "toy", branches=["A", "N"], run_kind="fixture")

    add_branch(cfg, results_path, "T", run_kind="fixture", concurrency=2)

    rows = load_rows(results_path)
    assert len(rows) == 3
    for row in rows:
        assert "T" in row["branches"]
        assert row["branches"]["T"]["probs"] is not None
        assert "T" in row["metrics"]
        assert set(row["metrics"]) == {"A", "N", "T"}

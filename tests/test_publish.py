from __future__ import annotations

import json

from jev_rag_bench.publish import (
    RESULTS_END,
    RESULTS_START,
    inject_into_readme,
    render_dataset_card,
    render_readme_block,
    render_text_tables,
    stage_published,
    validate_summaries,
    write_space,
)

REAL_SUMMARY = {
    "dataset": "xquad-en",
    "run_kind": "real",
    "n_queries": 1190,
    "results_file": "results/xquad-en-a-j-n-real.jsonl",
    "branches": {
        "A": {
            "label": "no reranker (hybrid order)",
            "n": 1190,
            "ndcg@10": 0.94314,
            "recall@5": 0.97605,
            "mrr@10": 0.93194,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
        },
        "J": {
            "label": "OpenJev batch noul",
            "n": 1190,
            "ndcg@10": 0.98097,
            "recall@5": 0.99425,
            "mrr@10": 0.97637,
            "latency_p50_ms": 532.0,
            "latency_p95_ms": 910.0,
        },
    },
    "calibration": {
        "ece_10bin": 0.0412,
        "brier": 0.0381,
        "top1_accuracy": 0.9385,
        "top1_mean_confidence_when_correct": 0.94,
        "top1_mean_confidence_when_wrong": 0.61,
    },
    "generation": {
        "models": ["meta/llama-3.3-70b-instruct"],
        "mean_f1": 0.29191,
        "exact_match": 0.00862,
        "success_rate_f1_ge_0.5": 0.1858,
        "abstention_rate": 0.0421,
        "citation_valid_rate": 0.9569,
        "file": "results/xquad-gen.jsonl",
    },
}


def test_validate_refuses_fixture_and_allows_preview():
    fixture = dict(REAL_SUMMARY, run_kind="fixture")
    try:
        validate_summaries([fixture])
        raise AssertionError("fixture summary should be refused")
    except ValueError:
        pass
    validate_summaries([fixture], allow_fixture=True)
    validate_summaries([REAL_SUMMARY])


def test_render_readme_block_contains_real_numbers():
    block = render_readme_block([REAL_SUMMARY])
    assert "Reranking" in block
    assert "98.10%" in block
    assert "OpenJev calibration" in block
    assert "Frozen-context answer generation" in block
    assert "from real runs" in block


def test_render_readme_block_marks_fixture_preview():
    block = render_readme_block([dict(REAL_SUMMARY, run_kind="fixture")])
    assert "FIXTURE PREVIEW" in block
    assert "not for publication" in block


def test_inject_into_readme_replaces_marker_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Title\n\n{RESULTS_START}\nold placeholder\n{RESULTS_END}\n\n## Next\n",
        encoding="utf-8",
    )
    inject_into_readme(readme, "NEW TABLE")
    text = readme.read_text(encoding="utf-8")
    assert "NEW TABLE" in text
    assert "old placeholder" not in text
    assert text.index(RESULTS_START) < text.index("NEW TABLE") < text.index(RESULTS_END)


def test_write_space_creates_static_page(tmp_path):
    index_path = write_space(tmp_path / "space", [REAL_SUMMARY], repo_url="https://github.com/x/y")
    html_text = index_path.read_text(encoding="utf-8")
    assert "xquad-en" in html_text
    space_readme = (tmp_path / "space" / "README.md").read_text(encoding="utf-8")
    assert "sdk: static" in space_readme
    assert "short_description" in space_readme


def test_dataset_card_has_metadata_and_results():
    card = render_dataset_card([REAL_SUMMARY], repo_url="https://github.com/x/y")
    assert card.startswith("---\n")
    assert "license: mit" in card
    assert "language:" in card
    assert "98.10%" in card
    assert "https://github.com/x/y" in card


def test_render_text_tables_is_plain_text():
    text = render_text_tables([REAL_SUMMARY])
    assert "Reranking" in text
    assert "nDCG@10" in text
    assert "|" not in text


def test_stage_published_copies_artifacts(tmp_path):
    source = tmp_path / "reports" / "generated" / "xquad-en-a-j-n-real"
    source.mkdir(parents=True)
    (source / "summary.json").write_text(json.dumps(REAL_SUMMARY), encoding="utf-8")
    (source / "report.md").write_text("# report", encoding="utf-8")
    results_file = tmp_path / "results" / "xquad-en-a-j-n-real.jsonl"
    results_file.parent.mkdir(parents=True)
    results_file.write_text("{}\n", encoding="utf-8")

    summary = dict(REAL_SUMMARY, results_file=str(results_file))
    out = stage_published(
        tmp_path / "reports" / "published",
        [summary],
        [source / "summary.json"],
    )
    assert (out / "xquad-en-a-j-n-real" / "summary.json").exists()
    assert (out / "xquad-en-a-j-n-real" / "report.md").exists()
    assert (out / "results" / "xquad-en-a-j-n-real.jsonl").exists()
    assert (out / "leaderboard.txt").exists()

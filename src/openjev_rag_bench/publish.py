from __future__ import annotations

import html
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"

FIXTURE_REFUSAL = (
    "refusing to publish fixture summaries: they verify the pipeline only. "
    "Run a real benchmark first, or pass --allow-fixture for a local preview."
)

GENERATION_HEADERS = [
    "Dataset",
    "Generator",
    "Token F1",
    "Exact match",
    "F1 >= 0.5",
    "Abstention",
    "Valid citations",
]


def load_summaries(paths: list[str | Path]) -> list[dict]:
    summaries = []
    for path in paths:
        summaries.append(json.loads(Path(path).read_text(encoding="utf-8")))
    return summaries


def find_summaries(reports_dir: str | Path) -> list[Path]:
    base = Path(reports_dir) / "generated"
    if not base.exists():
        return []
    return sorted(base.glob("*/summary.json"))


def validate_summaries(summaries: list[dict], allow_fixture: bool = False) -> None:
    if not summaries:
        raise ValueError("no summaries to publish")
    for summary in summaries:
        run_kind = summary.get("run_kind", "unknown")
        if run_kind != "real" and not allow_fixture:
            raise ValueError(FIXTURE_REFUSAL)


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.2f}%" if value is not None else "n/a"


def _fmt_ms(value: float | None) -> str:
    return f"{value:.0f} ms" if value is not None else "n/a"


def _method_label(branch: str, stats: dict) -> str:
    return f"{branch} — {stats.get('label', branch)}"


def rerank_rows(summary: dict) -> list[list[str]]:
    rows = []
    for branch in ("A", "J", "N", "T"):
        stats = summary.get("branches", {}).get(branch)
        if not stats:
            continue
        rows.append(
            [
                summary.get("dataset", "unknown"),
                str(summary.get("n_queries", "")),
                _method_label(branch, stats),
                _fmt_pct(stats.get("ndcg@10")),
                _fmt_pct(stats.get("recall@5")),
                _fmt_pct(stats.get("mrr@10")),
                _fmt_ms(stats.get("latency_p50_ms")),
            ]
        )
    return rows


def calibration_rows(summary: dict) -> list[list[str]]:
    calibration = summary.get("calibration") or {}
    if not calibration:
        return []
    correct = calibration.get("top1_mean_confidence_when_correct")
    wrong = calibration.get("top1_mean_confidence_when_wrong")
    return [
        [
            summary.get("dataset", "unknown"),
            f"{calibration.get('ece_10bin', 0.0):.4f}",
            f"{calibration.get('brier', 0.0):.4f}",
            _fmt_pct(calibration.get("top1_accuracy")),
            f"{correct:.3f}" if correct is not None else "n/a",
            f"{wrong:.3f}" if wrong is not None else "n/a",
        ]
    ]


def generation_rows(summary: dict) -> list[list[str]]:
    generation = summary.get("generation") or {}
    if not generation:
        return []
    models = ", ".join(generation.get("models") or []) or "n/a"
    return [
        [
            summary.get("dataset", "unknown"),
            models,
            _fmt_pct(generation.get("mean_f1")),
            _fmt_pct(generation.get("exact_match")),
            _fmt_pct(generation.get("success_rate_f1_ge_0.5")),
            _fmt_pct(generation.get("abstention_rate")),
            _fmt_pct(generation.get("citation_valid_rate")),
        ]
    ]


def gated_rows(summary: dict) -> list[list[str]]:
    gated = summary.get("gated") or {}
    if not gated:
        return []
    comparison = gated.get("partition_vs_baseline_ndcg@10") or {}
    diff = comparison.get("mean_diff")
    ci = comparison.get("ci") or [None, None]
    return [
        [
            summary.get("dataset", "unknown"),
            f"{gated.get('threshold', 0.5):.2f}",
            _fmt_pct(gated["baseline"]["ndcg@10"]),
            _fmt_pct(gated["always_on"]["ndcg@10"]),
            _fmt_pct(gated["partition"]["ndcg@10"]),
            f"{diff * 100:+.2f} pts" if diff is not None else "n/a",
            f"{ci[0] * 100:+.2f} to {ci[1] * 100:+.2f}" if ci[0] is not None else "n/a",
        ]
    ]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_readme_block(summaries: list[dict]) -> str:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    datasets = ", ".join(summary.get("dataset", "unknown") for summary in summaries)
    kinds = {summary.get("run_kind") for summary in summaries}
    if kinds == {"real"}:
        provenance = f"Generated {generated} from real runs ({datasets})."
    else:
        provenance = (
            f"FIXTURE PREVIEW generated {generated} — not for publication ({datasets})."
        )
    lines = [
        f"_{provenance} Fixture runs are never published._",
        "_Regenerate with:_ `uv run openjev-rag publish`",
        "",
    ]

    rows = [row for summary in summaries for row in rerank_rows(summary)]
    if rows:
        lines.append("### Reranking (frozen top-20 candidates, same pool for every method)")
        lines.append("")
        lines.append(
            _markdown_table(
                ["Dataset", "n", "Method", "nDCG@10", "Recall@5", "MRR@10", "Rerank p50"],
                rows,
            )
        )
        lines.append("")

    rows = [row for summary in summaries for row in calibration_rows(summary)]
    if rows:
        lines.append("### OpenJev calibration (candidate-level relevance probabilities)")
        lines.append("")
        lines.append(
            _markdown_table(
                [
                    "Dataset",
                    "ECE (10 bin)",
                    "Brier",
                    "Top-1 accuracy",
                    "Top-1 confidence (correct)",
                    "Top-1 confidence (wrong)",
                ],
                rows,
            )
        )
        lines.append("")

    rows = [row for summary in summaries for row in gated_rows(summary)]
    if rows:
        lines.append("### RAG optimization mode (confidence-partitioned OpenJev, fixed t = 0.50)")
        lines.append("")
        lines.append(
            _markdown_table(
                [
                    "Dataset",
                    "Threshold",
                    "A baseline nDCG@10",
                    "J always-on nDCG@10",
                    "G partitioned nDCG@10",
                    "Delta vs baseline",
                    "95% CI",
                ],
                rows,
            )
        )
        lines.append("")

    rows = [row for summary in summaries for row in generation_rows(summary)]
    if rows:
        lines.append("### Frozen-context answer generation (OpenJev top-5)")
        lines.append("")
        lines.append(_markdown_table(GENERATION_HEADERS, rows))
        lines.append("")

    lines.append(
        "_Metrics measure different stages: retrieval quality (nDCG, Recall) does not "
        "imply answer quality. See the per-run reports for confidence intervals, "
        "reliability bins, and risk-coverage tables._"
    )
    return "\n".join(lines)


def inject_into_readme(readme_path: str | Path, block: str) -> None:
    path = Path(readme_path)
    if not path.exists():
        raise FileNotFoundError(f"README not found: {path}")
    text = path.read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        raise ValueError(
            f"README is missing the {RESULTS_START} / {RESULTS_END} markers"
        )
    start = text.index(RESULTS_START) + len(RESULTS_START)
    end = text.index(RESULTS_END)
    updated = text[:start] + "\n\n" + block + "\n\n" + text[end:]
    path.write_text(updated, encoding="utf-8")


def render_text_tables(summaries: list[dict]) -> str:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"OpenJev RAG Benchmark - real runs - generated {generated}", ""]

    def ascii_table(headers: list[str], rows: list[list[str]], indent: str = "") -> None:
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))
        header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        lines.append(indent + header_line)
        lines.append(indent + "  ".join("-" * width for width in widths))
        for row in rows:
            lines.append(
                indent + "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
            )
        lines.append("")

    rows = [row for summary in summaries for row in rerank_rows(summary)]
    if rows:
        lines.append("Reranking (frozen top-20 candidates)")
        ascii_table(["Dataset", "n", "Method", "nDCG@10", "Recall@5", "MRR@10", "p50"], rows)

    rows = [row for summary in summaries for row in calibration_rows(summary)]
    if rows:
        lines.append("OpenJev calibration (candidate-level)")
        ascii_table(["Dataset", "ECE", "Brier", "Top-1 acc", "Conf correct", "Conf wrong"], rows)

    rows = [row for summary in summaries for row in generation_rows(summary)]
    if rows:
        lines.append("Frozen-context answer generation")
        ascii_table(
            ["Dataset", "Generator", "F1", "EM", "F1>=0.5", "Abstain", "Citations"], rows
        )
    return "\n".join(lines)


SPACE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 2rem 1rem; line-height: 1.5; }}
  main {{ max-width: 1020px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; }}
  h2 {{ font-size: 1.1rem; margin: 2rem 0 .5rem; }}
  p.note {{ opacity: .75; font-size: .9rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .9rem; margin: .5rem 0 1rem; }}
  th, td {{ border: 1px solid rgba(128,128,128,.35); padding: .45rem .6rem; text-align: left; }}
  th {{ background: rgba(128,128,128,.12); }}
  tr:nth-child(even) td {{ background: rgba(128,128,128,.06); }}
  footer {{ margin-top: 2.5rem; font-size: .85rem; opacity: .75; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<p class="note">{subtitle}</p>
{tables}
<footer>{footer}</footer>
</main>
</body>
</html>
"""


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    parts = ["<table>", "<thead><tr>"]
    parts += [f"<th>{html.escape(header)}</th>" for header in headers]
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def write_space(
    space_dir: str | Path,
    summaries: list[dict],
    title: str = "OpenJev RAG Benchmark",
    repo_url: str | None = None,
) -> Path:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []

    rows = [row for summary in summaries for row in rerank_rows(summary)]
    if rows:
        sections.append("<h2>Reranking (frozen top-20 candidates)</h2>")
        sections.append(
            _html_table(
                ["Dataset", "n", "Method", "nDCG@10", "Recall@5", "MRR@10", "Rerank p50"], rows
            )
        )

    rows = [row for summary in summaries for row in calibration_rows(summary)]
    if rows:
        sections.append("<h2>OpenJev calibration (candidate-level relevance probabilities)</h2>")
        sections.append(
            _html_table(
                [
                    "Dataset",
                    "ECE (10 bin)",
                    "Brier",
                    "Top-1 accuracy",
                    "Top-1 confidence (correct)",
                    "Top-1 confidence (wrong)",
                ],
                rows,
            )
        )

    rows = [row for summary in summaries for row in generation_rows(summary)]
    if rows:
        sections.append("<h2>Frozen-context answer generation (OpenJev top-5)</h2>")
        sections.append(_html_table(GENERATION_HEADERS, rows))

    rows = [row for summary in summaries for row in gated_rows(summary)]
    if rows:
        sections.append(
            "<h2>RAG optimization mode (confidence-partitioned OpenJev, fixed t = 0.50)</h2>"
        )
        sections.append(
            _html_table(
                [
                    "Dataset",
                    "Threshold",
                    "A baseline nDCG@10",
                    "J always-on nDCG@10",
                    "G partitioned nDCG@10",
                    "Delta vs baseline",
                    "95% CI",
                ],
                rows,
            )
        )

    footer_parts = [f"Generated {generated}. Real runs only; fixture runs are never published."]
    if repo_url:
        footer_parts.append(
            f'Code and raw results: <a href="{html.escape(repo_url)}">{html.escape(repo_url)}</a>.'
        )
    footer_parts.append(
        "OpenJev is an independent open-weights model served by Codiv; "
        "not affiliated with TypeSafe AI."
    )

    html_text = SPACE_TEMPLATE.format(
        title=html.escape(title),
        subtitle=(
            "Free English benchmark of an open System One model as a reranking "
            "and decision layer in RAG."
        ),
        tables="\n".join(sections),
        footer=" ".join(footer_parts),
    )
    out_dir = Path(space_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    (out_dir / "README.md").write_text(
        "---\n"
        f"title: {title}\n"
        "sdk: static\n"
        "app_file: index.html\n"
        "---\n\n"
        "Static leaderboard generated by `openjev-rag publish`.\n",
        encoding="utf-8",
    )
    return index_path


def stage_published(
    published_dir: str | Path,
    summaries: list[dict],
    summary_paths: list[str | Path],
) -> Path:
    out = Path(published_dir)
    out.mkdir(parents=True, exist_ok=True)
    for summary, summary_path in zip(summaries, summary_paths, strict=False):
        stem = Path(summary_path).parent.name
        target = out / stem
        target.mkdir(parents=True, exist_ok=True)
        source_dir = Path(summary_path).parent
        for name in ("report.md", "summary.json", "summary.csv"):
            source = source_dir / name
            if source.exists():
                shutil.copy2(source, target / name)
        results_file = summary.get("results_file")
        if results_file and Path(results_file).exists():
            results_target = out / "results"
            results_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(results_file, results_target / Path(results_file).name)
        generation_file = (summary.get("generation") or {}).get("file")
        if generation_file and Path(generation_file).exists():
            generation_target = out / "results"
            generation_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generation_file, generation_target / Path(generation_file).name)
    (out / "leaderboard.txt").write_text(render_text_tables(summaries), encoding="utf-8")
    return out


def push_to_huggingface(
    folder: str | Path,
    repo_id: str,
    repo_type: str = "dataset",
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is not installed; run: uv sync --extra hf"
        ) from error
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder))
    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"

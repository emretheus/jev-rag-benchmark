"""Benchmark charts as dependency-free SVG, with optional PNG conversion.

Charts are generated from the same summary.json files that feed the reports, so
they can never drift from the published numbers.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PALETTE = {
    "A": "#868e96",
    "T": "#f08c00",
    "N": "#4c6ef5",
    "bm25": "#868e96",
    "hybrid": "#4c6ef5",
}
DATASET_COLORS = {"xquad-en": "#f08c00", "scifact": "#4c6ef5"}
WIDTH = 960
HEIGHT = 520
PAD_LEFT = 80
PAD_RIGHT = 30
PAD_TOP = 50
PAD_BOTTOM = 70
FONT = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _text(
    x: float,
    y: float,
    content: str,
    size: int = 14,
    anchor: str = "start",
    fill: str = "#212529",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{_esc(content)}</text>'
    )


def _rect(
    x: float, y: float, width: float, height: float, fill: str, opacity: float = 1.0
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0.5):.1f}" '
        f'height="{max(height, 0.5):.1f}" fill="{fill}" opacity="{opacity}"/>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def _circle(x: float, y: float, radius: float, fill: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}"/>'


def _svg(body: str, title: str, width: int = WIDTH, height: int = HEIGHT) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_esc(title)}">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f"{_text(PAD_LEFT, 30, title, size=20, weight=600)}"
        f"{body}</svg>"
    )


def _plot_area() -> tuple[float, float, float, float]:
    return (PAD_LEFT, PAD_TOP, WIDTH - PAD_RIGHT, HEIGHT - PAD_BOTTOM)


def _y_axis(values: list[float], lo: float, hi: float, ticks: int = 5) -> tuple[list[str], object]:
    x0, y0, x1, y1 = _plot_area()
    parts = []
    span = hi - lo
    for index in range(ticks + 1):
        value = lo + span * index / ticks
        y = y1 - (value - lo) / span * (y1 - y0)
        parts.append(_line(x0, y, x1, y, "#dee2e6", 1))
        parts.append(_text(x0 - 10, y + 5, f"{value:.2f}", size=12, anchor="end", fill="#868e96"))
    return parts, lambda value: y1 - (value - lo) / span * (y1 - y0)


def _bar_chart(
    title: str,
    groups: list[str],
    series: list[tuple[str, list[float], list[float], list[float]]],
    value_range: tuple[float, float],
) -> str:
    """series: (label, values, ci_low, ci_high) — CI lists may be empty."""
    x0, y0, x1, y1 = _plot_area()
    lo, hi = value_range
    parts, to_y = _y_axis([], lo, hi)
    group_width = (x1 - x0) / max(len(groups), 1)
    bar_width = group_width / (len(series) + 1)
    for group_index, group in enumerate(groups):
        center = x0 + group_width * (group_index + 0.5)
        parts.append(_text(center, y1 + 25, group, size=13, anchor="middle", fill="#495057"))
        for series_index, (_label, values, ci_low, ci_high, color) in enumerate(series):
            value = values[group_index]
            offset = (series_index - (len(series) - 1) / 2) * bar_width
            left = center + offset - bar_width * 0.4
            top = to_y(value)
            parts.append(_rect(left, top, bar_width * 0.8, y1 - top, color))
            if ci_low and ci_high:
                ci_top = to_y(ci_high[group_index])
                ci_bottom = to_y(ci_low[group_index])
                mid = left + bar_width * 0.4
                parts.append(_line(mid, ci_top, mid, ci_bottom, "#343a40", 1.5))
                parts.append(_line(mid - 5, ci_top, mid + 5, ci_top, "#343a40", 1.5))
                parts.append(_line(mid - 5, ci_bottom, mid + 5, ci_bottom, "#343a40", 1.5))
    legend_x = x0
    for legend_label, _, _, _, color in series:
        parts.append(_rect(legend_x, HEIGHT - 30, 12, 12, color))
        parts.append(
            _text(legend_x + 18, HEIGHT - 20, legend_label, size=12, fill="#495057")
        )
        legend_x += 20 + len(legend_label) * 7.5
    return _svg("".join(parts), title)


def reranker_comparison(summaries: list[dict]) -> str:
    datasets = [summary["dataset"] for summary in summaries]
    branches = ["A", "T", "N", "L"]
    available = [
        branch
        for branch in branches
        if all(branch in summary["branches"] for summary in summaries)
    ]
    values = {
        branch: [summary["branches"][branch]["ndcg@10"] for summary in summaries]
        for branch in available
    }
    ci_low = {
        branch: [summary["branches"][branch]["ndcg@10_ci"][1] for summary in summaries]
        for branch in values
    }
    ci_high = {
        branch: [summary["branches"][branch]["ndcg@10_ci"][2] for summary in summaries]
        for branch in values
    }
    all_values = [value for branch_values in values.values() for value in branch_values]
    lo = min(all_values) - 0.05
    hi = min(1.0, max(all_values) + 0.05)
    series = []
    for branch in branches:
        if branch not in values:
            continue
        label = {"A": "no reranker", "T": "Jev 1.13", "N": "NVIDIA cross-encoder"}[branch]
        series.append((label, values[branch], ci_low[branch], ci_high[branch], PALETTE[branch]))
    return _bar_chart("nDCG@10 by method (frozen top-20 candidates)", datasets, series, (lo, hi))


def calibration_reliability(summaries: list[dict]) -> str:
    x0, y0, x1, y1 = _plot_area()
    parts = []
    for index in range(6):
        value = index / 5
        y = y1 - value * (y1 - y0)
        x = x0 + value * (x1 - x0)
        parts.append(_line(x0, y, x1, y, "#dee2e6", 1))
        parts.append(_text(x0 - 10, y + 5, f"{value:.1f}", size=12, anchor="end", fill="#868e96"))
        parts.append(_text(x, y1 + 22, f"{value:.1f}", size=12, anchor="middle", fill="#868e96"))
    parts.append(_line(x0, y1, x1, y0, "#adb5bd", 1.5, dash="6 6"))
    for summary in summaries:
        calibration = (
            (summary.get("calibration_by_branch") or {}).get("T")
            or summary.get("calibration")
            or {}
        )
        color = DATASET_COLORS.get(summary["dataset"], "#f08c00")
        for row in calibration.get("reliability", []):
            x = x0 + row["mean_confidence"] * (x1 - x0)
            y = y1 - row["accuracy"] * (y1 - y0)
            radius = 4 + min(10.0, row["count"] ** 0.5 / 3)
            parts.append(_circle(x, y, radius, color))
    legend_x = x0
    for summary in summaries:
        color = DATASET_COLORS.get(summary["dataset"], "#f08c00")
        parts.append(_circle(legend_x + 6, HEIGHT - 24, 6, color))
        parts.append(
            _text(legend_x + 18, HEIGHT - 20, summary["dataset"], size=12, fill="#495057")
        )
        legend_x += 40 + len(summary["dataset"]) * 8
    parts.append(
        _text(
            (x0 + x1) / 2,
            HEIGHT - 20,
            "mean confidence →",
            size=12,
            anchor="end",
            fill="#868e96",
        )
    )
    return _svg("".join(parts), "Jev calibration: mean confidence vs accuracy (candidate level)")


def risk_coverage(summaries: list[dict]) -> str:
    x0, y0, x1, y1 = _plot_area()
    parts = []
    for index in range(6):
        value = index / 5
        y = y1 - value * (y1 - y0)
        parts.append(_line(x0, y, x1, y, "#dee2e6", 1))
        parts.append(_text(x0 - 10, y + 5, f"{value:.1f}", size=12, anchor="end", fill="#868e96"))
    max_coverage = 0.2
    for summary in summaries:
        calibration = (
            (summary.get("calibration_by_branch") or {}).get("T")
            or summary.get("calibration")
            or {}
        )
        rows = calibration.get("risk_coverage", [])
        color = DATASET_COLORS.get(summary["dataset"], "#f08c00")
        points = [(row["coverage"], row["precision"]) for row in rows if row["count"]]
        points.sort()
        if points:
            max_coverage = max(max_coverage, max(coverage for coverage, _ in points) * 1.3)
        path = " ".join(
            f"{'M' if index == 0 else 'L'} {x0 + coverage / max_coverage * (x1 - x0):.1f} "
            f"{y1 - precision * (y1 - y0):.1f}"
            for index, (coverage, precision) in enumerate(points)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for coverage, precision in points:
            parts.append(
                _circle(
                    x0 + coverage / max_coverage * (x1 - x0),
                    y1 - precision * (y1 - y0),
                    4,
                    color,
                )
            )
    for index in range(6):
        value = index / 5
        x = x0 + value * (x1 - x0)
        parts.append(
            _text(
                x,
                y1 + 22,
                f"{value * max_coverage:.0%}",
                size=12,
                anchor="middle",
                fill="#868e96",
            )
        )
    legend_x = x0
    for summary in summaries:
        color = DATASET_COLORS.get(summary["dataset"], "#f08c00")
        parts.append(_rect(legend_x, HEIGHT - 32, 12, 12, color))
        parts.append(_text(legend_x + 18, HEIGHT - 22, summary["dataset"], size=12, fill="#495057"))
        legend_x += 30 + len(summary["dataset"]) * 8
    parts.append(
        _text(
            (x0 + x1) / 2,
            HEIGHT - 8,
            "coverage (share of candidates kept) →",
            size=12,
            anchor="middle",
            fill="#868e96",
        )
    )
    return _svg("".join(parts), "Precision vs coverage at a Jev probability threshold")


def latency_quality(summaries: list[dict]) -> str:
    x0, y0, x1, y1 = _plot_area()
    points = []
    for summary in summaries:
        for branch, stats in summary["branches"].items():
            latency = max(stats["latency_p50_ms"], 1.0)
            points.append((latency, stats["ndcg@10"], f"{summary['dataset']} · {branch}"))
    max_latency = max(point[0] for point in points) * 1.4
    min_ndcg = min(point[1] for point in points) - 0.05
    parts = []
    for index in range(6):
        value = index / 5
        y = y1 - value * (y1 - y0)
        parts.append(_line(x0, y, x1, y, "#dee2e6", 1))
        parts.append(
            _text(
                x0 - 10,
                y + 5,
                f"{min_ndcg + (1 - min_ndcg) * value:.2f}",
                size=12,
                anchor="end",
                fill="#868e96",
            )
        )
    for latency, ndcg, label in points:
        x = x0 + (latency / max_latency) ** 0.5 * (x1 - x0)
        y = y1 - (ndcg - min_ndcg) / (1 - min_ndcg) * (y1 - y0)
        color = {"T": "#f08c00", "N": "#4c6ef5"}.get(label.split(" · ")[-1], "#868e96")
        parts.append(_circle(x, y, 7, color))
        parts.append(_text(x + 10, y - 8, label, size=11, fill="#495057"))
    return _svg("".join(parts), "Quality vs rerank latency (p50, square-root x axis)")


def retrieval_errors(audits: list[dict]) -> str:
    groups = []
    bm25_errors = []
    hybrid_errors = []
    for audit in audits:
        for bm25_row, hybrid_row in zip(audit["bm25"], audit["hybrid"], strict=False):
            label = f"{audit['dataset']} · top-{bm25_row['depth']}"
            groups.append(label)
            bm25_errors.append(1 - bm25_row["recall"])
            hybrid_errors.append(1 - hybrid_row["recall"])
    hi = min(1.0, max([*bm25_errors, *hybrid_errors]) * 1.4 + 0.01)
    series = [
        ("BM25 only", bm25_errors, [], [], PALETTE["bm25"]),
        ("Hybrid (BM25 + dense)", hybrid_errors, [], [], PALETTE["hybrid"]),
    ]
    return _bar_chart("Gold passage not retrieved (lower is better)", groups, series, (0.0, hi))


def _find_chrome() -> str | None:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def svg_to_png(svg_path: Path, png_path: Path, width: int = WIDTH, height: int = HEIGHT) -> bool:
    chrome = _find_chrome()
    if chrome is None:
        return False
    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--window-size={width},{height}",
            f"--screenshot={png_path}",
            svg_path.resolve().as_uri(),
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    return png_path.exists() and png_path.stat().st_size > 0


CARD_WIDTH = 1200
CARD_HEIGHT = 630


def social_card(summaries: list[dict]) -> str:
    """1200x630 share card for LinkedIn / X / OG previews."""
    by_dataset = {summary["dataset"]: summary for summary in summaries}
    xquad = by_dataset.get("xquad-en")
    scifact = by_dataset.get("scifact")

    def delta(summary: dict | None) -> str:
        if not summary:
            return "n/a"
        comparison = (summary.get("paired_comparisons") or {}).get("T_vs_A") or {}
        value = comparison.get("ndcg@10_mean_diff")
        return f"{value * 100:+.2f}" if value is not None else "n/a"

    def ece(summary: dict | None) -> str:
        if not summary:
            return "n/a"
        calibration = (summary.get("calibration_by_branch") or {}).get("T") or {}
        return f"{calibration.get('ece_10bin', 0) * 100:.2f}%"

    parts = [
        _rect(0, 0, CARD_WIDTH, CARD_HEIGHT, "#0b0c0e"),
        _rect(0, 0, CARD_WIDTH, 8, "#f08c00"),
        _text(64, 108, "Jev 1.13 in a RAG pipeline", size=46, fill="#ffffff", weight=700),
        _text(
            64,
            150,
            "English benchmark · 1,490 queries · frozen candidate pools · $0 to reproduce",
            size=20,
            fill="#adb5bd",
        ),
    ]
    stats = [
        ("nDCG@10 vs no reranking", f"{delta(scifact)} pts", "SciFact"),
        ("nDCG@10 vs no reranking", f"{delta(xquad)} pts", "XQuAD-EN"),
        ("Calibration error (ECE)", ece(xquad), "XQuAD-EN"),
    ]
    box_width = (CARD_WIDTH - 128 - 2 * 24) / 3
    for index, (label, value, dataset) in enumerate(stats):
        x = 64 + index * (box_width + 24)
        parts.append(_rect(x, 210, box_width, 190, "#16181c"))
        parts.append(_text(x + 24, 260, label, size=16, fill="#adb5bd"))
        parts.append(_text(x + 24, 330, value, size=54, fill="#f08c00", weight=700))
        parts.append(_text(x + 24, 370, dataset, size=18, fill="#868e96"))
    parts.append(
        _text(
            64,
            470,
            "Jev improves retrieval where there is headroom, ties a trained",
            size=20,
            fill="#dee2e6",
        )
    )
    parts.append(
        _text(
            64,
            500,
            "cross-encoder on SciFact, and its calibrated probabilities enable",
            size=20,
            fill="#dee2e6",
        )
    )
    parts.append(
        _text(
            64,
            530,
            "high-precision filtering and cheaper generation.",
            size=20,
            fill="#dee2e6",
        )
    )
    parts.append(
        _text(
            64,
            570,
            "github.com/emretheus/jev-rag-benchmark",
            size=20,
            fill="#f08c00",
            weight=600,
        )
    )
    parts.append(
        _text(
            CARD_WIDTH - 64,
            570,
            "leaderboard: huggingface.co/spaces/emretheus/jev-rag-benchmark-leaderboard",
            size=15,
            fill="#868e96",
            anchor="end",
        )
    )
    return _svg("".join(parts), "Jev RAG Benchmark share card", CARD_WIDTH, CARD_HEIGHT)


def render_charts(
    summaries: list[dict],
    output_dir: str | Path,
    audits: list[dict] | None = None,
    png: bool = True,
    social_dir: str | Path | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts: dict[str, str] = {}
    if summaries:
        charts["reranker-comparison"] = reranker_comparison(summaries)
        charts["calibration-reliability"] = calibration_reliability(summaries)
        charts["risk-coverage"] = risk_coverage(summaries)
        charts["latency-quality"] = latency_quality(summaries)
    if audits:
        charts["retrieval-errors"] = retrieval_errors(audits)

    written = []
    for name, svg_text in charts.items():
        svg_path = output_dir / f"{name}.svg"
        svg_path.write_text(svg_text, encoding="utf-8")
        written.append(svg_path)
        if png:
            svg_to_png(svg_path, output_dir / f"{name}.png")

    if summaries and social_dir:
        social_path = Path(social_dir)
        social_path.mkdir(parents=True, exist_ok=True)
        svg_path = social_path / "jev-benchmark-card.svg"
        svg_path.write_text(social_card(summaries), encoding="utf-8")
        written.append(svg_path)
        if png:
            svg_to_png(
                svg_path,
                social_path / "jev-benchmark-card.png",
                width=CARD_WIDTH,
                height=CARD_HEIGHT,
            )
    return written

MODEL_COLORS = {"jev": "#f08c00", "laya": "#4c6ef5"}


def toolbench_accuracy(summary: dict) -> str:
    models = list((summary.get("models") or {}).keys())
    families = ["tool_selection", "call_approval", "arg_validation", "injection_risk"]
    series = []
    for model in models:
        model_summary = summary["models"][model]
        values = [
            model_summary.get("families", {}).get(family, {}).get("accuracy", 0.0)
            for family in families
        ]
        series.append(
            (
                model,
                values,
                [],
                [],
                MODEL_COLORS.get(model, "#868e96"),
            )
        )
    return _bar_chart(
        "Tool-calling governance: accuracy by task family",
        families,
        series,
        (0.0, 1.0),
    )


def toolbench_cardinality(summary: dict) -> str:
    models = list((summary.get("models") or {}).keys())
    sizes = sorted(
        {
            size
            for model in models
            for size in (summary["models"][model].get("cardinality_scaling") or {})
        },
        key=lambda value: int(value),
    )
    series = []
    for model in models:
        scaling = summary["models"][model].get("cardinality_scaling") or {}
        values = [float(scaling.get(size, 0.0)) for size in sizes]
        series.append((model, values, [], [], MODEL_COLORS.get(model, "#868e96")))
    return _bar_chart(
        "Tool selection accuracy by catalog size (choice question)",
        [f"{size} tools" for size in sizes],
        series,
        (0.0, 1.0),
    )


def render_toolbench_charts(
    summary: dict,
    output_dir: str | Path,
    png: bool = True,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, svg_text in (
        ("toolbench-accuracy", toolbench_accuracy(summary)),
        ("toolbench-cardinality", toolbench_cardinality(summary)),
    ):
        svg_path = output_dir / f"{name}.svg"
        svg_path.write_text(svg_text, encoding="utf-8")
        written.append(svg_path)
        if png:
            svg_to_png(svg_path, output_dir / f"{name}.png")
    return written

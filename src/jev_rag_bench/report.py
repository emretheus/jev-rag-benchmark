from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .calibration import (
    brier_score,
    expected_calibration_error,
    reliability_table,
    risk_coverage,
)
from .gate import answerability_summary, gating_summary
from .generate import load_rows
from .metrics import bootstrap_ci, paired_bootstrap_ci, percentile
from .rerank import BRANCH_LABELS

FIXTURE_WARNING = (
    "> **Fixture run.** These numbers come from deterministic offline stand-in clients "
    "(`run_kind=fixture`). They verify the pipeline only and must never be published as "
    "benchmark results."
)


def _metric_values(rows: list[dict], branch: str, metric: str) -> list[float]:
    values = []
    for row in rows:
        metrics = row.get("metrics") or {}
        if branch in metrics and metric in metrics[branch]:
            values.append(float(metrics[branch][metric]))
    return values


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def _fmt_ci(values: list[float], seed: int) -> str:
    point, low, high = bootstrap_ci(values, seed=seed)
    return f"{_fmt_pct(point)} (95% CI {_fmt_pct(low)} to {_fmt_pct(high)})"


def generate_report(
    results_path: str | Path,
    output_dir: str | Path,
    generation_path: str | Path | None = None,
    seed: int = 13,
) -> Path:
    rows = load_rows(results_path)
    if not rows:
        raise ValueError(f"no rows in {results_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = rows[0].get("dataset", "unknown")
    run_kind = rows[0].get("run_kind", "unknown")
    branches = [
        b for b in ("A", "J", "N", "T") if any(b in row.get("branches", {}) for row in rows)
    ]

    summary: dict = {
        "dataset": dataset,
        "run_kind": run_kind,
        "n_queries": len(rows),
        "results_file": str(results_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "branches": {},
        "paired_comparisons": {},
        "calibration": {},
        "generation": {},
    }

    candidate_ceiling = mean(1.0 if row.get("gold_in_candidates") else 0.0 for row in rows)
    summary["candidate_ceiling_recall"] = candidate_ceiling

    for branch in branches:
        ndcg = _metric_values(rows, branch, "ndcg@10")
        recall5 = _metric_values(rows, branch, "recall@5")
        recall10 = _metric_values(rows, branch, "recall@10")
        mrr = _metric_values(rows, branch, "mrr@10")
        latencies = [
            float(row["branches"][branch].get("latency_ms") or 0.0)
            for row in rows
            if branch in row.get("branches", {})
        ]
        tokens = [
            int(row["branches"][branch].get("input_tokens") or 0)
            for row in rows
            if branch in row.get("branches", {})
        ]
        models = sorted(
            {
                str(row["branches"][branch].get("resolved_model"))
                for row in rows
                if row.get("branches", {}).get(branch, {}).get("resolved_model")
            }
        )
        summary["branches"][branch] = {
            "label": BRANCH_LABELS.get(branch, branch),
            "n": len(ndcg),
            "recall@5": mean(recall5) if recall5 else 0.0,
            "recall@10": mean(recall10) if recall10 else 0.0,
            "ndcg@10": mean(ndcg) if ndcg else 0.0,
            "ndcg@10_ci": list(bootstrap_ci(ndcg, seed=seed)) if ndcg else [0.0, 0.0, 0.0],
            "mrr@10": mean(mrr) if mrr else 0.0,
            "latency_p50_ms": percentile(latencies, 0.5),
            "latency_p95_ms": percentile(latencies, 0.95),
            "mean_input_tokens": mean(tokens) if tokens else 0.0,
            "total_input_tokens": sum(tokens),
            "resolved_models": models,
        }

    if "J" in branches:
        j_ndcg = _metric_values(rows, "J", "ndcg@10")
        j_recall5 = _metric_values(rows, "J", "recall@5")
        for other in ("N", "A"):
            if other not in branches:
                continue
            other_ndcg = _metric_values(rows, other, "ndcg@10")
            other_recall5 = _metric_values(rows, other, "recall@5")
            if len(j_ndcg) != len(other_ndcg) or not j_ndcg:
                continue
            ndcg_diff = paired_bootstrap_ci(j_ndcg, other_ndcg, seed=seed)
            recall_diff = paired_bootstrap_ci(j_recall5, other_recall5, seed=seed)
            summary["paired_comparisons"][f"J_vs_{other}"] = {
                "ndcg@10_mean_diff": ndcg_diff[0],
                "ndcg@10_ci": [ndcg_diff[1], ndcg_diff[2]],
                "recall@5_mean_diff": recall_diff[0],
                "recall@5_ci": [recall_diff[1], recall_diff[2]],
                "n_pairs": len(j_ndcg),
            }

    if "T" in branches:
        for other in ("A", "J", "N"):
            if other not in branches:
                continue
            t_ndcg = _metric_values(rows, "T", "ndcg@10")
            other_ndcg = _metric_values(rows, other, "ndcg@10")
            t_recall5 = _metric_values(rows, "T", "recall@5")
            other_recall5 = _metric_values(rows, other, "recall@5")
            if len(t_ndcg) != len(other_ndcg) or not t_ndcg:
                continue
            ndcg_diff = paired_bootstrap_ci(t_ndcg, other_ndcg, seed=seed)
            recall_diff = paired_bootstrap_ci(t_recall5, other_recall5, seed=seed)
            summary["paired_comparisons"][f"T_vs_{other}"] = {
                "ndcg@10_mean_diff": ndcg_diff[0],
                "ndcg@10_ci": [ndcg_diff[1], ndcg_diff[2]],
                "recall@5_mean_diff": recall_diff[0],
                "recall@5_ci": [recall_diff[1], recall_diff[2]],
                "n_pairs": len(t_ndcg),
            }

    if "J" in branches:
        probs: list[float] = []
        labels: list[int] = []
        top1: list[tuple[float, int]] = []
        for row in rows:
            branch_row = row.get("branches", {}).get("J")
            if not branch_row or not branch_row.get("probs"):
                continue
            gold = set(row.get("gold_doc_ids") or [])
            order = branch_row["order"]
            for doc_id, probability in zip(order, branch_row["probs"], strict=False):
                probs.append(float(probability))
                labels.append(1 if doc_id in gold else 0)
            top1.append((float(branch_row["probs"][0]), 1 if order[0] in gold else 0))
        if probs:
            correct_top1 = [p for p, y in top1 if y == 1]
            wrong_top1 = [p for p, y in top1 if y == 0]
            summary["calibration"] = {
                "n_predictions": len(probs),
                "positive_rate": mean(labels),
                "brier": brier_score(probs, labels),
                "ece_10bin": expected_calibration_error(probs, labels, bins=10),
                "reliability": reliability_table(probs, labels, bins=10),
                "risk_coverage": risk_coverage(probs, labels),
                "top1_mean_confidence_when_correct": mean(correct_top1) if correct_top1 else None,
                "top1_mean_confidence_when_wrong": mean(wrong_top1) if wrong_top1 else None,
                "top1_accuracy": mean([y for _, y in top1]) if top1 else None,
            }

    if generation_path and Path(generation_path).exists():
        generation_rows = load_rows(generation_path)
        summary["generation"] = _generation_summary(Path(generation_path))
        summary["generation"]["file"] = str(generation_path)
        summary["generation_gating"] = answerability_summary(rows, generation_rows)

    gated = gating_summary(rows)
    if gated:
        summary["gated"] = gated

    report_path = output_dir / "report.md"
    report_path.write_text(_render_markdown(summary), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(output_dir, summary)
    return report_path


def _write_csv(output_dir: Path, summary: dict) -> None:
    path = output_dir / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "branch",
                "method",
                "n",
                "ndcg@10",
                "recall@5",
                "recall@10",
                "mrr@10",
                "latency_p50_ms",
                "latency_p95_ms",
                "mean_input_tokens",
                "total_input_tokens",
            ]
        )
        for branch, stats in summary["branches"].items():
            writer.writerow(
                [
                    branch,
                    stats["label"],
                    stats["n"],
                    f"{stats['ndcg@10']:.6f}",
                    f"{stats['recall@5']:.6f}",
                    f"{stats['recall@10']:.6f}",
                    f"{stats['mrr@10']:.6f}",
                    f"{stats['latency_p50_ms']:.3f}",
                    f"{stats['latency_p95_ms']:.3f}",
                    f"{stats['mean_input_tokens']:.1f}",
                    stats["total_input_tokens"],
                ]
            )


def _generation_summary(path: Path) -> dict:
    gen_rows = load_rows(path)
    if not gen_rows:
        return {}
    f1_values = [float(row["f1"]) for row in gen_rows if row.get("f1") is not None]
    em_values = [float(row["em"]) for row in gen_rows if row.get("em") is not None]
    latencies = [float(row.get("latency_ms") or 0.0) for row in gen_rows]
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in gen_rows)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in gen_rows)
    cited = sum(len(row.get("citations") or []) for row in gen_rows)
    invalid = sum(len(row.get("invalid_citations") or []) for row in gen_rows)
    return {
        "n": len(gen_rows),
        "branch": gen_rows[0].get("branch"),
        "run_kind": gen_rows[0].get("run_kind"),
        "models": sorted({str(row.get("model")) for row in gen_rows if row.get("model")}),
        "mean_f1": mean(f1_values) if f1_values else None,
        "exact_match": mean(em_values) if em_values else None,
        "success_rate_f1_ge_0.5": (
            mean([1.0 if value >= 0.5 else 0.0 for value in f1_values]) if f1_values else None
        ),
        "abstention_rate": mean([1.0 if row.get("abstained") else 0.0 for row in gen_rows]),
        "citation_valid_rate": (1 - invalid / cited) if cited else None,
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _render_markdown(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Jev RAG Benchmark report")
    lines.append("")
    if summary.get("run_kind") != "real":
        lines.append(FIXTURE_WARNING)
        lines.append("")
    lines.append(f"- Dataset: `{summary['dataset']}`")
    lines.append(f"- Queries: {summary['n_queries']}")
    lines.append(f"- Run kind: `{summary['run_kind']}`")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append("")

    lines.append("## 1. Candidate retrieval ceiling")
    lines.append("")
    lines.append(
        "Share of queries whose gold passage was present in the frozen candidate pool "
        "(hybrid BM25 + dense, reciprocal rank fusion)."
    )
    lines.append("")
    lines.append(f"- Gold in candidates: **{_fmt_pct(summary['candidate_ceiling_recall'])}**")
    lines.append("")

    lines.append("## 2. Reranker comparison")
    lines.append("")
    lines.append(
        "| Branch | Method | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | "
        "Rerank p50 | Rerank p95 | Mean input tokens |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for branch in ("A", "J", "N", "T"):
        stats = summary["branches"].get(branch)
        if not stats:
            continue
        lines.append(
            f"| {branch} | {stats['label']} | {_fmt_pct(stats['ndcg@10'])} | "
            f"{_fmt_pct(stats['recall@5'])} | {_fmt_pct(stats['recall@10'])} | "
            f"{_fmt_pct(stats['mrr@10'])} | {stats['latency_p50_ms']:.1f} ms | "
            f"{stats['latency_p95_ms']:.1f} ms | {stats['mean_input_tokens']:.0f} |"
        )
    lines.append("")
    for branch in ("A", "J", "N", "T"):
        stats = summary["branches"].get(branch)
        if stats and stats.get("resolved_models"):
            lines.append(
                f"- Branch {branch} resolved models: {', '.join(stats['resolved_models'])}"
            )
    lines.append("")

    comparisons = summary.get("paired_comparisons") or {}
    if comparisons:
        lines.append("### Paired comparisons (same queries, bootstrap 95% CI)")
        lines.append("")
        for key, comp in comparisons.items():
            ndcg_low, ndcg_high = comp["ndcg@10_ci"]
            recall_low, recall_high = comp["recall@5_ci"]
            lines.append(
                f"- {key}: nDCG@10 {comp['ndcg@10_mean_diff'] * 100:+.3f} pts "
                f"(95% CI {ndcg_low * 100:+.3f} to {ndcg_high * 100:+.3f}), "
                f"Recall@5 {comp['recall@5_mean_diff'] * 100:+.3f} pts "
                f"(95% CI {recall_low * 100:+.3f} to {recall_high * 100:+.3f}); "
                f"n={comp['n_pairs']}"
            )
        lines.append("")

    calibration = summary.get("calibration") or {}
    if calibration:
        lines.append("## 3. OpenJev calibration (candidate-level relevance)")
        lines.append("")
        lines.append(
            f"Each candidate passage receives a probability of relevance. "
            f"Predictions: {calibration['n_predictions']}, positive rate: "
            f"{_fmt_pct(calibration['positive_rate'])}."
        )
        lines.append("")
        lines.append(f"- Brier score: **{calibration['brier']:.4f}** (lower is better)")
        lines.append(f"- Expected calibration error (10 bins): **{calibration['ece_10bin']:.4f}**")
        if calibration.get("top1_accuracy") is not None:
            lines.append(f"- Top-1 accuracy: {_fmt_pct(calibration['top1_accuracy'])}")
        if calibration.get("top1_mean_confidence_when_correct") is not None:
            correct_confidence = calibration["top1_mean_confidence_when_correct"]
            wrong_confidence = calibration.get("top1_mean_confidence_when_wrong")
            wrong_display = f"{wrong_confidence:.3f}" if wrong_confidence is not None else "n/a"
            lines.append(
                "- Mean top-1 confidence when correct / wrong: "
                f"{correct_confidence:.3f} / {wrong_display}"
            )
        lines.append("")
        lines.append("Reliability:")
        lines.append("")
        lines.append("| Probability bin | Count | Mean confidence | Accuracy |")
        lines.append("|---|---|---|---|")
        for bin_row in calibration["reliability"]:
            lines.append(
                f"| {bin_row['bin']} | {bin_row['count']} | {bin_row['mean_confidence']:.3f} | "
                f"{bin_row['accuracy']:.3f} |"
            )
        lines.append("")
        lines.append("Risk-coverage (threshold on relevance probability):")
        lines.append("")
        lines.append("| Threshold | Covered | Coverage | Precision | Recall |")
        lines.append("|---|---|---|---|---|")
        for row in calibration["risk_coverage"]:
            lines.append(
                f"| {row['threshold']:.2f} | {row['count']} | {_fmt_pct(row['coverage'])} | "
                f"{_fmt_pct(row['precision'])} | {_fmt_pct(row['recall'])} |"
            )
        lines.append("")

    gated = summary.get("gated") or {}
    if gated:
        threshold = gated.get("threshold", 0.5)
        lines.append(
            f"## 3b. RAG optimization mode: confidence-partitioned OpenJev "
            f"(t = {threshold:.2f})"
        )
        lines.append("")
        lines.append(
            "Always-on reranking applies OpenJev to every candidate; the decision-layer mode "
            "reranks only candidates whose probability clears the threshold and keeps the hybrid "
            "order for the rest. Candidate coverage never drops below the baseline order."
        )
        lines.append("")
        lines.append("| Mode | nDCG@10 | Recall@5 | Recall@10 | MRR@10 |")
        lines.append("|---|---|---|---|---|")
        for key, label in (
            ("baseline", "A — hybrid order (baseline)"),
            ("always_on", "J — OpenJev always-on"),
            ("partition", "G — confidence-partitioned"),
            ("gate", "Gate — all-or-nothing switch"),
        ):
            stats = gated.get(key)
            if not stats:
                continue
            lines.append(
                f"| {label} | {_fmt_pct(stats['ndcg@10'])} | {_fmt_pct(stats['recall@5'])} | "
                f"{_fmt_pct(stats['recall@10'])} | {_fmt_pct(stats['mrr@10'])} |"
            )
        lines.append("")
        comparison = gated.get("partition_vs_baseline_ndcg@10")
        if comparison:
            low, high = comparison["ci"]
            lines.append(
                f"- Partitioned vs baseline nDCG@10: **{comparison['mean_diff'] * 100:+.3f} pts** "
                f"(95% CI {low * 100:+.3f} to {high * 100:+.3f}), n={gated.get('n', 0)}"
            )
            lines.append(
                "- This is a post-hoc (exploratory) analysis on already-collected data; the "
                "threshold is fixed at 0.5 a priori of this analysis but was not preregistered."
            )
            lines.append("")

    generation = summary.get("generation") or {}
    if generation:
        lines.append("## 4. Frozen-context answer generation")
        lines.append("")
        lines.append(
            f"Generator: `{', '.join(generation['models'])}`, branch "
            f"`{generation['branch']}`, n={generation['n']}."
        )
        lines.append("")
        if generation.get("mean_f1") is not None:
            lines.append(f"- Mean token F1: **{_fmt_pct(generation['mean_f1'])}**")
            lines.append(f"- Exact match: {_fmt_pct(generation['exact_match'])}")
            lines.append(
                "- Successful answers (F1 >= 0.5): "
                f"{_fmt_pct(generation['success_rate_f1_ge_0.5'])}"
            )
        lines.append(f"- Abstention rate: {_fmt_pct(generation['abstention_rate'])}")
        if generation.get("citation_valid_rate") is not None:
            lines.append(f"- Valid citation rate: {_fmt_pct(generation['citation_valid_rate'])}")
        lines.append(
            f"- Generation latency p50 / p95: {generation['latency_p50_ms']:.1f} ms / "
            f"{generation['latency_p95_ms']:.1f} ms"
        )
        lines.append(
            f"- Tokens: {generation['prompt_tokens']} prompt, "
            f"{generation['completion_tokens']} completion"
        )
        lines.append("")

        gating_rows = summary.get("generation_gating") or []
        if gating_rows:
            lines.append("### Answerability gating: generate only when top-1 probability >= t")
            lines.append("")
            lines.append(
                "Rows below the threshold would not call the generator at all; the remaining "
                "rows are the same frozen-context calls."
            )
            lines.append("")
            lines.append(
                "| Threshold | Coverage | Calls | Calls saved | Mean F1 | Success (F1 >= 0.5) |"
            )
            lines.append("|---|---|---|---|---|---|")
            for row in gating_rows:
                lines.append(
                    f"| {row['threshold']:.2f} | {_fmt_pct(row['coverage'])} | {row['calls']} | "
                    f"{_fmt_pct(row['calls_saved'])} | {_fmt_pct(row['mean_f1'])} | "
                    f"{_fmt_pct(row['success_rate'])} |"
                )
            lines.append("")

    lines.append("## 5. Interpretation limits")
    lines.append("")
    lines.append(
        "- Retrieval metrics (nDCG, Recall@k, MRR) and answer F1 measure different stages; "
        "high Recall@5 does not imply high answer quality."
    )
    lines.append(
        "- This report describes the exact model versions listed above (see the run manifest); "
        "do not generalize to other models or languages."
    )
    lines.append(
        "- XQuAD references are short extractive answers; token F1 is a proxy, not human "
        "factuality adjudication."
    )
    lines.append(
        "- Latency is observed API latency under free tiers and is not a universal throughput "
        "guarantee."
    )
    lines.append(
        "- The free-tier generator is DiffusionGemma 26B, the base model of OpenJev. Generation "
        "consumes frozen contexts, so retrieval and reranking metrics are unaffected, but "
        "end-to-end answer quality is not independent of the reranker's base model."
    )
    lines.append(
        "- Baseline model availability changes over time (several NVIDIA models were retired "
        "on 2026-08-25/26); the exact model versions are recorded in the run manifest."
    )
    lines.append(
        "- Branch T latency is observed through the Vercel AI Gateway free tier under client-side "
        "pacing (~1 request/s); it reflects gateway queueing and retry behavior, not the model's "
        "standalone latency (single-request pilot: ~0.6 s)."
    )
    lines.append("")
    return "\n".join(lines)

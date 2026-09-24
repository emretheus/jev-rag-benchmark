"""Answerability / abstention benchmark for the Jev decision layer.

Two case types per query:

- matched: the query's own frozen top-5 contexts (answerable)
- mismatched: another query's top-5 contexts that do not contain the gold
  passage (not answerable)

Jev answers a noul question ("the passages contain enough information to answer
the question") for every case. The results are then joined with the frozen
generation F1s to simulate skipping generation when answerability is low.
"""

from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from . import data as data_module
from .clients.systemone import extract_noul_probability
from .retrieval import corpus_text
from .run import build_clients

INSTRUCTION = "The passages contain enough information to answer the question"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build_cases(
    results_path: str | Path,
    negatives_per_query: int = 0,
    limit: int | None = None,
    seed: int = 13,
) -> list[dict]:
    rows = _load_jsonl(Path(results_path))
    if limit is not None:
        rows = rows[:limit]
    rng = random.Random(seed)
    cases = []
    for row in rows:
        order = list(row["branches"]["T"]["order"])[:5]
        cases.append(
            {
                "case_id": f"ans-{row['query_id']}",
                "query_id": str(row["query_id"]),
                "kind": "matched",
                "question": row["question"],
                "doc_ids": order,
                "gold": 1.0,
            }
        )
        for index in range(negatives_per_query):
            other = rows[rng.randrange(len(rows))]
            if str(other["query_id"]) == str(row["query_id"]):
                continue
            other_order = list(other["branches"]["T"]["order"])[:5]
            if set(row["gold_doc_ids"]) & set(other_order):
                continue
            cases.append(
                {
                    "case_id": f"ans-{row['query_id']}-neg{index}",
                    "query_id": str(row["query_id"]),
                    "kind": "mismatched",
                    "question": row["question"],
                    "doc_ids": other_order,
                    "gold": 0.0,
                }
            )
    return cases


def run_answerability(
    cfg: dict,
    results_path: str | Path,
    output_path: str | Path | None = None,
    negatives_per_query: int = 0,
    limit: int | None = None,
    concurrency: int = 4,
    run_kind: str = "real",
    seed: int = 13,
) -> Path:
    results_path = Path(results_path)
    rows = _load_jsonl(results_path)
    dataset = str(rows[0]["dataset"])
    corpus, _ = data_module.load_processed(dataset, Path(cfg["paths"]["data_dir"]))
    text_map = {row["doc_id"]: corpus_text(row) for row in corpus}

    cases = build_cases(results_path, negatives_per_query, limit, seed)
    output_path = (
        Path(output_path)
        if output_path
        else results_path.with_name(f"{results_path.stem}-answerability.jsonl")
    )
    done = (
        {str(row["case_id"]) for row in _load_jsonl(output_path)} if output_path.exists() else set()
    )
    pending = [case for case in cases if case["case_id"] not in done]

    clients = build_clients(cfg, run_kind, ["T"])
    systemone = clients["systemone"]
    lock = threading.Lock()
    stats = {"done": 0, "failed": 0}

    def work(case: dict) -> dict:
        passages = [text_map.get(doc_id, "") for doc_id in case["doc_ids"]]
        state = f"Question: {case['question']}\n\nPassages:\n\n" + "\n\n".join(
            f"[{index}] {text}" for index, text in enumerate(passages)
        )
        data, resolved_model, tokens, latency_ms = systemone.ask(
            state, {"answerable": {"type": "noul", "instructions": INSTRUCTION}}
        )
        probability = extract_noul_probability((data.get("answers") or {}).get("answerable") or {})
        return {
            "case_id": case["case_id"],
            "query_id": case["query_id"],
            "kind": case["kind"],
            "gold": case["gold"],
            "probability": probability,
            "resolved_model": resolved_model,
            "input_tokens": tokens,
            "latency_ms": round(latency_ms, 3),
        }

    print(
        f"answerability: {len(pending)} cases "
        f"({sum(1 for c in pending if c['kind'] == 'matched')} matched, "
        f"{sum(1 for c in pending if c['kind'] == 'mismatched')} mismatched)"
    )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(work, case): case for case in pending}
        for future in as_completed(futures):
            case = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001
                with lock:
                    stats["failed"] += 1
                print(f"  failed {case['case_id']}: {type(error).__name__}: {error}", flush=True)
                continue
            with lock:
                with open(output_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["done"] += 1
                if stats["done"] % 100 == 0:
                    print(f"  {stats['done']}/{len(pending)}", flush=True)
    print(f"done: {stats['done']} cases, {stats['failed']} failed -> {output_path}")
    return output_path


def summarize_answerability(
    answerability_path: str | Path,
    generation_path: str | Path | None = None,
) -> dict:
    rows = _load_jsonl(Path(answerability_path))
    if not rows:
        return {}
    probabilities = [float(row["probability"]) for row in rows]
    labels = [float(row["gold"]) for row in rows]
    accuracy = mean(
        [
            1.0 if (p >= 0.5) == (label >= 0.5) else 0.0
            for p, label in zip(probabilities, labels, strict=False)
        ]
    )
    brier = mean((p - label) ** 2 for p, label in zip(probabilities, labels, strict=False))
    bins = 10
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [
            (p, label)
            for p, label in zip(probabilities, labels, strict=False)
            if (low < p <= high) or (index == 0 and p <= 0.0)
        ]
        if not bucket:
            continue
        confidence = mean([p for p, _ in bucket])
        observed = mean([label for _, label in bucket])
        ece += len(bucket) / len(rows) * abs(confidence - observed)

    summary: dict = {
        "n": len(rows),
        "matched": sum(1 for row in rows if row["kind"] == "matched"),
        "mismatched": sum(1 for row in rows if row["kind"] == "mismatched"),
        "accuracy": accuracy,
        "brier": brier,
        "ece_10bin": ece,
        "mean_probability_matched": mean(
            [p for p, label in zip(probabilities, labels, strict=False) if label == 1.0]
        ),
        "mean_probability_mismatched": mean(
            [p for p, label in zip(probabilities, labels, strict=False) if label == 0.0]
        ),
        "risk_coverage": [],
    }
    total_positive = sum(labels)
    for threshold in (0.3, 0.5, 0.7, 0.9):
        kept = [(row, p) for row, p in zip(rows, probabilities, strict=False) if p >= threshold]
        if not kept:
            continue
        summary["risk_coverage"].append(
            {
                "threshold": threshold,
                "coverage": len(kept) / len(rows),
                "precision": sum(float(row["gold"]) for row, _ in kept) / len(kept),
                "recall": sum(float(row["gold"]) for row, _ in kept) / total_positive
                if total_positive
                else 0.0,
            }
        )

    if generation_path and Path(generation_path).exists():
        generation = {str(row["query_id"]): row for row in _load_jsonl(Path(generation_path))}
        matched = [
            row for row in rows if row["kind"] == "matched" and row["query_id"] in generation
        ]
        sweep = []
        for threshold in (0.0, 0.3, 0.5, 0.7, 0.9):
            kept = [row for row in matched if float(row["probability"]) >= threshold]
            if not kept:
                continue
            f1 = [float(generation[row["query_id"]].get("f1") or 0.0) for row in kept]
            success = [1.0 if value >= 0.5 else 0.0 for value in f1]
            sweep.append(
                {
                    "threshold": threshold,
                    "coverage": len(kept) / len(matched),
                    "mean_f1": mean(f1),
                    "success_rate": mean(success),
                    "success_per_1000_queries": mean(success) * len(kept) / len(matched) * 1000,
                }
            )
        summary["generation_gating"] = sweep
    return summary

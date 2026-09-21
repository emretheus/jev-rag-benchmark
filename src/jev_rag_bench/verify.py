"""Jev as a verifier: does it flag answers its own citations do not support?

For every generated answer, the same frozen top-5 contexts plus the answer text
are sent to Jev as a single noul question: "the passages support the factual
claims in the answer". The verification probability is then compared with the
answer's token F1, i.e. whether verification can separate correct from
incorrect answers.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import data as data_module
from .clients.systemone import extract_noul_probability
from .retrieval import corpus_text
from .run import build_clients

VERIFY_TEMPLATE = (
    "The passages support the factual claims made in the answer. "
    "An answer that adds facts not present in the passages is not supported."
)


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _auc(probabilities: list[float], labels: list[int]) -> float | None:
    positives = [p for p, y in zip(probabilities, labels, strict=False) if y == 1]
    negatives = [p for p, y in zip(probabilities, labels, strict=False) if y == 0]
    if not positives or not negatives:
        return None
    order = sorted(range(len(probabilities)), key=lambda index: probabilities[index])
    ranks = [0.0] * len(probabilities)
    for position, index in enumerate(order, start=1):
        ranks[index] = float(position)
    rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label == 1)
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def verify_generations(
    cfg: dict,
    results_path: str | Path,
    generation_path: str | Path,
    branch: str = "T",
    limit: int | None = None,
    concurrency: int = 4,
    run_kind: str = "real",
    output_path: str | Path | None = None,
    top_k: int | None = None,
) -> Path:
    results_path = Path(results_path)
    generation_path = Path(generation_path)
    rows_by_id = {str(row["query_id"]): row for row in _load_jsonl(results_path)}
    generations = _load_jsonl(generation_path)
    if limit is not None:
        generations = generations[:limit]

    dataset = str(next(iter(rows_by_id.values()))["dataset"]) if rows_by_id else "unknown"
    corpus, _ = data_module.load_processed(dataset, Path(cfg["paths"]["data_dir"]))
    text_map = {row["doc_id"]: corpus_text(row) for row in corpus}

    output_path = Path(output_path) if output_path else generation_path.with_name(
        f"{generation_path.stem}-verification.jsonl"
    )
    done: set[str] = set()
    if output_path.exists():
        done = {str(row.get("query_id")) for row in _load_jsonl(output_path)}

    pending = [row for row in generations if str(row["query_id"]) not in done]
    clients = build_clients(cfg, run_kind, [branch])
    systemone = clients["systemone"]
    top_k = int(top_k or cfg["generation"]["top_k"])

    state_lock = threading.Lock()
    stats = {"done": 0, "failed": 0}

    def work(generation: dict) -> dict:
        query_id = str(generation["query_id"])
        result_row = rows_by_id.get(query_id)
        if result_row is None:
            raise ValueError(f"no retrieval row for {query_id}")
        order = list(result_row["branches"][branch]["order"])[:top_k]
        contexts = [text_map.get(doc_id, "") for doc_id in order]
        answer = str(generation.get("text") or "")
        state = (
            f"Answer:\n{answer}\n\nCandidate passages:\n\n"
            + "\n\n".join(f"[{index}] {text}" for index, text in enumerate(contexts))
        )
        data, resolved_model, input_tokens, latency_ms = systemone.ask(
            state,
            {"supported": {"type": "noul", "instructions": VERIFY_TEMPLATE}},
        )
        probability = extract_noul_probability((data.get("answers") or {}).get("supported") or {})
        return {
            "query_id": query_id,
            "branch": branch,
            "probability": probability,
            "f1": generation.get("f1"),
            "em": generation.get("em"),
            "abstained": bool(generation.get("abstained")),
            "resolved_model": resolved_model,
            "input_tokens": input_tokens,
            "latency_ms": round(latency_ms, 3),
        }

    print(f"verifying {len(pending)} answers (concurrency {concurrency}) -> {output_path}")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(work, generation): generation for generation in pending}
        for future in as_completed(futures):
            generation = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001
                with state_lock:
                    stats["failed"] += 1
                print(
                    f"  failed {generation['query_id']}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            with state_lock:
                with open(output_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["done"] += 1
                if stats["done"] % 50 == 0:
                    print(f"  {stats['done']}/{len(pending)} verified", flush=True)

    print(f"done: {stats['done']} verified, {stats['failed']} failed; file: {output_path}")
    return output_path


def summarize_verification(verification_path: str | Path) -> dict:
    rows = _load_jsonl(Path(verification_path))
    scored = [row for row in rows if row.get("f1") is not None and not row.get("abstained")]
    probabilities = [float(row["probability"]) for row in scored]
    labels = [1 if float(row["f1"]) >= 0.5 else 0 for row in scored]
    if not scored:
        return {}
    correct = [p for p, y in zip(probabilities, labels, strict=False) if y == 1]
    wrong = [p for p, y in zip(probabilities, labels, strict=False) if y == 0]
    thresholds = [0.5, 0.7, 0.8, 0.9]
    sweep = []
    for threshold in thresholds:
        flagged = [row for row in scored if float(row["probability"]) >= threshold]
        if not flagged:
            continue
        sweep.append(
            {
                "threshold": threshold,
                "kept": len(flagged),
                "coverage": len(flagged) / len(scored),
                "precision_correct": sum(
                    1 for row in flagged if float(row["f1"]) >= 0.5
                )
                / len(flagged),
                "wrong_answers_flagged": sum(
                    1
                    for row in scored
                    if float(row["f1"]) < 0.5 and float(row["probability"]) < threshold
                ),
                "wrong_answers_total": sum(1 for row in scored if float(row["f1"]) < 0.5),
            }
        )
    return {
        "n": len(scored),
        "mean_probability_correct": sum(correct) / len(correct) if correct else None,
        "mean_probability_wrong": sum(wrong) / len(wrong) if wrong else None,
        "auc": _auc(probabilities, labels),
        "sweep": sweep,
    }

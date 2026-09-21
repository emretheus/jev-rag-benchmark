"""Candidate-order stability audit for the Jev batch-noul reranker.

Batch scoring can depend on the order candidates appear in. This audit re-scores
the same candidates in several shuffles and measures how much the resulting
ranking moves (Spearman, top-5 Jaccard, gold top-5 membership).
"""

from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import data as data_module
from .retrieval import corpus_text
from .run import build_clients


def _spearman(left: list[float], right: list[float]) -> float:
    n = len(left)
    if n < 2:
        return 1.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda index: -values[index])
        rank = [0.0] * n
        for position, index in enumerate(order, start=1):
            rank[index] = float(position)
        return rank

    left_rank, right_rank = ranks(left), ranks(right)
    mean_left = sum(left_rank) / n
    mean_right = sum(right_rank) / n
    cov = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(left_rank, right_rank, strict=False)
    )
    var_left = sum((a - mean_left) ** 2 for a in left_rank) ** 0.5
    var_right = sum((b - mean_right) ** 2 for b in right_rank) ** 0.5
    if var_left == 0 or var_right == 0:
        return 1.0
    return cov / (var_left * var_right)


def stability_audit(
    cfg: dict,
    results_path: str | Path,
    branch: str = "T",
    limit: int = 100,
    permutations: int = 3,
    concurrency: int = 4,
    run_kind: str = "real",
    seed: int = 13,
) -> dict:
    results_path = Path(results_path)
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:limit]
    if not rows:
        raise ValueError(f"no rows in {results_path}")

    dataset = str(rows[0]["dataset"])
    corpus, _ = data_module.load_processed(dataset, Path(cfg["paths"]["data_dir"]))
    text_map = {row["doc_id"]: corpus_text(row) for row in corpus}
    clients = build_clients(cfg, run_kind, [branch])
    systemone = clients["systemone"]

    state_lock = threading.Lock()
    stats = {"done": 0, "failed": 0}

    def work(row: dict, permutation: int) -> tuple[dict, int, list[float], list[str]]:
        candidates = [entry["doc_id"] for entry in row["candidates"]]
        rng = random.Random(seed + permutation * 7919 + hash(row["query_id"]) % 10_000)
        order = list(range(len(candidates)))
        rng.shuffle(order)
        shuffled_docs = [candidates[index] for index in order]
        passages = [text_map.get(doc_id, "") for doc_id in shuffled_docs]
        result = systemone.score_relevance(row["question"], passages)
        aligned = [0.0] * len(candidates)
        for shuffled_position, original_index in enumerate(order):
            aligned[original_index] = result.probabilities[shuffled_position]
        return row, permutation, aligned, candidates

    tasks = [(row, permutation) for row in rows for permutation in range(permutations)]
    print(f"stability audit: {len(rows)} queries x {permutations} shuffles = {len(tasks)} calls")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, row, permutation) for row, permutation in tasks]
        collected: dict[str, dict[int, list[float]]] = {}
        for future in as_completed(futures):
            try:
                row, permutation, aligned, _candidates = future.result()
            except Exception as error:  # noqa: BLE001
                with state_lock:
                    stats["failed"] += 1
                print(f"  failed: {type(error).__name__}: {error}", flush=True)
                continue
            with state_lock:
                collected.setdefault(str(row["query_id"]), {})[permutation] = aligned
                stats["done"] += 1
                if stats["done"] % 50 == 0:
                    print(f"  {stats['done']}/{len(tasks)} calls done", flush=True)

    spearmans: list[float] = []
    jaccards: list[float] = []
    membership_changes = 0
    queries_evaluated = 0
    for row in rows:
        query_id = str(row["query_id"])
        variants = collected.get(query_id)
        if not variants or len(variants) < 2:
            continue
        queries_evaluated += 1
        gold = set(row["gold_doc_ids"])
        candidates = [entry["doc_id"] for entry in row["candidates"]]
        top5_sets = []
        for permutation in sorted(variants):
            scores = variants[permutation]
            for other in sorted(variants):
                if other <= permutation:
                    continue
                spearmans.append(_spearman(scores, variants[other]))
            ranked = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
            top5_sets.append({candidates[index] for index in ranked[:5]})
        for index in range(len(top5_sets)):
            for other in range(index + 1, len(top5_sets)):
                intersection = len(top5_sets[index] & top5_sets[other])
                union = len(top5_sets[index] | top5_sets[other])
                jaccards.append(intersection / union if union else 1.0)
        gold_membership = {bool(gold & top5) for top5 in top5_sets}
        if len(gold_membership) > 1:
            membership_changes += 1

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "dataset": dataset,
        "branch": branch,
        "queries": queries_evaluated,
        "permutations": permutations,
        "calls": stats["done"],
        "failed_calls": stats["failed"],
        "mean_spearman": mean(spearmans),
        "mean_top5_jaccard": mean(jaccards),
        "queries_with_gold_membership_change": membership_changes,
        "gold_membership_change_rate": membership_changes / queries_evaluated
        if queries_evaluated
        else 0.0,
    }


def write_stability(audit: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{audit['dataset']}-order-stability.json"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md_path = output_dir / f"{audit['dataset']}-order-stability.md"
    md_path.write_text(
        "\n".join(
            [
                f"# Candidate-order stability: {audit['dataset']} (branch {audit['branch']})",
                "",
                f"- Queries: {audit['queries']} · shuffles per query: {audit['permutations']}",
                f"- Calls: {audit['calls']} ({audit['failed_calls']} failed)",
                f"- Mean Spearman rank correlation between shuffles: "
                f"**{audit['mean_spearman']:.3f}**",
                f"- Mean top-5 Jaccard between shuffles: **{audit['mean_top5_jaccard']:.3f}**",
                f"- Queries where gold top-5 membership changed across shuffles: "
                f"**{audit['queries_with_gold_membership_change']} "
                f"({audit['gold_membership_change_rate'] * 100:.1f}%)**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, md_path

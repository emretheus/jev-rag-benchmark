from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from . import data as data_module
from .clients.nvidia import NvidiaEmbeddings, NvidiaReranker, OpenAIChat
from .clients.systemone import SystemOneClient
from .config import api_key
from .fixtures import FixtureEmbeddings, FixtureGenerator, FixtureReranker, FixtureSystemOne
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k
from .rerank import ALL_BRANCHES, DEFAULT_BRANCHES, run_branch
from .retrieval import Candidate, Retriever, corpus_text


def build_clients(cfg: dict, run_kind: str, branches: list[str] | None = None) -> dict:
    branches = branches or DEFAULT_BRANCHES
    if run_kind == "fixture":
        return {
            "embeddings": FixtureEmbeddings(),
            "systemone": FixtureSystemOne(),
            "systemone_typesafe": FixtureSystemOne(),
            "reranker": FixtureReranker(),
            "generator": FixtureGenerator(),
        }

    needed = ["embedding", "reranker_nvidia", "generator"]
    if "J" in branches:
        needed.append("systemone")
    if "T" in branches:
        needed.append("systemone_typesafe")
    sections = {key: cfg["models"][key] for key in needed}

    missing = sorted(
        {
            str(section.get("api_key_env"))
            for section in sections.values()
            if section.get("api_key_env") and not api_key(section)
        }
    )
    if missing:
        raise RuntimeError(
            f"missing API keys for: {', '.join(missing)}. "
            "Set them in the environment (see .env.example) or run with --fixture."
        )

    clients: dict = {
        "embeddings": NvidiaEmbeddings(
            api_key(cfg["models"]["embedding"]),
            cfg["models"]["embedding"]["name"],
            base_url=cfg["models"]["embedding"]["base_url"],
            input_type=cfg["models"]["embedding"].get("input_type", ""),
            batch_size=int(cfg["models"]["embedding"].get("batch_size", 32)),
            requests_per_minute=float(cfg["models"]["embedding"].get("requests_per_minute", 40)),
        ),
        "reranker": NvidiaReranker(
            api_key(cfg["models"]["reranker_nvidia"]),
            cfg["models"]["reranker_nvidia"]["name"],
            base_url=cfg["models"]["reranker_nvidia"]["base_url"],
            requests_per_minute=float(
                cfg["models"]["reranker_nvidia"].get("requests_per_minute", 40)
            ),
        ),
        "generator": OpenAIChat(
            api_key(cfg["models"]["generator"]),
            cfg["models"]["generator"]["name"],
            base_url=cfg["models"]["generator"]["base_url"],
            max_tokens=int(cfg["models"]["generator"].get("max_tokens", 400)),
            temperature=float(cfg["models"]["generator"].get("temperature", 0.0)),
            requests_per_minute=float(cfg["models"]["generator"].get("requests_per_minute", 40)),
        ),
        "systemone": None,
        "systemone_typesafe": None,
    }

    def _systemone_client(section: dict) -> SystemOneClient:
        return SystemOneClient(
            section["base_url"],
            api_key(section),
            section["model"],
            section["instruction_template"],
            decisions_path=section.get("decisions_path", "/v1/systemone"),
            max_passage_chars=int(section.get("max_passage_chars", 4000)),
            max_state_chars=int(section.get("max_state_chars", 200000)),
            requests_per_minute=float(section.get("requests_per_minute", 600)),
        )

    if "J" in branches:
        clients["systemone"] = _systemone_client(cfg["models"]["systemone"])
    if "T" in branches:
        clients["systemone_typesafe"] = _systemone_client(cfg["models"]["systemone_typesafe"])
    return clients


def _branch_row(output) -> dict:
    return {
        "order": output.order,
        "scores": [round(score, 6) for score in output.scores],
        "probs": [round(p, 6) for p in output.probs] if output.probs is not None else None,
        "latency_ms": round(output.latency_ms, 3),
        "resolved_model": output.resolved_model,
        "input_tokens": output.input_tokens,
        "state_chars": output.state_chars,
        "cost_usd": output.cost_usd,
    }


def _branch_metrics(output, gold: set[str]) -> dict:
    return {
        "recall@5": recall_at_k(output.order, gold, 5),
        "recall@10": recall_at_k(output.order, gold, 10),
        "ndcg@10": ndcg_at_k(output.order, gold, 10),
        "mrr@10": mrr_at_k(output.order, gold, 10),
    }


def run_benchmark(
    cfg: dict,
    dataset: str,
    branches: list[str] | None = None,
    run_kind: str = "real",
    limit: int | None = None,
    resume: bool = True,
    output_path: str | Path | None = None,
) -> Path:
    branches = [branch.upper() for branch in (branches or DEFAULT_BRANCHES)]
    for branch in branches:
        if branch not in ALL_BRANCHES:
            raise ValueError(f"unknown branch: {branch}")

    data_root = Path(cfg["paths"]["data_dir"])
    corpus, queries = data_module.load_processed(dataset, data_root)
    if limit is not None:
        queries = queries[:limit]

    clients = build_clients(cfg, run_kind, branches)
    retriever = Retriever(dataset, corpus, cfg, clients["embeddings"])
    retriever.prepare()

    if output_path is None:
        results_dir = Path(cfg["paths"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            results_dir / f"{dataset}-{'-'.join(b.lower() for b in branches)}-{run_kind}.jsonl"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _check_request_caps(cfg, len(queries), branches)

    done: set[str] = set()
    if resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(str(json.loads(line)["query_id"]))

    run_id = (
        f"{dataset}-{'-'.join(b.lower() for b in branches)}-{run_kind}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    processed = 0
    skipped = 0
    systemone_tokens = 0
    typesafe_tokens = 0

    print(f"run_id: {run_id}")
    print(f"queries: {len(queries)} (already done: {len(done)}) -> {output_path}")

    with open(output_path, "a", encoding="utf-8") as handle:
        for query in queries:
            query_id = str(query["query_id"])
            if query_id in done:
                skipped += 1
                continue

            candidates = retriever.search(query["question"])
            passages = [retriever.text_map[candidate.doc_id] for candidate in candidates]
            gold = set(query.get("gold_doc_ids") or [])

            branch_rows: dict[str, dict] = {}
            branch_metrics: dict[str, dict] = {}
            for branch in branches:
                output = run_branch(
                    branch,
                    query["question"],
                    candidates,
                    passages,
                    clients["systemone"],
                    clients["reranker"],
                    clients["systemone_typesafe"],
                )
                branch_rows[branch] = _branch_row(output)
                branch_metrics[branch] = _branch_metrics(output, gold)
                if output.input_tokens and branch == "J":
                    systemone_tokens += output.input_tokens
                if output.input_tokens and branch == "T":
                    typesafe_tokens += output.input_tokens

            _check_token_caps(cfg, systemone_tokens, typesafe_tokens)

            row = {
                "run_id": run_id,
                "run_kind": run_kind,
                "dataset": dataset,
                "query_id": query_id,
                "question": query["question"],
                "gold_doc_ids": list(gold),
                "answers": query.get("answers") or [],
                "candidates": [
                    {
                        "doc_id": candidate.doc_id,
                        "bm25_rank": candidate.bm25_rank,
                        "dense_rank": candidate.dense_rank,
                        "rrf_rank": candidate.hybrid_rank,
                        "bm25_score": round(candidate.bm25_score, 6),
                        "dense_score": round(candidate.dense_score, 6),
                    }
                    for candidate in candidates
                ],
                "gold_in_candidates": bool(gold & {candidate.doc_id for candidate in candidates}),
                "branches": branch_rows,
                "metrics": branch_metrics,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
            if processed % 25 == 0:
                print(f"  {processed + skipped}/{len(queries)} rows written", flush=True)

    manifest = _build_manifest(
        cfg, dataset, branches, run_kind, output_path, run_id, len(queries), skipped
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: {processed} new rows, {skipped} skipped; manifest: {manifest_path}")
    return output_path


def add_branch(
    cfg: dict,
    results_path: str | Path,
    branch: str,
    run_kind: str = "real",
    limit: int | None = None,
    concurrency: int = 1,
) -> Path:
    """Add one branch to an existing results file without re-running other branches.

    Rows are checkpointed atomically every 100 completions, so an interrupted run
    resumes without losing work. Failed requests are counted and reported; rows
    without the branch are simply retried on the next invocation.
    """
    branch = branch.upper()
    if branch not in ALL_BRANCHES:
        raise ValueError(f"unknown branch: {branch}")
    results_path = Path(results_path)
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows in {results_path}")
    if all(branch in row.get("branches", {}) for row in rows):
        print(f"branch {branch} already present in all {len(rows)} rows; nothing to do")
        return results_path

    dataset = str(rows[0]["dataset"])
    data_root = Path(cfg["paths"]["data_dir"])
    corpus, _ = data_module.load_processed(dataset, data_root)
    text_map = {row["doc_id"]: corpus_text(row) for row in corpus}

    todo = [row for row in rows if branch not in row.get("branches", {})]
    if limit is not None:
        todo = todo[:limit]
    _check_request_caps(cfg, len(todo), [branch])

    clients = build_clients(cfg, run_kind, [branch])
    state_lock = threading.Lock()
    stats = {"done": 0, "failed": 0, "tokens": 0}
    failures: list[str] = []

    def checkpoint() -> None:
        temp_path = results_path.with_suffix(".jsonl.tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp_path.replace(results_path)

    def work(row: dict):
        candidates = [
            Candidate(
                doc_id=entry["doc_id"],
                bm25_rank=int(entry.get("bm25_rank", 0)),
                dense_rank=int(entry.get("dense_rank", 0)),
                hybrid_rank=int(entry.get("rrf_rank", 0)),
                bm25_score=float(entry.get("bm25_score", 0.0)),
                dense_score=float(entry.get("dense_score", 0.0)),
            )
            for entry in row["candidates"]
        ]
        if branch == "A":
            candidates.sort(key=lambda candidate: (candidate.hybrid_rank, candidate.doc_id))
        passages = [text_map.get(candidate.doc_id, "") for candidate in candidates]
        return run_branch(
            branch,
            row["question"],
            candidates,
            passages,
            clients["systemone"],
            clients["reranker"],
            clients["systemone_typesafe"],
        )

    def apply(row: dict, output) -> None:
        row.setdefault("branches", {})[branch] = _branch_row(output)
        row.setdefault("metrics", {})[branch] = _branch_metrics(
            output, set(row["gold_doc_ids"])
        )
        stats["tokens"] += output.input_tokens or 0
        _check_token_caps(
            cfg,
            stats["tokens"] if branch == "J" else 0,
            stats["tokens"] if branch == "T" else 0,
        )

    print(
        f"adding branch {branch} to {len(todo)} rows in {results_path} "
        f"(concurrency {concurrency})"
    )

    def record_success(row: dict, output) -> None:
        with state_lock:
            apply(row, output)
            stats["done"] += 1
            if stats["done"] % 100 == 0:
                checkpoint()
                print(f"  {stats['done']}/{len(todo)} rows updated", flush=True)

    def record_failure(row: dict, error: Exception) -> None:
        with state_lock:
            stats["failed"] += 1
            failures.append(str(row["query_id"]))
        print(f"  failed {row['query_id']}: {type(error).__name__}: {error}", flush=True)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(work, row): row for row in todo}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    record_success(row, future.result())
                except Exception as error:  # noqa: BLE001
                    record_failure(row, error)
    else:
        for row in todo:
            try:
                record_success(row, work(row))
            except Exception as error:  # noqa: BLE001
                record_failure(row, error)

    checkpoint()
    print(
        f"done: branch {branch} added to {stats['done']} rows, {stats['failed']} failed; "
        f"file rewritten: {results_path}"
    )
    if failures:
        print(f"failed query ids (rerun to retry): {', '.join(failures[:10])}"
              + (" ..." if len(failures) > 10 else ""))
    return results_path


def _check_request_caps(cfg: dict, query_count: int, branches: list[str]) -> None:
    if "J" in branches:
        cap = int(cfg["safety"]["max_systemone_requests"])
        if query_count > cap:
            raise RuntimeError(
                f"planned OpenJev requests ({query_count}) exceed safety cap ({cap}); "
                "raise safety.max_systemone_requests if this is intentional"
            )
    if "T" in branches:
        cap = int(cfg["safety"].get("max_typesafe_requests", 5000))
        if query_count > cap:
            raise RuntimeError(
                f"planned Jev 1.13 requests ({query_count}) exceed safety cap ({cap}); "
                "raise safety.max_typesafe_requests if this is intentional"
            )


def _check_token_caps(cfg: dict, systemone_tokens: int, typesafe_tokens: int) -> None:
    cap = int(cfg["safety"]["max_systemone_input_tokens"])
    if systemone_tokens > cap:
        raise RuntimeError(
            f"OpenJev input tokens exceeded safety cap ({cap}); "
            "raise safety.max_systemone_input_tokens if this is intentional"
        )
    cap_t = int(cfg["safety"].get("max_typesafe_input_tokens", 16000000))
    if typesafe_tokens > cap_t:
        raise RuntimeError(
            f"Jev 1.13 input tokens exceeded safety cap ({cap_t}); "
            "raise safety.max_typesafe_input_tokens if this is intentional"
        )


def _build_manifest(
    cfg: dict,
    dataset: str,
    branches: list[str],
    run_kind: str,
    output_path: Path,
    run_id: str,
    queries_total: int,
    skipped: int,
) -> dict:
    rows = []
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    systemone_tokens = 0
    typesafe_tokens = 0
    typesafe_cost = 0.0
    rerank_tokens = 0
    resolved_models: dict[str, set[str]] = {}
    for row in rows:
        for branch, branch_row in row.get("branches", {}).items():
            model = branch_row.get("resolved_model")
            if model:
                resolved_models.setdefault(branch, set()).add(model)
            tokens = branch_row.get("input_tokens") or 0
            cost = branch_row.get("cost_usd")
            if branch == "J":
                systemone_tokens += tokens
            elif branch == "T":
                typesafe_tokens += tokens
                typesafe_cost += float(cost or 0.0)
            elif branch == "N":
                rerank_tokens += tokens

    dataset_meta_path = Path(cfg["paths"]["data_dir"]) / "processed" / dataset / "meta.json"
    dataset_meta = (
        json.loads(dataset_meta_path.read_text(encoding="utf-8"))
        if dataset_meta_path.exists()
        else None
    )

    return {
        "run_id": run_id,
        "run_kind": run_kind,
        "dataset": dataset,
        "branches": branches,
        "results_file": str(output_path),
        "queries_total": queries_total,
        "queries_in_file": len(rows),
        "queries_skipped_this_run": skipped,
        "completed_at": datetime.now(UTC).isoformat(),
        "models": {
            "embedding": cfg["models"]["embedding"]["name"],
            "systemone_requested": cfg["models"]["systemone"]["model"],
            "systemone_resolved": sorted(resolved_models.get("J", [])),
            "systemone_typesafe_requested": cfg["models"]["systemone_typesafe"]["model"],
            "systemone_typesafe_resolved": sorted(resolved_models.get("T", [])),
            "reranker_requested": cfg["models"]["reranker_nvidia"]["name"],
            "reranker_resolved": sorted(resolved_models.get("N", [])),
            "generator": cfg["models"]["generator"]["name"],
        },
        "totals": {
            "systemone_input_tokens": systemone_tokens,
            "systemone_typesafe_input_tokens": typesafe_tokens,
            "systemone_typesafe_cost_usd": round(typesafe_cost, 6),
            "nvidia_rerank_prompt_tokens": rerank_tokens,
            "systemone_requests": sum(1 for row in rows if "J" in row.get("branches", {})),
            "systemone_typesafe_requests": sum(1 for row in rows if "T" in row.get("branches", {})),
        },
        "dataset_meta": dataset_meta,
        "config": cfg,
    }

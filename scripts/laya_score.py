"""Score frozen candidates with Laya (local, pointwise) for the RAG benchmark.

Run with the dedicated Laya environment (laya-mlx is not a dependency of the
main package):

    uv venv .venv-laya --python 3.12
    uv pip install --python .venv-laya laya-mlx
    .venv-laya/bin/python scripts/laya_score.py \
        results/xquad-en-a-n-t-real.jsonl results/xquad-en-laya-scores.jsonl

Laya's English checkpoint reads 512 tokens per question, so candidates are
scored one at a time with the passage truncated to --max-chars.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

INSTRUCTION = "Passage contains information that is needed to answer the question"


def load_env_token() -> str | None:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HF_TOKEN") and "=" in line:
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=1100)
    parser.add_argument("--model", default="convaiinnovations/laya")
    args = parser.parse_args()

    token = load_env_token()
    if token:
        os.environ["HF_TOKEN"] = token

    import laya_mlx

    results_path = Path(args.results)
    output_path = Path(args.output)
    rows = load_jsonl(results_path)
    if args.limit:
        rows = rows[: args.limit]
    dataset = rows[0]["dataset"]
    corpus_path = Path("data/processed") / dataset / "corpus.jsonl"
    text_map = {}
    for row in load_jsonl(corpus_path):
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        text_map[str(row["doc_id"])] = f"{title}. {text}".strip() if title else text

    done = (
        {str(row["query_id"]) for row in load_jsonl(output_path)}
        if output_path.exists()
        else set()
    )
    pending = [row for row in rows if str(row["query_id"]) not in done]

    agent = laya_mlx.Agent(args.model)
    question = {"relevant": {"type": "noul", "instructions": INSTRUCTION}}
    print(
        f"laya pointwise scoring: {len(pending)} queries "
        f"x {len(rows[0]['candidates'])} candidates"
    )
    print(f"model: {args.model} | max passage chars: {args.max_chars}")

    started = time.perf_counter()
    for index, row in enumerate(pending, start=1):
        probabilities = []
        latencies = []
        for candidate in row["candidates"]:
            passage = text_map.get(candidate["doc_id"], "")[: args.max_chars]
            state = f"Question: {row['question']}\n\nPassage: {passage}"
            call_started = time.perf_counter()
            result = agent.predict(state, question)
            latencies.append((time.perf_counter() - call_started) * 1000)
            probabilities.append(float(result["answers"]["relevant"]["noul"]))
        record = {
            "query_id": row["query_id"],
            "dataset": dataset,
            "model": args.model,
            "mode": "pointwise",
            "max_chars": args.max_chars,
            "probs": probabilities,
            "latency_ms": sum(latencies) / len(latencies),
        }
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if index % 25 == 0 or index == len(pending):
            elapsed = time.perf_counter() - started
            print(f"  {index}/{len(pending)} queries ({elapsed:.0f}s)", flush=True)

    print(f"done -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

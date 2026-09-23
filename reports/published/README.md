---
license: mit
language:
  - en
pretty_name: "Jev RAG Benchmark (English): scifact, xquad-en"
size_categories:
  - n<10K
task_categories:
  - question-answering
  - text-retrieval
tags:
  - rag
  - reranking
  - benchmark
  - jev
  - openjev
  - typesafe
  - calibration
  - system-one
  - retrieval-evaluation
---

# Jev RAG Benchmark (English)

Frozen-candidate-pool evaluation of **TypeSafe Jev 1.13** as the reranking
and decision layer of a RAG pipeline, compared with a **NVIDIA cross-encoder**
and with no reranking at all, on English XQuAD and SciFact. Every published
run used free tiers (total cost: $0).

This repository contains the raw per-query artifacts, per-run reports with
paired bootstrap confidence intervals, calibration tables, and a plain-text
leaderboard.

## Reranking (nDCG@10, identical frozen top-20 candidates)

| Dataset | n | Method | nDCG@10 | Recall@5 | MRR@10 | Rerank p50 |
|---|---|---|---|---|---|---|
| scifact | 300 | A — no reranker (hybrid order) | 71.67% | 79.67% | 68.75% | 0 ms |
| scifact | 300 | T — TypeSafe Jev 1.13 batch noul | 79.29% | 85.67% | 77.03% | 4044 ms |
| scifact | 300 | N — NVIDIA cross-encoder reranker | 78.70% | 87.33% | 76.16% | 307 ms |
| xquad-en | 1190 | A — no reranker (hybrid order) | 98.11% | 99.58% | 97.60% | 0 ms |
| xquad-en | 1190 | T — TypeSafe Jev 1.13 batch noul | 98.93% | 99.66% | 98.67% | 4000 ms |
| xquad-en | 1190 | N — NVIDIA cross-encoder reranker | 99.37% | 99.66% | 99.27% | 409 ms |

## Probability calibration (candidate-level relevance)

| Dataset | Model | ECE (10 bin) | Brier | Top-1 accuracy | Top-1 confidence (correct) | Top-1 confidence (wrong) |
|---|---|---|---|---|---|---|
| scifact | TypeSafe Jev 1.13 batch noul | 0.0625 | 0.0366 | 71.00% | 0.812 | 0.549 |
| xquad-en | TypeSafe Jev 1.13 batch noul | 0.0133 | 0.0045 | 97.90% | 0.960 | 0.740 |

## RAG optimization mode (confidence-partitioned, fixed t = 0.50)

| Dataset | Model | Threshold | A baseline nDCG@10 | Always-on nDCG@10 | Partitioned nDCG@10 | Delta vs baseline | 95% CI |
|---|---|---|---|---|---|---|---|
| scifact | TypeSafe Jev 1.13 batch noul | 0.50 | 71.67% | 79.29% | 75.81% | +4.14 pts | +2.13 to +6.29 |
| xquad-en | TypeSafe Jev 1.13 batch noul | 0.50 | 98.11% | 98.93% | 99.00% | +0.89 pts | +0.45 to +1.36 |

## Files

- `results/*.jsonl` — one row per query: candidates, every branch's order,
  probabilities, token usage, latency, resolved model ids.
- `results/*.manifest.json` — config snapshot, dataset SHA-256, totals.
- `<run>/report.md` — full report: paired CIs, reliability bins,
  risk-coverage, interpretation limits.
- `<run>/summary.json` / `summary.csv` — machine-readable aggregates.
- `leaderboard.txt` — plain-text tables.
- `toolbench/` — tool-calling governance results (tool selection,
  call approval, argument validation, injection detection) for Jev vs
  Laya, plus charts.
- `charts/` — SVG/PNG charts for the tables above.

## Reproduce

```bash
git clone https://github.com/emretheus/jev-rag-benchmark
uv sync --extra dev && uv run jev-rag data prepare --dataset all
uv run jev-rag estimate --dataset xquad-en   # zero API calls
```

Raw datasets are **not** redistributed here: XQuAD (CC BY-SA 4.0) and
SciFact/BEIR (abstracts ODC-By 1.0, annotations CC BY 4.0) are downloaded
by the harness. This repository's own artifacts are MIT.

## Links

- Code and protocol: https://github.com/emretheus/jev-rag-benchmark

_Results describe the exact resolved model versions recorded in each run manifest. This benchmark is independent and not affiliated with TypeSafe AI._
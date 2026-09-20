---
license: mit
language:
  - en
pretty_name: Jev RAG Benchmark (English): scifact, xquad-en
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
and decision layer of a RAG pipeline, compared with the free open-weights
**OpenJev** model and a **NVIDIA cross-encoder**, on English XQuAD and
SciFact. Every published run used free tiers (total cost: $0).

This repository contains the raw per-query artifacts, per-run reports with
paired bootstrap confidence intervals, calibration tables, and a plain-text
leaderboard.

## Reranking (nDCG@10, identical frozen top-20 candidates)

| Dataset | n | Method | nDCG@10 | Recall@5 | MRR@10 | Rerank p50 |
|---|---|---|---|---|---|---|
| scifact | 300 | A — no reranker (hybrid order) | 71.67% | 79.67% | 68.75% | 0 ms |
| scifact | 300 | J — OpenJev 0.1 batch noul (Codiv) | 63.92% | 68.67% | 62.59% | 1220 ms |
| scifact | 300 | N — NVIDIA reranker | 78.70% | 87.33% | 76.16% | 307 ms |
| scifact | 300 | T — TypeSafe Jev 1.13 batch noul (real Jev) | 79.29% | 85.67% | 77.03% | 4044 ms |
| xquad-en | 1190 | A — no reranker (hybrid order) | 98.11% | 99.58% | 97.60% | 0 ms |
| xquad-en | 1190 | J — OpenJev 0.1 batch noul (Codiv) | 98.03% | 98.82% | 97.65% | 923 ms |
| xquad-en | 1190 | N — NVIDIA reranker | 99.37% | 99.66% | 99.27% | 409 ms |
| xquad-en | 1190 | T — TypeSafe Jev 1.13 batch noul (real Jev) | 98.93% | 99.66% | 98.67% | 4000 ms |

## RAG optimization mode (confidence-partitioned, fixed t = 0.50)

| Dataset | Threshold | A baseline nDCG@10 | J always-on nDCG@10 | G partitioned nDCG@10 | Delta vs baseline | 95% CI |
|---|---|---|---|---|---|---|
| scifact | 0.50 | 71.67% | 63.92% | 71.49% | -0.18 pts | -1.93 to +1.54 |
| xquad-en | 0.50 | 98.11% | 98.03% | 98.76% | +0.66 pts | +0.27 to +1.07 |

## Files

- `results/*.jsonl` — one row per query: candidates, every branch's order,
  probabilities, token usage, latency, resolved model ids.
- `results/*.manifest.json` — config snapshot, dataset SHA-256, totals.
- `<run>/report.md` — full report: paired CIs, reliability bins,
  risk-coverage, interpretation limits.
- `<run>/summary.json` / `summary.csv` — machine-readable aggregates.
- `leaderboard.txt` — plain-text tables.

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

_OpenJev is an independent open-weights model served by Codiv; it is not TypeSafe's Jev. Results describe the exact resolved model versions recorded in each run manifest._
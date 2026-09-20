# Jev RAG Benchmark (English): Jev 1.13 vs OpenJev vs NVIDIA

A free, reproducible, framework-neutral benchmark of **TypeSafe Jev 1.13 as the
decision and reranking layer of a RAG pipeline** — on English data, with frozen
candidate pools, paired confidence intervals, and calibration analysis. The
free open-weights OpenJev model and a NVIDIA cross-encoder are compared on the
identical candidate pools.

It answers one question honestly: **does the model actually improve the system?**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Cost](https://img.shields.io/badge/benchmark%20cost-%240-brightgreen)
![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen)

## Key findings (2026-09-20)

| Stage | Result |
|---|---|
| **Real Jev 1.13 as reranker** | Improves nDCG@10 over the unranked baseline on both datasets: **+0.82 pts** on XQuAD-EN (95% CI +0.35 to +1.31) and **+7.62 pts** on SciFact (CI +4.88 to +10.38) |
| **Real Jev vs free OpenJev clone** | Real Jev is clearly better: **+0.90 pts** (XQuAD) and **+15.37 pts** (SciFact) |
| **Real Jev vs NVIDIA cross-encoder** | Statistically tied on SciFact (+0.59 pts, CI crosses zero); NVIDIA leads on XQuAD by 0.45 pts |
| **OpenJev (free clone) as reranker** | Does **not** help: flat on XQuAD, **−7.75 pts harm** on SciFact |
| **OpenJev as a decision layer** | Calibrated probabilities pay off: confidence-partitioned reranking **+0.66 pts** on XQuAD (CI +0.27 to +1.07); answerability gating cuts generation calls by **31%** while raising answer F1 (31.9% → 33.0%) |
| **Cost of every run above** | **$0** (Codiv free tier, NVIDIA NIM trial, Vercel AI Gateway free tier) |

Two of these analyses (confidence gating, real-Jev comparison) are labeled
**exploratory** in [`docs/protocol.md`](docs/protocol.md) — they were added
after the first ranking run. Everything is reported with that label.

## Results

Auto-generated from real runs by `uv run jev-rag publish`. Never
hand-edited; fixture runs are refused.

<!-- RESULTS:START -->

_Generated 2026-09-20 17:09 UTC from real runs (scifact, xquad-en). Fixture runs are never published._
_Regenerate with:_ `uv run jev-rag publish`

### Reranking (frozen top-20 candidates, same pool for every method)

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

### OpenJev calibration (candidate-level relevance probabilities)

| Dataset | ECE (10 bin) | Brier | Top-1 accuracy | Top-1 confidence (correct) | Top-1 confidence (wrong) |
|---|---|---|---|---|---|
| scifact | 0.0906 | 0.0521 | 57.67% | 0.606 | 0.514 |
| xquad-en | 0.0245 | 0.0086 | 96.72% | 0.765 | 0.455 |

### RAG optimization mode (confidence-partitioned OpenJev, fixed t = 0.50)

| Dataset | Threshold | A baseline nDCG@10 | J always-on nDCG@10 | G partitioned nDCG@10 | Delta vs baseline | 95% CI |
|---|---|---|---|---|---|---|
| scifact | 0.50 | 71.67% | 63.92% | 71.49% | -0.18 pts | -1.93 to +1.54 |
| xquad-en | 0.50 | 98.11% | 98.03% | 98.76% | +0.66 pts | +0.27 to +1.07 |

### Frozen-context answer generation (OpenJev top-5)

| Dataset | Generator | Token F1 | Exact match | F1 >= 0.5 | Abstention | Valid citations |
|---|---|---|---|---|---|---|
| xquad-en | diffusiongemma-26b | 31.89% | 2.35% | 19.16% | 2.77% | 99.85% |

_Metrics measure different stages: retrieval quality (nDCG, Recall) does not imply answer quality. See the per-run reports for confidence intervals, reliability bins, and risk-coverage tables._

<!-- RESULTS:END -->

## Pipeline

```
queries ──► BM25 (top-100) ─┐
                            ├─► RRF fusion ─► frozen top-20 candidates
queries ──► NVIDIA embeddings┘                       │
                                                     ├─► A: hybrid order (no reranking, baseline)
                                                     ├─► J: OpenJev 0.1 batch noul (Codiv, free)
                                                     ├─► N: NVIDIA cross-encoder reranker
                                                     └─► T: TypeSafe Jev 1.13 batch noul (real Jev)
                                                     │
                            frozen top-5 of branch J ┴─► answer generation (free-tier model)
```

Every branch sees the **identical frozen candidate pool**. Reranking, branch
comparison, and generation are measured separately; generation replays over
frozen contexts so a fresh retrieval draw cannot contaminate it.

## Branches

| Branch | Method | Provider / cost |
|---|---|---|
| A | No reranker; hybrid RRF order is used as-is | local, $0 |
| J | OpenJev `openjev-0.1` batch **noul** (one request per query, one yes/no question per candidate) | Codiv, free |
| N | NVIDIA `llama-nemotron-rerank-vl-1b-v2` cross-encoder logits | NVIDIA NIM trial, $0 |
| T | TypeSafe `typesafe-ai/jev` (= Jev 1.13) batch **noul**, the real model | [Vercel AI Gateway](configs/vercel.yaml) (free tier at run time) or OpenRouter |

Branch T is added to existing results without re-running the free branches:

```bash
uv run jev-rag --config configs/vercel.yaml add-branch \
  --results results/xquad-en-a-j-n-real.jsonl --branch T --concurrency 4
```

## Quickstart (free stack)

```bash
uv sync --extra dev
cp .env.example .env          # add keys: Codiv (free), NVIDIA (free). OpenRouter/Vercel only for branch T.

uv run jev-rag doctor
uv run jev-rag data prepare --dataset all
uv run jev-rag estimate --dataset xquad-en        # zero API calls

# retrieval + reranking branches A, J, N
uv run jev-rag run --dataset xquad-en --branches A,J,N

# frozen-context answer generation on the OpenJev top-5
uv run jev-rag generate --results results/xquad-en-a-j-n-real.jsonl --branch J

# report + README table + shareable artifacts
uv run jev-rag report --results results/xquad-en-a-j-n-real.jsonl \
  --generation results/xquad-en-a-j-n-real-gen-j-real.jsonl
uv run jev-rag publish --space-dir space
```

Offline pipeline check (no keys, no network, deterministic stand-ins):

```bash
uv run jev-rag run --dataset xquad-en --branches A,J,N --fixture --limit 25
```

Fixture runs are marked `run_kind=fixture` everywhere and are never published.

## Reproduce the published results

```bash
uv run jev-rag data prepare --dataset all
uv run jev-rag run --dataset xquad-en --branches A,J,N
uv run jev-rag run --dataset scifact --branches A,J,N
uv run jev-rag generate --results results/xquad-en-a-j-n-real.jsonl --branch J
uv run jev-rag --config configs/vercel.yaml add-branch \
  --results results/xquad-en-a-j-n-real.jsonl --branch T
uv run jev-rag --config configs/vercel.yaml add-branch \
  --results results/scifact-a-j-n-real.jsonl --branch T
uv run jev-rag report --results results/xquad-en-a-j-n-real.jsonl \
  --generation results/xquad-en-a-j-n-real-gen-j-real.jsonl
uv run jev-rag report --results results/scifact-a-j-n-real.jsonl
uv run jev-rag publish --space-dir space
```

## Metrics

- **Retrieval / reranking**: Recall@5, Recall@10, nDCG@10, MRR@10, plus the
  candidate-pool ceiling (`gold_in_candidates`).
- **Calibration** (candidate-level relevance probabilities): Brier score, ECE
  (10 bins), reliability table, risk-coverage, top-1 confidence split by
  correct/incorrect.
- **Selective prediction**: precision/coverage at probability thresholds.
- **Decision layer**: confidence-partitioned reranking (fixed t = 0.50) and
  answerability gating (generation calls saved at each threshold).
- **Generation**: token F1, exact match, success rate (F1 ≥ 0.5), abstention
  rate, valid citation rate.
- **Cost/latency**: p50/p95 per stage, tokens per model, resolved model ids,
  actual cost when the provider reports it.

Statistical comparison uses paired bootstrap CIs (5,000 resamples, seed 13).

## Reproducibility rules

- Datasets are downloaded at runtime and never committed (licenses below).
- The candidate pool is frozen before any reranker runs.
- Generation uses frozen contexts from exactly one branch.
- Model versions are pinned in config and resolved versions are recorded per row.
- Threshold-based analyses fix their threshold before evaluation; dev/test
  splits are used when a parameter is tuned.
- Safety caps exist for request counts and tokens per provider.
- Every run writes a manifest: config snapshot, dataset SHA-256, model ids,
  token totals, timestamps.

The preregistered protocol (research questions, hypotheses, decision rules,
changelog) is in [`docs/protocol.md`](docs/protocol.md).

## Differences from prior work

Compared with [`erendikmenn/jev-rag-benchmark`](https://github.com/erendikmenn/jev-rag-benchmark)
(the prior public Jev RAG benchmark; no code was copied):

| | Prior work | This repo |
|---|---|---|
| Language / data | Turkish XQuAD (1,044 q, 240 passages) | **English** XQuAD (1,190 q) + **SciFact** (300 claims / 5,183 abstracts) |
| Model tested | TypeSafe Jev 1.13 via OpenRouter (paid) | OpenJev 0.1 (free, open weights) **and** real Jev 1.13 |
| Baselines | Cohere Rerank 3.5 (paid) | NVIDIA cross-encoder (free) + no reranker |
| Calibration | Not measured | **Brier, ECE, reliability, risk-coverage, selective prediction** |
| Decision layer | Sample-based ablations | Confidence-partitioned reranking + answerability gating (full sets, CIs) |
| Cost | OpenRouter spend | **$0** for every published run |
| Framework | LlamaIndex app | Framework-neutral (3 runtime dependencies) |
| Sharing | Report + JSONL | Auto-updated README table, HF static Space, `leaderboard.txt`, publish pipeline that refuses fixture data |

## Publishing

```bash
uv run jev-rag publish --space-dir space --repo-url https://github.com/<you>/<repo>
uv sync --extra hf && export HF_TOKEN=hf_...
uv run jev-rag publish --push --repo-id <you>/jev-rag-benchmark --repo-type dataset
uv run jev-rag publish --space-dir space \
  --push --repo-id <you>/jev-rag-benchmark-leaderboard --repo-type space
```

Mint a DOI for the release with Zenodo (free) and cite it here.

## Data licenses

- XQuAD (English): CC BY-SA 4.0.
- SciFact / BEIR: abstracts ODC-By 1.0, annotations CC BY 4.0.

Raw data is downloaded by `jev-rag data prepare` and is git-ignored.

## Limitations

- **Exploratory analyses**: the gated mode and the real-Jev comparison were
  added after the first ranking run; they are labeled exploratory until
  replicated with a fresh candidate draw.
- **End-to-end answer quality for branch T is not yet measured** (generation
  currently uses OpenJev's top-5); it is a planned addition.
- Branch T latency reflects Vercel AI Gateway free-tier pacing (~1 request/s),
  not standalone model latency (single-request pilot: ~0.6 s).
- XQuAD references are short extractive answers; token F1 is a proxy, not human
  factuality adjudication.
- The generator (DiffusionGemma 26B) is OpenJev's base model; generation uses
  frozen contexts, so retrieval metrics are unaffected, but answer quality is
  not independent of that model family.
- Results describe the exact resolved model versions in each manifest. OpenJev
  is an independent open-weights model served by Codiv; it is **not** TypeSafe's
  Jev. Baseline model availability changes over time (several NVIDIA models were
  retired on 2026-08-25/26).

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this benchmark, cite the
repository and the archived DOI release once minted.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

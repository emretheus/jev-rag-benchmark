# Jev RAG Benchmark (English): Jev 1.13 vs NVIDIA cross-encoder vs no reranking

A free, reproducible, framework-neutral benchmark of **TypeSafe Jev 1.13 as the
reranking and decision layer of a RAG pipeline** — English data, frozen
candidate pools, paired bootstrap confidence intervals, and probability
calibration. Every published run used free tiers: **$0 total**.

It answers one question honestly: **does the model actually improve the system?**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Cost](https://img.shields.io/badge/benchmark%20cost-%240-brightgreen)
![Tests](https://img.shields.io/badge/tests-42%20passing-brightgreen)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-live%20leaderboard-yellow)](https://huggingface.co/spaces/emretheus/jev-rag-benchmark-leaderboard)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-raw%20artifacts-orange)](https://huggingface.co/datasets/emretheus/jev-rag-benchmark)

## Key findings (2026-09-21)

| Finding | Evidence |
|---|---|
| **Jev improves retrieval over the unranked baseline** | nDCG@10 **+0.82 pts** on XQuAD-EN (95% CI +0.35 to +1.31) and **+7.62 pts** on SciFact (CI +4.88 to +10.38) |
| **Against a dedicated cross-encoder** | Statistical tie on SciFact (+0.59 pts, CI −1.25 to +2.42); NVIDIA leads XQuAD by 0.45 pts |
| **Probabilities are well calibrated** | ECE **1.33%** (XQuAD) / 6.25% (SciFact); top-1 confidence 0.96 when correct vs 0.74 when wrong (XQuAD) |
| **Calibration enables high-precision filtering** | at p ≥ 0.7: **94.9% precision** at 5.1% coverage on XQuAD (19x the 5% base rate); SciFact 70.9% at p ≥ 0.8 (14x) |
| **Confidence-partitioned decisions beat baseline everywhere** | XQuAD **+0.89 pts** (CI +0.45 to +1.36); SciFact **+4.14 pts** (CI +2.13 to +6.29) while always-on is best there |
| **Answerability gating saves generation calls** | skipping generation below p = 0.9 drops 9.8% of calls and raises answer F1 from 30.6% to **31.3%** |
| **End-to-end answers from Jev top-5** | token F1 **30.6%**, exact match 1.3%, success (F1 ≥ 0.5) **17.3%**, abstention 2.6%, valid citations **99.85%** |
| **Cost** | **$0 actual** (free tiers); $0.33 at list price for all 1,490 reranked queries |

The decision-layer analyses (confidence partitioning, answerability gating) are
labeled **exploratory** in [`docs/protocol.md`](docs/protocol.md) — they were
added after the first ranking run and are reported with that label.

## Results

Auto-generated from real runs by `uv run jev-rag publish`. Never hand-edited;
fixture runs are refused. Charts for these tables live in
[`assets/benchmark/`](assets/benchmark).

<!-- RESULTS:START -->

_Generated 2026-09-24 12:09 UTC from real runs (scifact, xquad-en). Fixture runs are never published._
_Regenerate with:_ `uv run jev-rag publish`

### Reranking (frozen top-20 candidates, same pool for every method)

| Dataset | n | Method | nDCG@10 | Recall@5 | MRR@10 | Rerank p50 |
|---|---|---|---|---|---|---|
| scifact | 300 | A — no reranker (hybrid order) | 71.67% | 79.67% | 68.75% | 0 ms |
| scifact | 300 | T — TypeSafe Jev 1.13 batch noul | 79.29% | 85.67% | 77.03% | 4044 ms |
| scifact | 300 | N — NVIDIA cross-encoder reranker | 78.70% | 87.33% | 76.16% | 307 ms |
| xquad-en | 1190 | A — no reranker (hybrid order) | 98.11% | 99.58% | 97.60% | 0 ms |
| xquad-en | 1190 | T — TypeSafe Jev 1.13 batch noul | 98.93% | 99.66% | 98.67% | 4000 ms |
| xquad-en | 1190 | N — NVIDIA cross-encoder reranker | 99.37% | 99.66% | 99.27% | 409 ms |

### Probability calibration (candidate-level relevance)

| Dataset | Model | ECE (10 bin) | Brier | Top-1 accuracy | Top-1 confidence (correct) | Top-1 confidence (wrong) |
|---|---|---|---|---|---|---|
| scifact | TypeSafe Jev 1.13 batch noul | 0.0625 | 0.0366 | 71.00% | 0.812 | 0.549 |
| xquad-en | TypeSafe Jev 1.13 batch noul | 0.0133 | 0.0045 | 97.90% | 0.960 | 0.740 |

### RAG optimization mode (confidence-partitioned Jev, fixed t = 0.50)

| Dataset | Model | Threshold | A baseline nDCG@10 | Always-on nDCG@10 | Partitioned nDCG@10 | Delta vs baseline | 95% CI |
|---|---|---|---|---|---|---|---|
| scifact | TypeSafe Jev 1.13 batch noul | 0.50 | 71.67% | 79.29% | 75.81% | +4.14 pts | +2.13 to +6.29 |
| xquad-en | TypeSafe Jev 1.13 batch noul | 0.50 | 98.11% | 98.93% | 99.00% | +0.89 pts | +0.45 to +1.36 |

### Frozen-context answer generation

| Dataset | Generator | Token F1 | Exact match | F1 >= 0.5 | Abstention | Valid citations |
|---|---|---|---|---|---|---|
| xquad-en | T | diffusiongemma-26b | 30.58% | 1.26% | 17.31% | 2.61% | 99.85% |

### Cost at list price (actual published spend: $0, free tiers)

| Dataset | Branch | Input tokens | List price / 1M | Cost at list price |
|---|---|---|---|---|
| scifact | no reranker (hybrid order) | 0 | $0.000 | $0.0000 |
| scifact | NVIDIA cross-encoder reranker | 2,179,282 | $0.000 | $0.0000 |
| scifact | TypeSafe Jev 1.13 batch noul | 2,411,716 | $0.042 | $0.1013 |
| xquad-en | no reranker (hybrid order) | 0 | $0.000 | $0.0000 |
| xquad-en | NVIDIA cross-encoder reranker | 4,652,357 | $0.000 | $0.0000 |
| xquad-en | TypeSafe Jev 1.13 batch noul | 5,407,942 | $0.042 | $0.2271 |

_Metrics measure different stages: retrieval quality (nDCG, Recall) does not imply answer quality. See the per-run reports for confidence intervals, reliability bins, and risk-coverage tables._

<!-- RESULTS:END -->

### Charts

![nDCG@10 by method](assets/benchmark/reranker-comparison.svg)

![Calibration: confidence vs accuracy](assets/benchmark/calibration-reliability.svg)

![Precision vs coverage at a Jev probability threshold](assets/benchmark/risk-coverage.svg)

![Gold passage not retrieved, by candidate depth](assets/benchmark/retrieval-errors.svg)

![Quality vs rerank latency](assets/benchmark/latency-quality.svg)

Share card for social posts: [`assets/social/jev-benchmark-card.png`](assets/social/jev-benchmark-card.png) ·
ready-to-post text: [`assets/social/linkedin-post.md`](assets/social/linkedin-post.md).

## Answerability / abstention

Jev was asked one `noul` question per case: *the passages contain enough
information to answer the question*. Cases are the query's own frozen top-5
(matched, answerable) and another query's top-5 that excludes the gold passage
(mismatched, not answerable). Sampled run: **835 cases (420 matched / 415
mismatched)** — a subset of the 2,352 planned, due to free-tier pacing.

| Metric | Value |
|---|---|
| Accuracy | **97.5%** (95% CI 96.4–98.4) |
| Brier / ECE (10 bins) | 0.020 / 0.054 |
| Mean probability, answerable vs not | 0.894 vs 0.019 |

Joined with the frozen generation F1s, this score can gate generation:

| Gate | Coverage | Mean F1 | Success (F1 ≥ 0.5) | Successful answers per 1k queries |
|---|---|---|---|---|
| none | 100% | 29.94% | 13.33% | 133 |
| p ≥ 0.5 | 95.2% | **30.70%** | **14.00%** | **133** |
| p ≥ 0.7 | 90.0% | 30.90% | 13.76% | 124 |
| p ≥ 0.9 | 76.0% | 30.69% | 12.23% | 93 |

Reading: the check is accurate and calibrated, and gating at p ≥ 0.5 removes
~5% of generation calls **without losing a single successful answer per thousand
queries** — quality among kept calls even rises slightly. Past 0.7 the gate is
too aggressive for this dataset. Scope: negatives are synthetic (another
query's contexts), and XQuAD contains only answerable questions, so a real
unanswerable set (e.g. SQuAD v2) is the natural next step.

![Answerability gating](assets/benchmark/answerability-gating.svg)

## Bonus: tool-calling governance (Jev vs Laya vs an LLM)

Decision models can also govern tool calls: which tool to use, whether a call
may run without review, whether arguments are valid, and whether observed text
tries to manipulate the assistant. All families run on the same deterministic
synthetic cases (`jev-rag toolbench`, n = 40 per family). Jev runs through the
Vercel AI Gateway free tier, **Laya through its official Hugging Face demo
Space** (free ZeroGPU), and the LLM baseline is a free generative model
(DiffusionGemma 26B via Codiv) asked for the same decisions as JSON.

| Task family | Jev 1.13 | Laya | LLM baseline |
|---|---|---|---|
| Tool selection, 5 tools | 100% | 100% | 100% |
| Tool selection, 20 tools | 90% | 100% | 80% |
| Tool selection, 50 tools | 70% | 40% | 70% |
| Tool selection, 200 tools | **80%** | **refused**¹ | 80% |
| Call approval (`noul`) | 70% (Brier 0.198) | 32.5% (Brier 0.547) | **80%** (Brier 0.200) |
| Argument validation (`noul`) | 100% (Brier 0.019) | 65% (Brier 0.238) | 100% (Brier 0.000) |
| Injection detection (`noul`) | 100% (Brier 0.013) | 100% (Brier 0.014) | 100% (Brier 0.001) |
| Median decision latency | 1022 ms² | 121 ms² | 1018 ms² |
| No-answer cases | 0/160 | 10/160 | 0/160 |

¹ Laya's English checkpoint rejects 200-option questions outright:
`ValueError: options do not fit in head_max_len=192 tokens`. That is a
documented architectural budget, not a wrong answer. Its 50-option accuracy
(40%) shows the degradation begins well before the hard limit.

² Latency is end-to-end through each model's hosted route (shared Space queue
for Laya, gateways for Jev and the LLM); Laya's own reported model latency is
~121 ms, Jev's native model time is 70–500 ms.

![Tool selection accuracy by catalog size](assets/benchmark/toolbench-cardinality.svg)

![Accuracy by task family](assets/benchmark/toolbench-accuracy.svg)

### Cost comparison

Every published run executed on free tiers, so the actual spend was **$0**.
The table prices the same work at list rates so the comparison survives outside
free tiers.

| Workload | Tokens | Actual paid | List-price equivalent |
|---|---|---|---|
| RAG reranking, Jev (1,490 queries) | 7,819,658 input | $0 (Vercel free tier) | **$0.328** — $0.22 per 1k queries, $0.00022 per query |
| RAG reranking, NVIDIA cross-encoder | 6,831,639 input | $0 (NIM trial) | not published for this endpoint |
| Embeddings (corpus + queries) | ~5.4M input | $0 (NIM trial) | not published for this endpoint |
| Answer generation, DiffusionGemma 26B (1,190 answers) | 1.24M prompt + 53k output | $0 (Codiv free tier) | self-hosted: ~$0.10–0.30 of GPU time (estimate) |
| Toolbench, Jev (160 decisions) | 114,539 input (716/decision) | $0 (free tier) | **$0.0048** — $0.030 per 1k decisions |
| Toolbench, LLM baseline | 61,542 input + output | $0 (free tier) | self-hosted GPU time |
| Toolbench, Laya (160 decisions) | 13,989 input (87/decision) | $0 (HF Space) | self-hosted: **$0.0006 per 1k decisions** (7.2 ms/decision batched on a T4 at $0.30/h) |

Reading it honestly: Jev costs about **$0.22 per 1,000 reranked queries** and
**$0.03 per 1,000 decisions** at these prompt sizes — cheap in absolute terms,
and roughly 2x cheaper per query than the prior Turkish benchmark measured
($0.00039/query). Laya is roughly 50x cheaper per decision once self-hosted,
but it cannot serve the large-catalog cases at all. The LLM baseline has no
published per-token price at these providers; self-hosting it costs GPU time
like Laya. Estimates use a T4 at $0.30/hour; NVIDIA list prices for the
reranker and embedding endpoints are not published.

What the table shows, honestly: **a free LLM matches Jev on accuracy** here
(selection 82.5% vs 85%, approval 80% vs 70%, validation and injection tied at
100%). Jev's advantage is operational, not accuracy: typed answers with no
parsing, probabilities returned as a first-class API (the LLM had to be asked
for JSON and could have failed to produce it), many questions answered in one
parallel request, and input-only pricing. Laya trails on accuracy and has a hard
option budget, but its native latency is roughly an order of magnitude lower and
it self-hosts for free. Fairness notes: cases are synthetic and
template-generated (one seed); Laya was served as its English base checkpoint;
the LLM received the same state and questions with a JSON-only instruction. Full
artifacts: `reports/toolbench/`.

## Pipeline

```
queries ──► BM25 (top-100) ─┐
                            ├─► RRF fusion ─► frozen top-20 candidates
queries ──► NVIDIA embeddings┘                       │
                                                     ├─► A: hybrid order (baseline)
                                                     ├─► T: TypeSafe Jev 1.13 batch noul
                                                     └─► N: NVIDIA cross-encoder reranker
                                                     │
                            frozen top-5 of branch T ┴─► answer generation (free-tier model)
```

Every branch sees the **identical frozen candidate pool**. Reranking, branch
comparison, and generation are measured separately; generation replays over
frozen contexts so a fresh retrieval draw cannot contaminate it.

## Branches

| Branch | Method | Provider / cost |
|---|---|---|
| A | No reranker; hybrid RRF order is used as-is | local, $0 |
| T | TypeSafe `typesafe-ai/jev` (= Jev 1.13) batch **noul**: one request per query, one yes/no question per candidate | Vercel AI Gateway (free tier at run time) or OpenRouter |
| N | NVIDIA `llama-nemotron-rerank-vl-1b-v2` cross-encoder logits | NVIDIA NIM trial, $0 |

Adding a branch to existing results never re-runs the others:

```bash
uv run jev-rag add-branch --results results/xquad-en-a-n-t-real.jsonl --branch N --concurrency 4
```

## Quickstart (free stack)

```bash
uv sync --extra dev
cp .env.example .env     # keys: Vercel AI Gateway (Jev), NVIDIA NIM, Codiv (generator)

uv run jev-rag doctor
uv run jev-rag data prepare --dataset all
uv run jev-rag estimate --dataset xquad-en        # zero API calls

uv run jev-rag run --dataset xquad-en --branches A,T,N
uv run jev-rag generate --results results/xquad-en-a-n-t-real.jsonl --branch T
uv run jev-rag report --results results/xquad-en-a-n-t-real.jsonl \
  --generation results/xquad-en-a-n-t-real-gen-t-real.jsonl
uv run jev-rag charts --reports-dir reports/generated --output-dir assets/benchmark
uv run jev-rag publish --space-dir space
```

Offline pipeline check (no keys, no network, deterministic stand-ins):

```bash
uv run jev-rag run --dataset xquad-en --branches A,T,N --fixture --limit 25
```

Fixture runs are marked `run_kind=fixture` everywhere and are never published.

## Reproduce the published results

```bash
uv run jev-rag data prepare --dataset all
uv run jev-rag run --dataset xquad-en --branches A,T,N
uv run jev-rag run --dataset scifact --branches A,T,N
uv run jev-rag generate --results results/xquad-en-a-n-t-real.jsonl --branch T
uv run jev-rag report --results results/xquad-en-a-n-t-real.jsonl \
  --generation results/xquad-en-a-n-t-real-gen-t-real.jsonl
uv run jev-rag report --results results/scifact-a-n-t-real.jsonl
uv run jev-rag audit --dataset xquad-en
uv run jev-rag audit --dataset scifact
uv run jev-rag charts --reports-dir reports/generated --output-dir assets/benchmark
uv run jev-rag publish --space-dir space
```

## Metrics

- **Retrieval / reranking**: Recall@5, Recall@10, nDCG@10, MRR@10, candidate-pool
  ceiling, and a candidate-depth audit (BM25 vs hybrid) in `reports/audits/`.
- **Calibration** (candidate-level relevance probabilities): Brier score, ECE
  (10 bins), reliability table, risk-coverage, top-1 confidence split by
  correct/incorrect.
- **Decision layer**: confidence-partitioned reranking (fixed t = 0.50) and
  answerability gating (generation calls saved at each threshold).
- **Generation**: token F1, exact match, success rate (F1 ≥ 0.5), abstention
  rate, valid citation rate.
- **Cost/latency**: p50/p95 per stage, tokens per model, resolved model ids,
  actual cost when reported, list-price cost for budgeting.

Statistical comparison uses paired bootstrap CIs (5,000 resamples, seed 13).

## Reproducibility rules

- Datasets are downloaded at runtime and never committed (licenses below).
- The candidate pool is frozen before any reranker runs.
- Generation uses frozen contexts from exactly one branch.
- Model versions are pinned in config and resolved versions are recorded per row.
- Threshold analyses fix their threshold before evaluation; dev/test splits are
  used when a parameter is tuned.
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
| Model tested | TypeSafe Jev 1.13 via OpenRouter (paid) | TypeSafe Jev 1.13 via **Vercel AI Gateway** (free tier at run time) |
| Baselines | Cohere Rerank 3.5 (paid) | **NVIDIA cross-encoder** (free) + no reranker |
| Calibration | Not measured | **Brier, ECE, reliability, risk-coverage, selective prediction** |
| Decision layer | Sample-based ablations | Confidence-partitioned reranking + answerability gating |
| Generation | Gemini vs DeepSeek on frozen Jev top-5 | Free-tier generator on frozen Jev top-5, with answerability gating |
| Cost | OpenRouter spend | **$0** for every published run |
| Framework | LlamaIndex app | Framework-neutral (3 runtime dependencies) |
| Sharing | Report + JSONL | Auto-updated README table, **charts suite**, HF static Space, `leaderboard.txt`, publish pipeline that refuses fixture data |

## Publishing

```bash
uv run jev-rag publish --space-dir space --repo-url https://github.com/emretheus/jev-rag-benchmark
uv sync --extra hf && export HF_TOKEN=hf_...
uv run jev-rag publish --push --repo-id emretheus/jev-rag-benchmark --repo-type dataset
uv run jev-rag publish --space-dir space \
  --push --repo-id emretheus/jev-rag-benchmark-leaderboard --repo-type space
```

Mint a DOI for the release with Zenodo (free) and cite it here.

## Data licenses

- XQuAD (English): CC BY-SA 4.0.
- SciFact / BEIR: abstracts ODC-By 1.0, annotations CC BY 4.0.

Raw data is downloaded by `jev-rag data prepare` and is git-ignored.

## Limitations

- **Exploratory analyses**: the gated mode and answerability gating were added
  after the first ranking run; they are labeled exploratory until replicated
  with a fresh candidate draw or another dataset.
- Jev's always-on reranking helps most where the first stage leaves headroom:
  on XQuAD the baseline is already at 98.1% nDCG, so the absolute gain is small.
- Branch T latency reflects Vercel AI Gateway free-tier pacing (~1 request/s),
  not standalone model latency (single-request pilot: ~0.6 s).
- XQuAD references are short extractive answers; token F1 is a proxy, not human
  factuality adjudication.
- The generator is a free-tier model (DiffusionGemma 26B via Codiv); generation
  consumes frozen contexts, so retrieval metrics are unaffected by it.
- Results describe the exact resolved model versions in each manifest. This
  benchmark is independent and not affiliated with TypeSafe AI. Baseline model
  availability changes over time (several NVIDIA models were retired on
  2026-08-25/26).

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this benchmark, cite the
repository and the archived DOI release once minted.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

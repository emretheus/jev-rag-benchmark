# OpenJev RAG Benchmark report

- Dataset: `xquad-en`
- Queries: 1190
- Run kind: `real`
- Generated: 2026-09-20T16:57:40.162845+00:00

## 1. Candidate retrieval ceiling

Share of queries whose gold passage was present in the frozen candidate pool (hybrid BM25 + dense, reciprocal rank fusion).

- Gold in candidates: **99.664%**

## 2. Reranker comparison

| Branch | Method | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | Rerank p50 | Rerank p95 | Mean input tokens |
|---|---|---|---|---|---|---|---|---|
| A | no reranker (hybrid order) | 98.107% | 99.580% | 99.580% | 97.597% | 0.0 ms | 0.0 ms | 0 |
| J | OpenJev 0.1 batch noul (Codiv) | 98.028% | 98.824% | 99.160% | 97.655% | 922.7 ms | 1278.1 ms | 8011 |
| N | NVIDIA reranker | 99.374% | 99.664% | 99.664% | 99.272% | 409.0 ms | 579.1 ms | 3910 |
| T | TypeSafe Jev 1.13 batch noul (real Jev) | 98.926% | 99.664% | 99.664% | 98.672% | 4000.3 ms | 53583.5 ms | 4544 |

- Branch J resolved models: openjev-0.1
- Branch N resolved models: nvidia/llama-nemotron-rerank-vl-1b-v2
- Branch T resolved models: typesafe-ai/jev

### Paired comparisons (same queries, bootstrap 95% CI)

- J_vs_N: nDCG@10 -1.346 pts (95% CI -1.928 to -0.813), Recall@5 -0.840 pts (95% CI -1.429 to -0.336); n=1190
- J_vs_A: nDCG@10 -0.079 pts (95% CI -0.692 to +0.513), Recall@5 -0.756 pts (95% CI -1.261 to -0.336); n=1190
- T_vs_A: nDCG@10 +0.819 pts (95% CI +0.353 to +1.312), Recall@5 +0.084 pts (95% CI +0.000 to +0.252); n=1190
- T_vs_J: nDCG@10 +0.898 pts (95% CI +0.394 to +1.442), Recall@5 +0.840 pts (95% CI +0.336 to +1.429); n=1190
- T_vs_N: nDCG@10 -0.448 pts (95% CI -0.751 to -0.166), Recall@5 +0.000 pts (95% CI +0.000 to +0.000); n=1190

## 3. OpenJev calibration (candidate-level relevance)

Each candidate passage receives a probability of relevance. Predictions: 23800, positive rate: 4.983%.

- Brier score: **0.0086** (lower is better)
- Expected calibration error (10 bins): **0.0245**
- Top-1 accuracy: 96.723%
- Mean top-1 confidence when correct / wrong: 0.765 / 0.455

Reliability:

| Probability bin | Count | Mean confidence | Accuracy |
|---|---|---|---|
| 0.0-0.1 | 21536 | 0.015 | 0.001 |
| 0.1-0.2 | 761 | 0.135 | 0.022 |
| 0.2-0.3 | 230 | 0.241 | 0.096 |
| 0.3-0.4 | 96 | 0.346 | 0.344 |
| 0.4-0.5 | 86 | 0.448 | 0.616 |
| 0.5-0.6 | 105 | 0.554 | 0.771 |
| 0.6-0.7 | 154 | 0.653 | 0.903 |
| 0.7-0.8 | 218 | 0.755 | 0.950 |
| 0.8-0.9 | 290 | 0.856 | 0.986 |
| 0.9-1.0 | 324 | 0.950 | 0.997 |

Risk-coverage (threshold on relevance probability):

| Threshold | Covered | Coverage | Precision | Recall |
|---|---|---|---|---|
| 0.30 | 1273 | 5.349% | 88.138% | 94.604% |
| 0.50 | 1096 | 4.605% | 94.891% | 87.690% |
| 0.70 | 832 | 3.496% | 98.077% | 68.803% |
| 0.80 | 614 | 2.580% | 99.186% | 51.349% |
| 0.90 | 324 | 1.361% | 99.691% | 27.234% |

## 3b. RAG optimization mode: confidence-partitioned OpenJev (t = 0.50)

Always-on reranking applies OpenJev to every candidate; the decision-layer mode reranks only candidates whose probability clears the threshold and keeps the hybrid order for the rest. Candidate coverage never drops below the baseline order.

| Mode | nDCG@10 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|
| A — hybrid order (baseline) | 98.107% | 99.580% | 99.580% | 97.597% |
| J — OpenJev always-on | 98.028% | 98.824% | 99.160% | 97.655% |
| G — confidence-partitioned | 98.765% | 99.496% | 99.580% | 98.490% |
| Gate — all-or-nothing switch | 98.620% | 99.160% | 99.412% | 98.361% |

- Partitioned vs baseline nDCG@10: **+0.658 pts** (95% CI +0.266 to +1.069), n=1190
- This is a post-hoc (exploratory) analysis on already-collected data; the threshold is fixed at 0.5 a priori of this analysis but was not preregistered.

## 4. Frozen-context answer generation

Generator: `diffusiongemma-26b`, branch `J`, n=1190.

- Mean token F1: **31.886%**
- Exact match: 2.353%
- Successful answers (F1 >= 0.5): 19.160%
- Abstention rate: 2.773%
- Valid citation rate: 99.850%
- Generation latency p50 / p95: 948.6 ms / 1393.4 ms
- Tokens: 1238640 prompt, 51193 completion

### Answerability gating: generate only when top-1 probability >= t

Rows below the threshold would not call the generator at all; the remaining rows are the same frozen-context calls.

| Threshold | Coverage | Calls | Calls saved | Mean F1 | Success (F1 >= 0.5) |
|---|---|---|---|---|---|
| 0.50 | 88.487% | 1053 | 11.513% | 32.247% | 19.373% |
| 0.70 | 69.328% | 825 | 30.672% | 32.949% | 20.000% |
| 0.80 | 51.429% | 612 | 48.571% | 33.873% | 21.242% |
| 0.90 | 27.227% | 324 | 72.773% | 35.085% | 22.222% |

## 5. Interpretation limits

- Retrieval metrics (nDCG, Recall@k, MRR) and answer F1 measure different stages; high Recall@5 does not imply high answer quality.
- This report describes the exact model versions listed above (see the run manifest); do not generalize to other models or languages.
- XQuAD references are short extractive answers; token F1 is a proxy, not human factuality adjudication.
- Latency is observed API latency under free tiers and is not a universal throughput guarantee.
- The free-tier generator is DiffusionGemma 26B, the base model of OpenJev. Generation consumes frozen contexts, so retrieval and reranking metrics are unaffected, but end-to-end answer quality is not independent of the reranker's base model.
- Baseline model availability changes over time (several NVIDIA models were retired on 2026-08-25/26); the exact model versions are recorded in the run manifest.
- Branch T latency is observed through the Vercel AI Gateway free tier under client-side pacing (~1 request/s); it reflects gateway queueing and retry behavior, not the model's standalone latency (single-request pilot: ~0.6 s).

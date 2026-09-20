# OpenJev RAG Benchmark report

- Dataset: `scifact`
- Queries: 300
- Run kind: `real`
- Generated: 2026-09-20T16:57:52.208422+00:00

## 1. Candidate retrieval ceiling

Share of queries whose gold passage was present in the frozen candidate pool (hybrid BM25 + dense, reciprocal rank fusion).

- Gold in candidates: **90.000%**

## 2. Reranker comparison

| Branch | Method | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | Rerank p50 | Rerank p95 | Mean input tokens |
|---|---|---|---|---|---|---|---|---|
| A | no reranker (hybrid order) | 71.672% | 79.667% | 84.333% | 68.755% | 0.0 ms | 0.0 ms | 0 |
| J | OpenJev 0.1 batch noul (Codiv) | 63.917% | 68.667% | 74.667% | 62.587% | 1219.7 ms | 1529.8 ms | 13885 |
| N | NVIDIA reranker | 78.703% | 87.333% | 90.000% | 76.158% | 306.9 ms | 409.4 ms | 7264 |
| T | TypeSafe Jev 1.13 batch noul (real Jev) | 79.290% | 85.667% | 90.000% | 77.025% | 4043.5 ms | 54208.7 ms | 8039 |

- Branch J resolved models: openjev-0.1
- Branch N resolved models: nvidia/llama-nemotron-rerank-vl-1b-v2
- Branch T resolved models: typesafe-ai/jev

### Paired comparisons (same queries, bootstrap 95% CI)

- J_vs_N: nDCG@10 -14.786 pts (95% CI -18.156 to -11.659), Recall@5 -18.667 pts (95% CI -23.333 to -14.333); n=300
- J_vs_A: nDCG@10 -7.755 pts (95% CI -11.346 to -4.461), Recall@5 -11.000 pts (95% CI -15.333 to -7.000); n=300
- T_vs_A: nDCG@10 +7.619 pts (95% CI +4.881 to +10.376), Recall@5 +6.000 pts (95% CI +2.333 to +9.667); n=300
- T_vs_J: nDCG@10 +15.373 pts (95% CI +12.260 to +18.719), Recall@5 +17.000 pts (95% CI +12.667 to +21.333); n=300
- T_vs_N: nDCG@10 +0.587 pts (95% CI -1.254 to +2.420), Recall@5 -1.667 pts (95% CI -4.333 to +0.667); n=300

## 3. OpenJev calibration (candidate-level relevance)

Each candidate passage receives a probability of relevance. Predictions: 6000, positive rate: 5.000%.

- Brier score: **0.0521** (lower is better)
- Expected calibration error (10 bins): **0.0906**
- Top-1 accuracy: 57.667%
- Mean top-1 confidence when correct / wrong: 0.606 / 0.514

Reliability:

| Probability bin | Count | Mean confidence | Accuracy |
|---|---|---|---|
| 0.0-0.1 | 3395 | 0.036 | 0.010 |
| 0.1-0.2 | 1060 | 0.146 | 0.027 |
| 0.2-0.3 | 689 | 0.246 | 0.061 |
| 0.3-0.4 | 378 | 0.343 | 0.082 |
| 0.4-0.5 | 214 | 0.448 | 0.164 |
| 0.5-0.6 | 113 | 0.544 | 0.345 |
| 0.6-0.7 | 61 | 0.647 | 0.459 |
| 0.7-0.8 | 52 | 0.749 | 0.635 |
| 0.8-0.9 | 30 | 0.845 | 0.733 |
| 0.9-1.0 | 8 | 0.950 | 0.875 |

Risk-coverage (threshold on relevance probability):

| Threshold | Covered | Coverage | Precision | Recall |
|---|---|---|---|---|
| 0.30 | 856 | 14.267% | 22.780% | 65.000% |
| 0.50 | 264 | 4.400% | 48.864% | 43.000% |
| 0.70 | 90 | 1.500% | 68.889% | 20.667% |
| 0.80 | 38 | 0.633% | 76.316% | 9.667% |
| 0.90 | 8 | 0.133% | 87.500% | 2.333% |

## 3b. RAG optimization mode: confidence-partitioned OpenJev (t = 0.50)

Always-on reranking applies OpenJev to every candidate; the decision-layer mode reranks only candidates whose probability clears the threshold and keeps the hybrid order for the rest. Candidate coverage never drops below the baseline order.

| Mode | nDCG@10 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|
| A — hybrid order (baseline) | 71.672% | 79.667% | 84.333% | 68.755% |
| J — OpenJev always-on | 63.917% | 68.667% | 74.667% | 62.587% |
| G — confidence-partitioned | 71.488% | 79.000% | 84.000% | 68.778% |
| Gate — all-or-nothing switch | 68.222% | 75.000% | 77.667% | 66.926% |

- Partitioned vs baseline nDCG@10: **-0.183 pts** (95% CI -1.927 to +1.543), n=300
- This is a post-hoc (exploratory) analysis on already-collected data; the threshold is fixed at 0.5 a priori of this analysis but was not preregistered.

## 5. Interpretation limits

- Retrieval metrics (nDCG, Recall@k, MRR) and answer F1 measure different stages; high Recall@5 does not imply high answer quality.
- This report describes the exact model versions listed above (see the run manifest); do not generalize to other models or languages.
- XQuAD references are short extractive answers; token F1 is a proxy, not human factuality adjudication.
- Latency is observed API latency under free tiers and is not a universal throughput guarantee.
- The free-tier generator is DiffusionGemma 26B, the base model of OpenJev. Generation consumes frozen contexts, so retrieval and reranking metrics are unaffected, but end-to-end answer quality is not independent of the reranker's base model.
- Baseline model availability changes over time (several NVIDIA models were retired on 2026-08-25/26); the exact model versions are recorded in the run manifest.
- Branch T latency is observed through the Vercel AI Gateway free tier under client-side pacing (~1 request/s); it reflects gateway queueing and retry behavior, not the model's standalone latency (single-request pilot: ~0.6 s).

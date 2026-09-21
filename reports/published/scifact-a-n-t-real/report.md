# Jev RAG Benchmark report

- Dataset: `scifact`
- Queries: 300
- Run kind: `real`
- Generated: 2026-09-21T20:38:40.200317+00:00

## 1. Candidate retrieval ceiling

Share of queries whose gold passage was present in the frozen candidate pool (hybrid BM25 + dense, reciprocal rank fusion).

- Gold in candidates: **90.000%**

## 2. Reranker comparison

| Branch | Method | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | Rerank p50 | Rerank p95 | Mean input tokens |
|---|---|---|---|---|---|---|---|---|
| A | no reranker (hybrid order) | 71.672% | 79.667% | 84.333% | 68.755% | 0.0 ms | 0.0 ms | 0 |
| N | NVIDIA cross-encoder reranker | 78.703% | 87.333% | 90.000% | 76.158% | 306.9 ms | 409.4 ms | 7264 |
| T | TypeSafe Jev 1.13 batch noul | 79.290% | 85.667% | 90.000% | 77.025% | 4043.5 ms | 54208.7 ms | 8039 |

- Branch N resolved models: nvidia/llama-nemotron-rerank-vl-1b-v2
- Branch T resolved models: typesafe-ai/jev

### Paired comparisons (same queries, bootstrap 95% CI)

- T_vs_A: nDCG@10 +7.619 pts (95% CI +4.881 to +10.376), Recall@5 +6.000 pts (95% CI +2.333 to +9.667); n=300
- T_vs_N: nDCG@10 +0.587 pts (95% CI -1.254 to +2.420), Recall@5 -1.667 pts (95% CI -4.333 to +0.667); n=300

## 3. Probability calibration (candidate-level relevance)

Every candidate passage receives a probability of relevance. Lower Brier and ECE are better; positive rate and prediction counts are shown so ECE is interpretable.

| Model | Branch | Predictions | Positive rate | Brier | ECE (10 bin) | Top-1 accuracy | Confidence (correct) | Confidence (wrong) |
|---|---|---|---|---|---|---|---|---|
| TypeSafe Jev 1.13 batch noul | T | 6000 | 5.000% | 0.0366 | 0.0625 | 71.000% | 0.812 | 0.549 |

Reliability and risk-coverage for **TypeSafe Jev 1.13 batch noul** (branch T):

| Probability bin | Count | Mean confidence | Accuracy |
|---|---|---|---|
| 0.0-0.1 | 4760 | 0.032 | 0.003 |
| 0.1-0.2 | 467 | 0.144 | 0.064 |
| 0.2-0.3 | 168 | 0.250 | 0.089 |
| 0.3-0.4 | 114 | 0.348 | 0.149 |
| 0.4-0.5 | 85 | 0.453 | 0.118 |
| 0.5-0.6 | 50 | 0.551 | 0.180 |
| 0.6-0.7 | 52 | 0.655 | 0.231 |
| 0.7-0.8 | 56 | 0.756 | 0.286 |
| 0.8-0.9 | 84 | 0.858 | 0.440 |
| 0.9-1.0 | 164 | 0.959 | 0.854 |

| Threshold | Covered | Coverage | Precision | Recall |
|---|---|---|---|---|
| 0.30 | 618 | 10.300% | 39.159% | 80.667% |
| 0.50 | 414 | 6.900% | 51.691% | 71.333% |
| 0.70 | 310 | 5.167% | 62.581% | 64.667% |
| 0.80 | 254 | 4.233% | 70.866% | 60.000% |
| 0.90 | 174 | 2.900% | 84.483% | 49.000% |

## 3b. RAG optimization mode: confidence-partitioned reranking

Threshold t = 0.50: candidates whose probability clears the threshold are reranked by the model, the rest keep the hybrid order, so candidate coverage never drops below the baseline.

| Model | Mode | nDCG@10 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| — | A baseline (hybrid order) | 71.672% | 79.667% | 84.333% | 68.755% |
| TypeSafe Jev 1.13 batch noul | always-on | 79.290% | 85.667% | 90.000% | 77.025% |
| TypeSafe Jev 1.13 batch noul | partitioned | 75.815% | 82.333% | 86.000% | 73.677% |

| Model | Partitioned vs baseline nDCG@10 | 95% CI | n |
|---|---|---|---|
| TypeSafe Jev 1.13 batch noul | +4.143 pts | +2.134 to +6.286 | 300 |

- Post-hoc (exploratory) analysis on already-collected data; the threshold is fixed at 0.5 a priori of this analysis but was not preregistered.

## 5. Cost at list price

| Branch | Input tokens | List price / 1M | Cost at list price |
|---|---|---|---|
| no reranker (hybrid order) | 0 | $0.000 | $0.0000 |
| NVIDIA cross-encoder reranker | 2,179,282 | $0.000 | $0.0000 |
| TypeSafe Jev 1.13 batch noul | 2,411,716 | $0.042 | $0.1013 |

- Every published run executed on free tiers; the actual spend was **$0**. List prices are shown so the benchmark can be budgeted anywhere.

## 6. Interpretation limits

- Retrieval metrics (nDCG, Recall@k, MRR) and answer F1 measure different stages; high Recall@5 does not imply high answer quality.
- This report describes the exact model versions listed above (see the run manifest); do not generalize to other models or languages.
- XQuAD references are short extractive answers; token F1 is a proxy, not human factuality adjudication.
- Latency is observed API latency under free tiers and is not a universal throughput guarantee.
- The generator is a free-tier model (DiffusionGemma 26B via Codiv). Generation consumes frozen contexts, so retrieval and reranking metrics are unaffected by it.
- Baseline model availability changes over time (several NVIDIA models were retired on 2026-08-25/26); the exact model versions are recorded in the run manifest.
- Branch T latency is observed through the Vercel AI Gateway free tier under client-side pacing (~1 request/s); it reflects gateway queueing and retry behavior, not the model's standalone latency (single-request pilot: ~0.6 s).

# Jev RAG Benchmark report

- Dataset: `xquad-en`
- Queries: 1190
- Run kind: `real`
- Generated: 2026-09-21T20:38:34.085574+00:00

## 1. Candidate retrieval ceiling

Share of queries whose gold passage was present in the frozen candidate pool (hybrid BM25 + dense, reciprocal rank fusion).

- Gold in candidates: **99.664%**

## 2. Reranker comparison

| Branch | Method | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | Rerank p50 | Rerank p95 | Mean input tokens |
|---|---|---|---|---|---|---|---|---|
| A | no reranker (hybrid order) | 98.107% | 99.580% | 99.580% | 97.597% | 0.0 ms | 0.0 ms | 0 |
| N | NVIDIA cross-encoder reranker | 99.374% | 99.664% | 99.664% | 99.272% | 409.0 ms | 579.1 ms | 3910 |
| T | TypeSafe Jev 1.13 batch noul | 98.926% | 99.664% | 99.664% | 98.672% | 4000.3 ms | 53583.5 ms | 4544 |

- Branch N resolved models: nvidia/llama-nemotron-rerank-vl-1b-v2
- Branch T resolved models: typesafe-ai/jev

### Paired comparisons (same queries, bootstrap 95% CI)

- T_vs_A: nDCG@10 +0.819 pts (95% CI +0.353 to +1.312), Recall@5 +0.084 pts (95% CI +0.000 to +0.252); n=1190
- T_vs_N: nDCG@10 -0.448 pts (95% CI -0.751 to -0.166), Recall@5 +0.000 pts (95% CI +0.000 to +0.000); n=1190

## 3. Probability calibration (candidate-level relevance)

Every candidate passage receives a probability of relevance. Lower Brier and ECE are better; positive rate and prediction counts are shown so ECE is interpretable.

| Model | Branch | Predictions | Positive rate | Brier | ECE (10 bin) | Top-1 accuracy | Confidence (correct) | Confidence (wrong) |
|---|---|---|---|---|---|---|---|---|
| TypeSafe Jev 1.13 batch noul | T | 23800 | 4.983% | 0.0045 | 0.0133 | 97.899% | 0.960 | 0.740 |

Reliability and risk-coverage for **TypeSafe Jev 1.13 batch noul** (branch T):

| Probability bin | Count | Mean confidence | Accuracy |
|---|---|---|---|
| 0.0-0.1 | 22043 | 0.008 | 0.000 |
| 0.1-0.2 | 262 | 0.145 | 0.000 |
| 0.2-0.3 | 102 | 0.252 | 0.020 |
| 0.3-0.4 | 64 | 0.359 | 0.047 |
| 0.4-0.5 | 48 | 0.455 | 0.167 |
| 0.5-0.6 | 39 | 0.551 | 0.282 |
| 0.6-0.7 | 34 | 0.653 | 0.294 |
| 0.7-0.8 | 53 | 0.767 | 0.623 |
| 0.8-0.9 | 83 | 0.856 | 0.735 |
| 0.9-1.0 | 1072 | 0.981 | 0.985 |

| Threshold | Covered | Coverage | Precision | Recall |
|---|---|---|---|---|
| 0.30 | 1402 | 5.891% | 84.379% | 99.747% |
| 0.50 | 1288 | 5.412% | 90.916% | 98.735% |
| 0.70 | 1214 | 5.101% | 94.893% | 97.133% |
| 0.80 | 1165 | 4.895% | 96.481% | 94.772% |
| 0.90 | 1088 | 4.571% | 98.254% | 90.135% |

## 3b. RAG optimization mode: confidence-partitioned reranking

Threshold t = 0.50: candidates whose probability clears the threshold are reranked by the model, the rest keep the hybrid order, so candidate coverage never drops below the baseline.

| Model | Mode | nDCG@10 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| — | A baseline (hybrid order) | 98.107% | 99.580% | 99.580% | 97.597% |
| TypeSafe Jev 1.13 batch noul | always-on | 98.926% | 99.664% | 99.664% | 98.672% |
| TypeSafe Jev 1.13 batch noul | partitioned | 98.998% | 99.580% | 99.580% | 98.796% |

| Model | Partitioned vs baseline nDCG@10 | 95% CI | n |
|---|---|---|---|
| TypeSafe Jev 1.13 batch noul | +0.891 pts | +0.455 to +1.361 | 1190 |

- Post-hoc (exploratory) analysis on already-collected data; the threshold is fixed at 0.5 a priori of this analysis but was not preregistered.

## 4. Frozen-context answer generation

The generator answers only from the frozen top-5 of one branch; retrieval and reranking cannot influence this comparison.

| Branch (contexts) | Generator | n | Token F1 | Exact match | F1 >= 0.5 | Abstention | Valid citations | p50 |
|---|---|---|---|---|---|---|---|---|
| T | diffusiongemma-26b | 1190 | 30.578% | 1.261% | 17.311% | 2.605% | 99.853% | 928 ms |

Answerability gating (branch T): skip generation when the top-1 probability is below the threshold

| Threshold | Coverage | Calls | Calls saved | Mean F1 | Success (F1 >= 0.5) |
|---|---|---|---|---|---|
| 0.50 | 98.908% | 1177 | 1.092% | 30.843% | 17.502% |
| 0.70 | 97.311% | 1158 | 2.689% | 30.926% | 17.617% |
| 0.80 | 95.126% | 1132 | 4.874% | 31.198% | 17.845% |
| 0.90 | 90.168% | 1073 | 9.832% | 31.335% | 17.987% |

## 5. Cost at list price

| Branch | Input tokens | List price / 1M | Cost at list price |
|---|---|---|---|
| no reranker (hybrid order) | 0 | $0.000 | $0.0000 |
| NVIDIA cross-encoder reranker | 4,652,357 | $0.000 | $0.0000 |
| TypeSafe Jev 1.13 batch noul | 5,407,942 | $0.042 | $0.2271 |

- Every published run executed on free tiers; the actual spend was **$0**. List prices are shown so the benchmark can be budgeted anywhere.

## 6. Interpretation limits

- Retrieval metrics (nDCG, Recall@k, MRR) and answer F1 measure different stages; high Recall@5 does not imply high answer quality.
- This report describes the exact model versions listed above (see the run manifest); do not generalize to other models or languages.
- XQuAD references are short extractive answers; token F1 is a proxy, not human factuality adjudication.
- Latency is observed API latency under free tiers and is not a universal throughput guarantee.
- The generator is a free-tier model (DiffusionGemma 26B via Codiv). Generation consumes frozen contexts, so retrieval and reranking metrics are unaffected by it.
- Baseline model availability changes over time (several NVIDIA models were retired on 2026-08-25/26); the exact model versions are recorded in the run manifest.
- Branch T latency is observed through the Vercel AI Gateway free tier under client-side pacing (~1 request/s); it reflects gateway queueing and retry behavior, not the model's standalone latency (single-request pilot: ~0.6 s).

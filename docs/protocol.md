# Benchmark protocol (preregistered)

This protocol is frozen before the published English runs. Changes after the
first published run must be recorded in the changelog section at the bottom and
must not silently rewrite published numbers.

## Research questions

1. **RQ1 (retrieval ceiling).** Is the gold passage inside the frozen hybrid
   candidate pool, and how does the ceiling depend on candidate depth?
2. **RQ2 (reranking quality).** Does Jev 1.13 batch noul reranking improve
   Recall@5, nDCG@10, and MRR@10 over the unreranked hybrid order, and how does
   it compare with a dedicated cross-encoder (NVIDIA)?
3. **RQ3 (calibration).** Are Jev's noul probabilities calibrated for candidate
   relevance (Brier, ECE, reliability), and can a confidence threshold trade
   coverage for precision predictably (risk-coverage)?
4. **RQ4 (generation).** Given frozen Jev top-5 contexts, how well does a
   free-tier generative model answer extractive English questions (F1, EM,
   abstention, citation validity)?
5. **RQ5 (cost/latency).** What are observed p50/p95 latencies and token costs
   of each stage on free tiers, and what would they cost at list price?

## Hypotheses

- H1: Jev reranking improves nDCG@10 over branch A by at least 1 point on
  SciFact, with a paired bootstrap 95% CI excluding zero.
- H2: Jev is not assumed to beat the cross-encoder; the sign and CI are
  reported either way.
- H3: Jev probabilities are positively associated with relevance
  (top-1 confidence when correct > when incorrect).
- H4: A confidence threshold exists whose precision exceeds the base rate by at
  least 2x at non-trivial coverage.

## Datasets and splits

- XQuAD-EN (all answerable questions; gold = containing paragraph).
- SciFact (BEIR; qrels `test.tsv`, score > 0; gold = cited abstracts).
- No training or tuning on test questions. Prompt templates are fixed in config
  before the run and are published with it.

## Models (pinned)

| Role | Model | Provider |
|---|---|---|
| Embedding (hybrid first stage) | `nvidia/nemotron-3-embed-1b` | NVIDIA NIM |
| Tested reranker | `typesafe-ai/jev` (= Jev 1.13) | Vercel AI Gateway (free tier at run time) or OpenRouter |
| Baseline reranker | `nvidia/llama-nemotron-rerank-vl-1b-v2` | NVIDIA NIM |
| Answer generator | `diffusiongemma-26b` (configurable) | Codiv |

Model availability notes (recorded 2026-09-20): NVIDIA retired the text-only
reranker (`llama-nemotron-rerank-1b-v2`), the previous embedding model, and the
initial generator model on 2026-08-25/26. This protocol uses the live
replacements. The generator is a separate free-tier model; generation consumes
frozen contexts, so retrieval and reranking metrics are unaffected by it.

## Procedure

1. Normalize datasets; record raw SHA-256 hashes in the manifest.
2. Build hybrid retrieval: BM25 top-100 + dense top-100, RRF (k=60), take
   top-20 as the frozen candidate pool.
3. Run branches A, T, N on the identical pool.
4. Freeze branch T's top-5 contexts and replay the generator (resumable,
   concurrency 4, retries on transient errors).
5. Compute metrics and paired bootstrap CIs (5,000 resamples, seed 13).

## Decision rules

- Primary metric: nDCG@10. Secondary: Recall@5, MRR@10.
- A difference is reported as a difference; no claim of superiority is made
  unless the paired 95% CI excludes zero. When it does not, the result is
  reported as "no measurable difference on this dataset".
- Fixture runs (`run_kind=fixture`) are never cited as results.
- Negative and null results are published with the same prominence as positive
  ones.

## Reporting rules

- Every table states dataset, n, model versions, and run kind.
- Latency is observed API latency under free tiers, never throughput claims.
- Token F1 is labeled a proxy, not factuality adjudication.
- Calibration is reported at candidate level with the positive rate, so ECE is
  interpretable.

## Changelog

- 2026-09-20: initial protocol.
- 2026-09-20: **E2 added post-hoc (exploratory).** After the first ranking run,
  the confidence-partitioned decision mode was analyzed on the same data with a
  fixed threshold of 0.5: candidates with p >= 0.5 are reranked first, the rest
  keep the hybrid order. Results are exploratory, not confirmatory, until
  replicated on a fresh dataset or a different first stage.
- 2026-09-21: **E3 added post-hoc (exploratory).** Answerability gating:
  generation is skipped when the top-1 probability is below a threshold; call
  savings and answer quality are reported per threshold.
- 2026-09-21: **Scope narrowed to Jev 1.13** (branches A/T/N). The earlier
  open-weights comparison model was removed from the published benchmark; the
  harness supports additional branches through `add-branch`.

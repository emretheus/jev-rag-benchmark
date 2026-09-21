I benchmarked TypeSafe's **Jev 1.13** as the reranking and decision layer of a RAG pipeline — in English, on frozen candidate pools, with paired confidence intervals. Every published run cost **$0** (free tiers).

**Results (nDCG@10, identical top-20 candidates for every method):**

• **SciFact** (300 claims / 5,183 abstracts): no reranker 71.7% → Jev **79.3%** (+7.62 pts, 95% CI +4.9/+10.4)
• **XQuAD-EN** (1,190 questions): 98.1% → **98.9%** (+0.82 pts), vs **99.4%** for a trained NVIDIA cross-encoder
• **Calibrated probabilities**: ECE 1.33% on XQuAD; at p ≥ 0.7, **94.9% precision** at 5.1% coverage (19x the 5% base rate)
• **Confidence-gated decisions**: partitioned reranking +0.89 pts on XQuAD; skipping generation below p = 0.9 removes 9.8% of calls and *raises* answer F1 (30.6% → 31.3%)

**What I did not find:** Jev does not beat a dedicated cross-encoder everywhere — on XQuAD it is 0.45 pts behind. The honest summary is "useful decision layer with calibrated probabilities", not "universal ranker".

Everything is reproducible for free: code, raw per-query JSONL, reports with confidence intervals, charts, and a live leaderboard.

Code + protocol: github.com/emretheus/jev-rag-benchmark
Leaderboard: huggingface.co/spaces/emretheus/jev-rag-benchmark-leaderboard
Raw artifacts: huggingface.co/datasets/emretheus/jev-rag-benchmark

#RAG #LLM #Evaluation #Jev #Benchmark #Retrieval

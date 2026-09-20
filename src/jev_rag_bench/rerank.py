from __future__ import annotations

from dataclasses import dataclass

from .clients.nvidia import NvidiaReranker
from .clients.systemone import SystemOneClient
from .retrieval import Candidate

BRANCH_LABELS = {
    "A": "no reranker (hybrid order)",
    "J": "OpenJev 0.1 batch noul (Codiv)",
    "N": "NVIDIA reranker",
    "T": "TypeSafe Jev 1.13 batch noul (real Jev)",
}

DEFAULT_BRANCHES = ["A", "J", "N"]

ALL_BRANCHES = ["A", "J", "N", "T"]


@dataclass
class BranchOutput:
    order: list[str]
    scores: list[float]
    probs: list[float] | None
    latency_ms: float
    resolved_model: str | None
    input_tokens: int | None = None
    state_chars: int | None = None
    cost_usd: float | None = None


def _systemone_output(result, candidates: list[Candidate]) -> BranchOutput:
    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (-result.probabilities[index], candidates[index].hybrid_rank),
    )
    return BranchOutput(
        order=[candidates[index].doc_id for index in ranked],
        scores=[result.probabilities[index] for index in ranked],
        probs=[result.probabilities[index] for index in ranked],
        latency_ms=result.latency_ms,
        resolved_model=result.resolved_model,
        input_tokens=result.input_tokens,
        state_chars=result.state_chars,
        cost_usd=result.cost_usd,
    )


def run_branch(
    branch: str,
    question: str,
    candidates: list[Candidate],
    passages: list[str],
    systemone: SystemOneClient,
    nvidia_reranker: NvidiaReranker,
    systemone_typesafe: SystemOneClient | None = None,
) -> BranchOutput:
    if branch == "A":
        order = [candidate.doc_id for candidate in candidates]
        scores = [1.0 / (rank + 1) for rank in range(len(candidates))]
        return BranchOutput(
            order=order, scores=scores, probs=None, latency_ms=0.0, resolved_model=None
        )

    if branch in ("J", "T"):
        client = systemone if branch == "J" else systemone_typesafe
        if client is None:
            raise ValueError(f"branch {branch} requires its System One client")
        return _systemone_output(client.score_relevance(question, passages), candidates)

    if branch == "N":
        result = nvidia_reranker.rerank(question, passages)
        ranked = sorted(
            range(len(candidates)),
            key=lambda index: (-result.scores[index], candidates[index].hybrid_rank),
        )
        return BranchOutput(
            order=[candidates[index].doc_id for index in ranked],
            scores=[result.scores[index] for index in ranked],
            probs=None,
            latency_ms=result.latency_ms,
            resolved_model=result.resolved_model,
            input_tokens=result.prompt_tokens,
        )

    raise ValueError(f"unknown branch: {branch}")

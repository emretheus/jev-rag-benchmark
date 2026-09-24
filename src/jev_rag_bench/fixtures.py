"""Deterministic offline stand-ins for every API client.

Fixture clients are lexical proxies. They exist so the full pipeline can run in
tests and CI with no keys and no network. Runs produced with fixtures carry
run_kind=fixture and must never be published as benchmark results.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .clients.nvidia import ChatResult, NvidiaRerankResult
from .clients.systemone import SystemOneResult
from .text import split_sentences, tokenize

FIXTURE_DIMS = 256
ABSTAIN_MARKER = "INSUFFICIENT_CONTEXT"


def _hash_int(*parts: str) -> int:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _jitter(*parts: str) -> float:
    return (_hash_int(*parts) % 1000) / 1000.0


def _overlap(query: str, passage: str) -> float:
    query_terms = set(tokenize(query))
    passage_terms = set(tokenize(passage))
    if not query_terms or not passage_terms:
        return 0.0
    return len(query_terms & passage_terms) / len(query_terms)


class FixtureEmbeddings:
    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        vectors = np.zeros((len(texts), FIXTURE_DIMS), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in set(tokenize(text)):
                value = _hash_int(token)
                vectors[row, value % FIXTURE_DIMS] += 1.0 if (value >> 20) % 2 else -1.0
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class FixtureReranker:
    def rerank(self, query: str, passages: list[str]) -> NvidiaRerankResult:
        scores = [
            6.0 * _overlap(query, passage) - 2.0 + 0.05 * _jitter(query, passage, str(index))
            for index, passage in enumerate(passages)
        ]
        return NvidiaRerankResult(
            scores=scores,
            prompt_tokens=sum(len(passage) for passage in passages) // 4,
            latency_ms=0.0,
            resolved_model="fixture-reranker",
        )


class FixtureSystemOne:
    def ask(self, state: str, questions: dict):
        question_id = next(iter(questions))
        passages_block = state.split("Passages:", 1)[-1]
        question_line = state.split("Question:", 1)[-1].split("Passages:", 1)[0]
        overlap = _overlap(question_line, passages_block)
        probability = min(0.999, max(0.001, 0.1 + overlap))
        return (
            {"answers": {question_id: {"type": "noul", "noul": probability}}},
            "fixture-openjev",
            0,
            0.0,
        )

    def score_relevance(self, question: str, passages: list[str]) -> SystemOneResult:
        probabilities = []
        for index, passage in enumerate(passages):
            logit = 12.0 * (_overlap(question, passage) - 0.35)
            probability = 1.0 / (1.0 + float(np.exp(-logit)))
            probability += 0.002 * (_jitter(question, passage, str(index)) - 0.5)
            probabilities.append(min(0.999, max(0.001, probability)))
        return SystemOneResult(
            probabilities=probabilities,
            resolved_model="fixture-openjev",
            input_tokens=0,
            latency_ms=0.0,
            state_chars=0,
        )


class FixtureGenerator:
    def answer(self, question: str, contexts: list[tuple[str, str]]) -> ChatResult:
        best_sentence = ""
        best_doc_id = ""
        best_score = 0.0
        for doc_id, text in contexts:
            for sentence in split_sentences(text):
                score = _overlap(question, sentence)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
                    best_doc_id = doc_id
        if best_score < 0.15:
            text = ABSTAIN_MARKER
        else:
            text = f"[{best_doc_id}] {best_sentence}"
        return ChatResult(
            text=text,
            resolved_model="fixture-generator",
            prompt_tokens=sum(len(context) for _, context in contexts) // 4,
            completion_tokens=len(text) // 4,
            latency_ms=0.0,
            finish_reason="stop",
        )

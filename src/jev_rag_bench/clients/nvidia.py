from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .http import HttpClient


def _min_interval(requests_per_minute: float) -> float:
    return 60.0 / requests_per_minute if requests_per_minute else 0.0


@dataclass
class NvidiaRerankResult:
    scores: list[float]
    prompt_tokens: int
    latency_ms: float
    resolved_model: str


@dataclass
class ChatResult:
    text: str
    resolved_model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str


class NvidiaEmbeddings:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        input_type: str = "query_passage",
        batch_size: int = 32,
        requests_per_minute: float = 40,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.input_type = input_type
        self.batch_size = max(1, batch_size)
        self.http = HttpClient(
            base_url,
            api_key=api_key,
            timeout=timeout,
            min_interval_s=_min_interval(requests_per_minute),
        )

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload: dict = {"input": batch, "model": self.model}
            if self.input_type:
                payload["input_type"] = (
                    kind if self.input_type == "query_passage" else self.input_type
                )
            data = self.http.post_json("/embeddings", payload)
            rows = sorted(data["data"], key=lambda row: int(row["index"]))
            vectors.extend(row["embedding"] for row in rows)
        return np.asarray(vectors, dtype=np.float32)


class NvidiaReranker:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str = "https://ai.api.nvidia.com/v1",
        requests_per_minute: float = 40,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.http = HttpClient(
            base_url,
            api_key=api_key,
            timeout=timeout,
            min_interval_s=_min_interval(requests_per_minute),
        )

    def rerank(self, query: str, passages: list[str]) -> NvidiaRerankResult:
        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": passage} for passage in passages],
            "truncate": "END",
        }
        started = time.perf_counter()
        data = self.http.post_json(f"/retrieval/{self.model}/reranking", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        scores = [float("-inf")] * len(passages)
        for row in data.get("rankings", []):
            index = int(row["index"])
            if 0 <= index < len(scores):
                scores[index] = float(row["logit"])
        usage = data.get("usage") or {}
        return NvidiaRerankResult(
            scores=scores,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            latency_ms=latency_ms,
            resolved_model=self.model,
        )


class OpenAIChat:
    """Chat client for any OpenAI-compatible endpoint (NVIDIA NIM, Codiv)."""
    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_tokens: int = 400,
        temperature: float = 0.0,
        requests_per_minute: float = 40,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.http = HttpClient(
            base_url,
            api_key=api_key,
            timeout=timeout,
            min_interval_s=_min_interval(requests_per_minute),
        )

    def generate(self, messages: list[dict]) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        started = time.perf_counter()
        data = self.http.post_json("/chat/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            text=str(message.get("content") or ""),
            resolved_model=str(data.get("model", self.model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            finish_reason=str(choice.get("finish_reason") or ""),
        )

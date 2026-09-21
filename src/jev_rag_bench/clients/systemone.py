from __future__ import annotations

import time
from dataclasses import dataclass

from .http import HttpClient


@dataclass
class SystemOneResult:
    probabilities: list[float]
    resolved_model: str
    input_tokens: int
    latency_ms: float
    state_chars: int
    cost_usd: float | None = None


def build_state(
    question: str,
    passages: list[str],
    max_passage_chars: int = 4000,
    max_state_chars: int = 200000,
) -> str:
    blocks = []
    for index, passage in enumerate(passages):
        text = " ".join(passage.split())
        if len(text) > max_passage_chars:
            text = text[:max_passage_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"[{index}] {text}")
    state = f"Question: {question}\n\nCandidate passages:\n\n" + "\n\n".join(blocks)
    if len(state) > max_state_chars:
        state = state[:max_state_chars]
    return state


def extract_noul_probability(answer: dict) -> float:
    if answer.get("noul") is not None:
        return float(answer["noul"])
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        if "yes" in probabilities:
            return float(probabilities["yes"])
        return float(next(iter(probabilities.values())))
    raise ValueError(f"cannot read a noul probability from answer: {answer}")


class SystemOneClient:
    """System One client for Jev-compatible decisions endpoints.

    Works with the Vercel AI Gateway TypeSafe-compatible API
    (`/typesafe/v1/systemone`) and OpenRouter (`/alpha/decisions`) through a
    configurable base_url + decisions_path.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        instruction_template: str,
        decisions_path: str = "/v1/systemone",
        max_passage_chars: int = 4000,
        max_state_chars: int = 200000,
        requests_per_minute: float = 600,
        timeout: float = 180.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.instruction_template = instruction_template
        self.decisions_path = decisions_path
        self.max_passage_chars = max_passage_chars
        self.max_state_chars = max_state_chars
        min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.http = HttpClient(
            base_url,
            api_key=api_key,
            timeout=timeout,
            min_interval_s=min_interval,
            headers=extra_headers,
        )

    def ask(self, state: str, questions: dict) -> tuple[dict, str, int, float]:
        payload = {"model": self.model, "state": state, "questions": questions}
        started = time.perf_counter()
        data = self.http.post_json(self.decisions_path, payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
        return data, str(data.get("model", self.model)), input_tokens, latency_ms

    def score_relevance(self, question: str, passages: list[str]) -> SystemOneResult:
        if not passages:
            return SystemOneResult([], self.model, 0, 0.0, 0)
        state = build_state(
            question,
            passages,
            max_passage_chars=self.max_passage_chars,
            max_state_chars=self.max_state_chars,
        )
        questions = {
            f"passage_{index}": {
                "type": "noul",
                "instructions": self.instruction_template.format(index=index),
            }
            for index in range(len(passages))
        }
        data, resolved_model, input_tokens, latency_ms = self.ask(state, questions)
        answers = data.get("answers") or {}
        probabilities = [
            extract_noul_probability(answers.get(f"passage_{index}") or {})
            for index in range(len(passages))
        ]
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        return SystemOneResult(
            probabilities=probabilities,
            resolved_model=resolved_model,
            input_tokens=input_tokens,
            latency_ms=latency_ms,
            state_chars=len(state),
            cost_usd=float(cost) if cost is not None else None,
        )

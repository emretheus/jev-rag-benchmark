from __future__ import annotations

from jev_rag_bench.clients.systemone import (
    SystemOneClient,
    build_state,
    extract_noul_probability,
)


class _StubHttp:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def post_json(self, path: str, payload: dict) -> dict:
        self.calls.append((path, payload))
        return self.payload


def _docs_example_response() -> dict:
    return {
        "model": "openjev-0.1",
        "answers": {
            "passage_0": {"type": "noul", "noul": 0.91},
            "passage_1": {"type": "noul", "probabilities": {"yes": 0.22, "no": 0.78}},
        },
        "usage": {"input_tokens": 175, "output_tokens": 0},
    }


def test_extract_noul_probability_variants():
    assert extract_noul_probability({"type": "noul", "noul": 0.87}) == 0.87
    assert extract_noul_probability({"probabilities": {"yes": 0.7, "no": 0.3}}) == 0.7


def test_score_relevance_parses_response_and_builds_batch_questions():
    client = SystemOneClient(
        base_url="https://api.example.test",
        api_key="test-key",
        model="openjev-0.1",
        instruction_template="Passage [{index}] answers the question",
        decisions_path="/v1/systemone",
    )
    stub = _StubHttp(_docs_example_response())
    client.http = stub

    result = client.score_relevance(
        "What is the capital?", ["Paris is the capital.", "Mars is red."]
    )

    assert result.probabilities == [0.91, 0.22]
    assert result.resolved_model == "openjev-0.1"
    assert result.input_tokens == 175
    path, payload = stub.calls[0]
    assert path == "/v1/systemone"
    assert payload["model"] == "openjev-0.1"
    assert set(payload["questions"]) == {"passage_0", "passage_1"}
    assert payload["questions"]["passage_0"]["type"] == "noul"
    assert "Question: What is the capital?" in payload["state"]
    assert "[0] Paris is the capital." in payload["state"]


def test_openrouter_style_path_and_cost_are_used():
    response = _docs_example_response()
    response["usage"] = {"prompt_tokens": 300, "cost": 0.0000126}
    client = SystemOneClient(
        base_url="https://openrouter.ai/api",
        api_key="test-key",
        model="typesafe/jev-1.13",
        instruction_template="Passage [{index}] contains the answer",
        decisions_path="/alpha/decisions",
    )
    stub = _StubHttp(response)
    client.http = stub

    result = client.score_relevance("q", ["a", "b"])

    assert stub.calls[0][0] == "/alpha/decisions"
    assert result.input_tokens == 300
    assert abs(result.cost_usd - 0.0000126) < 1e-12


def test_build_state_truncates_passages_and_total():
    long_passage = "word " * 100
    state = build_state("q", [long_passage], max_passage_chars=20, max_state_chars=1000)
    assert " ..." in state
    assert len(state) < 200
    clipped = build_state("q", [long_passage], max_passage_chars=200, max_state_chars=50)
    assert len(clipped) <= 50

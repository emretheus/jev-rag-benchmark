"""Laya via the official Hugging Face Space (free ZeroGPU demo).

The Space exposes a generic `/run_playground` endpoint that accepts arbitrary
state and typed questions, so no local weights are needed. Requires the
`gradio_client` package and an HF token (the token is optional but raises the
ZeroGPU quota).
"""

from __future__ import annotations

import json
import threading
import time


class LayaHFSpace:
    def __init__(
        self,
        space: str = "convaiinnovations/laya-demo",
        token: str | None = None,
        api_name: str = "/run_playground",
        min_interval_s: float = 0.4,
    ) -> None:
        try:
            from gradio_client import Client
        except ImportError as error:
            raise RuntimeError(
                "gradio_client is not installed; run this with the Laya environment "
                "(uv pip install --python .venv-laya gradio_client)"
            ) from error
        self.api_name = api_name
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_call = 0.0
        self.client = Client(space, token=token)

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def ask(self, state: str, questions: dict) -> tuple[dict, str, int, float]:
        self._throttle()
        started = time.perf_counter()
        payload = self.client.predict(state, json.dumps(questions), api_name=self.api_name)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw = payload[1] if len(payload) > 1 else payload[0]
        if isinstance(raw, dict):
            data = raw
        else:
            try:
                data = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                data = {"answers": {}}
        answers = data.get("answers") or {}
        usage = data.get("usage") or {}
        return (
            answers,
            str(data.get("model", "laya")),
            int(usage.get("input_tokens", 0)),
            float(data.get("latency_ms", elapsed_ms)),
        )

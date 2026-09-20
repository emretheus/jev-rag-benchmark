from __future__ import annotations

import threading
import time

import httpx

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")
        self.status = status
        self.body = body
        self.url = url


class HttpClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 90.0,
        max_retries: int = 5,
        min_interval_s: float = 0.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            self.headers.update(headers)
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._client = httpx.Client(timeout=timeout)

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def post_json(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, json=payload)

    def get_json(self, path: str) -> dict:
        return self._request("GET", path)

    def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self._client.request(method, url, headers=self.headers, json=json)
            except httpx.TransportError as error:
                last_error = error
                if attempt == self.max_retries:
                    raise
                time.sleep(min(30.0, 1.5 * 2**attempt))
                continue
            if response.status_code in RETRY_STATUS and attempt < self.max_retries:
                delay = min(30.0, 1.5 * 2**attempt)
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                time.sleep(min(60.0, delay))
                continue
            if response.status_code >= 400:
                raise ApiError(response.status_code, response.text, url)
            return response.json()
        raise last_error if last_error else RuntimeError("request failed")

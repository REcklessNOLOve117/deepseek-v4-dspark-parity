from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class VllmClient:
    def __init__(self, endpoint: str, api_key: str = "EMPTY", timeout: float = 900.0):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            method="GET" if payload is None else "POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {path}: {body}") from exc

    def wait_ready(self, deadline_seconds: float = 1200.0) -> dict[str, Any]:
        deadline = time.monotonic() + deadline_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self._request("/v1/models")
            except Exception as exc:  # server startup is expected to refuse connections
                last_error = exc
                time.sleep(2)
        raise TimeoutError(f"server did not become ready: {last_error}")

    def tokenize(self, text: str) -> list[int]:
        response = self._request("/tokenize", {"prompt": text})
        token_ids = response.get("tokens") or response.get("token_ids")
        if not isinstance(token_ids, list):
            raise RuntimeError(f"unexpected /tokenize response: {response}")
        return [int(value) for value in token_ids]

    def detokenize(self, token_ids: list[int]) -> str:
        response = self._request("/detokenize", {"tokens": token_ids})
        text = response.get("prompt") or response.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"unexpected /detokenize response: {response}")
        return text

    def reset_prefix_cache(self) -> dict[str, Any]:
        return self._request("/reset_prefix_cache", {})

    def complete(
        self,
        *,
        model: str,
        prompt_token_ids: list[int],
        request_id: str,
        max_tokens: int,
        seed: int,
    ) -> dict[str, Any]:
        return self._request(
            "/v1/completions",
            {
                "model": model,
                "prompt": prompt_token_ids,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": seed,
                "request_id": request_id,
                "return_token_ids": True,
            },
        )

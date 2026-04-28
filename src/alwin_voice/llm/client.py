from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Iterator

import requests


def _strip_tilde_characters(text: str) -> str:
    return "".join(ch for ch in text if "TILDE" not in unicodedata.name(ch, ""))


@dataclass(slots=True)
class OllamaClient:
    endpoint: str
    model: str
    timeout_seconds: float = 45.0

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = self.endpoint.rstrip("/") + "/api/chat"
        request_options = {
            "proxies": {"http": "", "https": ""},
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
                **request_options,
            )
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"].strip()
            return _strip_tilde_characters(content)
        except requests.RequestException:
            # One retry for transient network hiccups.
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
                **request_options,
            )
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"].strip()
            return _strip_tilde_characters(content)

    def chat_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        url = self.endpoint.rstrip("/") + "/api/chat"
        request_options = {
            "proxies": {"http": "", "https": ""},
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        with requests.post(
            url,
            json=payload,
            timeout=self.timeout_seconds,
            stream=True,
            **request_options,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                data = json.loads(line)
                message = data.get("message") or {}
                token = message.get("content") or ""
                if token:
                    yield _strip_tilde_characters(token)

                if data.get("done"):
                    break

    def healthcheck(self) -> bool:
        base = self.endpoint.rstrip("/")
        request_options = {
            "proxies": {"http": "", "https": ""},
            "stream": True,
        }
        probe_paths = [
            "/api/tags",
            "/api/version",
            "/v1/models",
        ]

        for path in probe_paths:
            try:
                response = requests.get(base + path, timeout=5.0, **request_options)
                response.raise_for_status()
                response.close()
                return True
            except requests.RequestException:
                continue

        return False

import unittest
from unittest.mock import patch

import requests

from alwin_voice.llm.client import OllamaClient


class _FakeResponse:
    def __init__(self, payload: dict, lines: list[str] | None = None) -> None:
        self._payload = payload
        self._lines = lines or []

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload

    def iter_lines(self, decode_unicode: bool = True):
        del decode_unicode
        return iter(self._lines)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def close(self) -> None:
        return None


class _FailingResponse:
    def raise_for_status(self) -> None:
        raise requests.HTTPError("boom")


class TestOllamaClient(unittest.TestCase):
    def test_chat_strips_tilde_characters(self) -> None:
        client = OllamaClient(endpoint="http://127.0.0.1:11434", model="x")
        payload = {"message": {"content": "Hej~ h\u02dcall\u00e5 a\u0303!"}}

        with patch("requests.post", return_value=_FakeResponse(payload)):
            text = client.chat([{"role": "user", "content": "hej"}])

        self.assertEqual(text, "Hej hall\u00e5 a!")
        self.assertNotIn("~", text)

    def test_chat_stream_strips_tilde_characters(self) -> None:
        client = OllamaClient(endpoint="http://127.0.0.1:11434", model="x")
        lines = [
            '{"message": {"content": "Hej~ "}, "done": false}',
            '{"message": {"content": "a\\u0303"}, "done": false}',
            '{"message": {"content": "v\u00e4rlden"}, "done": true}',
        ]

        with patch("requests.post", return_value=_FakeResponse({}, lines=lines)):
            tokens = list(client.chat_stream([{"role": "user", "content": "hej"}]))

        joined = "".join(tokens)
        self.assertIn("Hej", joined)
        self.assertIn("v\u00e4rlden", joined)
        self.assertNotIn("~", joined)

    def test_healthcheck_uses_fallback_probe_paths(self) -> None:
        client = OllamaClient(endpoint="http://127.0.0.1:11434", model="x")
        responses = [
            _FailingResponse(),  # /api/tags
            _FakeResponse({"version": "0.7.0"}),  # /api/version
        ]
        seen_urls: list[str] = []

        def _fake_get(url: str, timeout: float, **kwargs: object):
            del timeout, kwargs
            seen_urls.append(url)
            return responses.pop(0)

        with patch("requests.get", side_effect=_fake_get):
            healthy = client.healthcheck()

        self.assertTrue(healthy)
        self.assertEqual(
            seen_urls,
            [
                "http://127.0.0.1:11434/api/tags",
                "http://127.0.0.1:11434/api/version",
            ],
        )

    def test_healthcheck_fails_when_all_probes_fail(self) -> None:
        client = OllamaClient(endpoint="http://127.0.0.1:11434", model="x")
        seen_urls: list[str] = []

        def _fake_get(url: str, timeout: float, **kwargs: object):
            del timeout, kwargs
            seen_urls.append(url)
            raise requests.ConnectionError("down")

        with patch("requests.get", side_effect=_fake_get):
            healthy = client.healthcheck()

        self.assertFalse(healthy)
        self.assertEqual(
            seen_urls,
            [
                "http://127.0.0.1:11434/api/tags",
                "http://127.0.0.1:11434/api/version",
                "http://127.0.0.1:11434/v1/models",
            ],
        )


if __name__ == "__main__":
    unittest.main()

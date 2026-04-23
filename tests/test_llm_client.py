import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()

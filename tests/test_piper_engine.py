import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alwin_voice.tts.piper_engine import PiperConfig, PiperEngine


class TestPiperEngine(unittest.TestCase):
    def test_synthesize_uses_utf8_input_for_sanitized_text(self) -> None:
        cfg = PiperConfig(
            executable="piper",
            model_path=Path("/tmp/model.onnx"),
            config_path=None,
            speaker=None,
            length_scale=1.0,
        )
        engine = PiperEngine(cfg)

        with patch("subprocess.run") as mock_run:
            wav_path = engine.synthesize_to_wav("Hej\u2011värld")

        self.assertTrue(wav_path.exists())
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["input"], "Hej-värld".encode("utf-8"))
        self.assertNotIn("text", kwargs)

        wav_path.unlink(missing_ok=True)

    def test_synthesize_strips_non_swedish_characters(self) -> None:
        cfg = PiperConfig(
            executable="piper",
            model_path=Path("/tmp/model.onnx"),
            config_path=None,
            speaker=None,
            length_scale=1.0,
        )
        engine = PiperEngine(cfg)

        with patch("subprocess.run") as mock_run:
            wav_path = engine.synthesize_to_wav("Hej 😊 Привет عربى åäö!")

        self.assertTrue(wav_path.exists())
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["input"], "Hej åäö!".encode("utf-8"))

        wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

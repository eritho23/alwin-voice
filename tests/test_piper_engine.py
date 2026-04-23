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
        self.assertEqual(kwargs["input"], "Hej-värld")
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["encoding"], "utf-8")

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
        self.assertEqual(kwargs["input"], "Hej åäö!")
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["encoding"], "utf-8")

        wav_path.unlink(missing_ok=True)

    def test_synthesize_strips_tilde_variants(self) -> None:
        cfg = PiperConfig(
            executable="piper",
            model_path=Path("/tmp/model.onnx"),
            config_path=None,
            speaker=None,
            length_scale=1.0,
        )
        engine = PiperEngine(cfg)

        with patch("subprocess.run") as mock_run:
            wav_path = engine.synthesize_to_wav("Hej~ h\u02dcall\u00e5 a\u0303!")

        self.assertTrue(wav_path.exists())
        kwargs = mock_run.call_args.kwargs
        self.assertIn("all\u00e5", kwargs["input"])
        self.assertIn("Hej", kwargs["input"])
        self.assertIn("a", kwargs["input"])
        self.assertNotIn("~", kwargs["input"])
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["encoding"], "utf-8")

        wav_path.unlink(missing_ok=True)

    def test_synthesize_uses_input_file_on_windows(self) -> None:
        cfg = PiperConfig(
            executable="piper",
            model_path=Path("/tmp/model.onnx"),
            config_path=None,
            speaker=None,
            length_scale=1.0,
        )
        engine = PiperEngine(cfg)

        with patch.object(engine, "_use_input_file_for_tts", return_value=True):
            with patch("subprocess.run") as mock_run:
                wav_path = engine.synthesize_to_wav("Vad händer?")

        self.assertTrue(wav_path.exists())
        args = mock_run.call_args.args[0]
        kwargs = mock_run.call_args.kwargs
        self.assertIn("--input_file", args)
        self.assertNotIn("input", kwargs)
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["encoding"], "utf-8")

        wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

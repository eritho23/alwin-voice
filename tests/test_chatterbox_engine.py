import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from alwin_voice.tts.chatterbox_engine import ChatterboxConfig, ChatterboxEngine


class TestChatterboxEngine(unittest.TestCase):
    def test_sanitize_strips_non_swedish_characters(self) -> None:
        cfg = ChatterboxConfig(
            device="cpu",
            language="sv",
            reference_audio_path=None,
            exaggeration=0.5,
            cfg_weight=0.5,
        )
        engine = ChatterboxEngine(cfg)
        self.assertEqual(engine._sanitize_tts_text("Hej 😊 Привет عربى åäö!"), "Hej åäö!")

    def test_sanitize_spells_out_numbers_in_swedish(self) -> None:
        cfg = ChatterboxConfig(
            device="cpu",
            language="sv",
            reference_audio_path=None,
            exaggeration=0.5,
            cfg_weight=0.5,
        )
        engine = ChatterboxEngine(cfg)
        self.assertEqual(
            engine._sanitize_tts_text("Jag har 2 äpplen och 14 bananer."),
            "Jag har två äpplen och fjorton bananer.",
        )

    def test_synthesize_uses_model_generate_and_writes_pcm16_wav(self) -> None:
        cfg = ChatterboxConfig(
            device="cpu",
            language="sv",
            reference_audio_path=Path("/tmp/ref.wav"),
            exaggeration=0.6,
            cfg_weight=0.4,
        )
        engine = ChatterboxEngine(cfg)

        fake_model = type("FakeModel", (), {})()
        fake_model.sr = 24000

        def _generate(**kwargs: object) -> np.ndarray:
            self.assertEqual(kwargs["language_id"], "sv")
            self.assertEqual(kwargs["audio_prompt_path"], "/tmp/ref.wav")
            self.assertEqual(kwargs["exaggeration"], 0.6)
            self.assertEqual(kwargs["cfg_weight"], 0.4)
            return np.array([[0.0, 0.3, -0.3]], dtype=np.float32)

        fake_model.generate = _generate

        with patch.object(engine, "_load_model", return_value=fake_model):
            wav_path = engine.synthesize_to_wav("Hej-värld")

        self.assertTrue(wav_path.exists())
        try:
            with wave.open(str(wav_path), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 24000)
                self.assertEqual(wf.getnframes(), 3)
        finally:
            wav_path.unlink(missing_ok=True)

    def test_synthesize_raises_on_empty_text_after_sanitize(self) -> None:
        cfg = ChatterboxConfig(
            device="cpu",
            language="sv",
            reference_audio_path=None,
            exaggeration=0.5,
            cfg_weight=0.5,
        )
        engine = ChatterboxEngine(cfg)

        with self.assertRaises(ValueError):
            engine.synthesize_to_wav("😊 Привет")


if __name__ == "__main__":
    unittest.main()

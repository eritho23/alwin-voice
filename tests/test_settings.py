import os
import tempfile
import unittest
from pathlib import Path

from alwin_voice.config.settings import load_config


class TestSettings(unittest.TestCase):
    def test_load_config_context_turns_default(self) -> None:
        original = os.environ.pop("ALWIN_CONTEXT_TURNS", None)
        try:
            cfg = load_config()
            self.assertEqual(cfg.context_turns, 12)
        finally:
            if original is not None:
                os.environ["ALWIN_CONTEXT_TURNS"] = original

    def test_load_config_model_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "voice.onnx"
            model.write_bytes(b"x")

            original = os.environ.get("ALWIN_PIPER_MODEL")
            os.environ["ALWIN_PIPER_MODEL"] = str(model)
            try:
                cfg = load_config()
                self.assertEqual(cfg.piper_model_path, model)
            finally:
                if original is None:
                    os.environ.pop("ALWIN_PIPER_MODEL", None)
                else:
                    os.environ["ALWIN_PIPER_MODEL"] = original

    def test_load_config_cpu_mode_default_and_env(self) -> None:
        original = os.environ.pop("ALWIN_CPU_MODE", None)
        try:
            cfg = load_config()
            self.assertFalse(cfg.cpu_mode)
            self.assertEqual(cfg.stt_device, "auto")
            self.assertEqual(cfg.stt_compute_type, "float16")

            os.environ["ALWIN_CPU_MODE"] = "true"
            cfg = load_config()
            self.assertTrue(cfg.cpu_mode)
            self.assertEqual(cfg.stt_device, "cpu")
            self.assertEqual(cfg.stt_compute_type, "int8")
        finally:
            if original is not None:
                os.environ["ALWIN_CPU_MODE"] = original
            else:
                os.environ.pop("ALWIN_CPU_MODE", None)

    def test_load_config_vad_preroll_default_and_env(self) -> None:
        original = os.environ.pop("ALWIN_VAD_PREROLL_SECONDS", None)
        try:
            cfg = load_config()
            self.assertEqual(cfg.vad_preroll_seconds, 0.30)

            os.environ["ALWIN_VAD_PREROLL_SECONDS"] = "0.5"
            cfg = load_config()
            self.assertEqual(cfg.vad_preroll_seconds, 0.5)
        finally:
            if original is not None:
                os.environ["ALWIN_VAD_PREROLL_SECONDS"] = original
            else:
                os.environ.pop("ALWIN_VAD_PREROLL_SECONDS", None)

    def test_load_config_barge_in_thresholds_default_and_env(self) -> None:
        original_rms = os.environ.pop("ALWIN_BARGE_IN_RMS_THRESHOLD", None)
        original_silero = os.environ.pop("ALWIN_BARGE_IN_SILERO_THRESHOLD", None)
        try:
            cfg = load_config()
            self.assertEqual(cfg.barge_in_rms_threshold, 0.03)
            self.assertEqual(cfg.barge_in_silero_threshold, 0.75)

            os.environ["ALWIN_BARGE_IN_RMS_THRESHOLD"] = "0.05"
            os.environ["ALWIN_BARGE_IN_SILERO_THRESHOLD"] = "0.85"
            cfg = load_config()
            self.assertEqual(cfg.barge_in_rms_threshold, 0.05)
            self.assertEqual(cfg.barge_in_silero_threshold, 0.85)
        finally:
            if original_rms is not None:
                os.environ["ALWIN_BARGE_IN_RMS_THRESHOLD"] = original_rms
            else:
                os.environ.pop("ALWIN_BARGE_IN_RMS_THRESHOLD", None)

            if original_silero is not None:
                os.environ["ALWIN_BARGE_IN_SILERO_THRESHOLD"] = original_silero
            else:
                os.environ.pop("ALWIN_BARGE_IN_SILERO_THRESHOLD", None)


if __name__ == "__main__":
    unittest.main()

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

    def test_load_config_model_default_falls_back_to_alma(self) -> None:
        original_env = os.environ.pop("ALWIN_PIPER_MODEL", None)
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model_dir = root / "models" / "piper"
                model_dir.mkdir(parents=True)
                fallback_model = model_dir / "sv_SE-alma-medium.onnx"
                fallback_model.write_bytes(b"x")

                os.chdir(root)
                cfg = load_config()
                self.assertEqual(
                    cfg.piper_model_path, Path("models/piper/sv_SE-alma-medium.onnx")
                )
        finally:
            os.chdir(original_cwd)
            if original_env is not None:
                os.environ["ALWIN_PIPER_MODEL"] = original_env

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

    def test_load_config_silero_vad_defaults_and_env(self) -> None:
        original_threshold = os.environ.pop("ALWIN_SILERO_THRESHOLD", None)
        original_min_silence = os.environ.pop("ALWIN_SILERO_MIN_SILENCE_MS", None)
        try:
            cfg = load_config()
            self.assertEqual(cfg.silero_threshold, 0.45)
            self.assertEqual(cfg.silero_min_silence_ms, 350)

            os.environ["ALWIN_SILERO_THRESHOLD"] = "0.55"
            os.environ["ALWIN_SILERO_MIN_SILENCE_MS"] = "500"
            cfg = load_config()
            self.assertEqual(cfg.silero_threshold, 0.55)
            self.assertEqual(cfg.silero_min_silence_ms, 500)
        finally:
            if original_threshold is not None:
                os.environ["ALWIN_SILERO_THRESHOLD"] = original_threshold
            else:
                os.environ.pop("ALWIN_SILERO_THRESHOLD", None)

            if original_min_silence is not None:
                os.environ["ALWIN_SILERO_MIN_SILENCE_MS"] = original_min_silence
            else:
                os.environ.pop("ALWIN_SILERO_MIN_SILENCE_MS", None)

    def test_load_config_unitree_local_mic_default_and_env(self) -> None:
        original = os.environ.pop("ALWIN_UNITREE_LOCAL_MIC", None)
        try:
            cfg = load_config()
            self.assertFalse(cfg.unitree_local_mic)

            os.environ["ALWIN_UNITREE_LOCAL_MIC"] = "true"
            cfg = load_config()
            self.assertTrue(cfg.unitree_local_mic)
        finally:
            if original is not None:
                os.environ["ALWIN_UNITREE_LOCAL_MIC"] = original
            else:
                os.environ.pop("ALWIN_UNITREE_LOCAL_MIC", None)


if __name__ == "__main__":
    unittest.main()

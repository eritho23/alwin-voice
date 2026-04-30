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

    def test_load_config_tts_language_default_and_env(self) -> None:
        original = os.environ.pop("ALWIN_TTS_LANGUAGE", None)
        try:
            cfg = load_config()
            self.assertEqual(cfg.tts_language, "sv")

            os.environ["ALWIN_TTS_LANGUAGE"] = "fr"
            cfg = load_config()
            self.assertEqual(cfg.tts_language, "fr")
        finally:
            if original is not None:
                os.environ["ALWIN_TTS_LANGUAGE"] = original
            else:
                os.environ.pop("ALWIN_TTS_LANGUAGE", None)

    def test_load_config_tts_reference_audio_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.wav"
            ref.write_bytes(b"x")
            original = os.environ.get("ALWIN_TTS_REFERENCE_AUDIO")
            os.environ["ALWIN_TTS_REFERENCE_AUDIO"] = str(ref)
            try:
                cfg = load_config()
                self.assertEqual(cfg.tts_reference_audio_path, ref)
            finally:
                if original is None:
                    os.environ.pop("ALWIN_TTS_REFERENCE_AUDIO", None)
                else:
                    os.environ["ALWIN_TTS_REFERENCE_AUDIO"] = original

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

    def test_load_config_tts_runtime_defaults_and_env(self) -> None:
        original_mode = os.environ.pop("ALWIN_TTS_RUNTIME_MODE", None)
        original_cmd = os.environ.pop("ALWIN_TTS_WORKER_COMMAND", None)
        original_timeout = os.environ.pop("ALWIN_TTS_WORKER_STARTUP_TIMEOUT_SECONDS", None)
        try:
            cfg = load_config()
            self.assertEqual(cfg.tts_runtime_mode, "inprocess")
            self.assertIsNone(cfg.tts_worker_command)
            self.assertEqual(cfg.tts_worker_startup_timeout_seconds, 20.0)

            os.environ["ALWIN_TTS_RUNTIME_MODE"] = "remote-stdio"
            os.environ["ALWIN_TTS_WORKER_COMMAND"] = "python -m alwin_voice.tts.worker"
            os.environ["ALWIN_TTS_WORKER_STARTUP_TIMEOUT_SECONDS"] = "7.5"
            cfg = load_config()
            self.assertEqual(cfg.tts_runtime_mode, "remote-stdio")
            self.assertEqual(cfg.tts_worker_command, "python -m alwin_voice.tts.worker")
            self.assertEqual(cfg.tts_worker_startup_timeout_seconds, 7.5)
        finally:
            if original_mode is not None:
                os.environ["ALWIN_TTS_RUNTIME_MODE"] = original_mode
            else:
                os.environ.pop("ALWIN_TTS_RUNTIME_MODE", None)

            if original_cmd is not None:
                os.environ["ALWIN_TTS_WORKER_COMMAND"] = original_cmd
            else:
                os.environ.pop("ALWIN_TTS_WORKER_COMMAND", None)

            if original_timeout is not None:
                os.environ["ALWIN_TTS_WORKER_STARTUP_TIMEOUT_SECONDS"] = original_timeout
            else:
                os.environ.pop("ALWIN_TTS_WORKER_STARTUP_TIMEOUT_SECONDS", None)


if __name__ == "__main__":
    unittest.main()

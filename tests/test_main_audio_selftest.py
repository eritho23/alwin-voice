import argparse
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

if "sounddevice" not in sys.modules:

    class _FakeInputStream:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> "_FakeInputStream":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb

    fake_sounddevice = types.SimpleNamespace(
        CallbackFlags=object,
        InputStream=_FakeInputStream,
        default=types.SimpleNamespace(device=(0, 0)),
        rec=lambda *args, **kwargs: np.zeros((16000, 1), dtype=np.float32),
        play=lambda *args, **kwargs: None,
        stop=lambda: None,
    )
    sys.modules["sounddevice"] = fake_sounddevice

from alwin_voice.config.settings import AppConfig
from alwin_voice.main import main, run_audio_selftest


class _FakeAudio:
    def __init__(
        self,
        check_errors: list[str] | None = None,
        mic_audio: np.ndarray | None = None,
    ) -> None:
        self._check_errors = check_errors or []
        self._mic_audio = (
            mic_audio if mic_audio is not None else np.ones(16000, dtype=np.float32) * 0.02
        )
        self.start_count = 0
        self.end_count = 0
        self.wav_count = 0

    @property
    def name(self) -> str:
        return "fake"

    def check(self) -> list[str]:
        return list(self._check_errors)

    def diagnostics(self) -> list[str]:
        return ["fake diagnostics"]

    def play_listen_start(self) -> None:
        self.start_count += 1

    def play_listen_end(self) -> None:
        self.end_count += 1

    def play_wav_file(self, path: Path) -> None:
        del path
        self.wav_count += 1

    def record_utterance(self) -> np.ndarray:
        return self._mic_audio

    def stop_playback(self) -> None:
        return None

    def start_barge_in_monitor(self) -> None:
        return None

    def stop_barge_in_monitor(self) -> None:
        return None

    def barge_in_detected(self) -> bool:
        return False


def _cfg() -> AppConfig:
    return AppConfig(
        ollama_endpoint="http://127.0.0.1:11434",
        ollama_model="llama3.1:8b",
        system_prompt="sys",
        cpu_mode=False,
        stt_model="small",
        stt_device="cpu",
        stt_compute_type="float16",
        stt_language="sv",
        piper_executable="piper",
        piper_model_path=Path("/tmp/missing-model.onnx"),
        piper_config_path=None,
        audio_sample_rate=16000,
        audio_channels=1,
        audio_blocksize=512,
        listen_max_seconds=12.0,
        vad_start_threshold=0.01,
        vad_end_threshold=0.016,
        vad_silence_seconds=0.2,
        vad_preroll_seconds=0.3,
        barge_in_rms_threshold=0.03,
        vad_engine="rms",
        silero_threshold=0.5,
        barge_in_silero_threshold=0.75,
        silero_min_silence_ms=150,
        silero_speech_pad_ms=20,
        context_turns=12,
        tts_speaker=None,
        tts_length_scale=1.0,
        audio_backend="local",
        unitree_network_mode=False,
        unitree_net_iface=None,
        unitree_multicast_group="239.168.123.161",
        unitree_multicast_port=5555,
        unitree_multicast_local_ip=None,
        unitree_mic_timeout_seconds=2.0,
    )


class TestMainAudioSelftest(unittest.TestCase):
    def test_run_audio_selftest_success(self) -> None:
        audio = _FakeAudio(mic_audio=np.ones(16000, dtype=np.float32) * 0.02)
        result = run_audio_selftest(
            config=_cfg(),
            audio=audio,
            notes=["note"],
            duration_seconds=1.0,
        )

        self.assertEqual(result, 0)
        self.assertEqual(audio.start_count, 1)
        self.assertEqual(audio.end_count, 1)
        self.assertEqual(audio.wav_count, 1)

    def test_run_audio_selftest_fails_on_silent_mic(self) -> None:
        audio = _FakeAudio(mic_audio=np.zeros(16000, dtype=np.float32))
        result = run_audio_selftest(
            config=_cfg(),
            audio=audio,
            notes=[],
            duration_seconds=1.0,
        )

        self.assertEqual(result, 2)

    def test_main_audio_selftest_skips_ollama_and_piper_validation(self) -> None:
        audio = _FakeAudio()

        args = argparse.Namespace(
            check=False,
            audio_selftest=True,
            selftest_seconds=1.0,
        )

        with patch("alwin_voice.main.parse_args", return_value=args):
            with patch("alwin_voice.main.load_config", return_value=_cfg()):
                with patch(
                    "alwin_voice.main.build_audio_backend", return_value=(audio, [])
                ):
                    with patch(
                        "alwin_voice.main.run_audio_selftest",
                        return_value=0,
                    ) as mocked_selftest:
                        with patch(
                            "alwin_voice.main.OllamaClient",
                            side_effect=AssertionError(
                                "OllamaClient should not be constructed"
                            ),
                        ):
                            result = main()

        self.assertEqual(result, 0)
        mocked_selftest.assert_called_once()


if __name__ == "__main__":
    unittest.main()

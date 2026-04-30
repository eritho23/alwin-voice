import unittest
from pathlib import Path
import sys
import tempfile
import types
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
        play=lambda *args, **kwargs: None,
        stop=lambda: None,
    )
    sys.modules["sounddevice"] = fake_sounddevice

from alwin_voice.config.settings import AppConfig
from alwin_voice.main import run_chat_loop


class _FakeTranscription:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeTranscriber:
    def __init__(self, text: str = "hej", texts: list[str] | None = None) -> None:
        self._fallback = text
        self._texts = list(texts) if texts is not None else []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> _FakeTranscription:
        del audio, sample_rate
        if self._texts:
            return _FakeTranscription(self._texts.pop(0))
        return _FakeTranscription(self._fallback)


class _FakeContext:
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        self.users: list[str] = []
        self.assistants: list[str] = []

    def add_user(self, text: str) -> None:
        self.users.append(text)

    def add_assistant(self, text: str) -> None:
        self.assistants.append(text)

    def as_ollama_messages(self, system_prompt: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system_prompt}]


class _FakeLLM:
    def healthcheck(self) -> bool:
        return True

    def chat_stream(self, messages: list[dict[str, str]]):
        del messages
        yield "Det"
        yield " blir avbrutet."


class _FakeChatterboxEngine:
    synthesized_texts: list[str] = []

    def __init__(self, config: object) -> None:
        self.config = config

    def synthesize_to_wav(self, chunk: str) -> Path:
        self.synthesized_texts.append(chunk)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            return Path(handle.name)


class _FakeAudio:
    def __init__(self, *, barge_checks_before_detect: int = 0) -> None:
        self.record_calls = 0
        self.playback_stops = 0
        self.monitor_starts = 0
        self.monitor_stops = 0
        self._monitor_active = False
        self._barge_in_checks = 0
        self._barge_checks_before_detect = barge_checks_before_detect
        self._detected_once = False

    @property
    def name(self) -> str:
        return "fake"

    def check(self) -> list[str]:
        return []

    def diagnostics(self) -> list[str]:
        return []

    def record_utterance(self) -> np.ndarray:
        self.record_calls += 1
        if self.record_calls <= 2:
            return np.ones(160, dtype=np.float32)
        raise KeyboardInterrupt()

    def play_listen_start(self) -> None:
        return None

    def play_listen_end(self) -> None:
        return None

    def play_wav_file(self, path: Path) -> None:
        del path

    def stop_playback(self) -> None:
        self.playback_stops += 1

    def start_barge_in_monitor(self) -> None:
        self._monitor_active = True
        self._barge_in_checks = 0
        self.monitor_starts += 1

    def stop_barge_in_monitor(self) -> None:
        self._monitor_active = False
        self.monitor_stops += 1

    def barge_in_detected(self) -> bool:
        if not self._monitor_active:
            return False
        self._barge_in_checks += 1
        if self._detected_once:
            return False
        if self._barge_in_checks > self._barge_checks_before_detect:
            self._detected_once = True
            return True
        return False


class _NoBargeAudio(_FakeAudio):
    def __init__(self) -> None:
        super().__init__(barge_checks_before_detect=10**9)

    def record_utterance(self) -> np.ndarray:
        self.record_calls += 1
        if self.record_calls == 1:
            return np.ones(160, dtype=np.float32)
        raise KeyboardInterrupt()


class TestMainBargeIn(unittest.TestCase):
    def test_voice_barge_in_stops_assistant_and_restarts_listening(self) -> None:
        fake_audio = _FakeAudio()
        fake_context = _FakeContext(max_turns=12)

        cfg = AppConfig(
            ollama_endpoint="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
            system_prompt="sys",
            cpu_mode=False,
            stt_model="small",
            stt_device="cpu",
            stt_compute_type="float16",
            stt_language="sv",
            tts_device="cpu",
            tts_language="sv",
            tts_reference_audio_path=None,
            tts_exaggeration=0.5,
            tts_cfg_weight=0.5,
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
            audio_backend="local",
            unitree_network_mode=False,
            unitree_net_iface=None,
            unitree_multicast_group="239.168.123.161",
            unitree_multicast_port=5555,
            unitree_multicast_local_ip=None,
            unitree_mic_timeout_seconds=2.0,
            unitree_local_mic=False,
        )

        with patch(
            "alwin_voice.main.build_audio_backend", return_value=(fake_audio, [])
        ):
            with patch(
                "alwin_voice.main.FasterWhisperTranscriber",
                return_value=_FakeTranscriber(texts=["Hej", "Kan du upprepa?"]),
            ):
                with patch(
                    "alwin_voice.main.ConversationContext", return_value=fake_context
                ):
                    with patch(
                        "alwin_voice.main.OllamaClient", return_value=_FakeLLM()
                    ):
                        with patch(
                            "alwin_voice.main.build_tts_backend",
                            return_value=_FakeChatterboxEngine(config=None),
                        ):
                            with self.assertRaises(KeyboardInterrupt):
                                run_chat_loop(cfg)

        self.assertEqual(fake_audio.playback_stops, 2)
        self.assertEqual(fake_audio.record_calls, 3)
        self.assertEqual(fake_audio.monitor_starts, 1)
        self.assertGreaterEqual(fake_audio.monitor_stops, 1)
        self.assertEqual(fake_context.users, ["Hej"])
        self.assertEqual(fake_context.assistants, [])

    def test_non_question_normal_turn_is_not_filtered(self) -> None:
        fake_audio = _NoBargeAudio()
        fake_context = _FakeContext(max_turns=12)
        _FakeChatterboxEngine.synthesized_texts = []

        cfg = AppConfig(
            ollama_endpoint="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
            system_prompt="sys",
            cpu_mode=False,
            stt_model="small",
            stt_device="cpu",
            stt_compute_type="float16",
            stt_language="sv",
            tts_device="cpu",
            tts_language="sv",
            tts_reference_audio_path=None,
            tts_exaggeration=0.5,
            tts_cfg_weight=0.5,
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
            audio_backend="local",
            unitree_network_mode=False,
            unitree_net_iface=None,
            unitree_multicast_group="239.168.123.161",
            unitree_multicast_port=5555,
            unitree_multicast_local_ip=None,
            unitree_mic_timeout_seconds=2.0,
            unitree_local_mic=False,
        )

        with patch(
            "alwin_voice.main.build_audio_backend", return_value=(fake_audio, [])
        ):
            with patch(
                "alwin_voice.main.FasterWhisperTranscriber",
                return_value=_FakeTranscriber("det här är ett påstående"),
            ):
                with patch(
                    "alwin_voice.main.ConversationContext", return_value=fake_context
                ):
                    with patch(
                        "alwin_voice.main.OllamaClient", return_value=_FakeLLM()
                    ):
                        with patch(
                            "alwin_voice.main.build_tts_backend",
                            return_value=_FakeChatterboxEngine(config=None),
                        ):
                            with self.assertRaises(KeyboardInterrupt):
                                run_chat_loop(cfg)

        self.assertEqual(fake_audio.monitor_starts, 1)
        self.assertEqual(fake_audio.playback_stops, 0)
        self.assertEqual(fake_context.users, ["det här är ett påstående"])
        self.assertEqual(fake_context.assistants, ["Det blir avbrutet."])
        self.assertEqual(_FakeChatterboxEngine.synthesized_texts, ["Det blir avbrutet."])

    def test_non_question_barge_in_candidate_is_ignored(self) -> None:
        fake_audio = _FakeAudio()
        fake_context = _FakeContext(max_turns=12)

        cfg = AppConfig(
            ollama_endpoint="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
            system_prompt="sys",
            cpu_mode=False,
            stt_model="small",
            stt_device="cpu",
            stt_compute_type="float16",
            stt_language="sv",
            tts_device="cpu",
            tts_language="sv",
            tts_reference_audio_path=None,
            tts_exaggeration=0.5,
            tts_cfg_weight=0.5,
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
            audio_backend="local",
            unitree_network_mode=False,
            unitree_net_iface=None,
            unitree_multicast_group="239.168.123.161",
            unitree_multicast_port=5555,
            unitree_multicast_local_ip=None,
            unitree_mic_timeout_seconds=2.0,
            unitree_local_mic=False,
        )

        with patch(
            "alwin_voice.main.build_audio_backend", return_value=(fake_audio, [])
        ):
            with patch(
                "alwin_voice.main.FasterWhisperTranscriber",
                return_value=_FakeTranscriber(texts=["Vad händer", "hmm"]),
            ):
                with patch(
                    "alwin_voice.main.ConversationContext", return_value=fake_context
                ):
                    with patch(
                        "alwin_voice.main.OllamaClient", return_value=_FakeLLM()
                    ):
                        with patch(
                            "alwin_voice.main.build_tts_backend",
                            return_value=_FakeChatterboxEngine(config=None),
                        ):
                            with self.assertRaises(KeyboardInterrupt):
                                run_chat_loop(cfg)

        self.assertGreaterEqual(fake_audio.playback_stops, 1)
        self.assertEqual(fake_context.users, ["Vad händer"])
        self.assertEqual(fake_context.assistants, ["Det blir avbrutet."])


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import types
import unittest
import wave
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
        play=lambda *args, **kwargs: None,
        stop=lambda: None,
    )
    sys.modules["sounddevice"] = fake_sounddevice

from alwin_voice.audio.backends import (
    UnitreeAudioBackend,
    UnitreeProbe,
    build_audio_backend,
    detect_unitree_runtime,
)
from alwin_voice.config.settings import AppConfig


def _cfg(audio_backend: str = "auto") -> AppConfig:
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
        piper_model_path=Path("/tmp/model.onnx"),
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
        silero_threshold=0.45,
        barge_in_silero_threshold=0.75,
        silero_min_silence_ms=350,
        silero_speech_pad_ms=20,
        context_turns=12,
        tts_speaker=None,
        tts_length_scale=1.0,
        audio_backend=audio_backend,
        unitree_network_mode=False,
        unitree_net_iface=None,
        unitree_multicast_group="239.168.123.161",
        unitree_multicast_port=5555,
        unitree_multicast_local_ip=None,
        unitree_mic_timeout_seconds=2.0,
        unitree_local_mic=False,
    )


class _FakeUnitreeAudioClient:
    def __init__(self) -> None:
        self.play_calls: list[tuple[str, str, bytes]] = []
        self.stop_calls: list[str] = []

    def PlayStream(self, app_name: str, stream_id: str, pcm_data: bytes):
        self.play_calls.append((app_name, stream_id, pcm_data))
        return 0, None

    def PlayStop(self, app_name: str) -> int:
        self.stop_calls.append(app_name)
        return 0


class TestAudioBackends(unittest.TestCase):
    def test_detect_unitree_runtime_env_override(self) -> None:
        with patch("os.getenv", return_value="true"):
            on_robot, marker = detect_unitree_runtime()
        self.assertTrue(on_robot)
        self.assertEqual(marker, "env:ALWIN_UNITREE_ROBOT")

    def test_auto_stays_local_when_sdk_present_off_robot(self) -> None:
        probe = UnitreeProbe(
            sdk_module="unitree_sdk2py",
            sdk_available=True,
            channel_module="unitree_sdk2py.core.channel",
            channel_api_available=True,
            g1_audio_client_module="unitree_sdk2py.g1.audio.g1_audio_client",
            g1_audio_api_available=True,
            vui_module="unitree_sdk2py.go2.vui.vui_client",
            vui_available=True,
            running_on_robot=False,
            robot_runtime_marker=None,
        )
        with patch("alwin_voice.audio.backends.probe_unitree_sdk", return_value=probe):
            backend, notes = build_audio_backend(_cfg(audio_backend="auto"))

        self.assertEqual(backend.name, "local")
        self.assertTrue(any("kept local" in note for note in notes))

    def test_auto_selects_unitree_on_robot(self) -> None:
        probe = UnitreeProbe(
            sdk_module="unitree_sdk2py",
            sdk_available=True,
            channel_module="unitree_sdk2py.core.channel",
            channel_api_available=True,
            g1_audio_client_module="unitree_sdk2py.g1.audio.g1_audio_client",
            g1_audio_api_available=True,
            vui_module="unitree_sdk2py.go2.vui.vui_client",
            vui_available=True,
            running_on_robot=True,
            robot_runtime_marker="path:/etc/unitree-release",
        )
        with patch("alwin_voice.audio.backends.probe_unitree_sdk", return_value=probe):
            backend, notes = build_audio_backend(_cfg(audio_backend="auto"))

        self.assertEqual(backend.name, "unitree")
        self.assertTrue(any("auto chose unitree" in note for note in notes))

    def test_auto_selects_unitree_when_network_mode_enabled_off_robot(self) -> None:
        probe = UnitreeProbe(
            sdk_module="unitree_sdk2py",
            sdk_available=True,
            channel_module="unitree_sdk2py.core.channel",
            channel_api_available=True,
            g1_audio_client_module="unitree_sdk2py.g1.audio.g1_audio_client",
            g1_audio_api_available=True,
            vui_module="unitree_sdk2py.go2.vui.vui_client",
            vui_available=True,
            running_on_robot=False,
            robot_runtime_marker=None,
        )
        cfg = _cfg(audio_backend="auto")
        cfg.unitree_network_mode = True
        with patch("alwin_voice.audio.backends.probe_unitree_sdk", return_value=probe):
            backend, notes = build_audio_backend(cfg)

        self.assertEqual(backend.name, "unitree")
        self.assertTrue(any("auto chose unitree" in note for note in notes))

    def test_unitree_backend_uses_sdk_speaker_stream(self) -> None:
        cfg = _cfg(audio_backend="unitree")
        backend = UnitreeAudioBackend(cfg)
        backend._probe = UnitreeProbe(
            sdk_module="unitree_sdk2py",
            sdk_available=True,
            channel_module="unitree_sdk2py.core.channel",
            channel_api_available=True,
            g1_audio_client_module="unitree_sdk2py.g1.audio.g1_audio_client",
            g1_audio_api_available=True,
            vui_module="unitree_sdk2py.go2.vui.vui_client",
            vui_available=True,
            running_on_robot=True,
            robot_runtime_marker="env:ALWIN_UNITREE_ROBOT",
        )

        fake_client = _FakeUnitreeAudioClient()
        backend._unitree_audio_client = fake_client
        backend._ensure_unitree_audio_client = lambda: True

        local_calls: list[Path] = []
        backend._local.play_wav_file = lambda path: local_calls.append(path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        samples = (0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(
            np.float32
        )
        pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm.tobytes())

        try:
            backend.play_wav_file(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)

        self.assertGreater(len(fake_client.play_calls), 0)
        self.assertEqual(local_calls, [])

    def test_unitree_backend_accepts_32bit_pcm_wav(self) -> None:
        cfg = _cfg(audio_backend="unitree")
        backend = UnitreeAudioBackend(cfg)
        backend._probe = UnitreeProbe(
            sdk_module="unitree_sdk2py",
            sdk_available=True,
            channel_module="unitree_sdk2py.core.channel",
            channel_api_available=True,
            g1_audio_client_module="unitree_sdk2py.g1.audio.g1_audio_client",
            g1_audio_api_available=True,
            vui_module="unitree_sdk2py.go2.vui.vui_client",
            vui_available=True,
            running_on_robot=True,
            robot_runtime_marker="env:ALWIN_UNITREE_ROBOT",
        )

        fake_client = _FakeUnitreeAudioClient()
        backend._unitree_audio_client = fake_client
        backend._ensure_unitree_audio_client = lambda: True

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        samples = (0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(
            np.float32
        )
        pcm32 = np.clip(samples * 2147483647.0, -2147483648.0, 2147483647.0).astype(
            np.int32
        )

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(4)
            wf.setframerate(22050)
            wf.writeframes(pcm32.tobytes())

        try:
            backend.play_wav_file(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)

        self.assertGreater(len(fake_client.play_calls), 0)

    def test_network_record_raises_only_when_no_packets(self) -> None:
        cfg = _cfg(audio_backend="unitree")
        cfg.unitree_network_mode = True
        backend = UnitreeAudioBackend(cfg)

        def _fake_record_no_packets() -> np.ndarray:
            backend._last_multicast_packet_count = 0
            return np.array([], dtype=np.float32)

        backend._record_utterance_via_multicast = _fake_record_no_packets
        with self.assertRaisesRegex(RuntimeError, "No microphone packets"):
            backend.record_utterance()

    def test_network_record_empty_with_packets_is_not_error(self) -> None:
        cfg = _cfg(audio_backend="unitree")
        cfg.unitree_network_mode = True
        backend = UnitreeAudioBackend(cfg)

        def _fake_record_with_packets() -> np.ndarray:
            backend._last_multicast_packet_count = 5
            return np.array([], dtype=np.float32)

        backend._record_utterance_via_multicast = _fake_record_with_packets
        recorded = backend.record_utterance()
        self.assertEqual(recorded.size, 0)

    def test_network_mode_can_use_local_microphone(self) -> None:
        cfg = _cfg(audio_backend="unitree")
        cfg.unitree_network_mode = True
        cfg.unitree_local_mic = True
        backend = UnitreeAudioBackend(cfg)

        expected = np.array([0.1, -0.1], dtype=np.float32)
        backend._local.record_utterance = lambda: expected
        backend._record_utterance_via_multicast = lambda: np.array([], dtype=np.float32)

        recorded = backend.record_utterance()
        self.assertTrue(np.array_equal(recorded, expected))

    def test_unitree_play_pcm_waits_after_stream_send(self) -> None:
        cfg = _cfg(audio_backend="unitree")
        backend = UnitreeAudioBackend(cfg)
        fake_client = _FakeUnitreeAudioClient()
        backend._unitree_audio_client = fake_client
        backend._ensure_unitree_audio_client = lambda: True

        pcm_bytes = (np.ones(3200, dtype=np.int16)).tobytes()
        with patch("alwin_voice.audio.backends.time.sleep") as mocked_sleep:
            ok = backend._play_pcm_via_unitree(pcm_bytes)

        self.assertTrue(ok)
        self.assertGreater(len(fake_client.play_calls), 0)
        self.assertGreaterEqual(mocked_sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()

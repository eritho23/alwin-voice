import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alwin_voice.config.settings import AppConfig
from alwin_voice.tts.backends import StdioTTSBackend, build_tts_backend
from alwin_voice.tts.chatterbox_engine import ChatterboxConfig


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, chunk: str) -> int:
        self.writes.append(chunk)
        return len(chunk)

    def flush(self) -> None:
        return None


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    def readline(self) -> str:
        if not self._lines:
            return ""
        return self._lines.pop(0)


class _FakeProc:
    def __init__(self, stdout_lines: list[str]) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_lines)
        self._terminated = False

    def poll(self) -> int | None:
        return 0 if self._terminated else None

    def terminate(self) -> None:
        self._terminated = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self._terminated = True
        return 0

    def kill(self) -> None:
        self._terminated = True


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
        tts_runtime_mode="inprocess",
        tts_worker_command=None,
        tts_worker_startup_timeout_seconds=2.0,
    )


class TestTTSBackends(unittest.TestCase):
    def test_stdio_backend_roundtrip(self) -> None:
        wav_bytes = b"RIFFdemo"
        response = json.dumps(
            {"id": 1, "result": {"wav_b64": base64.b64encode(wav_bytes).decode("ascii")}}
        )
        fake_proc = _FakeProc([response + "\n"])
        cfg = ChatterboxConfig(
            device="cpu",
            language="sv",
            reference_audio_path=None,
            exaggeration=0.5,
            cfg_weight=0.5,
        )

        with patch("alwin_voice.tts.backends.subprocess.Popen", return_value=fake_proc):
            backend = StdioTTSBackend(
                cfg=cfg,
                worker_command="python -m alwin_voice.tts.worker",
                timeout_seconds=2.0,
            )
            wav_path = backend.synthesize_to_wav("hej")

        self.assertTrue(wav_path.exists())
        try:
            self.assertEqual(wav_path.read_bytes(), wav_bytes)
        finally:
            wav_path.unlink(missing_ok=True)

        self.assertEqual(len(fake_proc.stdin.writes), 1)
        payload = json.loads(fake_proc.stdin.writes[0].strip())
        self.assertEqual(payload["method"], "synthesize")
        self.assertEqual(payload["params"]["text"], "hej")

    def test_stdio_backend_ignores_noise_and_stale_ids(self) -> None:
        wav_bytes = b"RIFFdemo2"
        response = json.dumps(
            {"id": 1, "result": {"wav_b64": base64.b64encode(wav_bytes).decode("ascii")}}
        )
        stale = json.dumps({"id": 99, "result": {"wav_b64": "ignored"}})
        fake_proc = _FakeProc(["not-json\n", stale + "\n", response + "\n"])
        cfg = ChatterboxConfig(
            device="cpu",
            language="sv",
            reference_audio_path=None,
            exaggeration=0.5,
            cfg_weight=0.5,
        )

        with patch("alwin_voice.tts.backends.subprocess.Popen", return_value=fake_proc):
            backend = StdioTTSBackend(
                cfg=cfg,
                worker_command="python -m alwin_voice.tts.worker",
                timeout_seconds=2.0,
            )
            wav_path = backend.synthesize_to_wav("hej")

        try:
            self.assertEqual(wav_path.read_bytes(), wav_bytes)
        finally:
            wav_path.unlink(missing_ok=True)

    def test_build_tts_backend_remote_requires_command(self) -> None:
        cfg = _cfg()
        cfg.tts_runtime_mode = "remote-stdio"
        cfg.tts_worker_command = None

        with self.assertRaisesRegex(RuntimeError, "ALWIN_TTS_WORKER_COMMAND"):
            build_tts_backend(cfg)

    def test_build_tts_backend_inprocess(self) -> None:
        cfg = _cfg()
        with patch("alwin_voice.tts.backends.ChatterboxEngine") as mocked_engine:
            backend = build_tts_backend(cfg)

        self.assertIsNotNone(backend)
        mocked_engine.assert_called_once()


if __name__ == "__main__":
    unittest.main()

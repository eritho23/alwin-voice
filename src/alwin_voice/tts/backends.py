from __future__ import annotations

import base64
import json
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Protocol

from alwin_voice.config.settings import AppConfig
from alwin_voice.tts.chatterbox_engine import ChatterboxConfig, ChatterboxEngine


class TTSBackend(Protocol):
    def synthesize_to_wav(self, text: str) -> Path: ...


@dataclass(slots=True)
class InProcessTTSBackend:
    engine: ChatterboxEngine

    def synthesize_to_wav(self, text: str) -> Path:
        return self.engine.synthesize_to_wav(text)


class StdioTTSBackend:
    def __init__(
        self,
        cfg: ChatterboxConfig,
        worker_command: str,
        timeout_seconds: float,
    ) -> None:
        self._cfg = cfg
        self._worker_command = worker_command
        self._timeout_seconds = timeout_seconds
        self._request_id = 0
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        cmd = shlex.split(self._worker_command)
        if not cmd:
            raise RuntimeError("ALWIN_TTS_WORKER_COMMAND is empty")

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        return self._proc

    def _readline_with_timeout(self, proc: subprocess.Popen[str]) -> str:
        assert proc.stdout is not None
        queue: Queue[str] = Queue(maxsize=1)

        def _target() -> None:
            line = proc.stdout.readline()
            queue.put(line)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(self._timeout_seconds)
        if thread.is_alive():
            raise RuntimeError("TTS worker timed out waiting for response")

        if queue.empty():
            raise RuntimeError("TTS worker returned no response")
        return queue.get()

    def _read_matching_response(self, proc: subprocess.Popen[str], req_id: int) -> dict:
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError("TTS worker timed out waiting for response")

            line = self._readline_with_timeout(proc)
            if not line:
                raise RuntimeError("TTS worker closed stdout unexpectedly")

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # Ignore non-JSON stdout noise from runtime/libs and keep waiting.
                continue

            if payload.get("id") != req_id:
                # Ignore stale/out-of-order frames and wait for current response.
                continue
            return payload

    def synthesize_to_wav(self, text: str) -> Path:
        with self._lock:
            proc = self._ensure_started()
            if proc.stdin is None:
                raise RuntimeError("TTS worker stdin unavailable")

            self._request_id += 1
            req = {
                "id": self._request_id,
                "method": "synthesize",
                "params": {
                    "text": text,
                    "device": self._cfg.device,
                    "language": self._cfg.language,
                    "reference_audio_path": (
                        str(self._cfg.reference_audio_path)
                        if self._cfg.reference_audio_path is not None
                        else None
                    ),
                    "exaggeration": self._cfg.exaggeration,
                    "cfg_weight": self._cfg.cfg_weight,
                },
            }
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            payload = self._read_matching_response(proc, self._request_id)
            if "error" in payload:
                raise RuntimeError(f"TTS worker error: {payload['error']}")

            result = payload.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("TTS worker returned invalid result")
            wav_b64 = result.get("wav_b64")
            if not isinstance(wav_b64, str) or not wav_b64:
                raise RuntimeError("TTS worker did not return WAV payload")

            wav_bytes = base64.b64decode(wav_b64.encode("ascii"))
            with tempfile.NamedTemporaryFile(
                prefix="alwin_tts_", suffix=".wav", delete=False
            ) as tmp:
                tmp.write(wav_bytes)
                return Path(tmp.name)

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)
        self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return


def build_tts_backend(config: AppConfig) -> TTSBackend:
    chatterbox_cfg = ChatterboxConfig(
        device=config.tts_device,
        language=config.tts_language,
        reference_audio_path=config.tts_reference_audio_path,
        exaggeration=config.tts_exaggeration,
        cfg_weight=config.tts_cfg_weight,
    )
    if config.tts_runtime_mode == "inprocess":
        return InProcessTTSBackend(engine=ChatterboxEngine(chatterbox_cfg))

    command = config.tts_worker_command
    if not command:
        raise RuntimeError(
            "ALWIN_TTS_WORKER_COMMAND is required for ALWIN_TTS_RUNTIME_MODE=remote-stdio"
        )

    return StdioTTSBackend(
        cfg=chatterbox_cfg,
        worker_command=command,
        timeout_seconds=config.tts_worker_startup_timeout_seconds,
    )

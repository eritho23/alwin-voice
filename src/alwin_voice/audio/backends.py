from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import sounddevice as sd

from alwin_voice.audio.player import AudioPlayer
from alwin_voice.audio.recorder import SileroVADRecorder, VADRecorder
from alwin_voice.config.settings import AppConfig


class AudioBackend(Protocol):
    @property
    def name(self) -> str: ...

    def check(self) -> list[str]: ...

    def record_utterance(self) -> np.ndarray: ...

    def play_listen_start(self) -> None: ...

    def play_listen_end(self) -> None: ...

    def play_wav_file(self, path: Path) -> None: ...

    def diagnostics(self) -> list[str]: ...


@dataclass(slots=True)
class UnitreeProbe:
    sdk_module: str | None
    sdk_available: bool
    audio_module: str | None
    audio_api_available: bool
    vui_module: str | None
    vui_available: bool


def probe_unitree_sdk() -> UnitreeProbe:
    sdk_candidates = [
        "unitree_sdk2py",
        "unitree_sdk2",
    ]
    audio_candidates = [
        "unitree_sdk2py.g1.audio.g1_audio_client",
        "unitree_sdk2py.g1.audio.g1_audio_api",
    ]
    vui_candidates = [
        "unitree_sdk2py.go2.vui.vui_client",
        "unitree_sdk2py.go2.vui.vui_api",
    ]

    sdk_module = _first_importable(sdk_candidates)
    audio_module = _first_importable(audio_candidates)
    vui_module = _first_importable(vui_candidates)

    return UnitreeProbe(
        sdk_module=sdk_module,
        sdk_available=sdk_module is not None,
        audio_module=audio_module,
        audio_api_available=audio_module is not None,
        vui_module=vui_module,
        vui_available=vui_module is not None,
    )


class LocalAudioBackend:
    def __init__(self, config: AppConfig) -> None:
        self._player = AudioPlayer(sample_rate=config.audio_sample_rate)
        self._vad_engine = config.vad_engine
        if config.vad_engine == "silero":
            self._recorder = SileroVADRecorder(
                sample_rate=config.audio_sample_rate,
                channels=config.audio_channels,
                blocksize=config.audio_blocksize,
                max_seconds=config.listen_max_seconds,
                threshold=config.silero_threshold,
                min_silence_ms=config.silero_min_silence_ms,
                speech_pad_ms=config.silero_speech_pad_ms,
            )
        else:
            self._recorder = VADRecorder(
                sample_rate=config.audio_sample_rate,
                channels=config.audio_channels,
                blocksize=config.audio_blocksize,
                start_threshold=config.vad_start_threshold,
                end_threshold=config.vad_end_threshold,
                silence_seconds=config.vad_silence_seconds,
                max_seconds=config.listen_max_seconds,
            )

    @property
    def name(self) -> str:
        return "local"

    def check(self) -> list[str]:
        errors: list[str] = []
        try:
            default_input, default_output = sd.default.device
            if default_input is None or default_input < 0:
                errors.append("No default input device found")
            if default_output is None or default_output < 0:
                errors.append("No default output device found")
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"Could not query audio devices: {exc}")

        return errors

    def record_utterance(self) -> np.ndarray:
        return self._recorder.record_utterance()

    def play_listen_start(self) -> None:
        self._player.play_tone(frequency_hz=880.0, duration_ms=120)

    def play_listen_end(self) -> None:
        self._player.play_tone(frequency_hz=660.0, duration_ms=120)

    def play_wav_file(self, path: Path) -> None:
        self._player.play_wav_file(path)

    def diagnostics(self) -> list[str]:
        return [
            "Audio backend local: using sounddevice for mic/speaker I/O",
            f"Audio backend local: VAD engine={self._vad_engine}",
        ]


class UnitreeAudioBackend:
    def __init__(self, config: AppConfig) -> None:
        self._local = LocalAudioBackend(config)
        self._probe = probe_unitree_sdk()

    @property
    def name(self) -> str:
        return "unitree"

    @property
    def probe(self) -> UnitreeProbe:
        return self._probe

    def check(self) -> list[str]:
        errors: list[str] = []
        if not self._probe.sdk_available:
            errors.append(
                "Unitree SDK not importable. Install unitree-sdk2 and dependencies."
            )
        errors.extend(self._local.check())
        return errors

    def record_utterance(self) -> np.ndarray:
        # Current implementation starts with local capture on robot compute.
        return self._local.record_utterance()

    def play_listen_start(self) -> None:
        self._local.play_listen_start()

    def play_listen_end(self) -> None:
        self._local.play_listen_end()

    def play_wav_file(self, path: Path) -> None:
        # Current implementation starts with local playback on robot compute.
        self._local.play_wav_file(path)

    def diagnostics(self) -> list[str]:
        sdk_status = self._probe.sdk_module if self._probe.sdk_module else "not found"
        audio_status = (
            self._probe.audio_module if self._probe.audio_module else "not found"
        )
        vui_status = self._probe.vui_module if self._probe.vui_module else "not found"

        return [
            f"Audio backend unitree: SDK module={sdk_status}",
            f"Audio backend unitree: Audio API module={audio_status}",
            f"Audio backend unitree: VUI module={vui_status}",
            "Audio backend unitree: mic/speaker currently routed via local sounddevice fallback",
        ]


def build_audio_backend(config: AppConfig) -> tuple[AudioBackend, list[str]]:
    mode = config.audio_backend
    notes: list[str] = []

    if mode == "local":
        backend: AudioBackend = LocalAudioBackend(config)
        notes.append("Audio backend selection: forced local")
        return backend, notes

    if mode == "unitree":
        backend = UnitreeAudioBackend(config)
        notes.append("Audio backend selection: forced unitree")
        return backend, notes

    # auto mode
    candidate = UnitreeAudioBackend(config)
    if candidate.probe.sdk_available:
        notes.append("Audio backend selection: auto chose unitree")
        return candidate, notes

    notes.append("Audio backend selection: auto fell back to local")
    return LocalAudioBackend(config), notes


def _first_importable(candidates: list[str]) -> str | None:
    for module_name in candidates:
        try:
            importlib.import_module(module_name)
            return module_name
        except Exception:  # pylint: disable=broad-except
            continue
    return None

from __future__ import annotations

import importlib
import os
import socket
import struct
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import sounddevice as sd

from alwin_voice.audio.player import AudioPlayer
from alwin_voice.audio.recorder import (
    RMSInterruptionMonitor,
    SileroInterruptionMonitor,
    SileroVADRecorder,
    VADRecorder,
)
from alwin_voice.config.settings import AppConfig


class AudioBackend(Protocol):
    @property
    def name(self) -> str: ...

    def check(self) -> list[str]: ...

    def record_utterance(self) -> np.ndarray: ...

    def play_listen_start(self) -> None: ...

    def play_listen_end(self) -> None: ...

    def play_wav_file(self, path: Path) -> None: ...

    def stop_playback(self) -> None: ...

    def start_barge_in_monitor(self) -> None: ...

    def stop_barge_in_monitor(self) -> None: ...

    def barge_in_detected(self) -> bool: ...

    def diagnostics(self) -> list[str]: ...


@dataclass(slots=True)
class UnitreeProbe:
    sdk_module: str | None
    sdk_available: bool
    channel_module: str | None
    channel_api_available: bool
    g1_audio_client_module: str | None
    g1_audio_api_available: bool
    vui_module: str | None
    vui_available: bool
    running_on_robot: bool
    robot_runtime_marker: str | None


def detect_unitree_runtime() -> tuple[bool, str | None]:
    explicit = os.getenv("ALWIN_UNITREE_ROBOT")
    if explicit is not None:
        enabled = explicit.strip().lower() in {"1", "true", "yes", "on"}
        marker = "env:ALWIN_UNITREE_ROBOT" if enabled else None
        return enabled, marker

    marker_paths = [
        "/etc/unitree-release",
        "/opt/unitree",
        "/home/unitree",
    ]
    for path in marker_paths:
        if os.path.exists(path):
            return True, f"path:{path}"

    model_paths = [
        "/proc/device-tree/model",
        "/sys/firmware/devicetree/base/model",
    ]
    for model_path in model_paths:
        try:
            raw = Path(model_path).read_bytes().decode("utf-8", errors="ignore")
        except OSError:
            continue

        if "unitree" in raw.lower():
            return True, f"model:{model_path}"

    return False, None


def probe_unitree_sdk() -> UnitreeProbe:
    sdk_candidates = [
        "unitree_sdk2py",
        "unitree_sdk2",
    ]
    channel_candidates = [
        "unitree_sdk2py.core.channel",
    ]
    g1_audio_candidates = [
        "unitree_sdk2py.g1.audio.g1_audio_client",
    ]
    vui_candidates = [
        "unitree_sdk2py.go2.vui.vui_client",
        "unitree_sdk2py.b2.vui.vui_client",
    ]

    sdk_module = _first_importable(sdk_candidates)
    channel_module = _first_importable(channel_candidates)
    g1_audio_client_module = _first_importable(g1_audio_candidates)
    vui_module = _first_importable(vui_candidates)
    running_on_robot, marker = detect_unitree_runtime()

    return UnitreeProbe(
        sdk_module=sdk_module,
        sdk_available=sdk_module is not None,
        channel_module=channel_module,
        channel_api_available=channel_module is not None,
        g1_audio_client_module=g1_audio_client_module,
        g1_audio_api_available=g1_audio_client_module is not None,
        vui_module=vui_module,
        vui_available=vui_module is not None,
        running_on_robot=running_on_robot,
        robot_runtime_marker=marker,
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
                preroll_seconds=config.vad_preroll_seconds,
            )
            self._barge_in_monitor = SileroInterruptionMonitor(
                sample_rate=config.audio_sample_rate,
                channels=config.audio_channels,
                blocksize=config.audio_blocksize,
                threshold=config.barge_in_silero_threshold,
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
                preroll_seconds=config.vad_preroll_seconds,
            )
            self._barge_in_monitor = RMSInterruptionMonitor(
                sample_rate=config.audio_sample_rate,
                channels=config.audio_channels,
                blocksize=config.audio_blocksize,
                start_threshold=config.barge_in_rms_threshold,
            )

        self._barge_in_stop_event = threading.Event()
        self._barge_in_detected_event = threading.Event()
        self._barge_in_thread: threading.Thread | None = None

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

    def stop_playback(self) -> None:
        self._player.stop()

    def start_barge_in_monitor(self) -> None:
        self.stop_barge_in_monitor()
        self._barge_in_stop_event.clear()
        self._barge_in_detected_event.clear()
        self._barge_in_thread = threading.Thread(
            target=self._barge_in_monitor.monitor,
            args=(self._barge_in_stop_event, self._barge_in_detected_event),
            daemon=True,
        )
        self._barge_in_thread.start()

    def stop_barge_in_monitor(self) -> None:
        self._barge_in_stop_event.set()
        if self._barge_in_thread is not None and self._barge_in_thread.is_alive():
            self._barge_in_thread.join(timeout=0.5)
        self._barge_in_thread = None

    def barge_in_detected(self) -> bool:
        return self._barge_in_detected_event.is_set()

    def diagnostics(self) -> list[str]:
        return [
            "Audio backend local: using sounddevice for mic/speaker I/O",
            f"Audio backend local: VAD engine={self._vad_engine}",
        ]


class UnitreeAudioBackend:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._local = LocalAudioBackend(config)
        self._probe = probe_unitree_sdk()
        self._unitree_audio_client: Any | None = None
        self._unitree_stream_name = "alwin_voice"
        self._unitree_channel_initialized = False
        self._unitree_play_lock = threading.Lock()
        self._stream_counter = 0
        self._network_mode = config.unitree_network_mode
        self._strict_unitree = config.audio_backend == "unitree" or self._network_mode
        self._last_multicast_packet_count = 0

    @property
    def name(self) -> str:
        return "unitree"

    @property
    def probe(self) -> UnitreeProbe:
        return self._probe

    def _can_use_unitree_speaker(self) -> bool:
        return (
            (self._probe.running_on_robot or self._network_mode)
            and self._probe.channel_api_available
            and self._probe.g1_audio_api_available
        )

    def _extract_call_code(self, result: Any) -> int:
        if isinstance(result, tuple) and result:
            first = result[0]
            if isinstance(first, int):
                return first
            return -1
        if isinstance(result, int):
            return result
        return -1

    def _ensure_unitree_audio_client(self) -> bool:
        if self._unitree_audio_client is not None:
            return True

        if not self._can_use_unitree_speaker():
            return False

        try:
            channel_module = importlib.import_module("unitree_sdk2py.core.channel")
            if not self._unitree_channel_initialized:
                init_fn = getattr(channel_module, "ChannelFactoryInitialize")
                iface = self._config.unitree_net_iface or ""
                if iface:
                    init_fn(0, iface)
                else:
                    init_fn(0)
                self._unitree_channel_initialized = True

            audio_module = importlib.import_module(
                "unitree_sdk2py.g1.audio.g1_audio_client"
            )
            audio_client_cls = getattr(audio_module, "AudioClient")
            client = audio_client_cls()
            if hasattr(client, "SetTimeout"):
                client.SetTimeout(10.0)
            client.Init()
            self._unitree_audio_client = client
            return True
        except Exception:  # pylint: disable=broad-except
            self._unitree_audio_client = None
            return False

    def _resample_to_16k(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == 16000:
            return audio

        if audio.size == 0:
            return audio

        source_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        target_size = max(1, int(audio.size * 16000 / source_rate))
        target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)

    def _load_wav_pcm16_mono_16k(self, path: Path) -> bytes:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sample_width != 2:
            raise ValueError("Unitree speaker stream requires 16-bit PCM WAV")

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        audio = self._resample_to_16k(audio, sample_rate)
        clipped = np.clip(audio, -32768.0, 32767.0).astype(np.int16)
        return clipped.tobytes()

    def _play_pcm_via_unitree(self, pcm_bytes: bytes) -> bool:
        if not pcm_bytes:
            return True
        if not self._ensure_unitree_audio_client():
            return False

        assert self._unitree_audio_client is not None
        with self._unitree_play_lock:
            self._stream_counter += 1
            stream_id = f"{int(time.time() * 1000)}-{self._stream_counter}"
            chunk_size = 96_000
            offset = 0
            while offset < len(pcm_bytes):
                chunk = pcm_bytes[offset : offset + chunk_size]
                ret = self._unitree_audio_client.PlayStream(
                    self._unitree_stream_name,
                    stream_id,
                    chunk,
                )
                if self._extract_call_code(ret) != 0:
                    return False
                offset += len(chunk)
            return True

    def _record_utterance_via_multicast(self) -> np.ndarray:
        group = self._config.unitree_multicast_group
        port = self._config.unitree_multicast_port
        local_ip = self._config.unitree_multicast_local_ip or "0.0.0.0"
        sample_rate = self._config.audio_sample_rate
        frame_duration = self._config.audio_blocksize / sample_rate
        preroll_blocks = max(0, int(self._config.vad_preroll_seconds / frame_duration))
        preroll: deque[np.ndarray] = deque(maxlen=preroll_blocks)
        captured: list[np.ndarray] = []
        started = False
        silence_accum = 0.0
        start_time = time.monotonic()
        packet_timeout_count = 0
        max_packets_without_data = max(
            1, int(self._config.unitree_mic_timeout_seconds / 0.2)
        )
        max_duration = self._config.listen_max_seconds
        packets_received = 0

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            membership = struct.pack(
                "4s4s",
                socket.inet_aton(group),
                socket.inet_aton(local_ip),
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
            sock.settimeout(0.2)

            while True:
                if time.monotonic() - start_time >= max_duration:
                    break

                try:
                    payload, _addr = sock.recvfrom(4096)
                except socket.timeout:
                    packet_timeout_count += 1
                    if not started and packet_timeout_count >= max_packets_without_data:
                        break
                    continue

                packet_timeout_count = 0
                if not payload:
                    continue
                packets_received += 1

                pcm = np.frombuffer(payload, dtype=np.int16)
                if pcm.size == 0:
                    continue

                mono = (pcm.astype(np.float32) / 32768.0).copy()
                energy = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
                duration = mono.size / sample_rate

                if not started and preroll_blocks > 0:
                    preroll.append(mono)

                if not started:
                    if energy >= self._config.vad_start_threshold:
                        started = True
                        captured.extend(preroll)
                        captured.append(mono)
                else:
                    captured.append(mono)
                    if energy < self._config.vad_end_threshold:
                        silence_accum += duration
                        if silence_accum >= self._config.vad_silence_seconds:
                            break
                    else:
                        silence_accum = 0.0
        finally:
            sock.close()
            self._last_multicast_packet_count = packets_received

        if not captured:
            return np.array([], dtype=np.float32)
        return np.concatenate(captured).astype(np.float32)

    def _play_tone_via_unitree(self, frequency_hz: float, duration_ms: int) -> bool:
        sample_rate = 16000
        duration_s = duration_ms / 1000.0
        samples = int(sample_rate * duration_s)
        if samples <= 0:
            return True

        t = np.linspace(0, duration_s, samples, endpoint=False, dtype=np.float32)
        waveform = 0.25 * np.sin(2 * np.pi * frequency_hz * t)
        pcm = np.clip(waveform * 32767.0, -32768.0, 32767.0).astype(np.int16).tobytes()
        return self._play_pcm_via_unitree(pcm)

    def _check_local_input_device(self) -> list[str]:
        errors: list[str] = []
        try:
            default_input, _ = sd.default.device
            if default_input is None or default_input < 0:
                errors.append("No default input device found")
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"Could not query input device: {exc}")
        return errors

    def check(self) -> list[str]:
        errors: list[str] = []
        if not self._probe.sdk_available:
            errors.append(
                "Unitree SDK not importable. Install unitree-sdk2 and dependencies."
            )

        if not self._probe.running_on_robot:
            if self._network_mode:
                pass
            else:
                errors.append(
                    "Unitree backend requested but Unitree robot runtime not detected. "
                    "Set ALWIN_UNITREE_NETWORK_MODE=true for external-PC network deployment."
                )

        if self._network_mode:
            if not self._config.unitree_net_iface:
                errors.append(
                    "ALWIN_UNITREE_NET_IFACE is required in Unitree network mode"
                )

        if self._network_mode:
            pass
        else:
            errors.extend(self._check_local_input_device())

        if self._can_use_unitree_speaker() and not self._ensure_unitree_audio_client():
            errors.append(
                "Unitree speaker client unavailable: failed to initialize unitree_sdk2py.g1.audio.g1_audio_client.AudioClient"
            )

        return errors

    def record_utterance(self) -> np.ndarray:
        if self._network_mode:
            recorded = self._record_utterance_via_multicast()
            if (
                recorded.size == 0
                and self._strict_unitree
                and self._last_multicast_packet_count == 0
            ):
                raise RuntimeError(
                    "No microphone packets received from Unitree multicast stream"
                )
            return recorded

        # Capture uses local ALSA/PortAudio device.
        return self._local.record_utterance()

    def play_listen_start(self) -> None:
        if not self._play_tone_via_unitree(frequency_hz=880.0, duration_ms=120):
            if self._strict_unitree:
                raise RuntimeError("Unitree speaker path unavailable for start tone")
            self._local.play_listen_start()

    def play_listen_end(self) -> None:
        if not self._play_tone_via_unitree(frequency_hz=660.0, duration_ms=120):
            if self._strict_unitree:
                raise RuntimeError("Unitree speaker path unavailable for end tone")
            self._local.play_listen_end()

    def play_wav_file(self, path: Path) -> None:
        try:
            pcm = self._load_wav_pcm16_mono_16k(path)
            if self._play_pcm_via_unitree(pcm):
                return
        except Exception as exc:  # pylint: disable=broad-except
            if self._strict_unitree:
                raise RuntimeError(f"Failed to play WAV via Unitree speaker: {exc}") from exc

        if self._strict_unitree:
            raise RuntimeError("Unitree speaker path unavailable for WAV playback")
        self._local.play_wav_file(path)

    def stop_playback(self) -> None:
        self._local.stop_playback()
        if self._unitree_audio_client is not None:
            try:
                self._unitree_audio_client.PlayStop(self._unitree_stream_name)
            except Exception:  # pylint: disable=broad-except
                pass

    def start_barge_in_monitor(self) -> None:
        self._local.start_barge_in_monitor()

    def stop_barge_in_monitor(self) -> None:
        self._local.stop_barge_in_monitor()

    def barge_in_detected(self) -> bool:
        return self._local.barge_in_detected()

    def diagnostics(self) -> list[str]:
        sdk_status = self._probe.sdk_module if self._probe.sdk_module else "not found"
        channel_status = (
            self._probe.channel_module if self._probe.channel_module else "not found"
        )
        g1_audio_status = (
            self._probe.g1_audio_client_module
            if self._probe.g1_audio_client_module
            else "not found"
        )
        vui_status = self._probe.vui_module if self._probe.vui_module else "not found"
        runtime_status = "detected" if self._probe.running_on_robot else "not detected"
        runtime_marker = self._probe.robot_runtime_marker or "none"
        speaker_path = (
            "unitree g1 audio client"
            if self._can_use_unitree_speaker()
            else "local sounddevice fallback"
        )
        mic_path = (
            f"network multicast {self._config.unitree_multicast_group}:{self._config.unitree_multicast_port}"
            if self._network_mode
            else "local sounddevice input"
        )

        return [
            f"Audio backend unitree: SDK module={sdk_status}",
            f"Audio backend unitree: Channel module={channel_status}",
            f"Audio backend unitree: G1 audio module={g1_audio_status}",
            f"Audio backend unitree: VUI module={vui_status}",
            f"Audio backend unitree: network_mode={self._network_mode}",
            f"Audio backend unitree: runtime={runtime_status}, marker={runtime_marker}",
            f"Audio backend unitree: microphone path={mic_path}",
            f"Audio backend unitree: speaker path={speaker_path}",
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
    if candidate.probe.sdk_available and (
        candidate.probe.running_on_robot or config.unitree_network_mode
    ):
        notes.append("Audio backend selection: auto chose unitree")
        return candidate, notes

    if candidate.probe.sdk_available and not candidate.probe.running_on_robot:
        notes.append(
            "Audio backend selection: auto kept local (Unitree SDK found but not running on Unitree robot)"
        )
        return LocalAudioBackend(config), notes

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

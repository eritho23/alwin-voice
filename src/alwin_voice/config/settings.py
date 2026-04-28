from __future__ import annotations

import os
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    ollama_endpoint: str
    ollama_model: str
    system_prompt: str
    cpu_mode: bool
    stt_model: str
    stt_device: str
    stt_compute_type: str
    stt_language: str
    piper_executable: str
    piper_model_path: Path
    piper_config_path: Path | None
    audio_sample_rate: int
    audio_channels: int
    audio_blocksize: int
    listen_max_seconds: float
    vad_start_threshold: float
    vad_end_threshold: float
    vad_silence_seconds: float
    vad_preroll_seconds: float
    barge_in_rms_threshold: float
    vad_engine: str
    silero_threshold: float
    barge_in_silero_threshold: float
    silero_min_silence_ms: int
    silero_speech_pad_ms: int
    context_turns: int
    tts_speaker: int | None
    tts_length_scale: float
    audio_backend: str
    unitree_network_mode: bool
    unitree_net_iface: str | None
    unitree_multicast_group: str
    unitree_multicast_port: int
    unitree_multicast_local_ip: str | None
    unitree_mic_timeout_seconds: float
    unitree_local_mic: bool


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser()


def _default_piper_model_path() -> Path:
    preferred = Path("./models/piper/sv_SE-nst-medium.onnx")
    if preferred.exists():
        return preferred

    fallback = Path("./models/piper/sv_SE-alma-medium.onnx")
    if fallback.exists():
        return fallback

    return preferred


def load_config() -> AppConfig:
    default_piper_bin = "piper.exe" if os.name == "nt" else "piper"
    model_path = _env_optional_path("ALWIN_PIPER_MODEL") or _default_piper_model_path()
    model_path = model_path.expanduser()
    cpu_mode = _env_bool("ALWIN_CPU_MODE", False)

    return AppConfig(
        ollama_endpoint=os.getenv("ALWIN_OLLAMA_ENDPOINT", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("ALWIN_OLLAMA_MODEL", "llama3.1:8b"),
        system_prompt=os.getenv(
            "ALWIN_SYSTEM_PROMPT",
            "Du ar en hjalpsam svensk robotassistent. Svara alltid i 1 till 2 korta meningar i samtalston. ANVAND ABSOLUT INGEN MARKDOWN-FORMATTERING. Ingen punktlista, ingen rubrik, inget kodblock, och anvand aldrig tecken som *, #, _, ` eller >.",
        ),
        cpu_mode=cpu_mode,
        stt_model=os.getenv("ALWIN_STT_MODEL", "small"),
        stt_device="cpu" if cpu_mode else os.getenv("ALWIN_STT_DEVICE", "auto"),
        stt_compute_type=(
            "int8" if cpu_mode else os.getenv("ALWIN_STT_COMPUTE", "float16")
        ),
        stt_language=os.getenv("ALWIN_STT_LANGUAGE", "sv"),
        piper_executable=os.getenv("ALWIN_PIPER_BIN", default_piper_bin),
        piper_model_path=model_path,
        piper_config_path=_env_optional_path("ALWIN_PIPER_CONFIG"),
        audio_sample_rate=_env_int("ALWIN_AUDIO_SAMPLE_RATE", 16000),
        audio_channels=_env_int("ALWIN_AUDIO_CHANNELS", 1),
        audio_blocksize=_env_int("ALWIN_AUDIO_BLOCKSIZE", 512),
        listen_max_seconds=_env_float("ALWIN_LISTEN_MAX_SECONDS", 12.0),
        vad_start_threshold=_env_float("ALWIN_VAD_START_THRESHOLD", 0.010),
        vad_end_threshold=_env_float("ALWIN_VAD_END_THRESHOLD", 0.016),
        vad_silence_seconds=_env_float("ALWIN_VAD_SILENCE_SECONDS", 0.20),
        vad_preroll_seconds=_env_float("ALWIN_VAD_PREROLL_SECONDS", 0.30),
        barge_in_rms_threshold=_env_float("ALWIN_BARGE_IN_RMS_THRESHOLD", 0.03),
        vad_engine=os.getenv("ALWIN_VAD_ENGINE", "silero").lower(),
        silero_threshold=_env_float("ALWIN_SILERO_THRESHOLD", 0.45),
        barge_in_silero_threshold=_env_float("ALWIN_BARGE_IN_SILERO_THRESHOLD", 0.75),
        silero_min_silence_ms=_env_int("ALWIN_SILERO_MIN_SILENCE_MS", 350),
        silero_speech_pad_ms=_env_int("ALWIN_SILERO_SPEECH_PAD_MS", 20),
        context_turns=_env_int("ALWIN_CONTEXT_TURNS", 12),
        tts_speaker=(
            int(os.getenv("ALWIN_TTS_SPEAKER"))
            if os.getenv("ALWIN_TTS_SPEAKER")
            else None
        ),
        tts_length_scale=_env_float("ALWIN_TTS_LENGTH_SCALE", 1.0),
        audio_backend=os.getenv("ALWIN_AUDIO_BACKEND", "auto").lower(),
        unitree_network_mode=_env_bool("ALWIN_UNITREE_NETWORK_MODE", False),
        unitree_net_iface=(os.getenv("ALWIN_UNITREE_NET_IFACE") or "").strip() or None,
        unitree_multicast_group=os.getenv(
            "ALWIN_UNITREE_MULTICAST_GROUP", "239.168.123.161"
        ),
        unitree_multicast_port=_env_int("ALWIN_UNITREE_MULTICAST_PORT", 5555),
        unitree_multicast_local_ip=(
            (os.getenv("ALWIN_UNITREE_MULTICAST_LOCAL_IP") or "").strip() or None
        ),
        unitree_mic_timeout_seconds=_env_float("ALWIN_UNITREE_MIC_TIMEOUT_SECONDS", 2.0),
        unitree_local_mic=_env_bool("ALWIN_UNITREE_LOCAL_MIC", False),
    )


def validate_config(config: AppConfig) -> list[str]:
    errors: list[str] = []

    if not config.ollama_endpoint.startswith("http"):
        errors.append("ALWIN_OLLAMA_ENDPOINT must start with http or https")

    if config.context_turns < 1:
        errors.append("ALWIN_CONTEXT_TURNS must be >= 1")

    if config.audio_channels != 1:
        errors.append("Only mono audio is currently supported (ALWIN_AUDIO_CHANNELS=1)")

    if not config.piper_model_path.exists():
        errors.append(
            f"Piper model not found at {config.piper_model_path}. Set ALWIN_PIPER_MODEL."
        )

    if config.piper_config_path and not config.piper_config_path.exists():
        errors.append(
            f"Piper config not found at {config.piper_config_path}. Set ALWIN_PIPER_CONFIG."
        )

    if shutil.which(config.piper_executable) is None:
        errors.append(
            f"Piper executable '{config.piper_executable}' was not found in PATH."
        )

    if config.listen_max_seconds <= 0:
        errors.append("ALWIN_LISTEN_MAX_SECONDS must be > 0")

    if config.vad_silence_seconds <= 0:
        errors.append("ALWIN_VAD_SILENCE_SECONDS must be > 0")

    if config.vad_preroll_seconds < 0:
        errors.append("ALWIN_VAD_PREROLL_SECONDS must be >= 0")

    if config.barge_in_rms_threshold <= 0:
        errors.append("ALWIN_BARGE_IN_RMS_THRESHOLD must be > 0")

    if config.vad_engine not in {"rms", "silero"}:
        errors.append("ALWIN_VAD_ENGINE must be one of: rms, silero")

    if config.vad_engine == "silero" and config.audio_sample_rate not in {8000, 16000}:
        errors.append("Silero VAD requires ALWIN_AUDIO_SAMPLE_RATE to be 8000 or 16000")

    if not (0.0 < config.silero_threshold < 1.0):
        errors.append("ALWIN_SILERO_THRESHOLD must be > 0 and < 1")

    if not (0.0 < config.barge_in_silero_threshold < 1.0):
        errors.append("ALWIN_BARGE_IN_SILERO_THRESHOLD must be > 0 and < 1")

    if config.silero_min_silence_ms < 0:
        errors.append("ALWIN_SILERO_MIN_SILENCE_MS must be >= 0")

    if config.silero_speech_pad_ms < 0:
        errors.append("ALWIN_SILERO_SPEECH_PAD_MS must be >= 0")

    if config.audio_backend not in {"auto", "unitree", "local"}:
        errors.append("ALWIN_AUDIO_BACKEND must be one of: auto, unitree, local")

    if config.unitree_multicast_port < 1 or config.unitree_multicast_port > 65535:
        errors.append("ALWIN_UNITREE_MULTICAST_PORT must be between 1 and 65535")

    try:
        socket.inet_aton(config.unitree_multicast_group)
    except OSError:
        errors.append("ALWIN_UNITREE_MULTICAST_GROUP must be a valid IPv4 address")

    if config.unitree_multicast_local_ip:
        try:
            socket.inet_aton(config.unitree_multicast_local_ip)
        except OSError:
            errors.append(
                "ALWIN_UNITREE_MULTICAST_LOCAL_IP must be a valid IPv4 address"
            )

    if config.unitree_mic_timeout_seconds <= 0:
        errors.append("ALWIN_UNITREE_MIC_TIMEOUT_SECONDS must be > 0")

    if config.unitree_network_mode and config.audio_sample_rate != 16000:
        errors.append("Unitree network mic path requires ALWIN_AUDIO_SAMPLE_RATE=16000")

    return errors

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    ollama_endpoint: str
    ollama_model: str
    system_prompt: str
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
    context_turns: int
    tts_speaker: int | None
    tts_length_scale: float
    audio_backend: str


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


def _env_optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser()


def load_config() -> AppConfig:
    default_piper_bin = "piper.exe" if os.name == "nt" else "piper"
    model_path = Path(
        os.getenv("ALWIN_PIPER_MODEL", "./models/piper/sv_SE-nst-medium.onnx")
    ).expanduser()

    return AppConfig(
        ollama_endpoint=os.getenv("ALWIN_OLLAMA_ENDPOINT", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("ALWIN_OLLAMA_MODEL", "llama3.1:8b"),
        system_prompt=os.getenv(
            "ALWIN_SYSTEM_PROMPT",
            "Du ar en hjalpsam svensk robotassistent. Svara kortfattat och tydligt.",
        ),
        stt_model=os.getenv("ALWIN_STT_MODEL", "small"),
        stt_device=os.getenv("ALWIN_STT_DEVICE", "auto"),
        stt_compute_type=os.getenv("ALWIN_STT_COMPUTE", "int8"),
        stt_language=os.getenv("ALWIN_STT_LANGUAGE", "sv"),
        piper_executable=os.getenv("ALWIN_PIPER_BIN", default_piper_bin),
        piper_model_path=model_path,
        piper_config_path=_env_optional_path("ALWIN_PIPER_CONFIG"),
        audio_sample_rate=_env_int("ALWIN_AUDIO_SAMPLE_RATE", 16000),
        audio_channels=_env_int("ALWIN_AUDIO_CHANNELS", 1),
        audio_blocksize=_env_int("ALWIN_AUDIO_BLOCKSIZE", 1024),
        listen_max_seconds=_env_float("ALWIN_LISTEN_MAX_SECONDS", 12.0),
        vad_start_threshold=_env_float("ALWIN_VAD_START_THRESHOLD", 0.010),
        vad_end_threshold=_env_float("ALWIN_VAD_END_THRESHOLD", 0.008),
        vad_silence_seconds=_env_float("ALWIN_VAD_SILENCE_SECONDS", 0.8),
        context_turns=_env_int("ALWIN_CONTEXT_TURNS", 12),
        tts_speaker=(
            int(os.getenv("ALWIN_TTS_SPEAKER"))
            if os.getenv("ALWIN_TTS_SPEAKER")
            else None
        ),
        tts_length_scale=_env_float("ALWIN_TTS_LENGTH_SCALE", 1.0),
        audio_backend=os.getenv("ALWIN_AUDIO_BACKEND", "auto").lower(),
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

    if config.audio_backend not in {"auto", "unitree", "local"}:
        errors.append("ALWIN_AUDIO_BACKEND must be one of: auto, unitree, local")

    return errors

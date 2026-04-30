from __future__ import annotations

import os
import re
import sys
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _strip_tilde_characters(text: str) -> str:
    return "".join(ch for ch in text if "TILDE" not in unicodedata.name(ch, ""))


def _swedish_integer_to_words(value: int) -> str:
    units = {
        0: "noll",
        1: "ett",
        2: "två",
        3: "tre",
        4: "fyra",
        5: "fem",
        6: "sex",
        7: "sju",
        8: "åtta",
        9: "nio",
        10: "tio",
        11: "elva",
        12: "tolv",
        13: "tretton",
        14: "fjorton",
        15: "femton",
        16: "sexton",
        17: "sjutton",
        18: "arton",
        19: "nitton",
    }
    tens = {
        20: "tjugo",
        30: "trettio",
        40: "fyrtio",
        50: "femtio",
        60: "sextio",
        70: "sjuttio",
        80: "åttio",
        90: "nittio",
    }

    def under_hundred(n: int) -> str:
        if n < 20:
            return units[n]
        base = (n // 10) * 10
        rest = n % 10
        if rest == 0:
            return tens[base]
        return f"{tens[base]} {units[rest]}"

    def under_thousand(n: int) -> str:
        if n < 100:
            return under_hundred(n)
        hundreds = n // 100
        rest = n % 100
        prefix = "ett hundra" if hundreds == 1 else f"{units[hundreds]} hundra"
        if rest == 0:
            return prefix
        return f"{prefix} {under_hundred(rest)}"

    if value == 0:
        return units[0]

    if value < 0:
        return f"minus {_swedish_integer_to_words(-value)}"

    parts: list[str] = []
    remaining = value
    scales = [
        (1_000_000_000, "miljard", "miljarder"),
        (1_000_000, "miljon", "miljoner"),
        (1_000, "tusen", "tusen"),
    ]

    for scale_value, singular_name, plural_name in scales:
        if remaining < scale_value:
            continue
        count = remaining // scale_value
        remaining %= scale_value
        if scale_value == 1_000:
            if count == 1:
                parts.append("ett tusen")
            else:
                parts.append(f"{_swedish_integer_to_words(count)} tusen")
            continue

        if count == 1:
            parts.append(f"en {singular_name}")
        else:
            parts.append(f"{_swedish_integer_to_words(count)} {plural_name}")

    if remaining > 0:
        parts.append(under_thousand(remaining))

    return " ".join(parts)


def _replace_numbers_with_swedish_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return _swedish_integer_to_words(int(match.group(0)))

    return re.sub(r"\d+", repl, text)


@dataclass(slots=True)
class ChatterboxConfig:
    device: str
    language: str
    reference_audio_path: Path | None
    exaggeration: float
    cfg_weight: float


class ChatterboxEngine:
    def __init__(self, cfg: ChatterboxConfig) -> None:
        self.cfg = cfg
        self._model: Any | None = None

    def _sanitize_tts_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = _strip_tilde_characters(normalized)
        replacements = {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2015": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2026": "...",
            "\xa0": " ",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        normalized = _replace_numbers_with_swedish_words(normalized)

        allowed_punctuation = set(" .,!?;:'\"-()[]/%&+=*@#\n\r\t")
        sanitized_chars: list[str] = []
        for ch in normalized:
            if ch.isascii() and (ch.isalnum() or ch in allowed_punctuation):
                sanitized_chars.append(ch)
                continue
            if ch in {"å", "ä", "ö", "Å", "Ä", "Ö"}:
                sanitized_chars.append(ch)

        sanitized = "".join(sanitized_chars)
        sanitized = re.sub(r"\s+", " ", sanitized)
        return sanitized.strip()

    def _resolve_device(self) -> str:
        if self.cfg.device != "auto":
            return self.cfg.device

        try:
            import torch
        except ImportError:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError as exc:
            raise RuntimeError(
                "Chatterbox is not installed. Install dependency 'chatterbox-tts'."
            ) from exc

        self._model = ChatterboxMultilingualTTS.from_pretrained(
            device=self._resolve_device()
        )
        return self._model

    def _synthesize_waveform(self, sanitized_text: str) -> np.ndarray:
        model = self._load_model()
        ref_path = (
            str(self.cfg.reference_audio_path)
            if self.cfg.reference_audio_path is not None
            else None
        )
        wav = model.generate(
            text=sanitized_text,
            language_id=self.cfg.language,
            audio_prompt_path=ref_path,
            exaggeration=self.cfg.exaggeration,
            cfg_weight=self.cfg.cfg_weight,
        )
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        waveform = np.asarray(wav, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform[0]
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Chatterbox generated invalid or empty waveform")
        return np.clip(waveform, -1.0, 1.0)

    def synthesize_to_wav(self, text: str) -> Path:
        sanitized_text = self._sanitize_tts_text(text)
        print(f"[Chatterbox input] {sanitized_text!r}", file=sys.stderr)

        if not sanitized_text:
            raise ValueError("Cannot synthesize empty text")

        waveform = self._synthesize_waveform(sanitized_text)
        model = self._load_model()
        sample_rate = int(getattr(model, "sr", 24000))

        with tempfile.NamedTemporaryFile(
            prefix="alwin_tts_", suffix=".wav", delete=False
        ) as tmp:
            output_path = Path(tmp.name)

        pcm = np.clip(waveform * 32767.0, -32768.0, 32767.0).astype(np.int16)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())

        if os.getenv("ALWIN_TTS_DEBUG_CHATTERBOX", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print(f"[Chatterbox language] {self.cfg.language}", file=sys.stderr)
            print(f"[Chatterbox device] {self._resolve_device()}", file=sys.stderr)
            print(
                f"[Chatterbox reference] {self.cfg.reference_audio_path}",
                file=sys.stderr,
            )

        return output_path

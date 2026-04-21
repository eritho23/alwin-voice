from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PiperConfig:
    executable: str
    model_path: Path
    config_path: Path | None
    speaker: int | None
    length_scale: float


class PiperEngine:
    def __init__(self, cfg: PiperConfig) -> None:
        self.cfg = cfg

    def _sanitize_tts_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
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

    def synthesize_to_wav(self, text: str) -> Path:
        sanitized_text = self._sanitize_tts_text(text)

        if not sanitized_text:
            raise ValueError("Cannot synthesize empty text")

        with tempfile.NamedTemporaryFile(
            prefix="alwin_tts_", suffix=".wav", delete=False
        ) as tmp:
            output_path = tmp.name

        cmd = [
            self.cfg.executable,
            "--model",
            str(self.cfg.model_path),
            "--output_file",
            output_path,
            "--length_scale",
            str(self.cfg.length_scale),
        ]

        if self.cfg.config_path:
            cmd.extend(["--config", str(self.cfg.config_path)])

        if self.cfg.speaker is not None:
            cmd.extend(["--speaker", str(self.cfg.speaker)])

        env = os.environ.copy()
        env.setdefault("LANG", "C.UTF-8")
        env.setdefault("LC_CTYPE", "C.UTF-8")

        # Send UTF-8 text explicitly and pin locale defaults for Piper so Swedish
        # characters are decoded consistently across host environments.
        subprocess.run(
            cmd,
            input=sanitized_text,
            text=True,
            encoding="utf-8",
            check=True,
            capture_output=True,
            env=env,
        )

        return Path(output_path)

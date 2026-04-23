from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def _strip_tilde_characters(text: str) -> str:
    return "".join(ch for ch in text if "TILDE" not in unicodedata.name(ch, ""))


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

    def _use_input_file_for_tts(self) -> bool:
        return os.name == "nt"

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
        print(f"[Piper input] {sanitized_text!r}", file=sys.stderr)

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

        effective_config_path = self.cfg.config_path
        if effective_config_path is None:
            sidecar_path = Path(f"{self.cfg.model_path}.json")
            if sidecar_path.exists():
                effective_config_path = sidecar_path

        if effective_config_path:
            cmd.extend(["--config", str(effective_config_path)])

        if self.cfg.speaker is not None:
            cmd.extend(["--speaker", str(self.cfg.speaker)])

        env = os.environ.copy()
        env.setdefault("LANG", "C.UTF-8")
        env.setdefault("LC_CTYPE", "C.UTF-8")

        # On Windows, prefer --input_file with explicit UTF-8 to avoid
        # stdin/codepage mismatches for non-ASCII characters.
        if self._use_input_file_for_tts():
            with tempfile.NamedTemporaryFile(
                prefix="alwin_tts_",
                suffix=".txt",
                delete=False,
                mode="w",
                encoding="utf-8",
            ) as txt_file:
                txt_file.write(sanitized_text)
                text_input_path = txt_file.name

            try:
                result = subprocess.run(
                    [*cmd, "--input_file", text_input_path],
                    text=True,
                    encoding="utf-8",
                    check=True,
                    capture_output=True,
                    env=env,
                )
            finally:
                try:
                    os.unlink(text_input_path)
                except FileNotFoundError:
                    pass
        else:
            # Send UTF-8 text explicitly and pin locale defaults for Piper so Swedish
            # characters are decoded consistently across host environments.
            result = subprocess.run(
                cmd,
                input=sanitized_text,
                text=True,
                encoding="utf-8",
                check=True,
                capture_output=True,
                env=env,
            )

        if os.getenv("ALWIN_TTS_DEBUG_PIPER", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print(f"[Piper model] {self.cfg.model_path}", file=sys.stderr)
            print(f"[Piper config] {effective_config_path}", file=sys.stderr)
            stderr_output = (result.stderr or "").strip()
            if stderr_output:
                print(f"[Piper stderr] {stderr_output}", file=sys.stderr)

        return Path(output_path)

from __future__ import annotations

import subprocess
import tempfile
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

    def synthesize_to_wav(self, text: str) -> Path:
        if not text.strip():
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

        subprocess.run(
            cmd,
            input=text,
            text=True,
            check=True,
            capture_output=True,
        )

        return Path(output_path)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel


@dataclass(slots=True)
class STTResult:
    text: str
    language: str | None


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        language: str,
    ) -> None:
        self._language = language
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        if audio.size == 0:
            return STTResult(text="", language=self._language)

        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return STTResult(text=text, language=info.language if info else self._language)

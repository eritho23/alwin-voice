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
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model = self._load_model_with_fallback()

    def _load_model_with_fallback(self) -> WhisperModel:
        try:
            return WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        except (RuntimeError, ValueError) as exc:
            message = str(exc)
            if (
                "CUBLAS_STATUS_NOT_SUPPORTED" in message
                and self._device in {"cuda", "auto"}
                and self._compute_type != "float16"
            ):
                # RTX 50xx and some CUDA stacks fail on mixed/int8 modes.
                self._compute_type = "float16"
                print(
                    "STT warning: cuBLAS compute mode unsupported; retrying with float16.",
                    flush=True,
                )
                return WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
            if (
                "float16" in message.lower()
                and "do not support efficient" in message.lower()
                and self._device in {"cpu", "auto"}
            ):
                # CPU backends often require int8/int8_float32 instead of float16.
                self._compute_type = "int8"
                print(
                    "STT warning: float16 is not supported efficiently on CPU; "
                    f"retrying with {self._compute_type}.",
                    flush=True,
                )
                return WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
            raise

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

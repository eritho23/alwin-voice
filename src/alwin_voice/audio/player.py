from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def play_array(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        sr = sample_rate or self.sample_rate
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        sd.play(audio, samplerate=sr, blocking=True)

    def play_tone(
        self, frequency_hz: float, duration_ms: int, volume: float = 0.25
    ) -> None:
        duration_s = duration_ms / 1000.0
        samples = int(self.sample_rate * duration_s)
        t = np.linspace(0, duration_s, samples, endpoint=False, dtype=np.float32)
        waveform = volume * np.sin(2 * np.pi * frequency_hz * t)
        self.play_array(waveform)

    def play_wav_file(self, path: Path) -> None:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported")

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        self.play_array(audio, sample_rate=sample_rate)

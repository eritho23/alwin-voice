from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import sounddevice as sd


class VADRecorder:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        blocksize: int,
        start_threshold: float,
        end_threshold: float,
        silence_seconds: float,
        max_seconds: float,
        preroll_seconds: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds
        self.preroll_seconds = preroll_seconds

    def _rms(self, block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))

    def record_utterance(self) -> np.ndarray:
        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(
            indata: np.ndarray, frames: int, t: object, status: sd.CallbackFlags
        ) -> None:
            if status:
                pass
            q.put(indata.copy())

        frame_duration = self.blocksize / self.sample_rate
        preroll_blocks = max(0, int(self.preroll_seconds / frame_duration))
        preroll_buffer: deque[np.ndarray] = deque(maxlen=preroll_blocks)
        captured: list[np.ndarray] = []
        started = False
        silence_accum = 0.0
        max_blocks = int(self.max_seconds / frame_duration)
        blocks_seen = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=callback,
        ):
            start_time = time.monotonic()
            while True:
                try:
                    block = q.get(timeout=0.5)
                except queue.Empty:
                    if time.monotonic() - start_time > self.max_seconds:
                        break
                    continue

                if self.channels > 1:
                    mono = block.mean(axis=1)
                else:
                    mono = block[:, 0]

                energy = self._rms(mono)
                blocks_seen += 1

                if not started and preroll_blocks > 0:
                    preroll_buffer.append(mono)

                if not started:
                    if energy >= self.start_threshold:
                        started = True
                        captured.extend(preroll_buffer)
                        captured.append(mono)
                else:
                    captured.append(mono)
                    if energy < self.end_threshold:
                        silence_accum += frame_duration
                        if silence_accum >= self.silence_seconds:
                            break
                    else:
                        silence_accum = 0.0

                if blocks_seen >= max_blocks:
                    break

        if not captured:
            return np.array([], dtype=np.float32)

        return np.concatenate(captured).astype(np.float32)


class RMSInterruptionMonitor:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        blocksize: int,
        start_threshold: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.start_threshold = start_threshold

    def _rms(self, block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))

    def monitor(
        self,
        stop_event: threading.Event,
        interrupted_event: threading.Event,
    ) -> None:
        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(
            indata: np.ndarray, frames: int, t: object, status: sd.CallbackFlags
        ) -> None:
            if status:
                pass
            q.put(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=callback,
        ):
            while not stop_event.is_set() and not interrupted_event.is_set():
                try:
                    block = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.channels > 1:
                    mono = block.mean(axis=1)
                else:
                    mono = block[:, 0]

                energy = self._rms(mono)
                if energy >= self.start_threshold:
                    interrupted_event.set()
                    break


class SileroVADRecorder:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        blocksize: int,
        max_seconds: float,
        threshold: float,
        min_silence_ms: int,
        speech_pad_ms: int,
        preroll_seconds: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.max_seconds = max_seconds
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.preroll_seconds = preroll_seconds

        self._torch: Any | None = None
        self._vad_iterator: Any | None = None

    def _ensure_model(self) -> None:
        if self._vad_iterator is not None:
            return

        try:
            import torch  # type: ignore
            from silero_vad import VADIterator, load_silero_vad  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD is not installed. Install silero-vad and torch, or set ALWIN_VAD_ENGINE=rms."
            ) from exc

        model = load_silero_vad()
        self._torch = torch
        self._vad_iterator = VADIterator(
            model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    def record_utterance(self) -> np.ndarray:
        self._ensure_model()
        assert self._torch is not None
        assert self._vad_iterator is not None

        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(
            indata: np.ndarray, frames: int, t: object, status: sd.CallbackFlags
        ) -> None:
            if status:
                pass
            q.put(indata.copy())

        frame_duration = self.blocksize / self.sample_rate
        preroll_blocks = max(0, int(self.preroll_seconds / frame_duration))
        preroll_buffer: deque[np.ndarray] = deque(maxlen=preroll_blocks)
        captured: list[np.ndarray] = []
        started = False
        max_blocks = int(self.max_seconds / frame_duration)
        blocks_seen = 0

        self._vad_iterator.reset_states()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=callback,
        ):
            start_time = time.monotonic()
            while True:
                try:
                    block = q.get(timeout=0.3)
                except queue.Empty:
                    if time.monotonic() - start_time > self.max_seconds:
                        break
                    continue

                if self.channels > 1:
                    mono = block.mean(axis=1)
                else:
                    mono = block[:, 0]

                event = self._vad_iterator(
                    self._torch.from_numpy(mono), return_seconds=False
                )
                blocks_seen += 1

                if not started and preroll_blocks > 0:
                    preroll_buffer.append(mono)

                if event and "start" in event:
                    started = True
                    captured.extend(preroll_buffer)

                if started:
                    captured.append(mono)

                if started and event and "end" in event:
                    break

                if blocks_seen >= max_blocks:
                    break

        self._vad_iterator.reset_states()

        if not captured:
            return np.array([], dtype=np.float32)

        return np.concatenate(captured).astype(np.float32)


class SileroInterruptionMonitor:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        blocksize: int,
        threshold: float,
        min_silence_ms: int,
        speech_pad_ms: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

        self._torch: Any | None = None
        self._vad_iterator: Any | None = None

    def _ensure_model(self) -> None:
        if self._vad_iterator is not None:
            return

        try:
            import torch  # type: ignore
            from silero_vad import VADIterator, load_silero_vad  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD is not installed. Install silero-vad and torch, or set ALWIN_VAD_ENGINE=rms."
            ) from exc

        model = load_silero_vad()
        self._torch = torch
        self._vad_iterator = VADIterator(
            model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    def monitor(
        self,
        stop_event: threading.Event,
        interrupted_event: threading.Event,
    ) -> None:
        self._ensure_model()
        assert self._torch is not None
        assert self._vad_iterator is not None

        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(
            indata: np.ndarray, frames: int, t: object, status: sd.CallbackFlags
        ) -> None:
            if status:
                pass
            q.put(indata.copy())

        self._vad_iterator.reset_states()
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.blocksize,
                callback=callback,
            ):
                while not stop_event.is_set() and not interrupted_event.is_set():
                    try:
                        block = q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if self.channels > 1:
                        mono = block.mean(axis=1)
                    else:
                        mono = block[:, 0]

                    event = self._vad_iterator(
                        self._torch.from_numpy(mono), return_seconds=False
                    )
                    if event and "start" in event:
                        interrupted_event.set()
                        break
        finally:
            self._vad_iterator.reset_states()

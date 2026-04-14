from __future__ import annotations

import queue
import time

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
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds

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

        captured: list[np.ndarray] = []
        started = False
        silence_accum = 0.0
        frame_duration = self.blocksize / self.sample_rate
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

                if not started:
                    if energy >= self.start_threshold:
                        started = True
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

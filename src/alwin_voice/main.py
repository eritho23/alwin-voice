from __future__ import annotations

import argparse
import queue
import re
import sys
import threading
from pathlib import Path

import sounddevice as sd

from alwin_voice.audio.player import AudioPlayer
from alwin_voice.audio.recorder import VADRecorder
from alwin_voice.config.settings import AppConfig, load_config, validate_config
from alwin_voice.llm.client import OllamaClient
from alwin_voice.llm.context import ConversationContext
from alwin_voice.stt.transcriber import FasterWhisperTranscriber
from alwin_voice.tts.piper_engine import PiperConfig, PiperEngine


def _print_config_errors(errors: list[str]) -> None:
    print("Configuration errors:", file=sys.stderr)
    for err in errors:
        print(f"- {err}", file=sys.stderr)


def _check_audio_devices() -> list[str]:
    errors: list[str] = []
    try:
        default_input, default_output = sd.default.device
        if default_input is None or default_input < 0:
            errors.append("No default input device found")
        if default_output is None or default_output < 0:
            errors.append("No default output device found")
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"Could not query audio devices: {exc}")
    return errors


def _extract_complete_sentences(text: str) -> tuple[list[str], str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) <= 1:
        return [], text
    completed = [p.strip() for p in parts[:-1] if p.strip()]
    remainder = parts[-1]
    return completed, remainder


def _tts_worker(
    tts: PiperEngine, player: AudioPlayer, tts_queue: queue.Queue[str | None]
) -> None:
    while True:
        chunk = tts_queue.get()
        if chunk is None:
            break
        if not chunk.strip():
            continue

        wav_path: Path | None = None
        try:
            wav_path = tts.synthesize_to_wav(chunk)
            player.play_wav_file(wav_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"TTS error: {exc}", file=sys.stderr)
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink(missing_ok=True)


def run_chat_loop(config: AppConfig) -> None:
    player = AudioPlayer(sample_rate=config.audio_sample_rate)
    recorder = VADRecorder(
        sample_rate=config.audio_sample_rate,
        channels=config.audio_channels,
        blocksize=config.audio_blocksize,
        start_threshold=config.vad_start_threshold,
        end_threshold=config.vad_end_threshold,
        silence_seconds=config.vad_silence_seconds,
        max_seconds=config.listen_max_seconds,
    )
    transcriber = FasterWhisperTranscriber(
        model_name=config.stt_model,
        device=config.stt_device,
        compute_type=config.stt_compute_type,
        language=config.stt_language,
    )
    context = ConversationContext(max_turns=config.context_turns)
    llm = OllamaClient(endpoint=config.ollama_endpoint, model=config.ollama_model)
    tts = PiperEngine(
        PiperConfig(
            executable=config.piper_executable,
            model_path=config.piper_model_path,
            config_path=config.piper_config_path,
            speaker=config.tts_speaker,
            length_scale=config.tts_length_scale,
        )
    )

    print("Voice chat started. Press Ctrl+C to stop.")
    while True:
        player.play_tone(frequency_hz=880.0, duration_ms=120)
        print("Listening...")

        audio = recorder.record_utterance()
        print("Stopped listening. Processing...")

        player.play_tone(frequency_hz=660.0, duration_ms=120)

        stt = transcriber.transcribe(audio=audio, sample_rate=config.audio_sample_rate)
        if not stt.text:
            print("No speech detected.")
            continue

        print(f"You: {stt.text}")
        context.add_user(stt.text)
        messages = context.as_ollama_messages(system_prompt=config.system_prompt)

        tts_queue: queue.Queue[str | None] = queue.Queue()
        worker = threading.Thread(
            target=_tts_worker,
            args=(tts, player, tts_queue),
            daemon=True,
        )
        worker.start()

        reply_parts: list[str] = []
        sentence_buffer = ""

        print("Assistant: ", end="", flush=True)
        try:
            for token in llm.chat_stream(messages):
                print(token, end="", flush=True)
                reply_parts.append(token)
                sentence_buffer += token

                ready, sentence_buffer = _extract_complete_sentences(sentence_buffer)
                for sentence in ready:
                    tts_queue.put(sentence)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\nLLM error: {exc}", file=sys.stderr)
            tts_queue.put(None)
            worker.join(timeout=2)
            continue

        print()

        if sentence_buffer.strip():
            tts_queue.put(sentence_buffer.strip())

        tts_queue.put(None)
        worker.join()

        reply = "".join(reply_parts).strip()
        if not reply:
            print("Assistant returned empty response.")
            continue

        context.add_assistant(reply)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STT -> Ollama -> Piper voice loop")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate runtime configuration and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    errors = validate_config(cfg)
    errors.extend(_check_audio_devices())

    client = OllamaClient(endpoint=cfg.ollama_endpoint, model=cfg.ollama_model)
    if not client.healthcheck():
        errors.append("Could not reach Ollama API")

    if errors:
        _print_config_errors(errors)
        return 2

    if args.check:
        print("Configuration check passed.")
        return 0

    try:
        run_chat_loop(cfg)
    except KeyboardInterrupt:
        print("\nStopping voice chat.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

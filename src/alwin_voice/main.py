from __future__ import annotations

import argparse
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

from alwin_voice.audio.backends import AudioBackend, build_audio_backend
from alwin_voice.config.settings import AppConfig, load_config, validate_config
from alwin_voice.interrupts import is_clear_question_or_clarification
from alwin_voice.llm.client import OllamaClient
from alwin_voice.llm.context import ConversationContext
from alwin_voice.stt.transcriber import FasterWhisperTranscriber
from alwin_voice.tts.piper_engine import PiperConfig, PiperEngine


def _print_config_errors(errors: list[str]) -> None:
    print("Configuration errors:", file=sys.stderr)
    for err in errors:
        print(f"- {err}", file=sys.stderr)


def _detect_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except OSError:
        return False


def _print_acceleration_info(config: AppConfig) -> None:
    gpu_available = _detect_nvidia_gpu()
    gpu_status = "detected" if gpu_available else "not detected"

    if config.cpu_mode:
        stt_mode = "CPU mode enabled"
    else:
        stt_device = config.stt_device.lower()
        if stt_device == "cuda":
            stt_mode = "CUDA requested"
        elif stt_device == "cpu":
            stt_mode = "CPU forced"
        else:
            stt_mode = "auto (runtime decides)"

    print("Acceleration:")
    print(f"- NVIDIA GPU: {gpu_status}")
    print(f"- STT (faster-whisper): {stt_mode}")
    print("- LLM (Ollama): managed by Ollama runtime")
    print("- TTS (Piper): CPU path in current implementation")


def _extract_complete_sentences(text: str) -> tuple[list[str], str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) <= 1:
        return [], text
    completed = [p.strip() for p in parts[:-1] if p.strip()]
    remainder = parts[-1]
    return completed, remainder


def _tts_worker(
    tts: PiperEngine,
    audio: AudioBackend,
    tts_queue: queue.Queue[str | None],
    stop_event: threading.Event,
) -> None:
    while True:
        if stop_event.is_set():
            break
        try:
            chunk = tts_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if chunk is None:
            break
        if not chunk.strip():
            continue

        wav_path: Path | None = None
        try:
            wav_path = tts.synthesize_to_wav(chunk)
            if stop_event.is_set():
                continue
            audio.play_wav_file(wav_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"TTS error: {exc}", file=sys.stderr)
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink(missing_ok=True)


def _confirm_interrupt_text(
    audio: AudioBackend,
    transcriber: FasterWhisperTranscriber,
    config: AppConfig,
) -> str:
    audio.stop_barge_in_monitor()
    audio.stop_playback()

    captured_audio = audio.record_utterance()
    stt = transcriber.transcribe(
        audio=captured_audio,
        sample_rate=config.audio_sample_rate,
    )
    return stt.text.strip()


def run_chat_loop(config: AppConfig) -> None:
    audio, notes = build_audio_backend(config)
    for note in notes:
        print(note)
    for line in audio.diagnostics():
        print(line)

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

    print(
        "Voice chat started. Speak while assistant is talking to interrupt and listen again; press Ctrl+C while listening/processing to stop."
    )
    while True:
        tts_queue: queue.Queue[str | None] | None = None
        stop_tts_event: threading.Event | None = None
        worker: threading.Thread | None = None
        assistant_phase = False
        try:
            audio.play_listen_start()
            print("Listening...")

            captured_audio = audio.record_utterance()
            print("Stopped listening. Processing...")

            audio.play_listen_end()

            stt = transcriber.transcribe(
                audio=captured_audio,
                sample_rate=config.audio_sample_rate,
            )
            if not stt.text:
                print("No speech detected.")
                continue

            print(f"You: {stt.text}")
            context.add_user(stt.text)
            messages = context.as_ollama_messages(system_prompt=config.system_prompt)

            tts_queue: queue.Queue[str | None] = queue.Queue()
            stop_tts_event = threading.Event()
            worker = threading.Thread(
                target=_tts_worker,
                args=(tts, audio, tts_queue, stop_tts_event),
                daemon=True,
            )
            worker.start()

            reply_parts: list[str] = []
            sentence_buffer = ""

            print("Assistant: ", end="", flush=True)
            assistant_phase = True
            interrupted_by_voice = False
            stream = llm.chat_stream(messages)
            audio.start_barge_in_monitor()
            try:
                for token in stream:
                    if audio.barge_in_detected():
                        candidate = _confirm_interrupt_text(audio, transcriber, config)
                        if is_clear_question_or_clarification(candidate):
                            interrupted_by_voice = True
                            break
                        if candidate:
                            print(
                                f"Ignoring non-question interruption candidate: {candidate}"
                            )
                        audio.start_barge_in_monitor()

                    print(token, end="", flush=True)
                    reply_parts.append(token)
                    sentence_buffer += token

                    ready, sentence_buffer = _extract_complete_sentences(
                        sentence_buffer
                    )
                    for sentence in ready:
                        tts_queue.put(sentence)

                print()

                if interrupted_by_voice:
                    if hasattr(stream, "close"):
                        stream.close()
                else:
                    if sentence_buffer.strip():
                        tts_queue.put(sentence_buffer.strip())

                    tts_queue.put(None)
                    while worker.is_alive():
                        worker.join(timeout=0.1)
                        if audio.barge_in_detected():
                            candidate = _confirm_interrupt_text(
                                audio, transcriber, config
                            )
                            if is_clear_question_or_clarification(candidate):
                                interrupted_by_voice = True
                                break
                            if candidate:
                                print(
                                    "Ignoring non-question interruption candidate: "
                                    f"{candidate}"
                                )
                            audio.start_barge_in_monitor()

                if interrupted_by_voice:
                    if stop_tts_event is not None:
                        stop_tts_event.set()
                    if tts_queue is not None:
                        tts_queue.put(None)
                    audio.stop_playback()
                    if worker.is_alive():
                        worker.join(timeout=1)
                    print("Assistant interrupted by speech. Listening again...")
                    continue
            finally:
                audio.stop_barge_in_monitor()

            reply = "".join(reply_parts).strip()
            if not reply:
                print("Assistant returned empty response.")
                continue

            context.add_assistant(reply)
        except KeyboardInterrupt:
            if not assistant_phase:
                raise

            if stop_tts_event is not None:
                stop_tts_event.set()
            if tts_queue is not None:
                tts_queue.put(None)
            audio.stop_playback()
            if worker is not None and worker.is_alive():
                worker.join(timeout=1)
            print("\nAssistant interrupted. Listening again...")
            continue
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\nLLM error: {exc}", file=sys.stderr)
            if stop_tts_event is not None:
                stop_tts_event.set()
            if tts_queue is not None:
                tts_queue.put(None)
            if worker is not None and worker.is_alive():
                worker.join(timeout=2)
            continue


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

    audio, _ = build_audio_backend(cfg)
    errors.extend(audio.check())

    client = OllamaClient(endpoint=cfg.ollama_endpoint, model=cfg.ollama_model)
    if not client.healthcheck():
        errors.append("Could not reach Ollama API")

    if errors:
        _print_config_errors(errors)
        return 2

    _print_acceleration_info(cfg)

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

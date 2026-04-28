from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

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


def _rms_level(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def _capture_mic_sample(
    sample_rate: int,
    channels: int,
    duration_seconds: float,
) -> np.ndarray:
    frames = max(1, int(sample_rate * duration_seconds))
    recording = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocking=True,
    )

    if channels > 1:
        mono = recording.mean(axis=1)
    else:
        mono = recording[:, 0]

    return mono.astype(np.float32)


def _write_selftest_wav(path: Path, sample_rate: int) -> None:
    duration_s = 0.4
    tone_hz = 740.0
    t = np.linspace(
        0,
        duration_s,
        int(sample_rate * duration_s),
        endpoint=False,
        dtype=np.float32,
    )
    waveform = 0.22 * np.sin(2 * np.pi * tone_hz * t)
    pcm = np.clip(waveform * 32767.0, -32768.0, 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def run_audio_selftest(
    config: AppConfig,
    audio: AudioBackend,
    notes: list[str],
    duration_seconds: float,
) -> int:
    print("Audio self-test:")
    for note in notes:
        print(note)
    for line in audio.diagnostics():
        print(line)

    errors = audio.check()
    if errors:
        _print_config_errors(errors)
        return 2

    try:
        print("- Speaker test: start/end tones")
        audio.play_listen_start()
        audio.play_listen_end()

        print("- Speaker test: WAV playback path")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            _write_selftest_wav(wav_path, sample_rate=config.audio_sample_rate)
            audio.play_wav_file(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Audio self-test speaker failure: {exc}", file=sys.stderr)
        return 2

    try:
        print(f"- Microphone test: capture {duration_seconds:.1f}s")
        mic_audio = _capture_mic_sample(
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            duration_seconds=duration_seconds,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Audio self-test microphone failure: {exc}", file=sys.stderr)
        return 2

    rms = _rms_level(mic_audio)
    peak = float(np.max(np.abs(mic_audio))) if mic_audio.size else 0.0
    print(f"- Microphone level: rms={rms:.5f}, peak={peak:.5f}")

    if mic_audio.size == 0:
        print("Audio self-test microphone failure: captured empty buffer", file=sys.stderr)
        return 2

    if peak < 1e-4:
        print(
            "Audio self-test microphone failure: captured near-silence only; check mute/device selection.",
            file=sys.stderr,
        )
        return 2

    print("Audio self-test passed.")
    return 0


def _play_tts_response(
    tts: PiperEngine,
    audio: AudioBackend,
    transcriber: FasterWhisperTranscriber,
    config: AppConfig,
    reply: str,
) -> bool:
    if not reply.strip():
        return False

    wav_path: Path | None = None
    playback_error: Exception | None = None

    try:
        wav_path = tts.synthesize_to_wav(reply)

        def _playback_target() -> None:
            nonlocal playback_error
            try:
                audio.play_wav_file(wav_path)
            except Exception as exc:  # pylint: disable=broad-except
                playback_error = exc

        playback_thread = threading.Thread(target=_playback_target, daemon=True)
        playback_thread.start()

        while playback_thread.is_alive():
            playback_thread.join(timeout=0.1)
            if audio.barge_in_detected():
                candidate = _confirm_interrupt_text(audio, transcriber, config)
                if is_clear_question_or_clarification(candidate):
                    return True
                if candidate:
                    print(f"Ignoring non-question interruption candidate: {candidate}")
                audio.start_barge_in_monitor()

        if playback_error is not None:
            raise playback_error

        return False
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

            reply_parts: list[str] = []

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

                print()

                if interrupted_by_voice:
                    if hasattr(stream, "close"):
                        stream.close()
                else:
                    interrupted_by_voice = _play_tts_response(
                        tts=tts,
                        audio=audio,
                        transcriber=transcriber,
                        config=config,
                        reply="".join(reply_parts),
                    )

                if interrupted_by_voice:
                    audio.stop_playback()
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

            audio.stop_playback()
            print("\nAssistant interrupted. Listening again...")
            continue
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\nLLM error: {exc}", file=sys.stderr)
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STT -> Ollama -> Piper voice loop")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate runtime configuration and exit",
    )
    parser.add_argument(
        "--audio-selftest",
        action="store_true",
        help="Run speaker/microphone self-tests and exit",
    )
    parser.add_argument(
        "--selftest-seconds",
        type=float,
        default=1.5,
        help="Microphone capture duration for --audio-selftest (seconds)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    audio, notes = build_audio_backend(cfg)

    if args.audio_selftest:
        if args.selftest_seconds <= 0:
            _print_config_errors(["--selftest-seconds must be > 0"])
            return 2
        return run_audio_selftest(
            config=cfg,
            audio=audio,
            notes=notes,
            duration_seconds=args.selftest_seconds,
        )

    errors = validate_config(cfg)
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

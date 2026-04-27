# alwin-voice

Single-process speech chat pipeline in Python:

1. Speech-to-text using `faster-whisper`
2. Chat with an Ollama model (configurable endpoint + model)
3. Text-to-speech using Piper (configurable voice model)
4. Immediate playback, with listen start/end tones around recording

Main target language is Swedish (`sv`), with configurable Piper voice model paths so you can swap voices without code changes.

## Requirements

- Python 3.10+
- Local Ollama instance running
- Piper runtime available as command in PATH (default: `piper` on Linux/macOS, `piper.exe` on Windows)
- Piper Swedish voice model `.onnx` file
- Working microphone and speaker

Linux note for audio:

```bash
sudo apt-get install -y libportaudio2 portaudio19-dev
```

Windows note for audio:

- `sounddevice` is supported with the prebuilt wheel on current CPython versions.
- If your microphone is not detected, verify the Windows input device in Sound settings and disable exclusive mode for that device.

## Install

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Configuration

All settings are environment-driven for minimal dependencies.

- `ALWIN_OLLAMA_ENDPOINT` default: `http://127.0.0.1:11434`
- `ALWIN_OLLAMA_MODEL` default: `llama3.1:8b`
- `ALWIN_SYSTEM_PROMPT` default: Swedish assistant prompt forcing 1-2 short conversational sentences and absolutely no Markdown formatting
- `ALWIN_STT_MODEL` default: `small`
- `ALWIN_STT_DEVICE` default: `auto`
- `ALWIN_STT_COMPUTE` default: `float16`
- `ALWIN_CPU_MODE` default: `false` (`true` uses CPU-suitable STT settings for dev/testing)
- `ALWIN_STT_LANGUAGE` default: `sv`
- `ALWIN_PIPER_BIN` default: `piper` on Linux/macOS, `piper.exe` on Windows
- `ALWIN_PIPER_MODEL` default: `./models/piper/sv_SE-nst-medium.onnx` (falls back to `./models/piper/sv_SE-alma-medium.onnx` when present)
- `ALWIN_PIPER_CONFIG` optional path to Piper config JSON
- `ALWIN_TTS_SPEAKER` optional speaker id
- `ALWIN_TTS_LENGTH_SCALE` default: `1.0`
- `ALWIN_AUDIO_SAMPLE_RATE` default: `16000`
- `ALWIN_AUDIO_BLOCKSIZE` default: `512`
- `ALWIN_LISTEN_MAX_SECONDS` default: `12.0`
- `ALWIN_VAD_ENGINE` default: `silero` (`rms`, `silero`)
- `ALWIN_VAD_START_THRESHOLD` default: `0.010`
- `ALWIN_VAD_END_THRESHOLD` default: `0.016`
- `ALWIN_VAD_SILENCE_SECONDS` default: `0.20`
- `ALWIN_VAD_PREROLL_SECONDS` default: `0.30` (keeps a short lead-in before speech start is detected)
- `ALWIN_BARGE_IN_RMS_THRESHOLD` default: `0.03` (higher = harder to interrupt on background voices)
- `ALWIN_SILERO_THRESHOLD` default: `0.45`
- `ALWIN_BARGE_IN_SILERO_THRESHOLD` default: `0.75` (higher = harder to interrupt on background voices)
- `ALWIN_SILERO_MIN_SILENCE_MS` default: `350`
- `ALWIN_SILERO_SPEECH_PAD_MS` default: `20`
- `ALWIN_CONTEXT_TURNS` default: `12`
- `ALWIN_AUDIO_BACKEND` default: `auto` (`auto`, `unitree`, `local`)
- `ALWIN_UNITREE_NETWORK_MODE` default: `false` (set `true` when running on an external PC connected to G1 over network)
- `ALWIN_UNITREE_NET_IFACE` optional network interface used for Unitree channel init (required in network mode)
- `ALWIN_UNITREE_MULTICAST_GROUP` default: `239.168.123.161` (G1 microphone multicast group)
- `ALWIN_UNITREE_MULTICAST_PORT` default: `5555` (G1 microphone multicast port)
- `ALWIN_UNITREE_MULTICAST_LOCAL_IP` optional local IPv4 to join multicast with (default `0.0.0.0`)
- `ALWIN_UNITREE_MIC_TIMEOUT_SECONDS` default: `2.0` (no-packet timeout before mic capture fails in network mode)

Example:

```bash
export ALWIN_OLLAMA_ENDPOINT="http://192.168.1.20:11434"
export ALWIN_OLLAMA_MODEL="qwen2.5:7b"
export ALWIN_PIPER_MODEL="$PWD/models/piper/sv_SE-alma-medium.onnx"
export ALWIN_VAD_ENGINE="silero"
```

Windows PowerShell example:

```powershell
$env:ALWIN_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
$env:ALWIN_OLLAMA_MODEL = "qwen2.5:7b"
$env:ALWIN_PIPER_MODEL = "$PWD\models\piper\sv_SE-alma-medium.onnx"
$env:ALWIN_CPU_MODE = "true"
```

## Run

Validation only:

```bash
alwin-voice --check

# Audio hardware/backend self-test
alwin-voice --audio-selftest
alwin-voice --audio-selftest --selftest-seconds 2.0
```

Start voice chat loop:

```bash
alwin-voice
```

Interruption behavior:

- Start speaking while assistant audio is playing to interrupt speech and immediately return to listening.
- Clear questions and clarification prompts are treated as interrupt-worthy; filler speech and background noise are ignored.
- Press `Ctrl+C` while assistant audio is playing as a fallback interruption method.
- Press `Ctrl+C` while listening or processing to stop the program.

If the console script is not available yet, run directly:

Linux/macOS:

```bash
PYTHONPATH=src python -m alwin_voice.main --check
PYTHONPATH=src python -m alwin_voice.main --audio-selftest
PYTHONPATH=src python -m alwin_voice.main
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m alwin_voice.main --check
python -m alwin_voice.main --audio-selftest
python -m alwin_voice.main
```

`--audio-selftest` behavior:

- Prints backend selection notes and diagnostics.
- Runs speaker tone playback and WAV-path playback through the active backend.
- Captures microphone audio for `--selftest-seconds` (default `1.5`) and reports
	RMS/peak levels.

Runtime flow per turn:

1. Start tone plays
2. Microphone recording starts
3. Recording auto-stops on silence (VAD)
4. Console shows `Stopped listening. Processing...`
5. End tone plays
6. Speech is transcribed to text
7. LLM response is generated with rolling chat context
8. Piper synthesizes response audio
9. Response audio plays immediately

## Unitree backend

The project has a pluggable audio backend layer:

- `local`: use `sounddevice` for microphone and speaker I/O
- `unitree`: force Unitree backend behavior
- `auto`: choose Unitree only when Unitree SDK2 is importable and runtime looks like a Unitree robot

Current implementation status:

- Unitree SDK2 probing validates `unitree_sdk2py`, `unitree_sdk2py.core.channel`, and
	`unitree_sdk2py.g1.audio.g1_audio_client`.
- Runtime auto-detection checks common Unitree robot markers and only auto-selects
	Unitree on robot.
- In explicit network mode (`ALWIN_UNITREE_NETWORK_MODE=true`), Unitree backend can run from an external PC.
- Microphone capture in network mode receives robot mic audio from multicast
	`239.168.123.161:5555` (configurable via env vars).
- Speaker playback uses Unitree G1 `AudioClient.PlayStream`.
- Forced Unitree mode (`ALWIN_AUDIO_BACKEND=unitree` or network mode enabled) surfaces
	errors instead of silently falling back to local speaker playback.

Environment overrides:

- `ALWIN_UNITREE_ROBOT=true|false` explicitly override robot runtime detection.
- `ALWIN_UNITREE_NET_IFACE=<iface>` set the network interface passed to
	`ChannelFactoryInitialize`.
- `ALWIN_UNITREE_NETWORK_MODE=true` enable external-PC network deployment.
- `ALWIN_UNITREE_MULTICAST_GROUP` / `ALWIN_UNITREE_MULTICAST_PORT` configure robot mic stream.
- `ALWIN_UNITREE_MULTICAST_LOCAL_IP` optionally set local IPv4 used for multicast join.

To prepare Unitree SDK2 Python on robot:

```bash
pip install unitree-sdk2
```

Depending on image/runtime, you may also need CycloneDDS and `CYCLONEDDS_HOME`
as documented by Unitree SDK2 Python.

### NVIDIA cuBLAS note

If you see `CUBLAS_STATUS_NOT_SUPPORTED` during STT model initialization, use:

```bash
export ALWIN_STT_COMPUTE="float16"
```

The transcriber also retries with `float16` automatically when this specific
cuBLAS error is detected.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Notes

- The implementation is intentionally single-process and synchronous for operational simplicity.
- Context is a rolling window of the latest turns to support conversational follow-up.
- You can switch Swedish voices by changing only `ALWIN_PIPER_MODEL` (and optional `ALWIN_PIPER_CONFIG`).

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
- `ALWIN_STT_LANGUAGE` default: `sv`
- `ALWIN_PIPER_BIN` default: `piper` on Linux/macOS, `piper.exe` on Windows
- `ALWIN_PIPER_MODEL` default: `./models/piper/sv_SE-nst-medium.onnx`
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
- `ALWIN_SILERO_THRESHOLD` default: `0.50`
- `ALWIN_SILERO_MIN_SILENCE_MS` default: `150`
- `ALWIN_SILERO_SPEECH_PAD_MS` default: `20`
- `ALWIN_CONTEXT_TURNS` default: `12`
- `ALWIN_AUDIO_BACKEND` default: `auto` (`auto`, `unitree`, `local`)

Example:

```bash
export ALWIN_OLLAMA_ENDPOINT="http://192.168.1.20:11434"
export ALWIN_OLLAMA_MODEL="qwen2.5:7b"
export ALWIN_PIPER_MODEL="$PWD/models/piper/sv_SE-nst-medium.onnx"
export ALWIN_VAD_ENGINE="silero"
```

Windows PowerShell example:

```powershell
$env:ALWIN_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
$env:ALWIN_OLLAMA_MODEL = "qwen2.5:7b"
$env:ALWIN_PIPER_MODEL = "$PWD\models\piper\sv_SE-nst-medium.onnx"
```

## Run

Validation only:

```bash
alwin-voice --check
```

Start voice chat loop:

```bash
alwin-voice
```

Keyboard interrupt behavior:

- Press `Ctrl+C` while assistant audio is playing to interrupt speech and immediately return to listening.
- Press `Ctrl+C` while listening or processing to stop the program.

If the console script is not available yet, run directly:

Linux/macOS:

```bash
PYTHONPATH=src python -m alwin_voice.main --check
PYTHONPATH=src python -m alwin_voice.main
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m alwin_voice.main --check
python -m alwin_voice.main
```

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

## Unitree R1 backend (initial)

The project now has a pluggable audio backend layer:

- `local`: use `sounddevice` for microphone and speaker I/O
- `unitree`: probe Unitree SDK2 Python modules and use robot backend path
- `auto`: prefer Unitree backend when SDK is importable, otherwise fall back to local

Current implementation status:

- Unitree SDK capability probing is implemented.
- Mic/speaker I/O still uses local `sounddevice` fallback while Unitree-specific
	audio API integration is completed incrementally.

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

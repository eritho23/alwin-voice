# alwin-voice

Single-process speech chat pipeline in Python:

1. Speech-to-text using `faster-whisper`
2. Chat with an Ollama model (configurable endpoint + model)
3. Text-to-speech using Chatterbox Multilingual TTS
4. Immediate playback, with listen start/end tones around recording

Main target language is Swedish (`sv`), with optional reference voice cloning via Chatterbox (`ALWIN_TTS_REFERENCE_AUDIO`).

## Requirements

- Python 3.10+
- Local Ollama instance running
- Chatterbox TTS runtime dependencies (`chatterbox-tts`)
- Working microphone and speaker

Linux note for audio:

```bash
sudo apt-get install -y libportaudio2 portaudio19-dev
```

Windows note for audio:

- `sounddevice` is supported with the prebuilt wheel on current CPython versions.
- If your microphone is not detected, verify the Windows input device in Sound settings and disable exclusive mode for that device.

## Install

Recommended (Linux/macOS and Windows with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv pip install -e .
```

Linux/macOS with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell with pip:

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
- `ALWIN_TTS_DEVICE` default: `auto` (`auto`, `cpu`, `cuda`, `mps`)
- `ALWIN_TTS_LANGUAGE` default: `sv` (Chatterbox multilingual language ID)
- `ALWIN_TTS_REFERENCE_AUDIO` optional path to a reference voice clip (when omitted, built-in default voice is used)
- `ALWIN_TTS_EXAGGERATION` default: `0.5`
- `ALWIN_TTS_CFG_WEIGHT` default: `0.5`
- `ALWIN_TTS_RUNTIME_MODE` default: `inprocess` (`inprocess`, `remote-stdio`)
- `ALWIN_TTS_WORKER_COMMAND` required when `ALWIN_TTS_RUNTIME_MODE=remote-stdio` (example: `/path/to/.venv-py314/bin/alwin-tts-worker`)
- `ALWIN_TTS_WORKER_STARTUP_TIMEOUT_SECONDS` default: `20.0`
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
- `ALWIN_UNITREE_NET_IFACE` optional network interface used for Unitree channel init (required in network mode). E.g., `enp0s1`. This sets up the control channel but does not affect the multicast socket routing.
- `ALWIN_UNITREE_MULTICAST_GROUP` default: `239.168.123.161` (G1 microphone multicast group). Must be a valid Class D multicast IP.
- `ALWIN_UNITREE_MULTICAST_PORT` default: `5555` (G1 microphone multicast port).
- `ALWIN_UNITREE_MULTICAST_LOCAL_IP` optional local IPv4 to join multicast with (default `0.0.0.0`). Set this to the IP of your computer running this code on the robot's network (e.g., `192.168.123.222`). This is crucial to ensure the OS routes the multicast subscription correctly!
- `ALWIN_UNITREE_MIC_TIMEOUT_SECONDS` default: `2.0` (no-packet timeout before mic capture fails in network mode)
- `ALWIN_UNITREE_LOCAL_MIC` default: `false` (set `true` to keep Unitree speaker playback while recording from this computer's local microphone)

Example:

```bash
export ALWIN_OLLAMA_ENDPOINT="http://192.168.1.20:11434"
export ALWIN_OLLAMA_MODEL="qwen2.5:7b"
export ALWIN_TTS_LANGUAGE="sv"
export ALWIN_TTS_REFERENCE_AUDIO="$PWD/models/voice-reference.wav"
export ALWIN_VAD_ENGINE="silero"
```

Windows PowerShell example:

```powershell
$env:ALWIN_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
$env:ALWIN_OLLAMA_MODEL = "qwen2.5:7b"
$env:ALWIN_TTS_LANGUAGE = "sv"
$env:ALWIN_TTS_REFERENCE_AUDIO = "$PWD\models\voice-reference.wav"
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
8. Chatterbox synthesizes response audio
9. Response audio plays immediately

## Split runtime for Unitree + RTX 50xx CUDA TTS

If Unitree SDK must run on Python 3.11 while GPU Chatterbox/Torch requires Python 3.14,
run a split setup:

```bash
uv venv .venv-py311 --python 3.11
uv venv .venv-py314 --python 3.14

uv pip install --python .venv-py311/bin/python -e .
uv pip install --python .venv-py314/bin/python -e .
```

Run the main app in Python 3.11 with remote stdio mode:

```bash
export ALWIN_TTS_RUNTIME_MODE="remote-stdio"
export ALWIN_TTS_WORKER_COMMAND="$PWD/.venv-py314/bin/alwin-tts-worker"
.venv-py311/bin/alwin-voice
```

`alwin-voice` launches the worker command as a child process when synthesis is requested.

### Startup example (Unitree network + split TTS runtime)

Use this when running Unitree from Python 3.11 and Chatterbox GPU TTS from Python 3.14:

```bash
ALWIN_OLLAMA_ENDPOINT="http://10.22.1.100:11434" \
ALWIN_OLLAMA_MODEL="gpt-oss:20b" \
ALWIN_CPU_MODE="true" \
ALWIN_AUDIO_BACKEND="unitree" \
ALWIN_UNITREE_NETWORK_MODE="true" \
ALWIN_UNITREE_NET_IFACE="enp129s0" \
ALWIN_UNITREE_LOCAL_MIC="true" \
ALWIN_TTS_RUNTIME_MODE="remote-stdio" \
ALWIN_TTS_WORKER_COMMAND="$PWD/.venv-py314/bin/python -m alwin_voice.tts.worker" \
PYTHONPATH="/opt/unitree_sdk2_python:$PWD/src" \
.venv-py311/bin/python -m alwin_voice.main
```

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
- In network mode, set `ALWIN_UNITREE_LOCAL_MIC=true` to use this computer's microphone instead of robot multicast mic input.
- Speaker playback uses Unitree G1 `AudioClient.PlayStream`.
- Forced Unitree mode (`ALWIN_AUDIO_BACKEND=unitree` or network mode enabled) surfaces
	errors instead of silently falling back to local speaker playback.

Environment overrides:

- `ALWIN_UNITREE_ROBOT=true|false` explicitly override robot runtime detection.
- `ALWIN_UNITREE_NET_IFACE=<iface>` set the network interface passed to
	`ChannelFactoryInitialize`.
- `ALWIN_UNITREE_NETWORK_MODE=true` enable external-PC network deployment.
- `ALWIN_UNITREE_LOCAL_MIC=true` keep Unitree speaker output but use local microphone capture.
- `ALWIN_UNITREE_MULTICAST_GROUP` / `ALWIN_UNITREE_MULTICAST_PORT` configure robot mic stream.
- `ALWIN_UNITREE_MULTICAST_LOCAL_IP` optionally set local IPv4 used for multicast join.

To prepare Unitree SDK2 Python on robot:

```bash
pip install unitree-sdk2
```

Depending on image/runtime, you may also need CycloneDDS and `CYCLONEDDS_HOME`
as documented by Unitree SDK2 Python.

## Troubleshooting

### Unitree G1 Microphone Multicast Issues (Network Mode)

If the robot operates normally but your Python application on an external computer receives no audio (times out gathering mic packets), it is usually caused by the OS silently rejecting the robot's multicast packets.

**1. Explicitly Set `ALWIN_UNITREE_MULTICAST_LOCAL_IP`**
You are required to provide `ALWIN_UNITREE_NET_IFACE` (e.g. `enp0s1`) to initialize the Unitree C++ SDK. Make sure you _also_ provide `ALWIN_UNITREE_MULTICAST_LOCAL_IP` with your control computer's IP address on that network (e.g., `192.168.123.222`). Without it, the underlying Python UDP socket uses `0.0.0.0` to join the `239.x.x.x` multicast group, often routing requests out of the wrong adapter (like your Wi-Fi).

**2. Configure the Firewall**
On Ubuntu/Linux deployments, local network traffic across a separate ethernet interface can still be halted by an inbound firewall configuration. You must specifically allow `239.168.123.161` on the `5555` port.
```bash
sudo ufw allow in on <robot_interface> to 239.168.123.161 port 5555 proto udp
```

**3. OS-Level Reception Verification (tcpdump)**
If Python still sees nothing, bypass to the OS level by using `tcpdump`. Since the robot sends "always-on" streams, you should be able to see the packets bypassing even firewall blocks.
```bash
sudo tcpdump -i <robot_interface> -n udp port 5555
```
If `tcpdump` shows traffic, but Python receives nothing, the OS firewall or reverse path filtering (`rp_filter`) is rejecting the packet. If `tcpdump` is empty, the traffic is never reaching your network adapter.


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
- You can keep default Swedish voice, or set `ALWIN_TTS_REFERENCE_AUDIO` to clone from a reference clip.

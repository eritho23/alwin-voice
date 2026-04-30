from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from alwin_voice.tts.chatterbox_engine import ChatterboxConfig, ChatterboxEngine


def _error_response(req_id: object, message: str) -> dict[str, object]:
    return {"id": req_id, "error": message}


def _result_response(req_id: object, wav_b64: str) -> dict[str, object]:
    return {"id": req_id, "result": {"wav_b64": wav_b64}}


def _build_engine(params: dict[str, object]) -> ChatterboxEngine:
    ref_audio = params.get("reference_audio_path")
    reference_audio_path = Path(ref_audio) if isinstance(ref_audio, str) else None
    cfg = ChatterboxConfig(
        device=str(params.get("device", "auto")),
        language=str(params.get("language", "sv")),
        reference_audio_path=reference_audio_path,
        exaggeration=float(params.get("exaggeration", 0.5)),
        cfg_weight=float(params.get("cfg_weight", 0.5)),
    )
    return ChatterboxEngine(cfg)


def main() -> int:
    engines: dict[str, ChatterboxEngine] = {}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        req_id: object = None
        try:
            payload = json.loads(line)
            req_id = payload.get("id")
            method = payload.get("method")
            params = payload.get("params")
            if method != "synthesize":
                response = _error_response(req_id, f"Unsupported method: {method}")
            elif not isinstance(params, dict):
                response = _error_response(req_id, "Invalid params")
            else:
                text = params.get("text")
                if not isinstance(text, str):
                    response = _error_response(req_id, "params.text must be a string")
                else:
                    cfg_key = json.dumps(
                        {
                            "device": params.get("device", "auto"),
                            "language": params.get("language", "sv"),
                            "reference_audio_path": params.get("reference_audio_path"),
                            "exaggeration": params.get("exaggeration", 0.5),
                            "cfg_weight": params.get("cfg_weight", 0.5),
                        },
                        sort_keys=True,
                    )
                    engine = engines.get(cfg_key)
                    if engine is None:
                        engine = _build_engine(params)
                        engines[cfg_key] = engine

                    wav_path = engine.synthesize_to_wav(text)
                    try:
                        wav_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
                    finally:
                        wav_path.unlink(missing_ok=True)
                    response = _result_response(req_id, wav_b64)
        except Exception as exc:
            response = _error_response(req_id, str(exc))

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

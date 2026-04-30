"""Text-to-speech adapter package."""

from .backends import TTSBackend, build_tts_backend

__all__ = ["TTSBackend", "build_tts_backend"]

import os
import tempfile
import unittest
from pathlib import Path

from alwin_voice.config.settings import load_config


class TestSettings(unittest.TestCase):
    def test_load_config_context_turns_default(self) -> None:
        original = os.environ.pop("ALWIN_CONTEXT_TURNS", None)
        try:
            cfg = load_config()
            self.assertEqual(cfg.context_turns, 12)
        finally:
            if original is not None:
                os.environ["ALWIN_CONTEXT_TURNS"] = original

    def test_load_config_model_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "voice.onnx"
            model.write_bytes(b"x")

            original = os.environ.get("ALWIN_PIPER_MODEL")
            os.environ["ALWIN_PIPER_MODEL"] = str(model)
            try:
                cfg = load_config()
                self.assertEqual(cfg.piper_model_path, model)
            finally:
                if original is None:
                    os.environ.pop("ALWIN_PIPER_MODEL", None)
                else:
                    os.environ["ALWIN_PIPER_MODEL"] = original


if __name__ == "__main__":
    unittest.main()

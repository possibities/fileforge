import os
import unittest
from pathlib import Path
from unittest.mock import patch

from config.config import _env_bool, _env_float, _env_int, _env_path, _env_str


class TestConfigEnvHelpers(unittest.TestCase):
    def test_env_str_returns_default_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_env_str("NOT_SET", "default"), "default")

    def test_env_int_returns_default_on_invalid_value(self):
        with patch.dict(os.environ, {"X_INT": "bad"}, clear=True):
            self.assertEqual(_env_int("X_INT", 7), 7)

    def test_env_float_parses_valid_number(self):
        with patch.dict(os.environ, {"X_FLOAT": "0.25"}, clear=True):
            self.assertEqual(_env_float("X_FLOAT", 0.1), 0.25)

    def test_env_bool_supports_common_values(self):
        with patch.dict(os.environ, {"X_BOOL": "On"}, clear=True):
            self.assertTrue(_env_bool("X_BOOL", False))
        with patch.dict(os.environ, {"X_BOOL": "0"}, clear=True):
            self.assertFalse(_env_bool("X_BOOL", True))

    def test_env_path_uses_default_when_empty(self):
        default = Path("/tmp/demo")
        with patch.dict(os.environ, {"X_PATH": ""}, clear=True):
            self.assertEqual(_env_path("X_PATH", default), str(default))


if __name__ == "__main__":
    unittest.main()

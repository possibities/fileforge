import sys
import types
import unittest

from constants import METADATA_SCHEMA


# Provide a lightweight stub so importing llm_client does not require runtime llama.cpp.
if "llama_cpp" not in sys.modules:
    llama_cpp_stub = types.ModuleType("llama_cpp")

    class _DummyLlama:  # pragma: no cover - constructor is never used in these tests.
        pass

    llama_cpp_stub.Llama = _DummyLlama
    sys.modules["llama_cpp"] = llama_cpp_stub

from infrastructure.llm_client import LlmClient


class TestLlmClientRegexFallback(unittest.TestCase):
    def _build_client(self):
        client = LlmClient.__new__(LlmClient)
        client.metadata_schema = METADATA_SCHEMA
        return client

    def test_extract_fields_by_regex_parses_null_string_and_number(self):
        client = self._build_client()
        fields = list(METADATA_SCHEMA.keys())
        key_text = fields[0]
        key_year = fields[1]
        key_note = fields[11]

        raw = (
            "{"
            f"\"{key_text}\": \"value\", "
            f"\"{key_year}\": 2020, "
            f"\"{key_note}\": null"
            "}"
        )
        metadata = client._extract_fields_by_regex(raw)

        self.assertEqual(metadata[key_text], "value")
        self.assertEqual(metadata[key_year], 2020)
        self.assertIsNone(metadata[key_note])

    def test_extract_fields_by_regex_parses_bool_float_and_array(self):
        client = self._build_client()
        fields = list(METADATA_SCHEMA.keys())
        key_bool = fields[12]
        key_float = fields[10]
        key_array = fields[9]

        raw = (
            "{"
            f"\"{key_bool}\": true, "
            f"\"{key_float}\": 1.5, "
            f"\"{key_array}\": [\"a\", \"b\"]"
            "}"
        )
        metadata = client._extract_fields_by_regex(raw)

        self.assertIs(metadata[key_bool], True)
        self.assertEqual(metadata[key_float], 1.5)
        self.assertEqual(metadata[key_array], ["a", "b"])

    def test_extract_fields_by_regex_ignores_unknown_keys(self):
        client = self._build_client()
        key_text = list(METADATA_SCHEMA.keys())[0]
        raw = "{ " f"\"{key_text}\": \"ok\", \"unknown_key\": \"drop\" " "}"

        metadata = client._extract_fields_by_regex(raw)

        self.assertIn(key_text, metadata)
        self.assertNotIn("unknown_key", metadata)


if __name__ == "__main__":
    unittest.main()

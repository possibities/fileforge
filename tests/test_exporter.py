import json
import shutil
import unittest
from pathlib import Path

from processors.exporter import Exporter


class TestExporter(unittest.TestCase):
    def setUp(self):
        Exporter.HEADERS = {}
        self.tmp_dir = Path("tests") / "_tmp_exporter"
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        Exporter.HEADERS = {}
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_headers_requires_initialize(self):
        with self.assertRaisesRegex(RuntimeError, "EXPORTER_NOT_INITIALIZED"):
            Exporter.get_headers()

    def test_get_headers_template_not_found(self):
        Exporter.HEADERS = {"default": ["a"]}
        with self.assertRaisesRegex(RuntimeError, "EXPORTER_TEMPLATE_NOT_FOUND"):
            Exporter.get_headers("missing")

    def test_validate_headers_rejects_empty_fields(self):
        with self.assertRaisesRegex(ValueError, "INVALID_HEADERS_CONFIG"):
            Exporter._validate_headers({"default": []})

    def test_initialize_raises_on_invalid_config_shape(self):
        cfg = self.tmp_dir / "bad_headers.json"
        cfg.write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "EXPORTER_INIT_FAILED"):
            Exporter.initialize(str(cfg))

    def test_export_to_json_writes_only_template_fields(self):
        Exporter.HEADERS = {"default": ["a", "b"]}
        output = self.tmp_dir / "result.json"
        results = [
            {"metadata": {"a": "v1", "b": None, "c": "drop"}},
            {"metadata": None},
        ]

        written = Exporter.export_to_json(results, str(output))

        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(written, 1)
        self.assertEqual(data, [{"a": "v1", "b": ""}])

    def test_export_to_csv_returns_zero_on_empty_results(self):
        Exporter.HEADERS = {"default": ["a"]}
        output = self.tmp_dir / "empty.csv"

        with self.assertLogs("processors.exporter", level="WARNING") as log_ctx:
            written = Exporter.export_to_csv([], str(output))

        self.assertEqual(written, 0)
        self.assertFalse(output.exists())
        self.assertTrue(any("no results to export" in line for line in log_ctx.output))


if __name__ == "__main__":
    unittest.main()

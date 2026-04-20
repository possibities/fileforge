import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from processors.batch_processor import BatchProcessor
from utils.batch_summary_validator import (
    BatchSummaryValidationError,
    Draft202012Validator,
    is_supported_major,
    parse_semver,
    validate_summary_file,
    validate_summary_payload,
)


class _MixedClassifier:
    def process_multi_page_document(self, archive_name, image_paths):
        if archive_name == "doc_success":
            return {
                "归档年度": "2020",
                "实体分类号": "YWL",
                "保管期限": "30年",
            }
        if archive_name == "doc_error":
            raise RuntimeError("boom")
        return {}


class TestBatchSummaryValidator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("tests") / f"_tmp_summary_validator_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.fake_image = self.tmp_dir / "1.jpg"
        self.fake_image.write_bytes(b"fake")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_parse_semver(self):
        self.assertEqual(parse_semver("1.2.3"), (1, 2, 3))
        self.assertIsNone(parse_semver("1.2"))
        self.assertIsNone(parse_semver("v1.2.3"))

    def test_is_supported_major(self):
        self.assertTrue(is_supported_major("1.0.0", 1))
        self.assertTrue(is_supported_major("1.9.9", 1))
        self.assertFalse(is_supported_major("2.0.0", 1))

    def test_validate_summary_file_passes_for_generated_summary(self):
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not available in current environment")

        processor = BatchProcessor(_MixedClassifier())
        out_dir = self.tmp_dir / "out"
        processor.batch_process_archives(
            {
                "doc_success": [str(self.fake_image)],
                "doc_empty": [str(self.fake_image)],
                "doc_error": [str(self.fake_image)],
                "doc_no_images": [],
            },
            output_dir=str(out_dir),
        )

        summary_path = out_dir / "batch_summary.json"
        schema_path = Path("config") / "batch_summary.schema.json"
        validate_summary_file(summary_path=summary_path, schema_path=schema_path, expected_major=1)

    def test_validate_summary_payload_rejects_incompatible_major(self):
        with self.assertRaisesRegex(BatchSummaryValidationError, "Unsupported summary schema major version"):
            validate_summary_payload(
                summary_payload={"summary_schema_version": "2.0.0"},
                schema_payload={},
                expected_major=1,
            )

    def test_validate_summary_payload_rejects_schema_mismatch(self):
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not available in current environment")

        schema_path = Path("config") / "batch_summary.schema.json"
        schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(BatchSummaryValidationError, "does not match schema"):
            validate_summary_payload(
                summary_payload={"summary_schema_version": "1.0.0"},
                schema_payload=schema_payload,
                expected_major=1,
            )

    def test_cli_returns_zero_for_valid_summary(self):
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not available in current environment")

        processor = BatchProcessor(_MixedClassifier())
        out_dir = self.tmp_dir / "cli_out"
        processor.batch_process_archives(
            {"doc_success": [str(self.fake_image)]},
            output_dir=str(out_dir),
        )
        summary_path = out_dir / "batch_summary.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "utils.batch_summary_validator",
                str(summary_path),
                "--schema",
                "config/batch_summary.schema.json",
                "--expected-major",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("[OK]", result.stdout)

    def test_cli_returns_nonzero_for_incompatible_major(self):
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not available in current environment")

        processor = BatchProcessor(_MixedClassifier())
        out_dir = self.tmp_dir / "cli_out_fail"
        processor.batch_process_archives(
            {"doc_success": [str(self.fake_image)]},
            output_dir=str(out_dir),
        )
        summary_path = out_dir / "batch_summary.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "utils.batch_summary_validator",
                str(summary_path),
                "--schema",
                "config/batch_summary.schema.json",
                "--expected-major",
                "2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[INVALID]", result.stdout)


if __name__ == "__main__":
    unittest.main()

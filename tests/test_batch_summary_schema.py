import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from processors.batch_processor import BatchProcessor

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - environment-dependent optional dependency
    Draft202012Validator = None


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


class TestBatchSummarySchema(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("tests") / f"_tmp_batch_schema_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.fake_image = self.tmp_dir / "1.jpg"
        self.fake_image.write_bytes(b"fake")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_supported_summary_version_policy(self):
        self.assertTrue(BatchProcessor._is_supported_summary_version("1.0.0"))
        self.assertTrue(BatchProcessor._is_supported_summary_version("1.2.3"))
        self.assertFalse(BatchProcessor._is_supported_summary_version("2.0.0"))
        self.assertFalse(BatchProcessor._is_supported_summary_version("1.0"))
        self.assertFalse(BatchProcessor._is_supported_summary_version(None))

    def test_generated_batch_summary_matches_json_schema(self):
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not available in current environment")

        processor = BatchProcessor(_MixedClassifier())
        output_dir = self.tmp_dir / "out"

        processor.batch_process_archives(
            {
                "doc_success": [str(self.fake_image)],
                "doc_empty": [str(self.fake_image)],
                "doc_error": [str(self.fake_image)],
                "doc_no_images": [],
            },
            output_dir=str(output_dir),
        )

        summary_path = output_dir / "batch_summary.json"
        schema_path = Path("config") / "batch_summary.schema.json"

        self.assertTrue(summary_path.exists())
        self.assertTrue(schema_path.exists())

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(summary), key=lambda err: list(err.path))
        if errors:
            details = "\n".join(
                f"{'/'.join(str(item) for item in err.path)}: {err.message}"
                for err in errors[:10]
            )
            self.fail(f"Summary does not match JSON schema:\n{details}")

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["fail_count"], 3)
        self.assertEqual(summary["failure_breakdown"][BatchProcessor.ERROR_NO_IMAGES], 1)
        self.assertEqual(summary["failure_breakdown"][BatchProcessor.ERROR_EMPTY_METADATA], 1)
        self.assertEqual(summary["failure_breakdown"][BatchProcessor.ERROR_PROCESS_EXCEPTION], 1)

    def test_validate_summary_data_rejects_incompatible_major(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported summary schema major version"):
            BatchProcessor._validate_summary_data({"summary_schema_version": "2.0.0"})

    def test_validate_summary_data_raises_for_invalid_payload(self):
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not available in current environment")

        with self.assertRaisesRegex(RuntimeError, "does not match schema"):
            BatchProcessor._validate_summary_data({"summary_schema_version": "1.0.0"})


if __name__ == "__main__":
    unittest.main()

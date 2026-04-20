import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from processors.batch_processor import BatchProcessor


class _EmptyClassifier:
    def process_multi_page_document(self, archive_name, image_paths):
        return {}


class _BrokenClassifier:
    def process_multi_page_document(self, archive_name, image_paths):
        raise RuntimeError("boom")


class TestBatchProcessorResultStatus(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("tests") / f"_tmp_batch_result_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.fake_image = self.tmp_dir / "1.jpg"
        self.fake_image.write_bytes(b"fake")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_batch_process_archives_marks_empty_image_list_as_failed(self):
        processor = BatchProcessor(_EmptyClassifier())
        results = processor.batch_process_archives({"doc_a": []})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], BatchProcessor.STATUS_FAILED)
        self.assertEqual(results[0]["error_code"], BatchProcessor.ERROR_NO_IMAGES)
        self.assertIn("No image files", results[0]["error_message"])
        self.assertEqual(results[0]["error"], results[0]["error_message"])
        self.assertIsNone(results[0]["metadata"])

    def test_batch_process_archives_marks_empty_metadata_as_failed(self):
        processor = BatchProcessor(_EmptyClassifier())
        results = processor.batch_process_archives({"doc_meta_empty": [str(self.fake_image)]})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], BatchProcessor.STATUS_FAILED)
        self.assertEqual(results[0]["error_code"], BatchProcessor.ERROR_EMPTY_METADATA)
        self.assertIn("empty metadata", results[0]["error_message"])

    def test_batch_process_archives_marks_classifier_exception_as_error(self):
        processor = BatchProcessor(_BrokenClassifier())
        results = processor.batch_process_archives({"doc_b": [str(self.fake_image)]})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], BatchProcessor.STATUS_ERROR)
        self.assertEqual(results[0]["error_code"], BatchProcessor.ERROR_PROCESS_EXCEPTION)
        self.assertIn("boom", results[0]["error_message"])
        self.assertIn("RuntimeError", results[0]["traceback"])

    def test_batch_summary_contains_schema_version_and_contract(self):
        processor = BatchProcessor(_EmptyClassifier())
        output_dir = self.tmp_dir / "out"

        processor.batch_process_archives(
            {
                "doc_with_image": [str(self.fake_image)],
                "doc_no_images": [],
            },
            output_dir=str(output_dir),
        )

        summary_path = output_dir / "batch_summary.json"
        self.assertTrue(summary_path.exists())

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            summary["summary_schema_version"],
            BatchProcessor.SUMMARY_SCHEMA_VERSION,
        )
        self.assertEqual(
            summary["summary_schema_ref"],
            BatchProcessor.SUMMARY_SCHEMA_REF,
        )
        self.assertEqual(
            summary["summary_changelog_ref"],
            BatchProcessor.SUMMARY_CHANGELOG_REF,
        )

        contract = summary["summary_contract"]
        self.assertIn("failure_breakdown", contract["summary_fields"])
        self.assertIn("summary_schema_ref", contract["summary_fields"])
        self.assertIn("summary_changelog_ref", contract["summary_fields"])
        self.assertIn("status", contract["result_required_fields"])
        self.assertIn("status", contract["result_field_descriptions"])
        self.assertIn(BatchProcessor.STATUS_FAILED, contract["status_values"])
        self.assertIn(BatchProcessor.ERROR_NO_IMAGES, contract["error_codes"])
        self.assertEqual(contract["schema_version_policy"]["scheme"], "semver")

        self.assertEqual(summary["failure_breakdown"][BatchProcessor.ERROR_NO_IMAGES], 1)
        self.assertEqual(
            summary["failure_breakdown"][BatchProcessor.ERROR_EMPTY_METADATA],
            1,
        )


if __name__ == "__main__":
    unittest.main()

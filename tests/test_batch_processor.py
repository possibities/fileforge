import shutil
import unittest
from pathlib import Path

from processors.batch_processor import BatchProcessor


class _DummyClassifier:
    def process_multi_page_document(self, archive_name, image_paths):
        return {}


class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = BatchProcessor(_DummyClassifier())
        self.tmp_root = Path("tests") / "_tmp_scan_case"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_scan_directory_structure_keeps_nested_prefix(self):
        nested = self.tmp_root / "a" / "b" / "c"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "0001.jpg").write_bytes(b"fake")

        archive_dict = self.processor.scan_directory_structure(str(self.tmp_root), max_depth=3)

        self.assertIn("a/b/c", archive_dict)
        self.assertEqual(len(archive_dict["a/b/c"]), 1)
        self.assertTrue(archive_dict["a/b/c"][0].endswith("0001.jpg"))

    def test_resolve_source_info_handles_empty_list(self):
        source_folder, created_time = self.processor._resolve_source_info([])

        self.assertIsNone(source_folder)
        self.assertIsNone(created_time)


if __name__ == "__main__":
    unittest.main()

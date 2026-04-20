import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


if "paddleocr" not in sys.modules:
    paddleocr_stub = types.ModuleType("paddleocr")

    class _DummyPaddleOCR:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def ocr(self, image, cls=True):
            self.calls.append((image, cls))
            return []

    paddleocr_stub.PaddleOCR = _DummyPaddleOCR
    sys.modules["paddleocr"] = paddleocr_stub

from infrastructure.ocr_client import OcrClient, OcrVariant, PageOcrCandidate


class TestOcrClientHelpers(unittest.TestCase):
    def _build_client(self):
        return OcrClient.__new__(OcrClient)

    def test_reconstruct_page_respects_layout(self):
        client = self._build_client()
        lines = [
            (
                [[0, 0], [20, 0], [20, 10], [0, 10]],
                ("标题", 0.99),
            ),
            (
                [[50, 0], [70, 0], [70, 10], [50, 10]],
                ("一", 0.99),
            ),
            (
                [[0, 30], [20, 30], [20, 40], [0, 40]],
                ("正文", 0.99),
            ),
        ]

        self.assertEqual(client._reconstruct_page(lines), "标题 一\n正文")

    def test_clean_ocr_text_normalizes_brackets_and_blank_lines(self):
        client = self._build_client()
        raw = "〔2026〕1号\n\n\n【附件】\xa0Ｏ"

        self.assertEqual(client._clean_ocr_text(raw), "[2026]1号\n[附件] 0")

    def test_should_retry_with_preprocess_for_empty_or_low_confidence_text(self):
        client = self._build_client()
        weak = PageOcrCandidate(
            label="original",
            raw_line_count=3,
            filtered_lines=[object()],
            low_conf_count=2,
            avg_confidence=0.7,
            filtered_char_count=8,
        )

        self.assertTrue(client._should_retry_with_preprocess(weak))


class TestOcrClientFallbackFlow(unittest.TestCase):
    def _build_client(self):
        client = OcrClient.__new__(OcrClient)
        client.ocr = object()
        return client

    def test_extract_best_candidate_prefers_preprocessed_result(self):
        client = self._build_client()
        original = PageOcrCandidate(
            label="original",
            raw_line_count=4,
            filtered_lines=[object()],
            low_conf_count=3,
            avg_confidence=0.75,
            filtered_char_count=6,
            page_text="弱结果",
            score=20.0,
        )
        improved = PageOcrCandidate(
            label="binary",
            raw_line_count=5,
            filtered_lines=[object(), object()],
            low_conf_count=1,
            avg_confidence=0.93,
            filtered_char_count=18,
            page_text="增强结果",
            score=70.0,
        )

        with patch.object(client, "_run_candidate", side_effect=[original, improved]):
            with patch.object(
                client,
                "_build_preprocessed_variants",
                return_value=[OcrVariant(label="binary", image="processed")],
            ):
                best = client._extract_best_candidate("demo.jpg")

        self.assertEqual(best.label, "binary")
        self.assertEqual(best.page_text, "增强结果")

    def test_extract_text_uses_cleaned_best_candidate(self):
        client = self._build_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "demo.jpg"
            image_path.write_bytes(b"placeholder")

            candidate = PageOcrCandidate(
                label="sharpen",
                raw_line_count=3,
                filtered_lines=[object()],
                low_conf_count=0,
                avg_confidence=0.91,
                filtered_char_count=10,
                page_text="〔2026〕\n\n【附件】",
                score=55.0,
            )

            with patch.object(client, "_extract_best_candidate", return_value=candidate):
                text = client.extract_text(str(image_path))

        self.assertEqual(text, "[2026]\n[附件]")


if __name__ == "__main__":
    unittest.main()

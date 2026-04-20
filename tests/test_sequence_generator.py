import unittest

from core.sequence_generator import SequenceGenerator


YEAR_KEY = "\u5f52\u6863\u5e74\u5ea6"
CLASS_KEY = "\u5b9e\u4f53\u5206\u7c7b\u53f7"
PERIOD_KEY = "\u4fdd\u7ba1\u671f\u9650"
SERIAL_KEY = "\u4ef6\u53f7"
DOC_ID_KEY = "\u6863\u53f7"


class TestSequenceGenerator(unittest.TestCase):
    def test_assign_generates_serial_and_doc_id(self):
        generator = SequenceGenerator()
        metadata = {
            YEAR_KEY: "2020",
            CLASS_KEY: "YWL",
            PERIOD_KEY: "30年",
        }

        result = generator.assign(metadata)

        self.assertEqual(result[SERIAL_KEY], "0001")
        self.assertEqual(result[DOC_ID_KEY], "2020-YWL-D30-0001")

    def test_assign_returns_none_when_required_fields_missing(self):
        generator = SequenceGenerator()
        metadata = {
            YEAR_KEY: "2020",
            CLASS_KEY: "",
            PERIOD_KEY: "30年",
        }

        with self.assertLogs("core.sequence_generator", level="WARNING") as log_ctx:
            result = generator.assign(metadata)

        self.assertGreaterEqual(len(log_ctx.output), 1)
        self.assertIsNone(result[SERIAL_KEY])
        self.assertIsNone(result[DOC_ID_KEY])


if __name__ == "__main__":
    unittest.main()

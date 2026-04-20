import unittest

from core.rules_engine import RulesEngine


class TestRulesEngine(unittest.TestCase):
    def test_rule11_marks_literary_briefing_title_for_review(self):
        engine = RulesEngine()
        metadata = {
            "题名": "春风行动简报",
            "备注": None,
        }

        with self.assertLogs("core.rules_engine", level="WARNING") as log_ctx:
            result = engine._clean_title(metadata)

        self.assertGreaterEqual(len(log_ctx.output), 1)
        self.assertTrue(result.get("_需重构简报题名"))
        # 规则 11 不再落备注，备注由 classifier 在 LLM 重写失败时兜底
        self.assertIsNone(result.get("备注"))

    def test_rule11_skips_when_title_has_substantive_verb(self):
        engine = RulesEngine()
        metadata = {
            "题名": "关于开展培训工作的简报",
            "备注": None,
        }

        result = engine._clean_title(metadata)

        self.assertNotIn("_需重构简报题名", result)
        self.assertIsNone(result.get("备注"))


if __name__ == "__main__":
    unittest.main()

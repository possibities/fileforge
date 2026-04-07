#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import Dict, List

from config.config import Config
from constants import METADATA_SCHEMA
from core.rules_engine import RulesEngine
from infrastructure.llm_client import LlmClient
from infrastructure.ocr_client import OcrClient
from utils.file import get_file_creation_time

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
logger = logging.getLogger(__name__)


class ArchiveClassifier:
    """
    档案智能分类器

    职责：
      - 协调OCR、LLM、规则引擎三个子系统
      - 构建提示词
      - 组装最终元数据
    """

    def __init__(
        self,
        ocr_lang: str = Config.OCR_LANG,
        model_path: str = Config.LLM_MODEL_PATH,   # 替换原来的base_model_path等参数
    ):
        self.ocr_client = OcrClient(lang=ocr_lang)
        self.llm_client = LlmClient(model_path=model_path)
        self.rules_engine = RulesEngine()
        self.metadata_schema = METADATA_SCHEMA
        self.extraction_prompt = self._build_extraction_prompt()

    # ── 公开接口 ───────────────────────────────────────────────────────────────

    def process_multi_page_document(
        self, archive_name: str, image_paths: List[str]
    ) -> Dict:
        """处理多页档案文件"""
        logger.info(f"\n{'='*70}")
        logger.info(f"处理档案: {archive_name}")
        logger.info(f"页数: {len(image_paths)} 页")
        logger.info(f"{'='*70}\n")

        ocr_text = self.ocr_client.extract_text_from_images(image_paths)

        if not ocr_text:
            logger.error("[错误] OCR未识别到任何文字")
            return {}

        logger.info("[OCR结果预览]")
        logger.info("-" * 70)
        preview_length = Config.OCR_PREVIEW_LENGTH
        logger.info(
            ocr_text[:preview_length] + f"\n...(共{len(ocr_text)}字符)"
            if len(ocr_text) > preview_length
            else ocr_text
        )
        logger.info("-" * 70)
        logger.info("")

        metadata = self._extract_metadata_from_text(ocr_text)

        if metadata:
            metadata['数字化时间'] = get_file_creation_time(image_paths[0])
            metadata['档案文件夹'] = archive_name

        return metadata

    def process_document(self, image_path: str) -> Dict:
        """处理单个图像文件（兼容旧接口）"""
        return self.process_multi_page_document(
            archive_name=Path(image_path).stem,
            image_paths=[image_path],
        )

    # ── 私有方法 ───────────────────────────────────────────────────────────────

    def _extract_metadata_from_text(self, ocr_text: str) -> Dict:
        """使用LLM从OCR文本中提取元数据，并应用规则修正"""
        metadata = self.llm_client.extract_metadata(ocr_text, self.extraction_prompt)

        if not metadata:
            return {}

        metadata = self.rules_engine.apply_all(metadata, ocr_text)

        logger.info(
            f"[LLM] 成功提取 "
            f"{len([v for v in metadata.values() if v is not None])} 个有效字段"
        )
        return metadata

    def _load_prompt_file(self, filename: str) -> str:
        """加载单个规则文件，文件不存在时快速失败"""
        path = PROMPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"[Prompt文件缺失] {path}")
        return path.read_text(encoding="utf-8").strip()

    def _load_examples(self) -> str:
        """从examples.json加载few-shot示例，转换为prompt字符串"""
        path = PROMPTS_DIR / "examples.json"
        if not path.exists():
            raise FileNotFoundError(f"[示例文件缺失] {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        blocks = []
        for i, ex in enumerate(data["examples"], 1):
            label = ex["label"]
            output = json.dumps(ex["output"], ensure_ascii=False, indent=2)
            output = output.replace("{", "{{").replace("}", "}}")
            blocks.append(f"【JSON输出示例{i} - {label}】\n{output}")

        return "\n\n".join(blocks)

    def _build_extraction_prompt(self) -> str:
        """拼装完整提示词，所有规则内容从外部文件加载"""
        fields_desc = "\n".join(
            [f"- {k}: {v}" for k, v in self.metadata_schema.items()]
        )

        rules_priority = self._load_prompt_file("rules_priority.txt")
        rules_category = self._load_prompt_file("rules_category.txt")
        rules_title    = self._load_prompt_file("rules_title.txt")
        rules_openness = self._load_prompt_file("rules_openness.txt")
        rules_fields   = self._load_prompt_file("rules_fields.txt")
        checklist      = self._load_prompt_file("checklist.txt")
        examples       = self._load_examples()

        return f"""你是专业档案整理员。你的任务是从OCR文本中提取档案元数据，以JSON格式输出。

【输出格式要求 - 最高优先级】
- 只输出一个JSON对象，不得包含任何其他文字、解释、markdown
- 不得输出规则说明、著录指南或任何非JSON内容
- 第一个字符必须是 {{{{，最后一个字符必须是 }}}}
- JSON的key必须与下方【需提取的字段】完全一致，禁止使用其他字段名

{rules_priority}

{rules_category}

{rules_openness}

{rules_title}

{rules_fields}

【需提取的字段】（key名称必须与此处完全一致）
{fields_desc}

【合法key列表】（JSON只能包含以下key，不得新增或改名）
{chr(10).join(f'- {k}' for k in self.metadata_schema.keys())}

【OCR识别文本】
{{ocr_text}}

{checklist}

{examples}

再次强调：直接输出JSON对象，第一个字符是 {{{{，不得有任何前置文字：
"""

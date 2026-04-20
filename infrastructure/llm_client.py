#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
档案智能分类系统 - LLM客户端（vLLM OpenAI 兼容接口）

本模块只持有一个 OpenAI SDK 客户端，实际推理在外部 vLLM server 中执行。
启动 server 的方式见 docs/vllm_server.md。
"""

import json
import logging
import re
from typing import Dict

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from config.config import Config
from constants import METADATA_SCHEMA

logger = logging.getLogger(__name__)


class LlmClient:
    """
    通过 OpenAI 兼容 API 调用远端 vLLM 服务。

    与旧版 llama-cpp 内嵌推理的差异：
      - 不再加载模型文件，只持有 HTTP 客户端
      - `model` 字段传服务端 `--served-model-name`，非本地路径
      - JSON 强制依赖 vLLM 的 `response_format={"type": "json_object"}`
      - Qwen3 系列通过 `chat_template_kwargs.enable_thinking=False` 关闭思考模式
    """

    def __init__(
        self,
        base_url: str = Config.LLM_BASE_URL,
        api_key: str = Config.LLM_API_KEY,
        model_name: str = Config.LLM_MODEL_NAME,
        timeout: float = Config.LLM_REQUEST_TIMEOUT,
    ):
        if OpenAI is None:
            raise RuntimeError(
                "openai SDK is not installed. Install with: pip install openai"
            )
        self.metadata_schema = METADATA_SCHEMA
        self.model_name = model_name

        logger.info(f"[LLM初始化] vLLM endpoint: {base_url}")
        logger.info(f"[LLM初始化] 模型名: {model_name}")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        logger.info("[LLM初始化] OpenAI 客户端就绪")

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def extract_metadata(self, ocr_text: str, prompt: str) -> Dict:
        logger.info("[LLM] 正在分析文本并提取元数据...")
        if not ocr_text or not ocr_text.strip():
            logger.warning("[LLM警告] OCR文本为空，跳过LLM提取")
            return {}

        try:
            formatted_prompt = prompt.replace("{ocr_text}", ocr_text)
            response = self._generate(formatted_prompt)

            logger.info(f"[LLM响应] 原始响应长度: {len(response)} 字符")

            response = self._clean_response(response)

            preview = (
                response[:Config.LLM_RESPONSE_PREVIEW_LENGTH]
                if len(response) > Config.LLM_RESPONSE_PREVIEW_LENGTH
                else response
            )
            logger.info(f"[JSON清理后] {preview}...")

            metadata = self._parse_json(response)
            return metadata

        except Exception as e:
            logger.exception(f"[LLM错误] {str(e)}")
            return {}

    # ── 推理调用 ──────────────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """
        调用 vLLM OpenAI 兼容 chat completions 接口。

        - response_format=json_object：vLLM 0.6+ 原生支持 guided JSON
        - chat_template_kwargs.enable_thinking：Qwen3 专有开关，JSON 场景强制 False
        """
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": Config.LLM_ENABLE_THINKING,
            }
        }

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是专业档案整理员，只输出JSON格式的元数据，不输出任何其他内容。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        return response.choices[0].message.content or ""

    # ── 响应清洗与解析（与推理后端无关，逻辑保留）──────────────────────────────

    def _clean_response(self, response: str) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        if '{' in response and '}' in response:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            response = response[start_idx:end_idx + 1]
        lines = response.split('\n')
        json_lines = []
        in_json = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{'):
                in_json = True
            if in_json:
                json_lines.append(line)
            if stripped.endswith('}') and in_json:
                break
        if json_lines:
            response = '\n'.join(json_lines)
        return response.strip()

    def _parse_json(self, response: str) -> Dict:
        try:
            metadata = json.loads(response)
            metadata = {k: v for k, v in metadata.items() if k in self.metadata_schema}
            return metadata
        except json.JSONDecodeError as e:
            logger.warning(f"[JSON解析失败] {str(e)}")
            logger.info("[尝试修复JSON格式...]")
        # 仅替换 JSON 结构位的单引号：key 周围（{'k': / ,'k':）与简单 value 位置（: 'v',）
        # 不做全局 replace，避免破坏字符串值中合法的单引号
        fixed = re.sub(r"([{,]\s*)'([^'\n]+?)'(\s*:)", r'\1"\2"\3', response)
        fixed = re.sub(r"(:\s*)'([^'\n]*?)'(\s*[,}])", r'\1"\2"\3', fixed)
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        try:
            metadata = json.loads(fixed)
            metadata = {k: v for k, v in metadata.items() if k in self.metadata_schema}
            logger.info("[修复成功] 成功提取字段")
            return metadata
        except Exception:
            pass
        logger.info("[尝试正则表达式提取...]")
        metadata = self._extract_fields_by_regex(response)
        if metadata:
            logger.info(f"[正则提取] 成功提取 {len(metadata)} 个字段")
            return metadata
        logger.warning("[完整响应内容]")
        logger.warning("-" * 70)
        logger.warning(response)
        logger.warning("-" * 70)
        return {}

    def _extract_fields_by_regex(self, text: str) -> Dict:
        metadata = {}
        pattern = re.compile(
            r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|null|true|false|-?\d+(?:\.\d+)?|\[.*?\]|\{.*?\})',
            re.S,
        )
        for key, raw_value in pattern.findall(text):
            if key not in self.metadata_schema:
                continue

            value = raw_value.strip()
            if value == "null":
                metadata[key] = None
            elif value == "true":
                metadata[key] = True
            elif value == "false":
                metadata[key] = False
            elif value.startswith('"') and value.endswith('"'):
                try:
                    metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    metadata[key] = value[1:-1]
            elif re.fullmatch(r'-?\d+', value):
                metadata[key] = int(value)
            elif re.fullmatch(r'-?\d+\.\d+', value):
                metadata[key] = float(value)
            elif value.startswith('[') or value.startswith('{'):
                try:
                    metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    metadata[key] = value
            else:
                metadata[key] = value
        return metadata

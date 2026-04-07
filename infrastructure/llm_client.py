#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
档案智能分类系统 - LLM客户端（llama.cpp，兼容NVIDIA/DCU）
"""

import json
import logging
import re
from typing import Dict

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from config.config import Config
from constants import METADATA_SCHEMA

logger = logging.getLogger(__name__)


class LlmClient:
    """
    封装llama.cpp本地推理，行为与原版Ollama完全一致。
    GGUF + Q4_K_M量化，chat template原生支持，效果等价于 ollama run qwen2.5:14b
    """

    def __init__(self, model_path: str = Config.LLM_MODEL_PATH):
        if Llama is None:
            raise RuntimeError("llama_cpp is not installed. Install dependencies in runtime environment first.")
        self.metadata_schema = METADATA_SCHEMA

        logger.info(f"[LLM初始化] 加载模型: {model_path}")
        logger.info(f"[LLM初始化] GPU层数: {Config.LLM_N_GPU_LAYERS}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=Config.LLM_N_CTX,
            n_gpu_layers=Config.LLM_N_GPU_LAYERS,
            verbose=False,
        )
        logger.info("[LLM初始化] 模型加载完成")

    # ── 公开接口（与原版完全一致）─────────────────────────────────────────────

    def extract_metadata(self, ocr_text: str, prompt: str) -> Dict:
        logger.info("[LLM] 正在分析文本并提取元数据...")
        if not ocr_text or not ocr_text.strip():
            logger.warning("[LLM警告] OCR文本为空，跳过LLM提取")
            return {}

        try:
            formatted_prompt = prompt.format(ocr_text=ocr_text)
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

    # ── 推理（等价于原版 self.llm.invoke，含chat template + JSON强制）─────────

    def _generate(self, prompt: str) -> str:
        """
        使用chat completion接口，与Ollama行为一致：
        - 自动应用Qwen2.5的chat template
        - response_format强制JSON输出，等价于Ollama的format:json
        """
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "你是专业档案整理员，只输出JSON格式的元数据，不输出任何其他内容。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            response_format={"type": "json_object"},  # 等价于Ollama format:json
        )
        return response["choices"][0]["message"]["content"]

    # ── 以下原版逻辑一字未改 ──────────────────────────────────────────────────

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
        fixed = response.replace("'", '"')
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




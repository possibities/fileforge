#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
档案智能分类系统 - OCR客户端
"""

from dataclasses import dataclass, field
import logging
import re
from pathlib import Path
from typing import Any, List, Sequence, Tuple

from paddleocr import PaddleOCR

from config.config import Config
from constants import (
    OCR_CONFIDENCE_NORMAL,
    OCR_CONFIDENCE_SHORT,
    OCR_REPLACEMENTS,
    OCR_SHORT_TEXT_LEN,
)

# 同行合并：两个文本框垂直中心点之差小于此阈值则视为同一行
# 单位：像素，可根据实际扫描件字号调整
_SAME_LINE_Y_THRESHOLD = 15

# 水平间距：同行两个文本框之间的像素间距超过此值则插入空格
# 单位：像素
_SPACE_INSERT_THRESHOLD = 20

logger = logging.getLogger(__name__)


@dataclass
class OcrVariant:
    label: str
    image: Any


@dataclass
class PageOcrCandidate:
    label: str
    raw_line_count: int = 0
    filtered_lines: List[Any] = field(default_factory=list)
    low_conf_count: int = 0
    avg_confidence: float = 0.0
    filtered_char_count: int = 0
    page_text: str = ""
    score: float = 0.0

    @property
    def filtered_line_count(self) -> int:
        return len(self.filtered_lines)

    @property
    def low_conf_ratio(self) -> float:
        if self.raw_line_count == 0:
            return 1.0 if self.filtered_line_count == 0 else 0.0
        return self.low_conf_count / self.raw_line_count


class OcrClient:
    """封装PaddleOCR，提供单页/多页文本提取能力"""

    def __init__(self, lang: str = Config.OCR_LANG):
        self.ocr = PaddleOCR(
            use_angle_cls=Config.OCR_USE_ANGLE_CLS,
            lang=lang,
            use_gpu=Config.OCR_USE_GPU,
            show_log=Config.OCR_SHOW_LOG,
            drop_score=Config.OCR_DROP_SCORE,
            det_db_thresh=Config.OCR_DET_DB_THRESH,
            det_db_box_thresh=Config.OCR_DET_DB_BOX_THRESH,
            det_db_unclip_ratio=Config.OCR_DET_DB_UNCLIP_RATIO,
        )

    # ── 公开接口 ───────────────────────────────────────────────────────────────

    def extract_text(self, image_path: str) -> str:
        """单页提取（兼容旧接口）"""
        logger.info(f"[OCR] 正在识别图像: {image_path}")
        path = Path(image_path)
        if not path.exists():
            logger.error(f"[OCR错误] 图像不存在: {image_path}")
            return ""

        candidate = self._extract_best_candidate(image_path)
        if not candidate.page_text:
            logger.warning("[OCR警告] 未识别到有效文字")
            return ""

        logger.info(
            "[OCR] 识别完成，采用 %s 版本，平均置信度 %.3f",
            candidate.label,
            candidate.avg_confidence,
        )
        return self._clean_ocr_text(candidate.page_text)

    def extract_text_from_images(self, image_paths: List[str]) -> str:
        """
        从多个图像提取文字并合并（用于多页档案）
        核心改动：使用 _reconstruct_page() 替换原始 join，
        利用 bbox 坐标还原空格和行结构。
        """
        if not image_paths:
            logger.warning("[OCR警告] 输入图像列表为空")
            return ""

        all_text = []

        logger.info(f"[OCR] 开始识别 {len(image_paths)} 页图像...")

        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"  正在识别第 {idx}/{len(image_paths)} 页: {Path(image_path).name}")
            if not Path(image_path).exists():
                logger.warning(f"    ✗ 图像不存在: {image_path}")
                continue

            candidate = self._extract_best_candidate(image_path)
            if candidate.page_text:
                all_text.append(f"===== 第 {idx} 页 =====")
                all_text.append(candidate.page_text)
                all_text.append("")
                logger.info(
                    "    ✓ 提取 %s 个文本框（过滤低置信度 %s 个，平均置信度 %.3f，采用 %s 版本）",
                    candidate.filtered_line_count,
                    candidate.low_conf_count,
                    candidate.avg_confidence,
                    candidate.label,
                )
            else:
                logger.warning("    ✗ 未识别到有效文字")

        full_text = "\n".join(all_text)
        full_text = self._clean_ocr_text(full_text)

        logger.info(f"[OCR] 完成，共处理 {len(image_paths)} 页\n")
        return full_text

    # ── 私有方法 ───────────────────────────────────────────────────────────────

    def _extract_best_candidate(self, image_path: str) -> PageOcrCandidate:
        primary = self._run_candidate(image_path, "original")
        best = primary

        if not self._should_retry_with_preprocess(primary):
            return best

        variants = self._build_preprocessed_variants(image_path)
        if variants:
            logger.info("    · 原图结果偏弱，尝试 %s 个预处理版本", len(variants))

        for variant in variants:
            candidate = self._run_candidate(variant.image, variant.label)
            if self._is_better_candidate(candidate, best):
                best = candidate

        return best

    def _run_candidate(self, image_input: Any, label: str) -> PageOcrCandidate:
        try:
            result = self.ocr.ocr(image_input, cls=Config.OCR_USE_ANGLE_CLS)
        except Exception as exc:
            logger.warning("    · %s 版本识别失败: %s", label, exc)
            return PageOcrCandidate(label=label)

        ocr_lines = result[0] if result and result[0] else []
        filtered_lines, low_conf_count, avg_confidence, filtered_char_count = self._filter_lines(
            ocr_lines
        )
        page_text = self._reconstruct_page(filtered_lines)
        score = self._score_candidate(
            filtered_char_count=filtered_char_count,
            filtered_line_count=len(filtered_lines),
            avg_confidence=avg_confidence,
            low_conf_count=low_conf_count,
        )

        return PageOcrCandidate(
            label=label,
            raw_line_count=len(ocr_lines),
            filtered_lines=filtered_lines,
            low_conf_count=low_conf_count,
            avg_confidence=avg_confidence,
            filtered_char_count=filtered_char_count,
            page_text=page_text,
            score=score,
        )

    def _filter_lines(
        self,
        ocr_lines: Sequence[Any],
    ) -> Tuple[List[Any], int, float, int]:
        filtered_lines: List[Any] = []
        low_conf_count = 0
        kept_confidences: List[float] = []
        filtered_char_count = 0

        for line in ocr_lines:
            text = str(line[1][0]).strip()
            confidence = float(line[1][1])
            if not text:
                low_conf_count += 1
                continue

            min_conf = (
                OCR_CONFIDENCE_SHORT
                if len(text) <= OCR_SHORT_TEXT_LEN
                else OCR_CONFIDENCE_NORMAL
            )
            if confidence >= min_conf:
                filtered_lines.append(line)
                kept_confidences.append(confidence)
                filtered_char_count += len(text)
            else:
                low_conf_count += 1

        avg_confidence = (
            sum(kept_confidences) / len(kept_confidences)
            if kept_confidences
            else 0.0
        )
        return filtered_lines, low_conf_count, avg_confidence, filtered_char_count

    def _should_retry_with_preprocess(self, candidate: PageOcrCandidate) -> bool:
        if not Config.OCR_ENABLE_PREPROCESS:
            return False
        if candidate.filtered_line_count == 0:
            return True
        if candidate.avg_confidence < Config.OCR_RETRY_LOW_AVG_CONFIDENCE:
            return True
        if candidate.low_conf_ratio > Config.OCR_RETRY_LOW_CONF_RATIO:
            return True
        return (
            candidate.filtered_char_count < Config.OCR_RETRY_MIN_TEXT_CHARS
            and candidate.avg_confidence < 0.9
        )

    def _build_preprocessed_variants(self, image_path: str) -> List[OcrVariant]:
        try:
            import numpy as np
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
        except ImportError as exc:
            logger.warning("    · 预处理依赖不可用，跳过增强: %s", exc)
            return []

        try:
            with Image.open(image_path) as image:
                working = ImageOps.exif_transpose(image).convert("L")
        except Exception as exc:
            logger.warning("    · 图像预处理失败，跳过增强: %s", exc)
            return []

        working = self._resize_for_preprocess(working, image_module=Image)
        enhanced = ImageOps.autocontrast(working)
        if Config.OCR_PREPROCESS_CONTRAST != 1.0:
            enhanced = ImageEnhance.Contrast(enhanced).enhance(
                Config.OCR_PREPROCESS_CONTRAST
            )

        sharpened = enhanced.filter(ImageFilter.MedianFilter(size=3)).filter(
            ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3)
        )
        binary = self._to_otsu_binary(sharpened)

        return [
            OcrVariant(label="autocontrast", image=np.asarray(enhanced)),
            OcrVariant(label="sharpen", image=np.asarray(sharpened)),
            OcrVariant(label="binary", image=np.asarray(binary)),
        ]

    def _resize_for_preprocess(self, image: Any, image_module: Any) -> Any:
        scale = max(1.0, Config.OCR_PREPROCESS_SCALE)
        max_side = max(image.size)
        if Config.OCR_PREPROCESS_MAX_SIDE > 0 and max_side > 0:
            scale = min(scale, Config.OCR_PREPROCESS_MAX_SIDE / max_side)
        if scale <= 1.0:
            return image

        resampling = getattr(image_module, "Resampling", image_module).LANCZOS
        resized = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        return image.resize(resized, resample=resampling)

    def _to_otsu_binary(self, image: Any) -> Any:
        import numpy as np
        from PIL import Image

        pixels = np.asarray(image, dtype=np.uint8)
        histogram = np.bincount(pixels.ravel(), minlength=256)
        total = pixels.size
        weighted_total = float(np.dot(np.arange(256), histogram))

        threshold = 0
        background_weight = 0.0
        background_sum = 0.0
        max_variance = -1.0

        for gray_value, count in enumerate(histogram):
            background_weight += count
            if background_weight == 0:
                continue

            foreground_weight = total - background_weight
            if foreground_weight == 0:
                break

            background_sum += gray_value * count
            mean_background = background_sum / background_weight
            mean_foreground = (weighted_total - background_sum) / foreground_weight
            between_variance = (
                background_weight
                * foreground_weight
                * (mean_background - mean_foreground) ** 2
            )
            if between_variance > max_variance:
                max_variance = between_variance
                threshold = gray_value

        binary = np.where(pixels >= threshold, 255, 0).astype("uint8")
        return Image.fromarray(binary, mode="L")

    def _is_better_candidate(
        self,
        candidate: PageOcrCandidate,
        best: PageOcrCandidate,
    ) -> bool:
        return (
            candidate.score,
            candidate.avg_confidence,
            candidate.filtered_char_count,
        ) > (
            best.score,
            best.avg_confidence,
            best.filtered_char_count,
        )

    def _score_candidate(
        self,
        *,
        filtered_char_count: int,
        filtered_line_count: int,
        avg_confidence: float,
        low_conf_count: int,
    ) -> float:
        return (
            filtered_char_count * 1.5
            + filtered_line_count * 4.0
            + avg_confidence * 50.0
            - low_conf_count * 1.5
        )

    def _reconstruct_page(self, ocr_lines: list) -> str:
        """
        基于 bbox 坐标将 OCR 文本框重建为自然阅读顺序的文本。

        PaddleOCR 每个 line 的结构：
          [ [[x0,y0],[x1,y1],[x2,y2],[x3,y3]], (text, confidence) ]
          bbox 四个顶点顺序：左上、右上、右下、左下

        算法：
          1. 计算每个文本框的中心点 (cx, cy)
          2. 按 cy 排序，cy 差值小于阈值的归为同一行
          3. 同行内按 cx 排序（从左到右）
          4. 同行相邻文本框水平间距超过阈值则插入空格
          5. 行间用换行符连接
        """
        if not ocr_lines:
            return ""

        boxes = []
        for line in ocr_lines:
            bbox = line[0]
            text = line[1][0]

            cx = sum(p[0] for p in bbox) / 4
            cy = sum(p[1] for p in bbox) / 4
            x_right = max(p[0] for p in bbox)
            x_left = min(p[0] for p in bbox)

            boxes.append(
                {
                    "text": text,
                    "cx": cx,
                    "cy": cy,
                    "x_left": x_left,
                    "x_right": x_right,
                }
            )

        boxes.sort(key=lambda b: b["cy"])

        rows: List[List[dict]] = []
        current_row: List[dict] = [boxes[0]]

        for box in boxes[1:]:
            if abs(box["cy"] - current_row[-1]["cy"]) <= _SAME_LINE_Y_THRESHOLD:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)

        text_lines = []
        for row in rows:
            row.sort(key=lambda b: b["cx"])

            line_parts = [row[0]["text"]]
            for i in range(1, len(row)):
                gap = row[i]["x_left"] - row[i - 1]["x_right"]
                if gap > _SPACE_INSERT_THRESHOLD:
                    line_parts.append(" ")
                line_parts.append(row[i]["text"])

            text_lines.append("".join(line_parts))

        return "\n".join(text_lines)

    def _clean_ocr_text(self, text: str) -> str:
        """
        清洗OCR常见噪声。
        在原有替换表基础上新增六角括号规范化。
        """
        text = text.replace("〔", "[").replace("〕", "]")
        text = text.replace("【", "[").replace("】", "]")

        for old, new in OCR_REPLACEMENTS.items():
            text = text.replace(old, new)

        text = re.sub(r"\n{3,}", "\n\n", text)

        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)

        return text

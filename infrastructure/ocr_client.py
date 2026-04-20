#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
档案智能分类系统 - OCR客户端
"""

from dataclasses import dataclass, field
import logging
import re
import statistics
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

# 同行合并阈值：以该页文本框中位高度的倍数衡量
# 两个文本框垂直中心点之差 < median_h * _SAME_LINE_Y_RATIO 视为同行
_SAME_LINE_Y_RATIO = 0.6

# 水平间距阈值：以该页文本框中位高度的倍数衡量
# 同行相邻文本框水平间距 > median_h * _SPACE_INSERT_RATIO 则插入空格
_SPACE_INSERT_RATIO = 0.8

# 中位字高取不到时的兜底（像素），主要防止空页或退化输入
_FALLBACK_LINE_HEIGHT = 20.0

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
                # 当前 best 已满足质量门槛（不再触发 retry 判定），
                # 剩余预处理版本无需再跑 OCR，省掉整轮推理
                if not self._should_retry_with_preprocess(best):
                    logger.info(
                        "    · %s 版本质量已达标，剩余 %s 个预处理版本跳过",
                        variant.label,
                        len(variants) - variants.index(variant) - 1,
                    )
                    break

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
        histogram = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
        total = float(pixels.size)

        # 累积直方图与加权累积和，整段向量化替代原 256 次 Python 循环
        levels = np.arange(256, dtype=np.float64)
        bg_weight = np.cumsum(histogram)
        fg_weight = total - bg_weight
        bg_sum = np.cumsum(histogram * levels)
        total_sum = bg_sum[-1]

        # 分母为 0 的两端用 0 填充，argmax 会自然忽略
        valid = (bg_weight > 0) & (fg_weight > 0)
        between = np.zeros_like(bg_weight)
        bg_mean = np.where(valid, bg_sum / np.where(bg_weight == 0, 1, bg_weight), 0.0)
        fg_mean = np.where(
            valid,
            (total_sum - bg_sum) / np.where(fg_weight == 0, 1, fg_weight),
            0.0,
        )
        between[valid] = (
            bg_weight[valid] * fg_weight[valid] * (bg_mean[valid] - fg_mean[valid]) ** 2
        )

        threshold = int(between.argmax())
        binary = np.where(pixels >= threshold, 255, 0).astype(np.uint8)
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
          1. 计算每个文本框的中心点 (cx, cy) 与字高 h
          2. 取所有字高的中位数作为该页的基准行高 median_h
          3. 按 cy 排序；cy 差 < median_h * _SAME_LINE_Y_RATIO 归为同一行
          4. 同行内按 cx 排序（从左到右）
          5. 相邻文本框水平间距 > median_h * _SPACE_INSERT_RATIO 则插入空格
          6. 行间用换行符连接

        阈值相对于字高，避免在不同 DPI / 预处理缩放下失配。
        """
        if not ocr_lines:
            return ""

        boxes = []
        heights = []
        for line in ocr_lines:
            bbox = line[0]
            text = line[1][0]

            cx = sum(p[0] for p in bbox) / 4
            cy = sum(p[1] for p in bbox) / 4
            x_right = max(p[0] for p in bbox)
            x_left = min(p[0] for p in bbox)
            y_top = min(p[1] for p in bbox)
            y_bot = max(p[1] for p in bbox)
            h = max(1.0, y_bot - y_top)

            heights.append(h)
            boxes.append(
                {
                    "text": text,
                    "cx": cx,
                    "cy": cy,
                    "x_left": x_left,
                    "x_right": x_right,
                }
            )

        median_h = statistics.median(heights) if heights else _FALLBACK_LINE_HEIGHT
        if median_h <= 0:
            median_h = _FALLBACK_LINE_HEIGHT

        same_line_threshold = median_h * _SAME_LINE_Y_RATIO
        space_insert_threshold = median_h * _SPACE_INSERT_RATIO

        boxes.sort(key=lambda b: b["cy"])

        rows: List[List[dict]] = []
        current_row: List[dict] = [boxes[0]]

        for box in boxes[1:]:
            if abs(box["cy"] - current_row[-1]["cy"]) <= same_line_threshold:
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
                if gap > space_insert_threshold:
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

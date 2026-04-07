#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
档案智能分类系统 - 规则引擎
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

from constants import (
    COMMERCIAL_EXEMPT_KEYWORDS,
    COMMERCIAL_KEYWORDS,
    CONTROLLED_SECURITY_LEVELS,
    NEGATIVE_PATTERNS,
    NEGATIVE_TITLE_KEYWORDS,
    PERIOD_ORDER,
    PRIVACY_KEYWORDS,
    CODE_NEW,
    CODE_OLD,
    CODE_SWITCH_YEAR,
    ADDRESS_CHANGE_KEYWORDS,
    BID_KEYWORDS,
    BRIEFING_BUSINESS_KEYWORDS,
    BRIEFING_PARTY_KEYWORDS,
    IMPORTANT_NOTICE_KEYWORDS,
    INTERNAL_ORG_KEYWORDS,
    MAINTENANCE_DOC_TYPES,
    MAINTENANCE_KEYWORDS,
    PARTY_BRANCH_ADJUST_KEYWORDS,
    PARTY_BRANCH_ELECTION_RESULT_KEYWORDS,
    PARTY_BRANCH_TARGET_KEYWORDS,
    PARTY_BRANCH_KEYWORDS,
    REGULATION_KEYWORDS,
    TRAINING_KEYWORDS,
    TRAINING_MGMT_KEYWORDS,
    PARTY_TRAINING_KEYWORDS,
    BUSINESS_FALSE_POSITIVE_DOC_TYPES,
    BUSINESS_LEGITIMATE_KEYWORDS,
)


class RulesEngine:
    """
    规则引擎：按优先级顺序执行五层规则修正

    执行顺序：
      0. _force_fix_fields             — 强制字段修正
      1. _apply_supplementary_rules    — 10条补充规则（最高优先级）
      2. _apply_open_status_rules      — 开放状态与延期开放理由判定
      3. _validate_classification_code — 编码格式校验
      4. _clean_title                  — 题名后处理（去除LLM拼入的编号/日期/重复内容）
    """

    def apply_all(self, metadata: Dict, ocr_text: str) -> Dict:
        logger.info("\n[开始应用规则修正]")
        metadata = self._force_fix_fields(metadata)
        metadata = self._apply_supplementary_rules(metadata, ocr_text)
        metadata = self._apply_open_status_rules(metadata, ocr_text)
        metadata = self._validate_classification_code(metadata)
        metadata = self._clean_title(metadata)
        logger.info("[规则修正完成]\n")
        return metadata

    # ── 优先级0：强制字段修正 ──────────────────────────────────────────────────

    def _force_fix_fields(self, metadata: Dict) -> Dict:
        """强制留空字段 + 立档单位名称同步责任者 + 密级合法值校验"""
        if not metadata:
            return metadata

        # 立档单位名称与责任者保持一致
        if not metadata.get("立档单位名称"):
            metadata["立档单位名称"] = metadata.get("责任者")

        # 密级合法值校验：不在允许范围内一律置null
        VALID_SECURITY_LEVELS = {"非涉密", "内部", "秘密", "机密", "绝密"}
        current_level = metadata.get("密级")
        if current_level and current_level not in VALID_SECURITY_LEVELS:
            logger.warning(f"[强制修正] 密级非法值: {current_level} → null")
            metadata["密级"] = None
            metadata["保密期限"] = None

        # 保密期限合法值校验
        VALID_SECRET_PERIODS = {"1年", "5年", "10年"}
        current_period = metadata.get("保密期限")
        if current_period and current_period not in VALID_SECRET_PERIODS:
            logger.warning(f"[强制修正] 保密期限非法值: {current_period} → null")
            metadata["保密期限"] = None

        return metadata

    # ── 优先级1：10条补充规则 ─────────────────────────────────────────────────

    def _apply_supplementary_rules(self, metadata: Dict, ocr_text: str) -> Dict:
        """
        10条补充规则，优先级高于LLM输出。

        执行顺序说明：
          - 规则2（简报→10年）最先执行，并设置 period_locked=True
          - period_locked=True 时，后续所有规则不得修改保管期限
          - 规则7（文件编号兜底）最后执行，但受 period_locked 保护
          - 规则1（培训）不直接覆盖保管期限，仅在LLM判定低于30年时提升
        """
        if not metadata:
            return metadata

        title = str(metadata.get("题名") or "").strip()
        text = ocr_text or ""
        content = title + " " + text

        # 期限锁标志：规则2触发后设True，后续规则不得修改保管期限
        period_locked = False

        # ── 规则2: 简报 → 10年（最先执行，锁定期限）─────────────────────────
        if "简报" in title:
            logger.info(f"[补充规则2] 简报，保管期限 → 10年（已锁定，后续规则不覆盖）")
            metadata["保管期限"] = "10年"
            period_locked = True

            # 规则2扩展：简报分类修正
            # 规则引擎接管分类判断，防止LLM将党务简报误归综合类/业务类
            if any(kw in content for kw in BRIEFING_PARTY_KEYWORDS):
                logger.info(f"[补充规则2-分类] 党务简报，分类 → 党群类")
                metadata["实体分类名称"] = "党群类"
            elif any(kw in content for kw in BRIEFING_BUSINESS_KEYWORDS):
                logger.info(f"[补充规则2-分类] 档案/培训简报，分类 → 业务类")
                metadata["实体分类名称"] = "业务类"
            else:
                logger.info(f"[补充规则2-分类] 一般简报，分类 → 综合类")
                metadata["实体分类名称"] = "综合类"
            metadata = self._validate_classification_code(metadata)

        # ── 规则1: 公司内部培训 → 业务类 ─────────────────────────────────────
        # 排除条件1：题名含管理类词汇（培训制度/经费/管理/考勤等）
        # 排除条件2：内容涉及党务/党员/党建等党务培训
        # 保管期限处理：不硬编码，仅在LLM判定低于30年时提升到30年（简报除外）
        is_training = any(kw in title for kw in TRAINING_KEYWORDS)
        is_training_mgmt = any(kw in title for kw in TRAINING_MGMT_KEYWORDS)
        is_party_training = any(kw in content for kw in PARTY_TRAINING_KEYWORDS)

        if is_training and not is_training_mgmt and not is_party_training:
            logger.info(f"[补充规则1] 公司内部培训，分类 → 业务类")
            metadata["实体分类名称"] = "业务类"
            # 期限：未锁定时，若LLM判定低于30年则提升；永久则保留
            if not period_locked:
                current_period = metadata.get("保管期限", "")
                if PERIOD_ORDER.get(current_period, 0) < PERIOD_ORDER["30年"]:
                    logger.info(f"[补充规则1] 培训类保管期限: {current_period} → 30年")
                    metadata["保管期限"] = "30年"
            metadata = self._validate_classification_code(metadata)

        # ── 规则3: 档案寄存地址变更 → 10年 ───────────────────────────────────
        if any(kw in content for kw in ADDRESS_CHANGE_KEYWORDS):
            if not period_locked:
                logger.info(f"[补充规则3] 档案寄存地址变更，保管期限 → 10年")
                metadata["保管期限"] = "10年"
            else:
                logger.info(f"[补充规则3] 档案寄存地址变更，期限已锁定（{metadata.get('保管期限')}），跳过")

        # ── 规则4: 公司内部安装/维修类函和通知 → 10年 ────────────────────────
        if any(kw in content for kw in MAINTENANCE_KEYWORDS):
            if any(doc_type in title for doc_type in MAINTENANCE_DOC_TYPES):
                if not period_locked:
                    logger.info(f"[补充规则4] 安装/维修类通知或函，保管期限 → 10年")
                    metadata["保管期限"] = "10年"
                else:
                    logger.info(f"[补充规则4] 安装/维修类，期限已锁定（{metadata.get('保管期限')}），跳过")

        # ── 规则5: 一般事务性通知 → 10年 ─────────────────────────────────────
        # 有文件编号 或 含重要关键词 → 不触发
        if "通知" in title:
            file_number = metadata.get("文件编号")
            has_number = (
                file_number
                and str(file_number).strip()
                and str(file_number) != "null"
            )
            is_important = any(kw in content for kw in IMPORTANT_NOTICE_KEYWORDS)
            if not has_number and not is_important:
                if not period_locked:
                    logger.info(f"[补充规则5] 一般事务通知，保管期限 → 10年")
                    metadata["保管期限"] = "10年"
                else:
                    logger.info(f"[补充规则5] 一般事务通知，期限已锁定（{metadata.get('保管期限')}），跳过")

        # ── 规则6: 公司内部制度/管理办法/条例/实施细则/章程 → 综合类，30年 ────
        if any(kw in title for kw in REGULATION_KEYWORDS):
            if any(kw in content for kw in INTERNAL_ORG_KEYWORDS):
                logger.info(f"[补充规则6] 公司内部制度，分类 → 综合类")
                metadata["实体分类名称"] = "综合类"
                if not period_locked:
                    logger.info(f"[补充规则6] 保管期限 → 30年")
                    metadata["保管期限"] = "30年"
                else:
                    logger.info(f"[补充规则6] 期限已锁定（{metadata.get('保管期限')}），不修改期限")
                metadata = self._validate_classification_code(metadata)

        # ── 规则8: 批评通报 → 30年 ────────────────────────────────────────────
        if any(kw in title for kw in ["批评通报", "通报批评"]):
            if not period_locked:
                logger.info(f"[补充规则8] 批评通报，保管期限 → 30年")
                metadata["保管期限"] = "30年"
            else:
                logger.info(f"[补充规则8] 批评通报，期限已锁定（{metadata.get('保管期限')}），跳过")

        # ── 规则9: 中标结果公示/中标通知函 → 30年 ────────────────────────────
        if "中标" in title and any(kw in title for kw in BID_KEYWORDS):
            if not period_locked:
                logger.info(f"[补充规则9] 中标结果/通知函，保管期限 → 30年")
                metadata["保管期限"] = "30年"
            else:
                logger.info(f"[补充规则9] 中标结果，期限已锁定（{metadata.get('保管期限')}），跳过")

        # ── 规则10: 党支部更换组织/委员/书记的请示 → 党群类，30年 ──────────
        # 排除：换届选举结果类文件（属永久，不得被降级）
        if any(kw in content for kw in PARTY_BRANCH_KEYWORDS):
            is_adjust = any(kw in content for kw in PARTY_BRANCH_ADJUST_KEYWORDS)
            is_target = any(kw in content for kw in PARTY_BRANCH_TARGET_KEYWORDS)
            is_request = "请示" in content
            is_election_result = any(kw in content for kw in PARTY_BRANCH_ELECTION_RESULT_KEYWORDS)

            if is_adjust and is_target and is_request and not is_election_result:
                logger.info(f"[补充规则10] 党支部调整请示，分类 → 党群类")
                metadata["实体分类名称"] = "党群类"
                if not period_locked:
                    logger.info(f"[补充规则10] 保管期限 → 30年")
                    metadata["保管期限"] = "30年"
                else:
                    logger.info(f"[补充规则10] 期限已锁定（{metadata.get('保管期限')}），不修改期限")
                metadata = self._validate_classification_code(metadata)
            elif is_election_result:
                logger.info(f"[补充规则10] 检测到换届选举结果，跳过（期限应为永久）")

        # ── 业务类误判兜底：非档案工作/非培训文件 → 强制纠正为综合类 ───────────
        # 触发条件：LLM将文件分到业务类，但文种/特征明显属于综合类
        # 排除条件：题名含明确的档案工作词或培训词（说明确实是业务类）
        if metadata.get("实体分类名称") == "业务类":
            is_legitimate_business = any(kw in content for kw in BUSINESS_LEGITIMATE_KEYWORDS)
            if not is_legitimate_business:
                is_false_positive = any(doc_type in title for doc_type in BUSINESS_FALSE_POSITIVE_DOC_TYPES)
                if is_false_positive:
                    logger.warning(f"[业务类兜底] 题名含综合类文种（{title}），非档案/培训文件，强制纠正 → 综合类")
                    metadata["实体分类名称"] = "综合类"
                    metadata = self._validate_classification_code(metadata)

        # ── 规则7: 本单位带文件编号 → 最少30年（兜底，最后执行）──────────────
        # [Fix3] 使用 PERIOD_ORDER 比较，任何低于30年的期限均提升
        file_number = metadata.get("文件编号")
        has_number = (
            file_number
            and str(file_number).strip()
            and str(file_number) != "null"
        )
        if has_number:
            current_period = metadata.get("保管期限", "")
            if PERIOD_ORDER.get(current_period, 0) < PERIOD_ORDER["30年"]:
                if not period_locked:
                    logger.info(f"[补充规则7] 本单位带文件编号，保管期限: {current_period} → 30年")
                    metadata["保管期限"] = "30年"
                else:
                    logger.info(f"[补充规则7] 带文件编号，但期限已锁定（{current_period}），跳过")

        return metadata

    # ── 优先级2：开放状态与延期开放理由 ──────────────────────────────────────

    def _apply_open_status_rules(self, metadata: Dict, ocr_text: str) -> Dict:
        """
        开放状态判定（默认开放）
        优先级：密级标注 > 文件主要内容
        工作秘密仅以密级标注字段（CONTROLLED_SECURITY_LEVELS）为准，不扫描正文
        延期开放理由只填最主要一个原因
        """
        if not metadata:
            return metadata

        title = str(metadata.get("题名") or "").strip()
        text = ocr_text or ""

        metadata["开放状态"] = "开放"
        metadata["延期开放理由"] = None

        # 第一优先级：密级字段标注
        # [Fix6] 仅使用 CONTROLLED_SECURITY_LEVELS，不再扫描正文 WORK_SECRET_KEYWORDS
        if metadata.get("密级") in CONTROLLED_SECURITY_LEVELS:
            metadata["开放状态"] = "控制"
            metadata["延期开放理由"] = "工作秘密"
            return metadata

        # 第二优先级：文件主要内容（命中即停止）

        # 个人隐私
        if any(kw in title or kw in text for kw in PRIVACY_KEYWORDS):
            metadata["开放状态"] = "控制"
            metadata["延期开放理由"] = "个人隐私"
            return metadata

        # 商业秘密（排除公开中标结果）
        if any(kw in title or kw in text for kw in COMMERCIAL_KEYWORDS):
            if not any(kw in title for kw in COMMERCIAL_EXEMPT_KEYWORDS):
                metadata["开放状态"] = "控制"
                metadata["延期开放理由"] = "商业秘密"
                return metadata

        # 负面信息：题名关键词
        # [Fix7] "约谈"已从 NEGATIVE_TITLE_KEYWORDS 移除，改为"诫勉约谈"精确匹配
        if any(kw in title for kw in NEGATIVE_TITLE_KEYWORDS):
            metadata["开放状态"] = "控制"
            metadata["延期开放理由"] = "负面信息"
            return metadata

        # 负面信息：正则精确匹配（处分类）
        for pattern in NEGATIVE_PATTERNS:
            if re.search(pattern, title) or re.search(pattern, text):
                metadata["开放状态"] = "控制"
                metadata["延期开放理由"] = "负面信息"
                return metadata

        return metadata

    # ── 优先级3：编码格式校验 ─────────────────────────────────────────────────

    def _validate_classification_code(self, metadata: Dict) -> Dict:
        """
        根据文件形成时间年份判断编码
        优先取文件形成时间前4位，降级使用归档年度
        """
        if not metadata:
            return metadata

        year = None

        # 优先：文件形成时间（格式YYYYMMDD）
        formed_time = str(metadata.get("文件形成时间") or "").strip()
        if formed_time and len(formed_time) >= 4:
            try:
                year = int(formed_time[:4])
            except ValueError:
                pass

        # 降级：归档年度
        if not year:
            try:
                year = int(str(metadata.get("归档年度", "")))
            except (ValueError, TypeError):
                return metadata

        category_name = metadata.get("实体分类名称", "")
        expected_code = self._resolve_code(year, category_name)

        if expected_code:
            current_code = metadata.get("实体分类号", "")
            if current_code != expected_code:
                logger.warning(
                    f"[编码校验] 实体分类号: {current_code} → {expected_code}"
                    f" (文件年份: {year})"
                )
            metadata["实体分类号"] = expected_code

        return metadata

    # ── 优先级4：题名后处理 ───────────────────────────────────────────────────

    def _clean_title(self, metadata: Dict) -> Dict:
        """
        题名字段后处理兜底（[Fix8-Fix13]）

        针对LLM高频确定性错误进行硬性清除，不涉及语义判断。
        处理规则（按执行顺序）：

          规则1  — 去除末尾纯数字日期        [20191106]、(20200527)
          规则2  — 去除末尾中文日期          [2019年9月3日]、(2020年5月27日)
          规则3  — 去除末尾年份版本标注      [2019年版]、[2020年号]
          规则4  — 去除开头年份标注          [2020]、(2019)
          规则5  — 去除开头中文日期          [2024年11月22日]
          规则6  — 去除末尾带括号文件编号    (黄脉源通政发[2020]2号)
          规则6b — 去除末尾裸露文件编号      黄脉源通政发(2019)23号（无外层括号）
          规则6c — 去除末尾 [YYYY]N号 编号   [2019]1号
          规则7  — 去除无意义重复另拟        题名[题名]（[ ]内容与主体完全相同）
          规则8a — 去除简报破折号后来源描述  ——金安集团高温慰问活动简报
          规则8b — 去除简报末尾期号（带符号）— 第3期
          规则8c — 去除简报末尾期号（带括号）（第3期）
          规则8d — 去除简报末尾期号（裸露）  第3期
          规则9  — 去除开头裸露编号前缀      26号 / 第26号
          规则10 — 去除冗余前置另拟          [X]关于印发《X》的通知 → 关于印发《X》的通知

        不处理的情形（属语义判断，保留给LLM）：
          - [ ]内容与原题名不同的合法另拟（如 通知[共青团中央关于…的通知]）
          - 合订件标注（如 关于XX的复函[及函]）
        """
        if not metadata:
            return metadata

        title = str(metadata.get("题名") or "").strip()
        if not title:
            return metadata

        original = title

        # ── 规则1: 末尾纯数字日期（6-8位） ───────────────────────────────────
        # 匹配：[20191106]、(20200527)、【20200101】
        title = re.sub(r'\s*[\[\(（【]\d{6,8}[\]\)）】]$', '', title)

        # ── 规则2: 末尾中文日期 ───────────────────────────────────────────────
        # 匹配：[2019年9月3日]、(2020年5月27日)
        title = re.sub(
            r'\s*[\[\(（【]\d{4}年\d{1,2}月\d{1,2}日[\]\)）】]$', '', title
        )

        # ── 规则3: 末尾年份版本标注 ───────────────────────────────────────────
        # 匹配：[2019年版]、[2020年号]、[2019年期]
        title = re.sub(r'\s*[\[\(（【]\d{4}年[版号期][\]\)）】]$', '', title)

        # ── 规则4: 开头年份标注 ───────────────────────────────────────────────
        # 匹配：[2020]、(2019)、【2020】
        title = re.sub(r'^[\[\(（【]\d{4}[\]\)）】]\s*', '', title)

        # ── 规则5: 开头中文日期 ───────────────────────────────────────────────
        # 匹配：[2024年11月22日]、(2020年5月27日)
        title = re.sub(
            r'^[\[\(（【]\d{4}年\d{1,2}月\d{1,2}日[\]\)）】]\s*', '', title
        )

        # ── 规则6: 末尾带括号文件编号 ─────────────────────────────────────────
        # 匹配：(黄脉源通政发[2020]2号)、（脉源通[2019]5号）
        # 特征：括号内含中文机构名 + [YYYY] + 数字 + 号
        title = re.sub(
            r'\s*[\(（][^\)）]{0,30}[\[【]\d{4}[\]】][^\)）]{0,10}[\)）]$', '', title
        )

        # ── 规则6b: 末尾裸露文件编号（无外层括号） ────────────────────────────
        # 匹配：黄脉源通政发(2019)23号、脉源通(2020)5号
        # 特征：空格后接中文机构简称 + (YYYY) + 数字 + 号，紧贴末尾
        title = re.sub(
            r'\s+[^\s\[\(]{1,10}[\(\[（【]\d{4}[\)\]）】]\d+号$', '', title
        )

        # ── 规则6c: 末尾 [YYYY]N号 形式编号 ──────────────────────────────────
        # 匹配：[2019]1号、[2020]12号
        title = re.sub(r'\s*\[\d{4}\]\d+号$', '', title)

        # ── 规则7: 无意义重复另拟 ─────────────────────────────────────────────
        # 匹配：关于春节放假的通知[关于春节放假的通知]
        # 仅处理[ ]内容与主体完全相同的情形，不误删合法另拟
        repeat_match = re.match(r'^(.+?)\s*\[(\1)\]$', title)
        if repeat_match:
            title = repeat_match.group(1)

        # ── 规则8: 简报专项清洗（仅在题名含"简报"时触发）────────────────────
        if "简报" in title:
            # 规则8a: 去除破折号后拼接的来源单位+简报描述
            # 匹配：……炎炎夏日"送清凉"——金安集团高温慰问活动简报
            # 保留：不忘初心……廉洁自律（简报在主体中不在尾部补充描述）
            title = re.sub(r'\s*[—－]{1,2}[^—]{2,30}简报$', '', title)
            # 规则8b: 去除末尾带连字符的期号：— 第3期、- 第3期
            title = re.sub(r'\s*[-－—]\s*第\d+期$', '', title)
            # 规则8c: 去除末尾带括号的期号：（第3期）、(第3期)
            title = re.sub(r'\s*[（(【]\s*第\d+期\s*[)）】]\s*$', '', title)
            # 规则8d: 去除末尾裸露期号：第3期
            title = re.sub(r'\s*第\d+期$', '', title)

        # ── 规则9: 开头裸露编号前缀 ───────────────────────────────────────────
        # 匹配：26号 关于…、第26号 关于…
        # 防止误删：仅匹配1-3位数字+号，后跟空格
        title = re.sub(r'^第?\d{1,3}号\s+', '', title)

        # ── 规则10: 冗余前置另拟 ──────────────────────────────────────────────
        # 匹配：[公司接待管理标准]关于印发《公司接待管理标准》的通知
        # 逻辑：[ ]内容与后文《》内容相同时，去除前置[ ]部分
        # 仅处理"关于印发《X》"结构，避免误删其他合法前置另拟
        title = re.sub(
            r'^\[([^\]]+)\](关于印发《\1》.*)', r'\2', title
        )

        title = title.strip()

        if title != original:
            logger.info(f"[题名清洗] {original!r} → {title!r}")
            metadata["题名"] = title

        # ── 规则11: 简报文学性标题重构 ──────────────────────────────────────
        # 触发条件：
        #   a) 题名含"简报"
        #   b) 题名中无"——"（说明LLM未完成重构）
        #   c) 题名不含动词+宾语结构（即不含"关于"、"开展"、"召开"等实质事由词）
        #   d) 题名不以"["开头（已另拟的不再处理）
        # 处理方式：
        #   仅标记 metadata["_需重构简报题名"] = True，交由LLM二次调用或人工核查
        #   规则引擎不自动拼接责任者（责任者字段来源不稳定，硬拼接易出错）
        SUBSTANTIVE_VERBS = [
            "关于", "开展", "召开", "组织", "举办", "开办",
            "推进", "落实", "部署", "传达", "学习", "讨论",
        ]
        if (
            "简报" in title
            and "——" not in title
            and not title.startswith("[")
            and not any(v in title for v in SUBSTANTIVE_VERBS)
        ):
            logger.warning(
                f"[规则11] 疑似文学性简报标题，未含机构+事由，建议人工核查: {title!r}"
            )
            metadata["备注"] = (
                (metadata.get("备注") or "")
                + f"【待核查】简报题名疑为文学性标题，需补充责任者及活动事由: {title}"
            ).strip()

        return metadata

    @staticmethod
    def _resolve_code(year: int, category_name: str) -> str:
        """
        根据文件年份和分类名称解析编码
        [Fix5] 使用精确匹配（key == category_name），防止"业务管理类"误匹配"业务类"
        """
        mapping = CODE_NEW if year >= CODE_SWITCH_YEAR else CODE_OLD
        return mapping.get(category_name, "")


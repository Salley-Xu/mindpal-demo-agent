#!/usr/bin/env python3
"""
风险语义识别 — 数据统一处理脚本 v2

功能：
1. 加载所有公开数据源（SOS-1K fine-grained, SOS-HL-1K, MentalGLM）
2. 统一样本 schema
3. 文本清洗
4. 去重与近重复处理
5. 初始标签映射（→ cssrs_lite_level 0-3）
6. 自动标注 subject_context（规则检测）
7. 自动标注 evidence_tags（规则检测 + 源标签映射）
8. silver/review 分流
9. group-safe train/dev/test 切分
10. 输出标签分布报告

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --output-dir processed
"""

import os
import json
import re
import hashlib
import argparse
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

random.seed(42)

# ============================================================
# 路径配置
# ============================================================
RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "processed"


# ============================================================
# 规则库：subject_context 检测
# ============================================================

# 否定/拒绝模式
NEGATION_PATTERNS = [
    r"没有(想|自杀|自伤|死的想法|勇气|能力)",
    r"不(会|想|要|敢|能|打算|愿意)(自杀|自伤|去死|跳|割腕|上吊|离开)",
    r"不(会|想|要|敢)(去死|死掉|自杀|寻死)",
    r"别(担心|怕|这样|想不开)",
    r"只是(吐槽|发泄|说说|抱怨)",
    r"不是真的想死",
    r"开玩笑的?",
    r"我才不(会|去|要)",
    r"又没(想|有|打算)自杀",
    r"根本没(想|有)自杀",
    r"打消.*念头",
    r"不会自杀",
    r"不想死",
    r"没有.*自杀.*念头",
    r"没有死的勇气",
]

# 第三方/他人模式
THIRD_PARTY_PATTERNS = [
    r"他[们]?(说|的|家|就|也|还|想|要|在).*(自杀|自残|跳|割|上吊)",
    r"她[们]?(说|的|家|就|也|还|想|要|在).*(自杀|自残|跳|割|上吊)",
    r"有[个位]?(朋友|同学|同事|网友).*(自杀|跳|割|离开|想不开)",
    r"朋友.*(自杀|跳|割|上吊|想不开|离开)",
    r"听说.*(自杀|跳|死了|想不开)",
    r"新闻.*(自杀|跳|坠)",
    r"看.*(跳楼|自杀|割腕|上吊).*新闻",
    r"室友.*(自杀|跳|割)",
    r"替.*(朋友|同学|他|她).*求助",
    r"关[心注]?那些.*(自杀|抑郁|不开心).*人",
    r"有.*人.*(跳楼|自杀|想不开)",
    r"(男[友朋]|女[友朋]|亲戚|邻居).*(自杀|跳|割|上吊|离开|想不开)",
]

# 过去经历模式
PAST_PATTERNS = [
    r"曾[经]?(想|有过|自杀|自残|割|跳)",
    r"以前.*(想死|自杀|割|跳|自残)",
    r"那[时年段].*(想死|自杀|割|跳|自残)",
    r"过[去往].*(想死|自杀|割|跳|自残)",
    r"当时.*(想|要|去)(死|自杀|跳)",
    r"曾经.*(自|割|跳|吞|烧)",
    r"几年前.*(自|割|跳|吞|烧)",
]

# 假设/讨论/引用模式
HYPOTHETICAL_PATTERNS = [
    r"如果.*(自杀|跳|割腕|上吊|死).*会(怎么样|怎样|多痛苦)",
    r"假如.*(自杀|死|跳|离开)",
    r"讨论.*自杀",
    r"科普.*自杀",
    r"电影.*(自杀|跳|死)",
    r"电视.*(自杀|跳|死)",
    r"新闻.*跳楼.*女孩",
    r"[说问]起.*自杀话题",
    r"引用.*(数据|研究|文章).*自杀",
    r"(高分|推荐).*电影.*(涉及|讨论).*自杀",
    r"纯属虚构", r"小说情节",
    r"学术讨论", r"研究.*指出",
    r"写论文",
]

# 保护因素模式
PROTECTIVE_FACTOR_PATTERNS = [
    r"为了.*(家人|父母|孩子|朋友).*(活|撑|坚持|忍)",
    r"放不下.*(家人|父母|孩子|牵挂)",
    r"还有.*(牵挂|留恋|不舍)",
    r"舍不得.*(家人|父母|孩子|朋友|猫|狗)",
    r"怕.*(家人|父母).*伤心",
    r"不能.*让.*(家人|父母).*难过",
    r"还想.*(撑|活|坚持|看看)",
    r"求(救|助|帮助|帮帮我)",
    r"有谁.*(帮|陪|救).*我",
    r"救救.*我",
    r"(打电话|联系).*(求助|热线|咨询)",
    r"预约.*(咨询|医生|治疗)",
    r"在看医生",
    r"吃药.*(治疗|控制|好|)",
    r"希望.*(帮|救|好起|走出)",
    r"不想.*让.*(担心|难过|伤心)",
    r"还有.*(意义|希望) *(吧|吗|么)",
    r"为了.*(活|忍|撑).*下去",
    r"也想.*(活|好起|坚持|撑)",
    r"活下[去来]",
    r"我还.*(年轻|有.*未来)",
]

# 求助模式
HELP_SEEKING_PATTERNS = [
    r"怎么?办",
    r"帮.*我",
    r"求.*(助|救|帮)",
    r"有[没]?有人.*(帮|陪|救)",
    r"救救",
    r"支持.*我",
    r"我该怎?么(办|做)",
    r"有.*(推荐|建议|办法|方法).*吗",
    r"(倾诉|树洞).*谢谢",
    r"谢谢.*(听|看|倾)",
]

# 被动死亡愿望
PASSIVE_DEATH_WISH_PATTERNS = [
    r"想(离开|消失|不存在|睡过去|永不醒来|走)",
    r"希望.*(不存?在|消失|离开|死|结束)",
    r"不想(醒|存在|活|继续|面对)",
    r"想.*(死|离开).*但.*不(敢|会|能|行)",
    r"活.*没[意义意思]",
    r"活.*有[什啥]?么?意思",
    r"不如.*死",
    r"死了算[了吧]",
    r"死.*解脱",
    r"想.*(早点|尽快).*(离开|结束|死)",
    r"盼.*(死|结束|离开)",
    r"还有.*多[久长].*(熬|撑|忍|活)",
    r"活.*太.*累",
    r"累.*想.*(死|离开|结束)",
    r"受够了.*(生活|一切|这)",
    r"不想.*活.*了",
    r"离开.*世界",
    r"永远.*(离开|消失)",
    r"消失.*(就好|就好了|最好)",
    r"好想.*(消失|离开|不存在)",
]

# 主动自杀意念
ACTIVE_IDEATION_PATTERNS = [
    r"想.*(自杀|自残|自伤|割腕|上吊|跳|烧炭|吞药|安眠药|跳河|跳海|跳楼|坠|卧轨)",
    r"想.*(死|死掉|死去)",
    r"(自杀|自残|自伤).*念头",
    r"有.*自杀.*(想法|念头|冲动|倾向)",
    r"想.*(结束|了结|终结).*(自己|生命|一切)",
    r"想去死",
    r"想死.*了",
    r"去死一死",
    r"一了百了",
    r"不想活了",
    r"活不下去",
    r"撑不下去",
    r"坚持不下去",
    r"准备.*(自杀|死|离开)",
    r"打算.*(自杀|死|离开)",
]

# 方法信号
METHOD_PATTERNS = [
    r"(跳|坠).*(楼|河|海|崖|桥)",
    r"(割|划|切).*(腕|脉|手|脖子|手腕|动脉|血管)",
    r"(上吊|吊死|自缢|悬梁)",
    r"(烧炭|一氧化碳|煤气)",
    r"(吞|吃|喝).*(药|安眠药|毒药|农药|老鼠药)",
    r"(卧轨|撞车|撞墙|跳车)",
    r"(自焚|引火|点火)",
    r"(跳楼|跳桥|跳河|跳海|跳崖)",
    r"(割腕|割脉|切腕)",
    r"(安眠药|过量服药|药物过[量]?)",
    r"(方法|方式|手段|办法).*(死|自杀|离开)",
    r"怎么.*死.*(不痛|不痛苦|舒服|快)",
    r"有[什啥]?么(办法|方法|方式).*(死|自杀|离开)",
]

# 计划信号
PLAN_PATTERNS = [
    r"(遗书|遗言|告别信|告别)",
    r"(计划|准备|安排).*(自杀|死|后事|离开)",
    r"(留了|写了).*(遗书|信|话)",
    r"(处理|安排).*(后事|东西|遗产)",
    r"(最后.*次|最后.*话|最后.*见)",
    r"(等我|当我|如果).*就.*(死|自杀|离开)",
    r"打算.*(方法|方式|计划).*死",
    r"想好.*(怎么|如何).*死",
    r"等.*(时候|那天|完).*(就|再).*(死|自杀|离开)",
]

# 紧迫性信号
URGENCY_PATTERNS = [
    r"(今晚|今天.*就[要去]|现在|马上|立刻|即刻|立即).*(死|自杀|跳|割|离开|行动|动手)",
    r"最后.*(一晚|一次|时刻|告别)",
    r"等不到.*了",
    r"坚持不到.*了",
    r"(没几天|没多少时间|时日无多)",
    r"(最近|近日|近期).*(实施|行动|动手|进行)",
    r"我走\s*了",
    r"再见.*(世界|大家|各位)",
    r"永别",
    r"最后.*(话|帖|发|留)",
    r"正在.*(死|自杀|割|跳)",
    r"现在就去死",
    r"这就去死",
]

# 准备行为信号
PREPARATION_PATTERNS = [
    r"买[了好]?(刀|绳|炭|药|工具|材料)",
    r"准备.*(工具|绳子|刀|药|炭)",
    r"(找|踩|观察|选).*地点",
    r"(查|搜索|百度|问).*(方法|方式|怎么).*死",
    r"(存[了好]?|攒[了好]?)药",
    r"(准备好|准备着|准备了)",
]

# 失控信号
LOSS_OF_CONTROL_PATTERNS = [
    r"控制不住.*(自己|想|自杀|伤害|念头)",
    r"害怕.*自己.*(会|做|伤害|控制不住)",
    r"担心.*自己.*(失控|控制不住|会出事)",
    r"快(崩|撑|坚)持不住",
    r"快要.*(疯|崩|爆|撑不住)",
    r"随时.*(会|可能).*(失控|崩|做傻事|伤害)",
    r"感觉.*(失控|控制不了|控制不住)",
    r"无法控制.*(自己|情绪|念头|冲动)",
    r"冲动.*想.*(死|自杀|伤害)",
]

# 严重痛苦信号
SEVERE_DISTRESS_PATTERNS = [
    r"太痛[苦了]",
    r"受不了[了]",
    r"熬不下[去了]",
    r"撑不下[去了]",
    r"崩[溃了]",
    r"绝望",
    r"没[有]?(希望|意义|意思|盼头)",
    r"痛苦.*(死|活|结束)",
    r"受够了",
    r"好难过",
    r"好痛[苦]",
    r"难受.*想死",
    r"烦躁.*想死",
    r"煎熬",
    r"生不如死",
]

# 讨论/引用语境（额外的，检测非风险讨论）
DISCUSSION_CONTEXT_PATTERNS = [
    r"这个研究", r"据报道", r"统计.*显示",
    r"有研究", r"论文.*说", r"文献.*指出",
    r"科普.*(内容|文章|视频)",
]

# 否定死亡/自杀
NEGATION_SUICIDE_PATTERNS = [
    r"没.*想.*自杀",
    r"不.*会.*自杀",
    r"开.*玩笑.*死",
    r"只.*是.*(吐槽|发泄|说).*(想死|自杀)",
]


# ============================================================
# 工具函数
# ============================================================

def detect_subject_context(text: str) -> str:
    """检测 subject_context（主体与语境）"""
    if not text:
        return "unclear"

    # 按优先级检测
    # 1. 否定风险
    if any(re.search(p, text) for p in NEGATION_SUICIDE_PATTERNS):
        return "negated"

    # 2. 假设/讨论/科普语境
    if any(re.search(p, text) for p in HYPOTHETICAL_PATTERNS):
        return "hypothetical"

    # 3. 第三人称
    if any(re.search(p, text) for p in THIRD_PARTY_PATTERNS):
        return "third_party"

    # 4. 过去经历
    if any(re.search(p, text) for p in PAST_PATTERNS):
        return "self_past"

    # 5. 默认本人当前
    return "self_current"


def extract_evidence_tags(text: str) -> list:
    """基于关键词规则提取 evidence_tags"""
    if not text:
        return []

    tags = set()

    if any(re.search(p, text) for p in PASSIVE_DEATH_WISH_PATTERNS):
        tags.add("passive_death_wish")
    if any(re.search(p, text) for p in ACTIVE_IDEATION_PATTERNS):
        tags.add("active_ideation")
    if any(re.search(p, text) for p in METHOD_PATTERNS):
        tags.add("method_signal")
    if any(re.search(p, text) for p in INTENT_PATTERNS):
        tags.add("intent_signal")
    if any(re.search(p, text) for p in PLAN_PATTERNS):
        tags.add("plan_signal")
    if any(re.search(p, text) for p in URGENCY_PATTERNS):
        tags.add("urgency_signal")
    if any(re.search(p, text) for p in PREPARATION_PATTERNS):
        tags.add("preparation_signal")
    if any(re.search(p, text) for p in LOSS_OF_CONTROL_PATTERNS):
        tags.add("loss_of_control")
    if any(re.search(p, text) for p in HELP_SEEKING_PATTERNS):
        tags.add("help_seeking")
    if any(re.search(p, text) for p in PROTECTIVE_FACTOR_PATTERNS):
        tags.add("protective_factor")
    if any(re.search(p, text) for p in SEVERE_DISTRESS_PATTERNS):
        tags.add("severe_distress")
    if any(re.search(p, text) for p in NEGATION_PATTERNS):
        tags.add("negation")
    if any(re.search(p, text) for p in THIRD_PARTY_PATTERNS):
        tags.add("third_party")
    if any(re.search(p, text) for p in DISCUSSION_CONTEXT_PATTERNS + HYPOTHETICAL_PATTERNS):
        tags.add("discussion_context")

    return sorted(tags)


def refine_level_from_tags(cssrs_level: int, subject_context: str, evidence_tags: list,
                           source=None, original_label=None) -> int:
    """
    根据 evidence_tags 和 subject_context 校正 level。

    v2 修复：对 SOS-1K 原始标签更保守，防止关键词规则过度升级。
    """
    context = set(evidence_tags)

    # =======================================
    # 优先级0: SOS-1K fine_6+ 专家标签优先
    # 禁止被 subject_context 降级（只有 negated 除外）
    # =======================================

    if source == "sos-1k-fine" and original_label and original_label.startswith("fine_"):
        fine_level = int(original_label.split("_")[1])
        if fine_level >= 6:
            if subject_context == "negated":
                return 0
            # 允许升级，禁止降级
            if "preparation_signal" in context:
                return max(cssrs_level, 3)
            if "urgency_signal" in context and "intent_signal" in context:
                return max(cssrs_level, 3)
            return cssrs_level

    # =======================================
    # 优先级1: subject_context 强制规则
    # =======================================

    # 否定表达 → level_0
    if subject_context == "negated":
        return 0

    # 假设/讨论/第三方 → level_0（除非有非常强的本人当前信号）
    if subject_context in ("third_party", "hypothetical"):
        if "urgency_signal" in context and "intent_signal" in context:
            return max(cssrs_level, 2)
        return 0

    # 过去经历 → 降级到不超过 level_1
    if subject_context == "self_past":
        if "urgency_signal" in context and "intent_signal" in context:
            return cssrs_level
        return min(cssrs_level, 1)

    # =======================================
    # 优先级2: SOS-1K 原始标签约束
    # =======================================

    if source == "sos-1k-fine" and original_label and original_label.startswith("fine_"):
        fine_level = int(original_label.split("_")[1])

        # fine_0/1/2: 无自杀意念 → 除非有极强信号否则保持 level_0
        if fine_level in (0, 1, 2):
            if "urgency_signal" in context and "intent_signal" in context:
                return max(cssrs_level, 1)
            if "active_ideation" in context and ("method_signal" in context or "plan_signal" in context):
                return max(cssrs_level, 1)
            return 0

        # fine_3: 强烈生存痛苦，无自杀愿望 → level_0
        if fine_level == 3:
            if "urgency_signal" in context and "intent_signal" in context:
                return max(cssrs_level, 1)
            return 0

        # fine_4/5: 自杀愿望已表达 → level_1
        if fine_level in (4, 5):
            if "preparation_signal" in context:
                return max(cssrs_level, 3)
            if "urgency_signal" in context and "intent_signal" in context:
                return max(cssrs_level, 2)
            if "plan_signal" in context and "method_signal" in context:
                return max(cssrs_level, 2)
            return cssrs_level

    # =======================================
    # 优先级3: 非 SOS-1K 数据（SOS-HL-1K, MentalGLM）
    # =======================================

    if "preparation_signal" in context:
        return 3
    if "urgency_signal" in context and "intent_signal" in context:
        return 3
    if "intent_signal" in context and ("method_signal" in context or "plan_signal" in context):
        return 3
    if "plan_signal" in context or "method_signal" in context:
        return 2
    if "active_ideation" in context:
        return 2
    if "passive_death_wish" in context:
        return 1
    if "severe_distress" in context:
        return 0

    return cssrs_level


def generate_level_from_tags(evidence_tags: list, subject_context: str) -> int:
    """从 evidence_tags 推导初始 level（用于 SOS-HL-1K 等无原始细粒度标签的数据）"""
    ctx = set(evidence_tags)

    if subject_context in ("negated",):
        return 0

    if "urgency_signal" in ctx or "preparation_signal" in ctx:
        return 3
    if "intent_signal" in ctx and ("method_signal" in ctx or "plan_signal" in ctx):
        return 3
    if "plan_signal" in ctx or "method_signal" in ctx:
        return 2
    if "active_ideation" in ctx:
        return 2
    if "passive_death_wish" in ctx:
        return 1
    if "severe_distress" in ctx:
        return 0

    return 0


def clean_text(text: str) -> str:
    """文本清洗：保留风险语义，去除格式噪声"""
    if not text:
        return ""
    text = text.strip()
    # URL 替换为占位符
    text = re.sub(r'https?://\S+', '[URL]', text)
    # 微博 @ 用户
    text = re.sub(r'@[\w一-鿿_-]+', '[USER]', text)
    # 话题标签
    text = re.sub(r'#[^#]+#', '[TOPIC]', text)
    # HTML 实体
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)
    # 控制字符清理
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # 重复空格压缩
    text = re.sub(r' +', ' ', text)
    # 重复空行压缩
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def generate_sample_id(source: str, idx: int) -> str:
    return f"{source}-{idx:06d}"


def generate_group_id(text: str) -> str:
    """基于文本内容生成 group_id（用于去重）"""
    normalized = re.sub(r'\s+', '', text.lower())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12]


def make_record(source, raw_id, text, raw_text, original_label,
                subject_context=None,
                cssrs_lite_level=None,
                evidence_tags=None,
                label_quality="silver",
                mapping_method="rule",
                review_reason="",
                group_id=None,
                sample_id=None,
                idx=0):
    """构造统一 schema 记录（自动检测 subject_context 和 evidence_tags）"""
    cleaned = clean_text(text)

    # 自动检测 subject_context
    if subject_context is None:
        subject_context = detect_subject_context(cleaned)

    # 自动提取 evidence_tags
    if evidence_tags is None:
        evidence_tags = extract_evidence_tags(cleaned)

    # 如果没有初始 level，从 evidence_tags 推导
    if cssrs_lite_level is None:
        cssrs_lite_level = generate_level_from_tags(evidence_tags, subject_context)
    else:
        cssrs_lite_level = refine_level_from_tags(
            cssrs_lite_level, subject_context, evidence_tags,
            source=source, original_label=original_label,
        )

    if group_id is None:
        group_id = generate_group_id(cleaned)
    if sample_id is None:
        sample_id = generate_sample_id(source, idx)

    return {
        "sample_id": sample_id,
        "source": source,
        "raw_id": str(raw_id),
        "text": cleaned,
        "raw_text": raw_text[:500] if raw_text else cleaned,
        "subject_context": subject_context,
        "cssrs_lite_level": cssrs_lite_level,
        "evidence_tags": sorted(evidence_tags) if evidence_tags else [],
        "original_label": str(original_label),
        "label_quality": label_quality,
        "mapping_method": mapping_method,
        "review_reason": review_reason,
        "group_id": group_id,
    }


# 修正：需要在 extract_evidence_tags 前定义 INTENT_PATTERNS
INTENT_PATTERNS = [
    r"(我|自己).*要.*(死|自杀|离开|结束)",
    r"真的.*(想|要).*(死|自杀|离开|结束)",
    r"决定.*(死|自杀|离开|结束)",
    r"下[定].*决心.*(死|自杀|离开)",
    r"(打算|计划|准备).*(自杀|死|离开|行动|实行|实施)",
    r"我.*(就|会|要|必须|一定).*(死|自杀|离开|结束)",
    r"(非死|一定要死|必须死)",
    r"我.*(不|再).*(忍|撑|熬|坚持).*(下去|住了)",
    r"我.*(放弃|不想).*(活|坚持|努力)",
    r"我还会.*(再|尝试).*(自杀|死)",
    r"现在.*就.*(结束|了结|死)",
    r"这次.*(一定|真的).*(死|自杀|离开)",
    r"越来越.*(想|强烈).*(死|自杀)",
    r"(选择|选).*自杀",
    r"迈出.*那一步",
]


# ============================================================
# 数据加载器
# ============================================================

def load_sos1k_finegrained() -> list:
    """加载 SOS-1K 细粒度数据（11 级标签）—— 项目主训练数据"""
    records = []
    base = RAW_DIR / "sos-1k" / "suicideDataProcessing" / "data" / "fine-grained"
    idx = 0

    train_files = sorted(base.glob("train_data*.tsv"))
    val_files = sorted(base.glob("val_data*.tsv"))
    test_file = base / "test_data.tsv"

    seen_texts = set()
    for fpath in train_files + val_files + ([test_file] if test_file.exists() else []):
        fold_name = fpath.stem
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("comment", "").strip()
                raw_label = row.get("myLabel", "").strip()
                if not text or not raw_label.isdigit():
                    continue
                dedup_key = text[:100]
                if dedup_key in seen_texts:
                    continue
                seen_texts.add(dedup_key)

                label_int = int(raw_label)

                # SOS-1K 0-10 → 项目 0-3 映射
                level = SOS1K_FINE_TO_LEVEL.get(label_int, 0)

                # 源标签 evidence_tags
                src_tags = list(SOS1K_FINE_TO_TAGS.get(label_int, []))
                if label_int >= 7 and "intent_signal" not in src_tags:
                    src_tags.append("intent_signal")

                # 不再直接传 tags，让 make_record 合并源标签和规则标签
                # make_record 会做规则检测，但源标签作为参考
                # 这里我们预先用源标签覆盖
                record = make_record(
                    source="sos-1k-fine",
                    raw_id=f"{fold_name}-{row.get('id', idx)}",
                    text=text,
                    raw_text=text,
                    original_label=f"fine_{label_int}",
                    subject_context=None,  # 自动检测
                    cssrs_lite_level=level,  # 基于源标签映射
                    evidence_tags=None,  # 自动提取
                    label_quality="silver",
                    mapping_method="rule",
                    review_reason="",
                    idx=idx,
                )

                # 合并源标签（SOS-1K 专家的标注比规则更可靠）
                current_tags = set(record["evidence_tags"])
                merged_tags = current_tags | set(src_tags)
                record["evidence_tags"] = sorted(merged_tags)

                records.append(record)
                idx += 1

    print(f"  [SOS-1K Fine] 加载 {len(records)} 条（去重后）")
    return records


def load_sos_hl_1k() -> list:
    """加载 SOS-HL-1K 数据（高低风险二分类）"""
    records = []
    base = RAW_DIR / "sos-hl-1k" / "data" / "suicide"
    idx = 200000

    seen_texts = set()
    for split_name in ["train", "val"]:
        jsonl_path = base / f"suicide_{split_name}.jsonl"
        if not jsonl_path.exists():
            continue
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = data.get("messages", [])
                if len(messages) < 3:
                    continue
                text = messages[1].get("content", "").strip()
                label = messages[2].get("content", "").strip()

                if not text:
                    continue
                dedup_key = text[:100]
                if dedup_key in seen_texts:
                    continue
                seen_texts.add(dedup_key)

                is_high = "高" in label

                # SOS-HL-1K 没有细粒度标签，让 make_record 从文本检测
                record = make_record(
                    source="sos-hl-1k",
                    raw_id=f"hl-{split_name}-{idx}",
                    text=text,
                    raw_text=text,
                    original_label=f"hl_{'high' if is_high else 'low'}",
                    subject_context=None,  # 自动检测
                    cssrs_lite_level=None,  # 自动从 evidence_tags 推导
                    evidence_tags=None,  # 自动提取
                    label_quality="review",
                    mapping_method="rule",
                    review_reason="",
                    idx=idx,
                )

                # 基于原始高/低风险标签确认 review 原因
                subject = record["subject_context"]
                level = record["cssrs_lite_level"]

                if is_high and level < 2:
                    record["review_reason"] = "hl_high_but_low_level_tag"
                    record["label_quality"] = "review"
                elif not is_high and level >= 2:
                    record["review_reason"] = "hl_low_but_high_level_tag"
                    record["label_quality"] = "review"
                elif is_high:
                    record["review_reason"] = "hl_high_need_verification"
                    record["label_quality"] = "review"
                elif level >= 1:
                    record["review_reason"] = "hl_low_but_passive_signal"
                    record["label_quality"] = "review"
                else:
                    # 低风险且 level_0 → silver
                    record["review_reason"] = ""
                    record["label_quality"] = "silver"

                records.append(record)
                idx += 1

    print(f"  [SOS-HL-1K] 加载 {len(records)} 条（去重后）")
    return records


def load_mentalglm() -> list:
    """加载 MentalGLM 原始数据（跳过 SOS-HL-1K 重叠）"""
    records = []
    base = RAW_DIR / "mentalglm" / "Raw data" / "Suiside"
    idx = 300000

    # 收集 SOS-HL-1K 已有文本
    hl_texts = set()
    hl_base = RAW_DIR / "sos-hl-1k" / "data" / "suicide"
    for split_name in ["train", "val"]:
        jsonl_path = hl_base / f"suicide_{split_name}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        messages = data.get("messages", [])
                        if len(messages) >= 2:
                            hl_texts.add(messages[1].get("content", "")[:100])
                    except json.JSONDecodeError:
                        pass

    seen_texts = set()
    for split_name in ["train_set", "val_set", "test_set"]:
        tsv_path = base / f"{split_name}.tsv"
        if not tsv_path.exists():
            continue
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                label_raw = parts[0].strip()
                text = parts[1].strip()
                if not text or label_raw not in ("0", "1"):
                    continue
                dedup_key = text[:100]
                if dedup_key in seen_texts:
                    continue
                seen_texts.add(dedup_key)
                if dedup_key in hl_texts:
                    continue

                is_high = label_raw == "1"

                record = make_record(
                    source="mentalglm",
                    raw_id=f"mg-{split_name}-{idx}",
                    text=text,
                    raw_text=text,
                    original_label=f"mg_{'high' if is_high else 'low'}",
                    subject_context=None,
                    cssrs_lite_level=None,
                    evidence_tags=None,
                    label_quality="review",
                    mapping_method="rule",
                    review_reason="mentalglm_only_supplement" if is_high else "",
                    idx=idx,
                )
                if not is_high and record["cssrs_lite_level"] == 0:
                    record["label_quality"] = "silver"
                    record["review_reason"] = ""

                records.append(record)
                idx += 1

    print(f"  [MentalGLM] 加载 {len(records)} 条（去重后，排除了 SOS-HL-1K 重叠）")
    return records


# ============================================================
# 标签映射表
# ============================================================

SOS1K_FINE_TO_LEVEL = {
    0: 0, 1: 0, 2: 0, 3: 0,
    4: 1, 5: 1,
    6: 2, 7: 2,
    8: 3, 9: 3, 10: 3,
}

SOS1K_FINE_TO_TAGS = {
    0: ["severe_distress"],
    1: ["severe_distress"],
    2: ["severe_distress"],
    3: ["severe_distress"],
    4: ["passive_death_wish"],
    5: ["passive_death_wish", "active_ideation"],
    6: ["active_ideation", "plan_signal"],
    7: ["active_ideation", "method_signal", "plan_signal"],
    8: ["active_ideation", "plan_signal", "urgency_signal"],
    9: ["active_ideation", "method_signal", "intent_signal", "urgency_signal"],
    10: ["intent_signal", "urgency_signal", "loss_of_control"],
}


# ============================================================
# 去重与分流
# ============================================================

def deduplicate(records: list) -> list:
    """基于 group_id 去重"""
    seen = set()
    unique = []
    for r in records:
        gid = r["group_id"]
        if gid not in seen:
            seen.add(gid)
            unique.append(r)
    print(f"  去重: {len(records)} → {len(unique)}（移除 {len(records)-len(unique)} 条）")
    return unique


def classify_silver_review(records: list) -> (list, list):
    """将记录分为 silver（可进入训练）和 review（需人工复核）"""
    silver = []
    review = []

    for r in records:
        if r["review_reason"]:
            r["label_quality"] = "review"
            review.append(r)
            continue

        source = r["source"]

        # 默认 silver
        is_silver = True

        # SOS-1K fine-grained 数据质量高，默认 silver
        if source == "sos-1k-fine":
            # 除非有明确问题
            if r.get("subject_context") == "unclear":
                is_silver = False
                r["review_reason"] = "unclear_subject"
            elif r.get("cssrs_lite_level") == 3:
                # level_3 全部保留 silver（高优先级安全信号）
                pass

        # SOS-HL-1K 和 MentalGLM 已在上游处理
        elif source in ("sos-hl-1k", "mentalglm"):
            is_silver = False
            if not r["review_reason"]:
                r["review_reason"] = f"{source}_supplement"

        if is_silver:
            r["label_quality"] = "silver"
            silver.append(r)
        else:
            r["label_quality"] = "review"
            review.append(r)

    print(f"  分流: silver={len(silver)}, review={len(review)}")
    return silver, review


# ============================================================
# 导出 & 统计
# ============================================================

def export_records(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写入 {len(records)} 条 → {path}")


def print_stats(records: list, title: str):
    """打印标签分布统计"""
    print(f"\n{'='*60}")
    print(f"  {title}（共 {len(records)} 条）")
    print(f"{'='*60}")

    src_counter = Counter(r["source"] for r in records)
    print(f"\n  来源分布:")
    for src, cnt in src_counter.most_common():
        print(f"    {src}: {cnt}")

    level_counter = Counter(r["cssrs_lite_level"] for r in records)
    print(f"\n  cssrs_lite_level:")
    for lv in sorted(level_counter.keys()):
        pct = level_counter[lv]/len(records)*100
        print(f"    level_{lv}: {level_counter[lv]} ({pct:.1f}%)")

    tags_counter = Counter()
    for r in records:
        for tag in r.get("evidence_tags", []):
            tags_counter[tag] += 1
    if tags_counter:
        total = len(records)
        print(f"\n  evidence_tags:")
        for tag, cnt in tags_counter.most_common():
            print(f"    {tag}: {cnt} ({cnt/total*100:.1f}%)")

    ctx_counter = Counter(r["subject_context"] for r in records)
    print(f"\n  subject_context:")
    for ctx, cnt in ctx_counter.most_common():
        print(f"    {ctx}: {cnt} ({cnt/len(records)*100:.1f}%)")

    quality_counter = Counter(r["label_quality"] for r in records)
    print(f"\n  质量等级:")
    for q, cnt in quality_counter.most_common():
        print(f"    {q}: {cnt}")
    print()


def generate_label_report(records: list, path: Path):
    """生成标签分布 Markdown 报告"""
    lines = []
    lines.append("# 标签分布报告\n")
    lines.append(f"生成日期: 2026-07-02\n")
    lines.append(f"总样本数: {len(records)}\n")

    # level 分布
    level_c = Counter(r["cssrs_lite_level"] for r in records)
    lines.append("## cssrs_lite_level 分布\n")
    lines.append("| level | 样本数 | 占比 |")
    lines.append("|-------|--------|------|")
    for lv in range(4):
        cnt = level_c.get(lv, 0)
        pct = cnt / len(records) * 100
        lines.append(f"| {lv} | {cnt} | {pct:.1f}% |")
    lines.append("")

    # subject_context 分布
    ctx_c = Counter(r["subject_context"] for r in records)
    lines.append("## subject_context 分布\n")
    lines.append("| 标签 | 样本数 | 占比 |")
    lines.append("|------|--------|------|")
    for ctx, cnt in ctx_c.most_common():
        lines.append(f"| {ctx} | {cnt} | {cnt/len(records)*100:.1f}% |")
    lines.append("")

    # evidence_tags 分布
    tags_c = Counter()
    for r in records:
        for tag in r.get("evidence_tags", []):
            tags_c[tag] += 1
    lines.append("## evidence_tags 分布\n")
    lines.append("| 标签 | 样本数 | 占比 |")
    lines.append("|------|--------|------|")
    for tag, cnt in tags_c.most_common():
        lines.append(f"| {tag} | {cnt} | {cnt/len(records)*100:.1f}% |")
    lines.append("")

    # 来源分布
    src_c = Counter(r["source"] for r in records)
    lines.append("## 数据来源分布\n")
    lines.append("| 来源 | 样本数 | 占比 |")
    lines.append("|------|--------|------|")
    for src, cnt in src_c.most_common():
        lines.append(f"| {src} | {cnt} | {cnt/len(records)*100:.1f}% |")
    lines.append("")

    # 质量分布
    qual_c = Counter(r["label_quality"] for r in records)
    lines.append("## 标签质量分布\n")
    lines.append("| 质量 | 样本数 | 占比 |")
    lines.append("|------|--------|------|")
    for q, cnt in qual_c.most_common():
        lines.append(f"| {q} | {cnt} | {cnt/len(records)*100:.1f}% |")
    lines.append("")

    # 交叉分析：level × subject
    lines.append("## 交叉分析：level × subject_context\n")
    lines.append("| level | self_current | self_past | third_party | negated | hypothetical | unclear |")
    lines.append("|-------|-------------|-----------|-------------|---------|--------------|---------|")
    for lv in range(4):
        lv_records = [r for r in records if r["cssrs_lite_level"] == lv]
        ctx_counts = Counter(r["subject_context"] for r in lv_records)
        row = [f"**{lv}** ({len(lv_records)})"]
        for ctx in ["self_current", "self_past", "third_party", "negated", "hypothetical", "unclear"]:
            row.append(str(ctx_counts.get(ctx, 0)))
        lines.append(" | ".join(row))
    lines.append("")

    # 边界样本分析
    lines.append("## 边界样本清单（review 队列）\n")
    review_records = [r for r in records if r["label_quality"] == "review"]
    if review_records:
        lines.append(f"共 {len(review_records)} 条需人工复核：\n")
        for r in review_records[:20]:
            text_short = r["text"][:80].replace("\n", " ")
            lines.append(f"- `{r['sample_id']}` level_{r['cssrs_lite_level']} | {r['subject_context']} | {r['review_reason']}")
            lines.append(f"  > {text_short}\n")
        if len(review_records) > 20:
            lines.append(f"  ... 还有 {len(review_records) - 20} 条\n")
    else:
        lines.append("无待复核样本。\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  报告已写入 → {path}")


# ============================================================
# 训练集切分
# ============================================================

def split_train_dev_test(
    silver_records: list,
    review_records: list,
    test_ratio: float = 0.1,
    dev_ratio: float = 0.15,
) -> dict:
    """Group-safe stratified split"""
    groups = defaultdict(list)
    for r in silver_records:
        groups[r["group_id"]].append(r)

    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    level_groups = defaultdict(list)
    for gid in group_ids:
        level = groups[gid][0]["cssrs_lite_level"]
        level_groups[level].append(gid)

    train_gids = set()
    dev_gids = set()
    test_gids = set()

    for level, gids in level_groups.items():
        random.shuffle(gids)
        n_total = len(gids)
        n_test = max(1, round(n_total * test_ratio))
        n_dev = max(1, round(n_total * dev_ratio))
        test_gids.update(gids[:n_test])
        dev_gids.update(gids[n_test:n_test + n_dev])
        train_gids.update(gids[n_test + n_dev:])

    train = [r for r in silver_records if r["group_id"] in train_gids]
    dev = [r for r in silver_records if r["group_id"] in dev_gids]
    test = [r for r in silver_records if r["group_id"] in test_gids]

    # dev 中加入边界 review 样本
    remaining_review = list(review_records)
    if review_records:
        boundary = [r for r in review_records
                    if "level" in r.get("review_reason", "")
                    or "passive" in r.get("review_reason", "")]
        random.shuffle(boundary)
        n_boundary = min(30, len(boundary))
        dev.extend(boundary[:n_boundary])
        remaining_review = [r for r in review_records if r not in boundary[:n_boundary]]

    print(f"\n  切分结果:")
    print(f"    train: {len(train)}")
    print(f"    dev:   {len(dev)}（含 {len(dev)-len(silver_records)-len([s for s in dev if s['label_quality']=='review'])} silver + 边界样本）")
    print(f"    test:  {len(test)}")
    print(f"    review_queue: {len(remaining_review)}")
    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        lc = Counter(r["cssrs_lite_level"] for r in data)
        print(f"    {name} level: {dict(sorted(lc.items()))}")

    return {
        "train": train,
        "dev": dev,
        "test": test,
        "review_queue": remaining_review,
    }


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="风险语义识别数据统一处理 v2")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("=" * 60)
    print("  风险语义识别 — 数据统一处理 v2")
    print("  规则检测: subject_context + evidence_tags")
    print("=" * 60)

    # ---- 1. 加载 ----
    print("\n[1/7] 加载数据源...")
    all_records = []
    all_records.extend(load_sos1k_finegrained())
    all_records.extend(load_sos_hl_1k())
    all_records.extend(load_mentalglm())
    print(f"  总计加载: {len(all_records)} 条")

    # ---- 2. 去重 ----
    print("\n[2/7] 去重...")
    all_records = deduplicate(all_records)

    # ---- 3. 归一化导出 ----
    print("\n[3/7] 导出 unified schema...")
    if not args.skip_export:
        export_records(all_records, output_dir / "normalized_all.jsonl")

    # ---- 4. 统计 ----
    print("\n[4/7] 全量统计...")
    print_stats(all_records, "全量数据")

    # ---- 5. silver/review ----
    print("\n[5/7] silver/review 分流...")
    silver, review = classify_silver_review(all_records)

    quality_map = {}
    for r in silver:
        quality_map[r["sample_id"]] = "silver"
    for r in review:
        quality_map[r["sample_id"]] = "review"
    for r in all_records:
        r["label_quality"] = quality_map.get(r["sample_id"], "review")

    if not args.skip_export:
        export_records(all_records, output_dir / "mapped_v3_all.jsonl")
        export_records(silver, output_dir / "silver_v3.jsonl")
        export_records(review, output_dir / "review_queue.jsonl")

    # ---- 6. 切分 ----
    print("\n[6/7] group-safe train/dev/test 切分...")
    split_result = split_train_dev_test(silver, review)

    if not args.skip_export:
        for name in ["train", "dev", "test"]:
            export_records(split_result[name], output_dir / f"{name}_v3.jsonl")
        export_records(split_result["review_queue"], output_dir / "review_queue_remaining.jsonl")

    # ---- 7. 报告 ----
    print("\n[7/7] 生成报告...")
    if not args.skip_export:
        generate_label_report(all_records, output_dir / "label_distribution_report.md")

    print(f"\n输出目录: {output_dir}")
    print(f"  normalized_all.jsonl         - 统一 schema 全量数据")
    print(f"  mapped_v3_all.jsonl           - 自动映射后全量数据")
    print(f"  silver_v3.jsonl               - 高置信自动标签")
    print(f"  review_queue.jsonl            - 待人工复核")
    print(f"  train/dev/test_v3.jsonl       - 训练集切分")
    print(f"  label_distribution_report.md  - 标签分布报告")

    print_stats(silver, "Silver")
    print_stats(review, "Review")


if __name__ == "__main__":
    main()

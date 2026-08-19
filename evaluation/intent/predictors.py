# -*- coding: utf-8 -*-
"""
Intent 预测器（Phase 1）。

冻结接口 IntentResult = {labels, confidence, is_open_set, source}。

包含：
  - IntentPredictor    基类
  - LegacyRulePredictor 当前系统 user_intent 关键词规则（baseline A）
  - LLMPredictor        LLM-only（baseline B，性能上界 + 数据质量检查器）
  - RuleClassifier      Phase 1 新的轻量规则分类器（Hybrid 的 rule 兜底，可选）
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


# 当前系统 emotion_analyzer._detect_user_intent 的 4 类规则（原样复刻）
USER_INTENT_RULES = {
    "seeking_help": ["怎么办", "怎么做", "帮帮我", "我该怎么办", "有什么办法"],
    "planning": ["计划", "安排", "拆分", "步骤", "怎么开始"],
    "seeking_relief": ["缓解", "放松", "平静", "睡着", "减轻"],
    "sharing": [],
}

# 用户意图 -> Intent Taxonomy（v1.1 的 10 类）映射（lossy，Legacy baseline 用）
USER_INTENT_TO_TAXONOMY = {
    "seeking_help": ["explicit_help_request"],
    "planning": ["explicit_help_request"],
    "seeking_relief": ["explicit_help_request", "resource_request"],
    "sharing": [],  # 由情绪决定 emotional_expression / casual_chat
}

# 简易情绪关键词（用于 sharing 分派）
_NEGATIVE_EMOTION_WORDS = ["焦虑", "紧张", "累", "难过", "伤心", "压力", "烦", "抑郁",
                           "失眠", "孤独", "害怕", "低落", "想哭", "崩溃", "绝望", "撑不住"]
_POSITIVE_EMOTION_WORDS = ["开心", "高兴", "快乐", "不错", "好", "棒"]


def legacy_user_intent(text: str) -> str:
    for intent, keywords in USER_INTENT_RULES.items():
        if keywords and any(k in text for k in keywords):
            return intent
    return "sharing"


class IntentPredictor(ABC):
    """Intent 预测器基类。"""

    name: str = "base"

    @abstractmethod
    def predict(self, text: str, context: Optional[List[Dict]] = None) -> Dict:
        """返回 IntentResult 结构。"""

    def predict_batch(self, texts, contexts=None):
        return [self.predict(t, (contexts[i] if contexts else None)) for i, t in enumerate(texts)]


class LegacyRulePredictor(IntentPredictor):
    """Legacy Rule Baseline（当前系统 user_intent 关键词规则）。"""

    name = "legacy_rule"

    def predict(self, text: str, context: Optional[List[Dict]] = None) -> Dict:
        ui = legacy_user_intent(text)
        labels = list(USER_INTENT_TO_TAXONOMY.get(ui, []))
        if ui == "sharing":
            if any(w in text for w in _NEGATIVE_EMOTION_WORDS):
                labels = ["emotional_expression"]
            else:
                labels = ["casual_chat"]
        return {
            "labels": labels,
            "confidence": 0.9 if ui != "sharing" else 0.5,
            "is_open_set": False,
            "source": "rule",
            "label_scores": {lab: 1.0 for lab in labels},
        }


class RuleClassifierPredictor(IntentPredictor):
    """
    Phase 1 轻量规则分类器（Hybrid 的 rule 成分，确定性）。
    覆盖高频/稳定模式：资源请求、信息请求、meta、高危、记忆引用、反馈。
    作为 Small Model 之前的快速起点，也是 Small Model 的对比/兜底。
    """

    name = "rule_classifier"

    # 每类关键词（可扩充）
    RULES = {
        "resource_request": ["推荐", "推荐点", "有没有", "有没有推荐的", "给", "音频", "文章", "练习", "课程", "书", "App", "工具", "内容"],
        "information_request": ["什么是", "是什么", "为什么", "有什么区别", "怎么区分", "原理", "概念", "含义", "正常吗", "为什么"],
        "meta_question": ["你是", "你是不", "你会", "你能", "你的", "你和", "你属于", "你是不是", "记得", "懂"],
        "high_risk_expression": ["自杀", "想死", "不想活", "结束生命", "离开世界", "跳楼", "活不下去", "伤害自己", "不想醒", "解脱", "一了百了", "撑不住", "想不开"],
        "memory_reference": ["上次", "之前", "以前", "还记得", "你记得", "你说过", "我说过", "我跟你说过"],
        "feedback": ["试了", "没用", "用了", "效果", "推荐", "你上次", "我试", "听了", "做了", "感觉不错", "谢谢"],
        "follow_up": ["刚才", "你刚才", "你说的方法", "那个方法", "那这个", "然后呢", "接下来"],
        "explicit_help_request": ["怎么办", "怎么做", "帮帮我", "想办法", "建议", "教我", "怎么才能", "该怎么做", "帮我想", "有什么办法", "什么办法"],
        "casual_chat": ["天气", "吃饭", "电影", "随便聊聊", "陪你", "下班", "周末", "散步"],
        "emotional_expression": ["焦虑", "紧张", "累", "难过", "伤心", "压力", "烦", "抑郁", "孤独", "害怕", "低落", "想哭", "崩溃"],
    }

    def __init__(self, high_risk_recall_first: bool = True):
        self.high_risk_recall_first = high_risk_recall_first

    def predict(self, text: str, context: Optional[List[Dict]] = None) -> Dict:
        labels: List[str] = []
        # 高危优先（Recall 优先）
        if self.high_risk_recall_first and self._match(text, "high_risk_expression"):
            labels.append("high_risk_expression")
        for lab in self.RULES:
            if lab == "high_risk_expression" and labels:
                continue
            if lab == "casual_chat" and labels:
                continue  # 有具体任务意图时不额外标 casual
            if self._match(text, lab):
                labels.append(lab)
        labels = list(dict.fromkeys(labels))
        return {
            "labels": labels,
            "confidence": 0.8,
            "is_open_set": False,
            "source": "rule",
            "label_scores": {lab: 1.0 for lab in labels},
        }

    def _match(self, text: str, lab: str) -> bool:
        return any(k in text for k in self.RULES.get(lab, []))


class LLMPredictor(IntentPredictor):
    """
    LLM-only Baseline（DeepSeek）。
    作用：性能上界参考 + 数据质量检查器 + fallback baseline。
    只允许输出已有 10 类标签。
    """

    name = "llm"

    def __init__(self, model: Optional[str] = None):
        from dotenv import load_dotenv
        load_dotenv(_BACKEND_DIR / ".env", override=True)  # 从 backend/.env 加载真实密钥（覆盖测试 key）
        from openai import AsyncOpenAI
        from config import config
        self.client = AsyncOpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.API_BASE_URL)
        self.model = model or config.CHAT_MODEL
        self._sys = (
            "你是意图分类器。从给定列表中为用户的输入标注 1-N 个意图标签（multi-label）。"
            f"标签列表：{', '.join(sorted([
                'casual_chat','emotional_expression','explicit_help_request','information_request',
                'resource_request','feedback','follow_up','memory_reference','high_risk_expression','meta_question']))}。"
            "只输出 JSON，格式：{\"labels\": [...], \"confidence\": 0.0-1.0}。"
            "禁止输出列表外的标签。"
        )

    def predict(self, text: str, context: Optional[List[Dict]] = None) -> Dict:
        import asyncio
        try:
            return asyncio.run(self._predict_async(text, context))
        except Exception as e:  # noqa: BLE001
            logger.error(f"LLM intent 预测失败: {e}")
            return {"labels": [], "confidence": 0.0, "is_open_set": True, "source": "llm",
                    "label_scores": {}, "error": str(e)}

    async def _predict_async(self, text: str, context: Optional[List[Dict]] = None) -> Dict:
        messages = [{"role": "system", "content": self._sys}]
        if context:
            messages.append({"role": "user", "content": f"对话历史：{json.dumps(context, ensure_ascii=False)}"})
        messages.append({"role": "user", "content": f"当前输入：{text}"})
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 容错：尝试提取 JSON
            m = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
        labels = data.get("labels", [])
        # 只保留合法标签
        from data.intent.intent_schema import INTENT_LABELS
        labels = [l for l in labels if l in INTENT_LABELS]
        confidence = float(data.get("confidence", 0.7))
        return {
            "labels": labels,
            "confidence": confidence,
            "is_open_set": len(labels) == 0,
            "source": "llm",
            "label_scores": {l: confidence for l in labels},
        }

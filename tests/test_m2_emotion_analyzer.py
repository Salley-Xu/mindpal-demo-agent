"""
情绪分析器综合测试 — BERT 预测、缓存行为、LLM 回退、情绪-风险交互

覆盖：
  - BertEmotionPredictor 推理（mock 模型）       [m2_unit]
  - EmotionAnalyzer 缓存（命中/过期/LRU）         [m2_unit]
  - LLM 回退路径 & 置信度估算                     [m2_unit]
  - build_emotion_state_payload 完整构造          [m2_unit]
  - 情绪-风险交互（编排器级）                      [m2_integration]
"""

import asyncio
import os
import sys
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CHAT_MODEL", "deepseek-chat")
os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com/v1")

from bert_emotion_predictor import BertEmotionPredictor, EMOTION_TO_CHINESE
from emotion_analyzer import emotion_analyzer, _get_bert_predictor, EmotionAnalyzer
from agent_orchestrator import agent_orchestrator
from config import config


# =========================================================================
# 第一部分：BERT 预测器测试（mock PyTorch 模型，不加载真实权重）
# =========================================================================

class TestBertEmotionPredictor:
    """BertEmotionPredictor 推理测试 — mock 模型 forward 输出"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """每个测试前创建 mock predictor"""
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

        self.mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        import torch
        # logits: [neutral=-0.5, positive=-1.0, anxiety=2.5, sadness=0.3, anger=-0.8]
        mock_logits = torch.tensor([[-0.5, -1.0, 2.5, 0.3, -0.8]])
        self.mock_model.return_value = mock_logits

        with patch("bert_emotion_predictor.AutoTokenizer.from_pretrained", return_value=self.mock_tokenizer), \
             patch("bert_emotion_predictor.EmotionBERT", return_value=self.mock_model), \
             patch("bert_emotion_predictor.torch.load"):
            self.predictor = BertEmotionPredictor(
                model_path="dummy/path",
                device="cpu",
                confidence_threshold=0.6,
            )

    def test_predict_anxiety(self):
        """BERT 应正确预测 anxiety 类（logits 中索引 2 最高）"""
        label, conf = self.predictor.predict("最近压力很大，晚上睡不着")
        assert label == "anxiety"
        assert 0.8 <= conf <= 1.0

    def test_predict_empty_text(self):
        """空文本应返回 neutral + 0.0 置信度"""
        label, conf = self.predictor.predict("")
        assert label == "neutral"
        assert conf == 0.0

    def test_predict_whitespace_text(self):
        """纯空白文本应返回 neutral + 0.0 置信度"""
        label, conf = self.predictor.predict("   ")
        assert label == "neutral"
        assert conf == 0.0

    def test_predict_chinese_mapping(self):
        """predict_chinese 应返回正确的中文标签"""
        label_cn, conf = self.predictor.predict_chinese("感觉一切都没有意义")
        assert label_cn == "焦虑"
        assert 0.8 <= conf <= 1.0

    def test_predict_with_probs_returns_distribution(self):
        """predict_with_probs 应返回完整概率分布"""
        result = self.predictor.predict_with_probs("考试考砸了，很沮丧")
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["label"] == "anxiety"
        assert set(result["probabilities"].keys()) == {"neutral", "positive", "anxiety", "sadness", "anger"}
        total_prob = sum(result["probabilities"].values())
        assert abs(total_prob - 1.0) < 0.01

    def test_temperature_scaling_applied(self):
        """Temperature Scaling 参数应影响 softmax 输出"""
        import torch
        orig_temp = self.predictor.temperature

        self.predictor.temperature = 5.0
        result_hot = self.predictor.predict_with_probs("测试文本")

        self.predictor.temperature = 0.5
        result_cold = self.predictor.predict_with_probs("测试文本")

        max_hot = max(result_hot["probabilities"].values())
        max_cold = max(result_cold["probabilities"].values())
        assert max_hot <= max_cold

        self.predictor.temperature = orig_temp


# =========================================================================
# 第二部分：EmotionAnalyzer 缓存行为测试（禁用 BERT，走 LLM 回退）
# =========================================================================

class TestEmotionAnalyzerCache:
    """缓存命中/过期/LRU 行为测试（走 LLM 回退路径）"""

    pytestmark = pytest.mark.asyncio

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """创建隔离分析器，禁用 BERT 以确保走 LLM 路径"""
        self.analyzer = EmotionAnalyzer(cache_size=3, cache_ttl=2)
        self.analyzer._call_llm = AsyncMock(return_value="焦虑")
        # 禁用 BERT 以确保走 LLM 路径
        self._orig_bert_flag = config.USE_BERT_EMOTION
        config.USE_BERT_EMOTION = False
        yield
        config.USE_BERT_EMOTION = self._orig_bert_flag

    async def test_cache_hit_returns_same_result(self):
        """相同文本应命中缓存，返回相同结果"""
        r1 = await self.analyzer.analyze_with_context_async("最近压力很大")
        r2 = await self.analyzer.analyze_with_context_async("最近压力很大")
        assert r1 == r2
        assert self.analyzer._call_llm.call_count == 1

    async def test_cache_miss_different_text(self):
        """不同文本应触发新分析"""
        await self.analyzer.analyze_with_context_async("文本A")
        await self.analyzer.analyze_with_context_async("文本B")
        assert self.analyzer._call_llm.call_count == 2

    async def test_cache_expiry(self):
        """超过 TTL 的缓存条目应过期"""
        await self.analyzer.analyze_with_context_async("测试文本")
        time.sleep(2.1)
        await self.analyzer.analyze_with_context_async("测试文本")
        assert self.analyzer._call_llm.call_count >= 2

    async def test_lru_eviction(self):
        """超出 cache_size 应淘汰最久未使用的条目"""
        self.analyzer._call_llm = AsyncMock(side_effect=["焦虑", "抑郁", "愤怒", "压力"])
        await self.analyzer.analyze_with_context_async("文本1")
        await self.analyzer.analyze_with_context_async("文本2")
        await self.analyzer.analyze_with_context_async("文本3")  # 缓存满
        await self.analyzer.analyze_with_context_async("文本4")  # 淘汰 1

        # 文本1 已被淘汰
        self.analyzer._call_llm.reset_mock()
        self.analyzer._call_llm.return_value = "焦虑的回访"
        await self.analyzer.analyze_with_context_async("文本1")
        assert self.analyzer._call_llm.call_count == 1

        # 文本3 仍在缓存中
        self.analyzer._call_llm.reset_mock()
        await self.analyzer.analyze_with_context_async("文本3")
        assert self.analyzer._call_llm.call_count == 0

    async def test_cache_key_with_context(self):
        """相同文本但不同上下文应生成不同缓存键"""
        self.analyzer._call_llm = AsyncMock(return_value="焦虑")
        ctx_a = {"turn_count": 3, "conversation_stage": "deepening", "key_concerns": ["academic"]}
        ctx_b = {"turn_count": 10, "conversation_stage": "intervention", "key_concerns": ["relationship"]}
        # 每次 analyze_with_context_async 会调用 2 次 LLM（base + context），
        # 但只要缓存键不同则总调用数 = 2 次/次 × 2 次调用 = 4
        await self.analyzer.analyze_with_context_async("相同文本", ctx_a)
        await self.analyzer.analyze_with_context_async("相同文本", ctx_b)
        assert self.analyzer._call_llm.call_count == 4  # 2 次/analyze × 2 个上下文

    async def test_cache_clear_on_ttl_boundary(self):
        """TTL 边界上的缓存条目应被清理"""
        self.analyzer._call_llm = AsyncMock(return_value="中性")
        await self.analyzer.analyze_with_context_async("旧文本")
        cache_key = next(iter(self.analyzer._emotion_cache))
        self.analyzer._emotion_cache[cache_key]["timestamp"] = time.time() - 10
        result = self.analyzer._check_cache(cache_key)
        assert result is None
        assert cache_key not in self.analyzer._emotion_cache


# =========================================================================
# 第三部分：LLM 回退 & 置信度估算测试
# =========================================================================

class TestLlmFallbackAndConfidence:
    """LLM 回退路径和 _estimate_llm_confidence 测试"""

    pytestmark = pytest.mark.asyncio

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """创建隔离分析器，禁用 BERT"""
        self.analyzer = EmotionAnalyzer(cache_size=100, cache_ttl=3600)
        self.analyzer._call_llm = AsyncMock(return_value="焦虑")
        self._orig_bert_flag = config.USE_BERT_EMOTION
        config.USE_BERT_EMOTION = False
        yield
        config.USE_BERT_EMOTION = self._orig_bert_flag

    async def test_llm_fallback_returns_tuple(self):
        """LLM 回退路径应返回 (str, str, float) 元组"""
        result = await self.analyzer.analyze_with_context_async("最近很烦")
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)
        assert isinstance(result[2], float)

    async def test_llm_confidence_baseline(self):
        """LLM 回退的置信度应在合理范围内"""
        _, _, conf = await self.analyzer.analyze_with_context_async("今天心情不错")
        assert 0.35 <= conf <= 0.90


class TestConfidenceEstimation:
    """置信度估算函数的单元测试（纯同步，无需 mock）"""

    def setup_method(self):
        self.analyzer = EmotionAnalyzer()

    def test_confidence_short_text_lower(self):
        """极短文本应降低置信度"""
        conf = self.analyzer._estimate_llm_confidence("好", "中性")
        assert conf <= 0.55  # 0.65 - 0.15(短) - 0.08(中性) = 0.42

    def test_confidence_long_text_higher(self):
        """较长文本应略微提高置信度"""
        conf = self.analyzer._estimate_llm_confidence("我" * 101, "焦虑")
        assert conf >= 0.70

    def test_confidence_neutral_penalty(self):
        """中性情绪应降低置信度"""
        conf_neutral = self.analyzer._estimate_llm_confidence("还好吧", "中性")
        conf_anxiety = self.analyzer._estimate_llm_confidence("很焦虑", "焦虑")
        assert conf_neutral < conf_anxiety

    def test_confidence_unknown_emotion_penalty(self):
        """未知情绪标签应进一步降低置信度"""
        conf_unknown = self.analyzer._estimate_llm_confidence("测试", "未知情绪!!")
        assert conf_unknown <= 0.60

    def test_confidence_clamped_range(self):
        """置信度应始终在 [0.35, 0.90] 范围内"""
        conf = self.analyzer._estimate_llm_confidence("a", "乱七八糟")
        assert 0.35 <= conf <= 0.90
        conf2 = self.analyzer._estimate_llm_confidence("a" * 200, "抑郁")
        assert 0.35 <= conf2 <= 0.90


# =========================================================================
# 第四部分：build_emotion_state_payload 测试（纯同步）
# =========================================================================

class TestEmotionStatePayload:
    """情绪状态载荷构造测试"""

    def test_payload_contains_all_required_fields(self):
        """载荷应包含所有 EmotionState 必需字段"""
        payload = emotion_analyzer.build_emotion_state_payload(
            text="考试压力大，睡不着",
            current_emotion="焦虑",
            context_emotion="深层焦虑",
            confidence=0.85,
            conversation_summary={
                "emotion_trend": "escalating",
                "recent_emotions": ["压力", "焦虑", "焦虑"],
                "key_concerns": ["academic"],
            },
        )
        for key in ("current_emotion", "emotion_type", "context_emotion",
                     "emotion_intensity", "stress_source", "user_intent",
                     "negative_trend", "confidence", "emotion_trend"):
            assert key in payload, f"缺少字段: {key}"

    def test_emotion_type_mapping_anxiety(self):
        """焦虑 应映射为 anxiety"""
        payload = emotion_analyzer.build_emotion_state_payload("焦虑", "焦虑", "焦虑", 0.8)
        assert payload["emotion_type"] == "anxiety"

    def test_emotion_type_mapping_positive(self):
        """快乐/平静/放松 应映射为 positive"""
        for emo in ["快乐", "平静", "放松"]:
            payload = emotion_analyzer.build_emotion_state_payload("test", emo, emo, 0.8)
            assert payload["emotion_type"] == "positive", f"{emo} → positive failed"

    def test_stress_source_detection_academic(self):
        """包含学业关键词应触发 学业/求职压力"""
        payload = emotion_analyzer.build_emotion_state_payload(
            "这次考试没考好，论文也没写完", "焦虑", "焦虑", 0.8
        )
        assert payload["stress_source"] == "学业/求职压力"

    def test_stress_source_detection_relationship(self):
        """包含关系关键词应触发 人际关系压力"""
        payload = emotion_analyzer.build_emotion_state_payload(
            "和室友吵架了，不想沟通", "愤怒", "愤怒", 0.8
        )
        assert payload["stress_source"] == "人际关系压力"

    def test_stress_source_none(self):
        """无匹配关键词时 stress_source 应为 None"""
        payload = emotion_analyzer.build_emotion_state_payload("今天天气不错", "中性", "中性", 0.5)
        assert payload["stress_source"] is None

    def test_user_intent_seeking_help(self):
        """包含求助关键词应识别为 seeking_help"""
        payload = emotion_analyzer.build_emotion_state_payload("我该怎么办？帮帮我", "焦虑", "焦虑", 0.8)
        assert payload["user_intent"] == "seeking_help"

    def test_user_intent_default_sharing(self):
        """无特别意图应默认为 sharing"""
        payload = emotion_analyzer.build_emotion_state_payload("今天去了公园", "快乐", "快乐", 0.8)
        assert payload["user_intent"] == "sharing"

    def test_emotion_intensity_negative_boost(self):
        """负面情绪应提高强度基准"""
        i_n = emotion_analyzer._estimate_emotion_intensity("还行", "中性", None)
        i_a = emotion_analyzer._estimate_emotion_intensity("很焦虑", "焦虑", None)
        assert i_a > i_n

    def test_emotion_intensity_strong_markers(self):
        """强烈情绪词应增加强度"""
        base = emotion_analyzer._estimate_emotion_intensity("有点焦虑", "焦虑", None)
        strong = emotion_analyzer._estimate_emotion_intensity("非常焦虑，完全崩溃", "焦虑", None)
        assert strong > base

    def test_negative_trend_escalating(self):
        """escalating 趋势 + 负面情绪 → negative_trend=True"""
        payload = emotion_analyzer.build_emotion_state_payload(
            "越来越糟了", "焦虑", "焦虑", 0.8,
            conversation_summary={"emotion_trend": "escalating", "recent_emotions": ["压力", "焦虑", "焦虑"]},
        )
        assert payload["negative_trend"] is True

    def test_negative_trend_improving(self):
        """improving 趋势 → negative_trend=False"""
        payload = emotion_analyzer.build_emotion_state_payload(
            "感觉好多了", "快乐", "快乐", 0.8,
            conversation_summary={"emotion_trend": "improving"},
        )
        assert payload["negative_trend"] is False


# =========================================================================
# 第五部分：情绪-风险交互测试（编排器级）
# =========================================================================

class TestEmotionRiskInteraction:
    """情绪分析与风险评估的交互正确性"""

    pytestmark = pytest.mark.asyncio

    @pytest.fixture(autouse=True)
    def disable_external_calls(self):
        """关键 mock：禁用所有真实外部调用"""
        # 禁用 BERT 情绪模型
        self._orig_bert_flag = config.USE_BERT_EMOTION
        config.USE_BERT_EMOTION = False
        # mock emotion analyzer 的 LLM 调用
        self._emotion_llm = patch.object(emotion_analyzer, "_call_llm", new=AsyncMock(return_value="焦虑"))
        self._emotion_llm.start()
        # mock orchestrator 的 LLM 调用（agent 循环中使用 self.client.chat.completions.create）
        self._orch_completion = patch.object(
            agent_orchestrator.client.chat.completions, "create",
            new=AsyncMock(return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content="这是测试回复",
                    tool_calls=None,
                ))]
            )),
        )
        self._orch_completion.start()
        # mock 风险评估器的 _get_predictor（避免加载 ~400MB BERT 模型）
        from risk_evaluator import risk_evaluator as risk_eval_instance
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("level_0", 0.12)
        mock_predictor.predict_with_probs.return_value = {"label": "level_0", "confidence": 0.12}
        self._risk_patch = patch.object(risk_eval_instance, "_get_predictor", return_value=mock_predictor)
        self._risk_patch.start()
        yield
        config.USE_BERT_EMOTION = self._orig_bert_flag
        self._emotion_llm.stop()
        self._orch_completion.stop()
        self._risk_patch.stop()

    async def test_emotion_analysis_passes_to_risk_evaluator(self):
        """验证情绪状态被正确传递到 risk_evaluator"""
        from risk_evaluator import risk_evaluator
        from models import AgentRunRequest

        captured_kwargs = {}
        original_evaluate = risk_evaluator.evaluate

        def capturing_evaluate(text, **kwargs):
            captured_kwargs.update(kwargs)
            return original_evaluate(text, **kwargs)

        risk_evaluator.evaluate = capturing_evaluate
        try:
            user_id = f"test-user-{uuid.uuid4().hex[:8]}"
            session_id = f"test-session-{uuid.uuid4().hex[:8]}"

            request = AgentRunRequest(
                text="最近压力很大，完全不知道怎么办，感觉要崩溃了",
                user_id=user_id,
                session_id=session_id,
            )
            result = await agent_orchestrator.run_agent(request)

            assert "emotion_state" in captured_kwargs
            es = captured_kwargs["emotion_state"]
            assert es.get("current_emotion") is not None
            assert es.get("emotion_intensity", 0) > 0
            assert result.chat.emotion_state is not None
        finally:
            risk_evaluator.evaluate = original_evaluate

    async def test_emotion_analysis_does_not_crash_on_empty_text(self):
        """空文本不应导致情绪分析崩溃"""
        from models import AgentRunRequest
        user_id = f"test-user-{uuid.uuid4().hex[:8]}"
        session_id = f"test-session-{uuid.uuid4().hex[:8]}"

        request = AgentRunRequest(
            text="",
            user_id=user_id,
            session_id=session_id,
        )
        result = await agent_orchestrator.run_agent(request)
        assert result is not None


# =========================================================================
# 第六部分：EMOTION_TO_CHINESE / 标签体系测试（纯同步）
# =========================================================================

class TestEmotionLabelSystem:
    """情绪标签映射体系测试"""

    def test_all_bert_labels_have_chinese_mapping(self):
        """每个 BERT 英文标签都应有中文映射"""
        from bert_emotion_predictor import IDX_TO_EMOTION
        for label in IDX_TO_EMOTION.values():
            assert label in EMOTION_TO_CHINESE, f"{label} 缺少中文映射"
            assert isinstance(EMOTION_TO_CHINESE[label], str) and len(EMOTION_TO_CHINESE[label]) > 0

    def test_merged_labels_map_to_existing_chinese(self):
        """被合并的标签（stress/confusion/helplessness）应映射到已有中文"""
        for merged in ["stress", "confusion", "helplessness", "压力", "困惑", "无助"]:
            assert merged in EMOTION_TO_CHINESE, f"{merged} 缺少映射"
            cn = EMOTION_TO_CHINESE[merged]
            assert cn in ["焦虑", "中性", "快乐", "抑郁", "愤怒"], \
                f"{merged} 映射到 {cn}，不在规范中文标签中"

    def test_chinese_roundtrip(self):
        """中文标签映射回 emotion_type 应有合理结果"""
        for cn_label in ["焦虑", "抑郁", "愤怒", "压力", "学业压力", "困惑", "不确定",
                         "无助", "孤独", "中性", "快乐", "平静", "放松"]:
            emotion_type = emotion_analyzer._normalize_emotion_type(cn_label)
            assert emotion_type in ["anxiety", "sadness", "anger", "stress", "confusion",
                                    "helplessness", "neutral", "positive"], \
                f"{cn_label} → {emotion_type} 不在规范类型中"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

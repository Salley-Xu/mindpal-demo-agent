import asyncio
import os
import sys

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

from agent_orchestrator import AgentOrchestrator  # noqa: E402


@pytest.mark.asyncio
async def test_build_initial_messages_includes_memory_context():
    orchestrator = AgentOrchestrator()
    messages = await orchestrator._build_initial_messages(
        text="我明天面试，越想越乱。",
        history=[
            {"user_input": "我最近一直在准备面试。", "ai_response": "听起来这段时间压力持续存在。"},
            {"user_input": "我怕讲不清项目。", "ai_response": "我们可以把项目表达先拆成几个固定部分。"},
        ],
        summary={
            "conversation_stage": "deepening",
            "primary_emotion": "焦虑",
            "emotion_trend": "escalating",
            "key_concerns": ["future", "self"],
            "current_topic": "学业与求职",
            "recent_intents": ["seeking_help", "planning"],
            "stress_sources": ["求职", "面试"],
            "accepted_recommendations": ["tool_001"],
            "rejected_recommendations": ["audio_003"],
            "compressed_context": "历史对话摘要: 用户多次提到面试和项目表达压力。",
        },
        urgent_issue={
            "level": "medium",
            "message": "需要持续关注压力升级",
            "triggers": ["sleep_problem", "interview_anxiety"],
            "risk_score": 0.68,
        },
        emotion_state={
            "current_emotion": "焦虑",
            "emotion_type": "anxiety",
            "context_emotion": "自我怀疑",
            "emotion_intensity": 0.82,
            "user_intent": "seeking_help",
            "stress_source": "求职",
            "negative_trend": True,
        },
        user_profile={
            "risk_level": "medium",
            "preferred_support_style": "direct_actionable",
            "avoid_style": ["empty_comfort"],
            "main_stress_sources": ["求职", "面试"],
            "recommendation_feedback": {"tool_001": "helpful", "audio_003": "avoid"},
        },
        recommendation_decision={
            "should_recommend": True,
            "recommend_type": "soft",
            "reason_codes": ["high_emotion_intensity", "negative_trend"],
        },
    )

    system_prompt = messages[0]["content"]
    assert "【会话摘要】" in system_prompt
    assert "【压缩上下文】" in system_prompt
    assert "【用户长期画像】" in system_prompt
    assert "【当前情绪状态】" in system_prompt
    assert "【当前风险状态】" in system_prompt
    assert "历史对话摘要: 用户多次提到面试和项目表达压力。" in system_prompt
    assert "偏好支持风格: direct_actionable" in system_prompt
    assert "已接受推荐: tool_001" in system_prompt
    assert "已拒绝推荐: audio_003" in system_prompt
    assert "当前情绪: 焦虑" in system_prompt
    assert "风险等级: medium" in system_prompt
    assert "当前仅允许在自然语言回复中给出软建议" in system_prompt

    # 有压缩摘要时，应缩小原始历史窗口，但仍保留当前输入
    assert messages[-1] == {"role": "user", "content": "我明天面试，越想越乱。"}
    assert len(messages) == 6


def main():
    asyncio.run(test_build_initial_messages_includes_memory_context())
    print("PASS: prompt includes summary, memory, and state context")


if __name__ == "__main__":
    main()

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

from urgent_detector import urgent_detector  # noqa: E402


def test_detect_returns_normalized_four_level_semantics():
    high = urgent_detector.detect("我真的不想活了，今晚想结束生命。", "抑郁")
    medium = urgent_detector.detect("最近太痛苦了，感觉自己快撑不住。", "焦虑")
    low = urgent_detector.detect("这周有点累，但我想慢慢调整。", "压力")

    assert high["level"] == "level_3"
    assert high["legacy_level"] == "high"
    assert medium["level"] in {"level_1", "level_2"}
    assert medium["legacy_level"] == "medium"
    assert low["level"] == "level_0"
    assert low["legacy_level"] == "low"
    assert high["risk_score"] >= medium["risk_score"] >= low["risk_score"]


@pytest.mark.asyncio
async def test_generate_crisis_response_keeps_legacy_level_compatibility():
    original_warning = urgent_detector._generate_warning_response_async

    async def fake_warning(user_input, urgent_issue):
        assert urgent_issue["level"] == "warning_high"
        return "warning-response"

    urgent_detector._generate_warning_response_async = fake_warning
    try:
        warning_result = await urgent_detector.generate_crisis_response_async(
            user_input="我感觉快撑不住了。",
            urgent_issue={"level": "warning_high", "triggers": ["撑不住"]},
            conversation_summary={},
        )
        urgent_result = await urgent_detector.generate_crisis_response_async(
            user_input="我不想活了。",
            urgent_issue={"level": "urgent", "triggers": ["不想活了"]},
            conversation_summary={},
        )
    finally:
        urgent_detector._generate_warning_response_async = original_warning

    assert warning_result == "warning-response"
    assert "立即联系专业帮助" in urgent_result
    assert "400-161-9995" in urgent_result


def main():
    test_detect_returns_normalized_four_level_semantics()
    print("PASS: urgent detector returns four-level semantics")

    asyncio.run(test_generate_crisis_response_keeps_legacy_level_compatibility())
    print("PASS: urgent detector keeps legacy crisis-response compatibility")


if __name__ == "__main__":
    main()

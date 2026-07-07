"""AI Rerank Mock 测试 — 验证 _ai_based_recommendation 的 ML 调用与降级"""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

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

from content_recommender import content_recommender
from models import ContentItem


# ============================================================
# 辅助
# ============================================================

def _make_item(id: str = "test_001", title: str = "", type: str = "article",
               category: str = "stress", difficulty: str = "beginner",
               tags: list = None, emotion_tags: list = None) -> ContentItem:
    return ContentItem(
        id=id,
        title=title or f"Test {id}",
        type=type,
        category=category,
        description="测试内容",
        difficulty=difficulty,
        tags=tags or [],
        emotion_tags=emotion_tags or [],
    )


def _make_fake_response(content: str):
    """构建模仿 OpenAI chat.completions.create 返回的 SimpleNamespace"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ]
    )


# ============================================================
# rerank 调用 mock 测试
# ============================================================

@pytest.mark.asyncio
async def test_rerank_parses_valid_response():
    """正常 JSON 响应应解析并返回对应内容"""
    candidates = [
        _make_item(id="a", title="压力管理"),
        _make_item(id="b", title="放松练习"),
        _make_item(id="c", title="正念冥想"),
    ]
    original_create = content_recommender.client.chat.completions.create

    async def fake_create(*a, **kw):
        return _make_fake_response(json.dumps({"ids": ["b", "c"]}, ensure_ascii=False))

    content_recommender.client.chat.completions.create = fake_create
    try:
        result = await content_recommender._ai_based_recommendation(
            user_input="我压力很大",
            current_emotion="压力",
            conversation_summary={"conversation_stage": "initial", "key_concerns": []},
            candidates=candidates,
            user_profile={},
            limit=3,
        )
        assert len(result) == 2
        assert result[0].id == "b"
        assert result[1].id == "c"
    finally:
        content_recommender.client.chat.completions.create = original_create


@pytest.mark.asyncio
async def test_rerank_fallback_on_json_error():
    """JSON 解析失败应回退到正则提取，仍失败则返回空"""
    candidates = [
        _make_item(id="a", title="压力管理"),
        _make_item(id="b", title="放松练习"),
    ]
    original_create = content_recommender.client.chat.completions.create

    async def fake_create(*a, **kw):
        return _make_fake_response("invalid json")

    content_recommender.client.chat.completions.create = fake_create
    try:
        result = await content_recommender._ai_based_recommendation(
            user_input="我压力很大",
            current_emotion="压力",
            conversation_summary={},
            candidates=candidates,
            user_profile={},
            limit=3,
        )
        # JSON 解析 + 正则都失败 → 空列表
        assert result == []
    finally:
        content_recommender.client.chat.completions.create = original_create


@pytest.mark.asyncio
async def test_rerank_fallback_on_empty_ids():
    """LLM 返回空 ID 列表应回退到空（调用方会 fallback 到规则结果）"""
    candidates = [_make_item(id="a")]
    original_create = content_recommender.client.chat.completions.create

    async def fake_create(*a, **kw):
        return _make_fake_response(json.dumps({"ids": []}))

    content_recommender.client.chat.completions.create = fake_create
    try:
        result = await content_recommender._ai_based_recommendation(
            user_input="测试",
            current_emotion="压力",
            conversation_summary={},
            candidates=candidates,
            user_profile={},
            limit=3,
        )
        assert result == []
    finally:
        content_recommender.client.chat.completions.create = original_create


@pytest.mark.asyncio
async def test_rerank_fallback_on_exception():
    """LLM 调用异常应返回空"""
    candidates = [_make_item(id="a")]
    original_create = content_recommender.client.chat.completions.create

    async def failing_create(*args, **kwargs):
        raise RuntimeError("API timeout")

    content_recommender.client.chat.completions.create = failing_create
    try:
        result = await content_recommender._ai_based_recommendation(
            user_input="测试",
            current_emotion="压力",
            conversation_summary={},
            candidates=candidates,
            user_profile={},
            limit=3,
        )
        assert result == []
    finally:
        content_recommender.client.chat.completions.create = original_create


@pytest.mark.asyncio
async def test_rerank_respects_limit():
    """rerank 返回结果应受 limit 限制"""
    candidates = [
        _make_item(id=f"item_{i}", title=f"内容{i}")
        for i in range(5)
    ]
    original_create = content_recommender.client.chat.completions.create

    async def fake_create(*a, **kw):
        return _make_fake_response(json.dumps({"ids": ["item_0", "item_1", "item_2", "item_3"]}))

    content_recommender.client.chat.completions.create = fake_create
    try:
        result = await content_recommender._ai_based_recommendation(
            user_input="测试",
            current_emotion="压力",
            conversation_summary={},
            candidates=candidates,
            user_profile={},
            limit=2,
        )
        assert len(result) == 2
    finally:
        content_recommender.client.chat.completions.create = original_create


@pytest.mark.asyncio
async def test_rerank_alternative_json_format():
    """兼容 different key: content_ids 或 recommendations"""
    candidates = [_make_item(id="x"), _make_item(id="y")]
    original_create = content_recommender.client.chat.completions.create

    async def fake_create(*a, **kw):
        return _make_fake_response(json.dumps({"content_ids": ["x", "y"]}))

    content_recommender.client.chat.completions.create = fake_create
    try:
        result = await content_recommender._ai_based_recommendation(
            user_input="测试",
            current_emotion="压力",
            conversation_summary={},
            candidates=candidates,
            user_profile={},
            limit=5,
        )
        assert len(result) == 2
    finally:
        content_recommender.client.chat.completions.create = original_create


@pytest.mark.asyncio
async def test_rerank_ignores_unknown_ids():
    """LLM 返回的 ID 不在候选列表中应忽略"""
    candidates = [_make_item(id="a")]
    original_create = content_recommender.client.chat.completions.create

    async def fake_create(*a, **kw):
        return _make_fake_response(json.dumps({"ids": ["a", "unknown_id"]}))

    content_recommender.client.chat.completions.create = fake_create
    try:
        result = await content_recommender._ai_based_recommendation(
            user_input="测试",
            current_emotion="压力",
            conversation_summary={},
            candidates=candidates,
            user_profile={},
            limit=5,
        )
        assert len(result) == 1
        assert result[0].id == "a"
    finally:
        content_recommender.client.chat.completions.create = original_create


def main():
    asyncio.run(test_rerank_parses_valid_response())
    print("PASS: rerank parses valid response")
    asyncio.run(test_rerank_fallback_on_json_error())
    print("PASS: rerank fallback on json error")
    asyncio.run(test_rerank_fallback_on_empty_ids())
    print("PASS: rerank fallback on empty ids")
    asyncio.run(test_rerank_fallback_on_exception())
    print("PASS: rerank fallback on exception")
    asyncio.run(test_rerank_respects_limit())
    print("PASS: rerank respects limit")
    asyncio.run(test_rerank_alternative_json_format())
    print("PASS: rerank alternative json format")
    asyncio.run(test_rerank_ignores_unknown_ids())
    print("PASS: rerank ignores unknown ids")


if __name__ == "__main__":
    main()

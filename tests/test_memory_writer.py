"""
Phase 2 测试：MemoryPolicy 写入门控 + MemoryWriter 写入 pipeline。

验证:
  - 各类候选记忆的正确门控
  - 重要性 / 置信度 / TTL 计算
  - 完整写入 pipeline：extract → gate → write
"""

import os, sys

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.extend([PROJECT_ROOT, BACKEND_DIR])

import asyncio
import tempfile
import pytest
from datetime import datetime

from memory_policy import MemoryPolicy
from memory_writer import MemoryWriter, MemoryCandidateExtractor
from models import MemoryCandidate, TurnContext, MemoryItem
from memory_store import MemoryStore


# ============================================================
# MemoryPolicy 测试
# ============================================================

def test_policy_should_write_explicit_preference():
    """显式表达的支持偏好应被写入。"""
    policy = MemoryPolicy()
    cand = MemoryCandidate(
        user_id="u1", memory_type="support_preference",
        content="test", source="explicit", confidence=0.9,
    )
    ctx = TurnContext(user_id="u1", session_id="s1", user_input="直接告诉我怎么做")
    assert policy.should_write(cand, ctx)
    print("PASS: explicit preference should_write=True")


def test_policy_should_skip_low_intensity_mood():
    """低强度情绪事件不应写入 memory_items。"""
    policy = MemoryPolicy()
    cand = MemoryCandidate(
        user_id="u1", memory_type="mood_event",
        content="test", emotion_intensity=0.4, risk_level="level_0",
        confidence=0.5, source="inferred",
    )
    ctx = TurnContext(user_id="u1", session_id="s1", user_input="今天还行")
    assert not policy.should_write(cand, ctx)
    print("PASS: low-intensity mood should_write=False")


def test_policy_should_write_high_intensity_mood():
    """高强度情绪事件应被写入。"""
    policy = MemoryPolicy()
    cand = MemoryCandidate(
        user_id="u1", memory_type="mood_event",
        content="test", emotion_intensity=0.8, risk_level="level_0",
        confidence=0.7, source="inferred",
    )
    ctx = TurnContext(user_id="u1", session_id="s1", user_input="最近太焦虑了")
    assert policy.should_write(cand, ctx)
    print("PASS: high-intensity mood should_write=True")


def test_policy_importance_explicit_higher():
    """显式表达的偏好应获得更高重要性。"""
    policy = MemoryPolicy()
    explicit = MemoryCandidate(
        user_id="u1", memory_type="support_preference",
        content="test", source="explicit", confidence=0.9,
    )
    inferred = MemoryCandidate(
        user_id="u1", memory_type="support_preference",
        content="test", source="inferred", confidence=0.65,
    )
    ctx = TurnContext(user_id="u1", session_id="s1", user_input="test")
    assert policy.assign_importance(explicit, ctx) > policy.assign_importance(inferred, ctx)
    print("PASS: explicit importance > inferred importance")


def test_policy_confidence_by_source():
    """置信度应按来源类型正确分级。"""
    policy = MemoryPolicy()
    assert policy.assign_confidence(MemoryCandidate(user_id="u1", memory_type="test", content="", source="explicit")) >= 0.85
    assert policy.assign_confidence(MemoryCandidate(user_id="u1", memory_type="test", content="", source="inferred")) < 0.85
    print("PASS: confidence by source type")


def test_policy_ttl_by_type():
    """TTL 应按记忆类型正确分配。"""
    policy = MemoryPolicy()
    pref_cand = MemoryCandidate(user_id="u1", memory_type="support_preference", content="")
    assert policy.assign_ttl(pref_cand) is None, "preference should have no TTL"
    mood_cand = MemoryCandidate(user_id="u1", memory_type="mood_event", content="")
    ttl = policy.assign_ttl(mood_cand)
    assert ttl is not None, "mood_event should have TTL"
    print(f"PASS: preference TTL=None, mood_event TTL={ttl}")


# ============================================================
# MemoryCandidateExtractor 测试
# ============================================================

def test_extractor_explicit_preference():
    """用户说'直接告诉我怎么做'应提取 support_preference 候选。"""
    extractor = MemoryCandidateExtractor()
    ctx = TurnContext(
        user_id="u1", session_id="s1", user_input="直接告诉我怎么做才对",
        emotion_state={"user_intent": "seeking_help"},
        conversation_summary={"stress_sources": [], "recent_intents": []},
    )
    candidates = extractor.extract(ctx)
    types = [c.memory_type for c in candidates]
    assert "support_preference" in types, f"Expected support_preference, got {types}"
    print(f"PASS: extracted preference from explicit keyword, types={types}")


def test_extractor_avoid_preference():
    """用户说'别安慰我'应提取 avoid_preference 候选。"""
    extractor = MemoryCandidateExtractor()
    ctx = TurnContext(
        user_id="u1", session_id="s1", user_input="别安慰我，我不喜欢空泛安慰",
    )
    candidates = extractor.extract(ctx)
    types = [c.memory_type for c in candidates]
    assert "avoid_preference" in types
    print(f"PASS: extracted avoid_preference, types={types}")


def test_extractor_skips_low_intensity():
    """低强度情绪不应产生 mood_event 候选。"""
    extractor = MemoryCandidateExtractor()
    ctx = TurnContext(
        user_id="u1", session_id="s1", user_input="今天还行",
        emotion_state={"emotion_intensity": 0.3, "risk_level": "level_0"},
    )
    candidates = extractor.extract(ctx)
    types = [c.memory_type for c in candidates]
    assert "mood_event" not in types, f"mood_event should not be extracted for low intensity, got {types}"
    print(f"PASS: no mood_event for low intensity, types={types}")


# ============================================================
# MemoryWriter 集成测试
# ============================================================

@pytest.mark.asyncio
async def test_writer_writes_to_memory_items():
    """MemoryWriter.process_turn 应写入 memory_items 表。"""
    import tempfile
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = db.name
    db.close()

    store = MemoryStore(db_path=db_path)
    writer = MemoryWriter(store=store)

    ctx = TurnContext(
        user_id="u1", session_id="s1", user_input="直接告诉我怎么做，我最近压力很大",
        emotion_state={
            "current_emotion": "焦虑", "emotion_intensity": 0.8,
            "stress_source": "求职压力", "risk_level": "level_1",
            "user_intent": "seeking_help",
        },
    )
    ids = await writer.process_turn(ctx)
    assert len(ids) > 0, f"Expected at least 1 memory_id, got {ids}"
    print(f"PASS: writer returned {len(ids)} memory_ids: {ids}")

    # 验证写入的内容可读回
    for mid in ids:
        item = await store.get(mid)
        assert item is not None, f"memory_id {mid} not found after write"
        print(f"  - {item.memory_type}: importance={item.importance:.2f}, confidence={item.confidence:.2f}")

    await store.close()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_writer_skips_low_value():
    """MemoryWriter 应跳过低价值输入（不写入任何 memory_items）。"""
    store = MemoryStore(db_path=":memory:")
    writer = MemoryWriter(store=store)

    ctx = TurnContext(
        user_id="u1", session_id="s1", user_input="今天感觉还行",
        emotion_state={
            "current_emotion": "中性", "emotion_intensity": 0.3,
            "risk_level": "level_0",
        },
    )
    ids = await writer.process_turn(ctx)
    assert len(ids) == 0, f"Expected no writes for low-value input, got {ids}"
    print("PASS: no writes for low-value input")

    await store.close()


@pytest.mark.asyncio
async def test_memory_store_persistence():
    """MemoryStore 文件模式：写入 → 读取 → 更新 → 归档 完整流程。"""
    import tempfile
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = db.name
    db.close()

    store = MemoryStore(db_path=db_path)

    # write
    mid = await store.write(MemoryItem(
        user_id="u1", memory_type="stress_source", content="求职压力",
        importance=0.8,
    ))
    assert mid.startswith("mem_")
    print(f"1. write: {mid}")

    # get
    item = await store.get(mid)
    assert item is not None
    print(f"2. get: memory_type={item.memory_type}, importance={item.importance}")

    # second get should increment access_count
    item2 = await store.get(mid)
    assert item2 is not None
    print(f"3. get again: access_count={item2.access_count}")

    # update
    await store.update(mid, {"importance": 0.9})
    item3 = await store.get(mid)
    assert item3.importance == 0.9
    print(f"4. update importance: {item3.importance}")

    # archive
    await store.archive(mid, reason="test")
    item4 = await store.get(mid)
    assert item4.status == "archived"
    print(f"5. archive: status={item4.status}")

    # batch_write
    items = [
        MemoryItem(user_id="u1", memory_type="mood_event", content="e1"),
        MemoryItem(user_id="u1", memory_type="mood_event", content="e2"),
    ]
    mids = await store.batch_write(items)
    assert len(mids) == 2
    print(f"6. batch_write: {len(mids)} items")

    # get_active_items
    active = await store.get_active_items("u1", memory_types=["mood_event"])
    assert len(active) == 2
    print(f"7. get_active_items: {len(active)} mood_events")

    await store.close()
    os.unlink(db_path)


# ============================================================
if __name__ == "__main__":
    print("=== MemoryPolicy Tests ===")
    test_policy_should_write_explicit_preference()
    test_policy_should_skip_low_intensity_mood()
    test_policy_should_write_high_intensity_mood()
    test_policy_importance_explicit_higher()
    test_policy_confidence_by_source()
    test_policy_ttl_by_type()

    print("\n=== MemoryCandidateExtractor Tests ===")
    test_extractor_explicit_preference()
    test_extractor_avoid_preference()
    test_extractor_skips_low_intensity()

    print("\n=== MemoryWriter Integration Tests ===")
    asyncio.run(test_writer_writes_to_memory_items())
    asyncio.run(test_writer_skips_low_value())
    asyncio.run(test_memory_store_persistence())

    print("\n=== ALL PHASE 2 TESTS PASSED ===")

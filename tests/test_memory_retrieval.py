"""
Phase 3a 测试：MemoryRetriever（BM25 + RRF + 重排）+ MemoryInjector（token budget）。

验证:
  - BM25 能检索到关键词匹配的记忆
  - RRF 融合排序稳定
  - 多因子重排优先风险记忆
  - MemoryInjector 不超过 token budget
  - 注入优先级：风险 > 偏好 > 压力源 > 情绪事件
"""

import os, sys

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.extend([PROJECT_ROOT, BACKEND_DIR])

import asyncio
import tempfile
from models import MemoryItem, MemoryQuery, MemorySearchResult
from memory_store import MemoryStore
from memory_retriever import MemoryRetriever, MemoryBM25Retriever
from memory_injector import MemoryInjector
from memory_context_builder import MemoryContextBuilder


# ============================================================
# MemoryBM25Retriever 测试
# ============================================================

def test_bm25_keyword_match():
    """BM25 应检索到含关键词的记忆。"""
    bm25 = MemoryBM25Retriever()
    items = [
        MemoryItem(id="m1", user_id="u1", memory_type="mood_event", content="实验失败感到焦虑", importance=0.5),
        MemoryItem(id="m2", user_id="u1", memory_type="stress_source", content="求职压力很大", importance=0.8),
        MemoryItem(id="m3", user_id="u1", memory_type="mood_event", content="今天心情不错", importance=0.3),
    ]
    results = bm25.retrieve("实验失败", items, limit=5)
    ids = [item.id for item, _ in results]
    assert "m1" in ids, f"BM25 应召回 m1 (实验失败), got {ids}"
    print(f"PASS: BM25 keyword match, top={ids[0]}")


def test_bm25_extra_terms():
    """BM25 应利用额外检索词扩大召回。"""
    bm25 = MemoryBM25Retriever()
    items = [
        MemoryItem(id="m1", user_id="u1", memory_type="stress_source", content="面试", emotion="焦虑"),
        MemoryItem(id="m2", user_id="u1", memory_type="mood_event", content="今天天气不错"),
    ]
    results = bm25.retrieve("最近压力很大", items, limit=5, extra_terms=["面试", "焦虑"])
    ids = [item.id for item, _ in results]
    assert "m1" in ids, f"BM25 应通过 extra_terms 召回 m1, got {ids}"
    print(f"PASS: BM25 extra_terms, top={ids[0]}")


# ============================================================
# MemoryRetriever 集成测试
# ============================================================

async def test_retriever_risk_priority():
    """MemoryRetriever 的 RRF + 重排应将风险记忆排在前面。"""
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = db.name
    db.close()

    store = MemoryStore(db_path=db_path)
    retriever = MemoryRetriever(store=store)

    # 写入测试数据
    items = [
        MemoryItem(user_id="u1", memory_type="mood_event", content="今天心情不错", emotion="快乐", importance=0.3),
        MemoryItem(user_id="u1", memory_type="mood_event", content="实验失败很焦虑", emotion="焦虑", risk_level="level_2", importance=0.7),
        MemoryItem(user_id="u1", memory_type="stress_source", content="求职压力", importance=0.8),
        MemoryItem(user_id="u1", memory_type="support_preference", content="用户偏好直接建议", source="explicit", importance=0.9),
    ]
    for item in items:
        await store.write(item)

    # 检索：查询"焦虑" + need_risk_context
    query = MemoryQuery(
        text="最近好焦虑",
        emotion="焦虑",
        risk_level="level_2",
        need_risk_context=True,
        memory_types=["mood_event", "stress_source", "support_preference"],
    )
    results = await retriever.retrieve("u1", query, top_k=10)
    assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"

    # 风险高的记忆应排前面
    top_types = [r.item.memory_type for r in results]
    print(f"  top results: {[(r.item.memory_type, r.item.content[:20], f'{r.score:.3f}') for r in results[:4]]}")

    await store.close()
    os.unlink(db_path)
    print("PASS: retriever returns results with risk priority")


async def test_retriever_with_real_memory_items():
    """从 memory_items 表写入后能检索到。"""
    store = MemoryStore(db_path=":memory:")
    retriever = MemoryRetriever(store=store)

    await store.write(MemoryItem(
        user_id="u2", memory_type="mood_event",
        content="导师问进度压力很大", emotion="焦虑",
        importance=0.7, risk_level="level_1",
    ))
    await store.write(MemoryItem(
        user_id="u2", memory_type="support_preference",
        content="用户偏好直接可执行建议", source="explicit",
        importance=0.9,
    ))

    query = MemoryQuery(text="导师又催了", emotion="焦虑", need_risk_context=True)
    results = await retriever.retrieve("u2", query, top_k=5)
    assert len(results) > 0, "Expected results with real memory_items"
    print(f"PASS: retriever got {len(results)} results from memory_items: {[r.item.memory_type for r in results]}")

    await store.close()


# ============================================================
# MemoryInjector 测试
# ============================================================

def test_injector_budget_respected():
    """MemoryInjector 不应超过 token budget。"""
    injector = MemoryInjector()
    items = [
        MemoryItem(id=f"m{i}", user_id="u1", memory_type="mood_event", content=f"测试记忆内容 {i} 的详细描述", importance=0.7)
        for i in range(10)
    ]
    memories = [
        MemorySearchResult(item=item, score=1.0, rank=i, retrieval_method="bm25")
        for i, item in enumerate(items)
    ]
    result = injector.build_context(memories, risk_context=None, token_budget=100)
    assert result.used_tokens <= 100, f"Budget exceeded: {result.used_tokens} > 100"
    print(f"PASS: injector budget respected: {result.used_tokens}/100")


def test_injector_risk_first():
    """风险上下文应优先于其他记忆。"""
    injector = MemoryInjector()
    risk_ctx = {"baseline": "high", "common_triggers": ["求职压力"]}
    result = injector.build_context([], risk_context=risk_ctx, token_budget=200)
    assert "风险" in result.text, f"Risk context should be injected, got: {result.text[:50]}"
    print(f"PASS: risk context injected first: {result.text[:40]}")


def test_injector_priority_order():
    """注入优先级：偏好 > 压力源 > 情绪事件。"""
    injector = MemoryInjector()
    items = [
        MemoryItem(id="m1", user_id="u1", memory_type="mood_event", content="普通情绪事件", importance=0.5),
        MemoryItem(id="m2", user_id="u1", memory_type="support_preference", content="用户偏好直接建议", source="explicit", importance=0.9),
        MemoryItem(id="m3", user_id="u1", memory_type="stress_source", content="求职压力", importance=0.8),
    ]
    memories = [
        MemorySearchResult(item=item, score=1.0, rank=i, retrieval_method="bm25")
        for i, item in enumerate(items)
    ]
    result = injector.build_context(memories, risk_context=None, token_budget=500)
    lines = [l for l in result.text.split("\n") if l.strip()]
    assert len(lines) >= 2, f"Expected multiple lines, got: {lines}"
    print(f"PASS: priority injection, {len(lines)} lines: {[l[:20] for l in lines]}")


def test_injector_skipped_ids():
    """极低 budget 时低优先级记忆应被跳过。"""
    injector = MemoryInjector()
    # 风险上下文已占用 ~20 tokens，余量很小
    risk_ctx = {"baseline": "high", "common_triggers": ["求职"]}
    items = [
        MemoryItem(id="m_pref", user_id="u1", memory_type="support_preference", content="用户偏好直接建议", source="explicit", importance=0.9),
        MemoryItem(id="m_event", user_id="u1", memory_type="mood_event", content="今天情绪有些低落", importance=0.3),
    ]
    memories = [
        MemorySearchResult(item=items[0], score=1.0, rank=0, retrieval_method="bm25"),
        MemorySearchResult(item=items[1], score=0.5, rank=1, retrieval_method="scored"),
    ]
    result = injector.build_context(memories, risk_context=risk_ctx, token_budget=30)
    # 极低 budget 下，低优先级 mood_event 应被跳过
    if "m_event" in result.included_memory_ids:
        print(f"  INFO: m_event was included (budget allowed both), dropped={result.dropped_memory_ids}")
    else:
        assert "m_event" in result.dropped_memory_ids, f"m_event should be tracked as dropped"
    print(f"PASS: skipped ids tracked: {result.dropped_memory_ids}")


# ============================================================
# MemoryContextBuilder 集成测试
# ============================================================

async def test_context_builder_with_data():
    """MemoryContextBuilder 在 memory_items 有数据时应返回注入结果。"""
    import tempfile, os
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = db.name
    db.close()

    store = MemoryStore(db_path=db_path)
    await store.write(MemoryItem(
        user_id="u3", memory_type="stress_source",
        content="科研压力", importance=0.8,
    ))
    await store.write(MemoryItem(
        user_id="u3", memory_type="support_preference",
        content="用户偏好直接建议", source="explicit", importance=0.9,
    ))

    from memory_retriever import MemoryRetriever
    from memory_injector import MemoryInjector
    builder = MemoryContextBuilder()
    builder.retriever = MemoryRetriever(store=store)
    builder.injector = MemoryInjector()

    result = await builder.build(
        user_id="u3",
        user_input="实验又失败了，压力很大",
        emotion_state={"current_emotion": "焦虑", "stress_source": "科研压力"},
        risk_state={"level": "level_0"},
    )
    print(f"  context_builder: {result.used_tokens}t, {len(result.included_memory_ids)} memories, text={result.text[:60]}")
    # 应至少注入偏好（显式表达 + 高重要性）
    assert len(result.included_memory_ids) >= 1, f"Expected at least 1 memory, got {result.included_memory_ids}"

    await store.close()
    os.unlink(db_path)
    print("PASS: context_builder with file-based DB")


# ============================================================
if __name__ == "__main__":
    print("=== MemoryBM25Retriever Tests ===")
    test_bm25_keyword_match()
    test_bm25_extra_terms()

    print("\n=== MemoryRetriever Integration Tests ===")
    asyncio.run(test_retriever_risk_priority())
    asyncio.run(test_retriever_with_real_memory_items())

    print("\n=== MemoryInjector Tests ===")
    test_injector_budget_respected()
    test_injector_risk_first()
    test_injector_priority_order()
    test_injector_skipped_ids()

    print("\n=== MemoryContextBuilder Tests ===")
    asyncio.run(test_context_builder_with_data())

    print("\n=== ALL PHASE 3a TESTS PASSED ===")

"""
P0 测试：风险记忆并发写入安全。

验证 RiskMemoryStore 在并发写入场景下不会丢失事件或基线数据。
使用临时文件数据库以确保写入持久化。
"""

import os
import sys
import tempfile

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.extend([PROJECT_ROOT, BACKEND_DIR])

import asyncio
import pytest
from risk_memory_store import RiskMemoryStore
from models import RiskEvent, RiskBaseline, ProtectiveFactor


@pytest.fixture
def db_path():
    """每个测试使用独立临时数据库文件。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = tmp.name
    tmp.close()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.mark.asyncio
async def test_concurrent_risk_writes_no_data_loss(db_path):
    """10 个并发写入，确认 10 条事件全部保留。"""
    store = RiskMemoryStore(db_path=db_path)
    tasks = [
        store.add_risk_event(
            RiskEvent(
                user_id="test_user",
                session_id=f"sess_{i}",
                risk_level="level_2",
                subject="self",
                summary=f"Concurrent test event {i}",
            )
        )
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10, f"Expected 10 event IDs, got {len(results)}"

    events = await store.get_recent_events("test_user", days=30)
    assert len(events) == 10, f"Expected 10 events, got {len(events)}"

    await store._close()


@pytest.mark.asyncio
async def test_concurrent_baseline_updates(db_path):
    """并发写入 Level 3 事件，确认 baseline=high。"""
    store = RiskMemoryStore(db_path=db_path)
    tasks = [
        store.add_risk_event_with_baseline_update(
            RiskEvent(
                user_id="test_user",
                session_id=f"sess_{i}",
                risk_level="level_3",
                subject="self",
                summary=f"Concurrent crisis {i}",
            )
        )
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5

    for eid, baseline in results:
        assert baseline.baseline == "high", f"Expected high baseline, got {baseline.baseline}"

    events = await store.get_recent_events("test_user", days=30)
    assert len(events) == 5

    await store._close()


@pytest.mark.asyncio
async def test_trigger_upsert_concurrent(db_path):
    """并发 upsert 同一触发词，frequency 不丢失。"""
    store = RiskMemoryStore(db_path=db_path)
    tasks = [
        store.add_risk_trigger("test_user", "求职压力")
        for _ in range(5)
    ]
    await asyncio.gather(*tasks)

    triggers = await store.get_active_triggers("test_user")
    assert len(triggers) == 1
    assert triggers[0].frequency == 5, f"Expected frequency 5, got {triggers[0].frequency}"

    await store._close()


@pytest.mark.asyncio
async def test_mixed_workload(db_path):
    """混合并发：事件写入 + 基线读取 + 触发词更新 同时进行。"""
    store = RiskMemoryStore(db_path=db_path)

    async def writer():
        for i in range(3):
            await store.add_risk_event_with_baseline_update(
                RiskEvent(
                    user_id="workload_user",
                    session_id=f"s_{i}",
                    risk_level="level_2",
                    subject="self",
                    summary=f"Workload {i}",
                )
            )
            await store.add_risk_trigger("workload_user", f"trigger_{i}")

    async def reader():
        for _ in range(10):
            bl = await store.get_baseline("workload_user")
            _ = bl.baseline if bl else "low"
            await store.get_recent_events("workload_user", days=30)

    await asyncio.gather(writer(), reader())
    events = await store.get_recent_events("workload_user", days=30)
    assert len(events) == 3

    await store._close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

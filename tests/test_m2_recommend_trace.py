"""RecommendationTrace 单元测试"""
import os
import sys
import tempfile
import shutil
import json

PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from recommendation_trace import TraceEvent, write_trace, read_traces


def test_trace_event_defaults():
    """默认字段值应合理"""
    t = TraceEvent()
    assert t.request_id == ""
    assert t.gate_inputs == {}
    assert t.gate_output == {}
    assert t.recommendation_ids == []
    assert t.recommendation_scores == []
    assert t.safety_overridden is False
    assert t.latency_ms == 0
    assert t.timestamp is not None


def test_trace_event_serialization():
    """TraceEvent 可序列化为 JSON 并反序列化"""
    t = TraceEvent(
        request_id="test_001",
        user_id="user_1",
        session_id="session_1",
        turn_id=5,
        gate_inputs={"emotion_intensity": 0.72, "risk_score": 0.4, "intent_score": 0.45},
        gate_output={"should_recommend": True, "recommend_type": "soft", "score": 0.51},
        recommendation_ids=["article_001", "audio_001"],
        recommendation_scores=[0.85, 0.72],
        safety_overridden=False,
        persisted=True,
        latency_ms=320,
    )
    data = json.loads(t.model_dump_json())
    assert data["request_id"] == "test_001"
    assert data["gate_output"]["recommend_type"] == "soft"
    assert data["recommendation_ids"] == ["article_001", "audio_001"]
    assert data["latency_ms"] == 320


def test_trace_write_and_read():
    """write_trace 写入 JSONL 后 read_traces 应能读取"""
    # 使用临时路径
    import recommendation_trace as rt
    original_path = rt._TRACE_LOG_PATH
    tmpdir = tempfile.mkdtemp()
    try:
        rt._TRACE_LOG_PATH = os.path.join(tmpdir, "test_trace.jsonl")

        t1 = TraceEvent(request_id="r1", user_id="u1", turn_id=1,
                         gate_output={"should_recommend": True, "recommend_type": "hard"})
        t2 = TraceEvent(request_id="r2", user_id="u1", turn_id=2,
                         gate_output={"should_recommend": False, "recommend_type": "none"})
        write_trace(t1)
        write_trace(t2)

        events = read_traces()
        assert len(events) == 2
        ids = [e.request_id for e in events]
        assert "r1" in ids
        assert "r2" in ids
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        rt._TRACE_LOG_PATH = original_path


def test_trace_read_limit():
    """read_traces(limit) 应限制返回数量"""
    import recommendation_trace as rt
    original_path = rt._TRACE_LOG_PATH
    tmpdir = tempfile.mkdtemp()
    try:
        rt._TRACE_LOG_PATH = os.path.join(tmpdir, "test_limit.jsonl")
        for i in range(10):
            write_trace(TraceEvent(request_id=f"r{i}"))
        events = read_traces(limit=3)
        assert len(events) == 3
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        rt._TRACE_LOG_PATH = original_path


def test_trace_read_empty_file():
    """无日志文件时返回空列表"""
    import recommendation_trace as rt
    original_path = rt._TRACE_LOG_PATH
    tmpdir = tempfile.mkdtemp()
    try:
        rt._TRACE_LOG_PATH = os.path.join(tmpdir, "nonexistent.jsonl")
        events = read_traces()
        assert events == []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        rt._TRACE_LOG_PATH = original_path


def test_trace_handles_partial_data():
    """TraceEvent 应接受部分字段"""
    t = TraceEvent(request_id="partial", user_id="u1")
    assert t.request_id == "partial"
    assert t.turn_id == 0
    data = json.loads(t.model_dump_json())
    assert data["gate_inputs"] == {}
    assert data["gate_output"] == {}


def test_trace_disabled():
    """禁用 trace 时不写入文件"""
    import recommendation_trace as rt
    original_path = rt._TRACE_LOG_PATH
    original_enabled = rt._TRACE_ENABLED
    tmpdir = tempfile.mkdtemp()
    try:
        rt._TRACE_LOG_PATH = os.path.join(tmpdir, "disabled.jsonl")
        rt._TRACE_ENABLED = False
        write_trace(TraceEvent(request_id="should_not_appear"))
        assert not os.path.exists(rt._TRACE_LOG_PATH)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        rt._TRACE_LOG_PATH = original_path
        rt._TRACE_ENABLED = original_enabled


def main():
    test_trace_event_defaults()
    print("PASS: trace event defaults")
    test_trace_event_serialization()
    print("PASS: trace event serialization")
    test_trace_write_and_read()
    print("PASS: trace write and read")
    test_trace_read_limit()
    print("PASS: trace read limit")
    test_trace_read_empty_file()
    print("PASS: trace read empty file")
    test_trace_handles_partial_data()
    print("PASS: trace handles partial data")
    test_trace_disabled()
    print("PASS: trace disabled")


if __name__ == "__main__":
    main()

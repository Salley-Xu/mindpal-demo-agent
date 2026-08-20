import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.extend([PROJECT_ROOT, BACKEND_DIR])

import state.persistence as state_persistence  # noqa: E402
from agent_orchestrator import agent_orchestrator  # noqa: E402
from agent_tools import TOOL_DEFINITIONS  # noqa: E402
from config import config  # noqa: E402
from main import app  # noqa: E402
from risk_evaluator import RiskEvaluator, _FallbackPredictor  # noqa: E402
from state.persistence import AgentStateStore  # noqa: E402
from state.schema import AgentState, IdentityState  # noqa: E402


def test_risk_fallback_fails_safe_for_explicit_crisis_language():
    evaluator = RiskEvaluator()
    fallback = _FallbackPredictor(evaluator._analyze_context)

    crisis = fallback.predict("我正在割腕，真的不想活了")
    denial = fallback.predict("我没有想自杀，只是压力很大")
    discussion = fallback.predict("电影里主角最后自杀了")
    third_party = fallback.predict("我朋友说他不想活了，我该怎么办")

    assert crisis["level"] == "level_3"
    assert crisis["degraded_mode"] is True
    assert denial["level"] == "level_0"
    assert discussion["level"] == "level_0"
    assert third_party["level"] == "level_1"

    evaluator._predictor = fallback
    integrated = evaluator.evaluate("我正在割腕，真的不想活了")
    assert integrated["level"] == "level_3"
    assert integrated["risk_evidence"]["bert"]["degraded_mode"] is True


@pytest.mark.asyncio
async def test_tool_identity_is_always_bound_to_request_context():
    captured = {}
    original = agent_orchestrator.available_tools["get_user_profile"]

    async def fake_tool(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    agent_orchestrator.available_tools["get_user_profile"] = fake_tool
    tool_call = SimpleNamespace(
        id="tool-security",
        function=SimpleNamespace(
            name="get_user_profile",
            arguments='{"user_id":"victim","session_id":"other-session"}',
        ),
    )
    try:
        result = await agent_orchestrator._execute_single_tool(
            tool_call=tool_call,
            conversation_summary={},
            request_text="test",
            user_id="current-user",
            session_id="current-session",
        )
    finally:
        agent_orchestrator.available_tools["get_user_profile"] = original

    assert result["tool_step_info"].success is True
    assert captured["user_id"] == "current-user"
    assert captured["session_id"] == "current-session"


def test_tool_schemas_do_not_expose_identity_fields_to_model():
    for definition in TOOL_DEFINITIONS:
        properties = definition["function"]["parameters"].get("properties", {})
        assert "user_id" not in properties
        assert "session_id" not in properties


def test_user_token_cannot_access_another_users_resources():
    original_admin = config.API_AUTH_TOKEN
    original_required = config.API_AUTH_REQUIRED
    original_users = config.API_USER_TOKENS_JSON
    config.API_AUTH_TOKEN = "admin-token"
    config.API_AUTH_REQUIRED = True
    config.API_USER_TOKENS_JSON = '{"user-a-token":"user-a"}'
    try:
        with TestClient(app) as client:
            cross_user = client.get(
                "/session/user-b/list",
                headers={"Authorization": "Bearer user-a-token"},
            )
            admin_only = client.get(
                "/urgent/cases",
                headers={"Authorization": "Bearer user-a-token"},
            )
            invalid = client.get(
                "/session/user-a/list",
                headers={"Authorization": "Bearer invalid-token"},
            )
    finally:
        config.API_AUTH_TOKEN = original_admin
        config.API_AUTH_REQUIRED = original_required
        config.API_USER_TOKENS_JSON = original_users

    assert cross_user.status_code == 403
    assert admin_only.status_code == 403
    assert invalid.status_code == 401


def test_state_history_uses_hashed_filename_for_untrusted_session_id():
    original_dir = state_persistence._DEBUG_HISTORY_DIR
    with tempfile.TemporaryDirectory() as temp_dir:
        state_persistence._DEBUG_HISTORY_DIR = Path(temp_dir)
        try:
            store = AgentStateStore(persist_history=True)
            state = AgentState(
                identity=IdentityState(
                    user_id="user-a",
                    session_id="../../outside",
                )
            )
            store.save_latest(state)
            files = list(Path(temp_dir).glob("*.jsonl"))
        finally:
            state_persistence._DEBUG_HISTORY_DIR = original_dir

    assert len(files) == 1
    assert files[0].name != "outside.jsonl"
    assert ".." not in files[0].name

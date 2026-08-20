import os
import sys

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.extend([PROJECT_ROOT, BACKEND_DIR])

from application import create_app  # noqa: E402
from conversation_manager import ConversationManager  # noqa: E402


class FakeScheduler:
    def __init__(self):
        self.running = False
        self.jobs = []
        self.shutdown_wait = None

    def add_job(self, **kwargs):
        self.jobs.append(kwargs)

    def start(self):
        self.running = True

    def shutdown(self, wait=True):
        self.shutdown_wait = wait
        self.running = False


class FakeConversationManager:
    def __init__(self):
        self.waited = False

    async def cleanup_expired_sessions_async(self, days):
        return 0

    async def wait_for_persistence(self):
        self.waited = True


def test_application_factory_owns_scheduler_lifecycle():
    scheduler = FakeScheduler()
    manager = FakeConversationManager()
    app = create_app(
        APIRouter(),
        manager=manager,
        scheduler_factory=lambda: scheduler,
    )

    with TestClient(app):
        assert scheduler.running is True
        assert len(scheduler.jobs) == 1
        assert app.state.scheduler is scheduler

    assert manager.waited is True
    assert scheduler.running is False
    assert scheduler.shutdown_wait is True


@pytest.mark.asyncio
async def test_conversation_manager_drains_owned_persistence_tasks():
    manager = ConversationManager(use_persistence=False)
    completed = []

    async def persist():
        completed.append("done")

    manager._schedule_persistence(persist())
    await manager.wait_for_persistence()

    assert completed == ["done"]

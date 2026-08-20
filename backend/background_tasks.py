"""Ownership and shutdown semantics for in-process background work."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundTaskRegistry:
    """Track fire-and-forget tasks so they are observable and drainable."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()

    def schedule(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coroutine)
            return

        task = loop.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except (asyncio.CancelledError, RuntimeError):
            return
        if error is not None:
            logger.error(
                "后台任务失败",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def drain(self) -> None:
        current_loop = asyncio.get_running_loop()
        tasks = [
            task
            for task in tuple(self._tasks)
            if not task.done() and task.get_loop() is current_loop
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


background_tasks = BackgroundTaskRegistry()

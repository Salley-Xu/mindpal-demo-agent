# -*- coding: utf-8 -*-
"""
Memory Lifecycle Updater（Phase 4 Task 4.8）。

处理：EXPIRE（TTL 到期）/ DECAY（长期未访问）/ MERGE（合并回写）/ SUPERSEDE（归档）。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

from memory_v2.schema import MEMORY_TYPE_CONFIGS, MemoryOperation, MemoryType, MemoryWriteEvent
from models import MemoryCandidate, MemoryItem


class MemoryLifecycle:
    """生命周期治理。"""

    def apply_expiry(self, item: MemoryItem, now: Optional[datetime] = None) -> Optional[MemoryOperation]:
        """检查 TTL / 长期未访问 → EXPIRE。"""
        now = now or datetime.now()
        if item.expires_at and item.expires_at < now:
            return MemoryOperation.EXPIRE
        # 长期未访问（>180 天）且低重要性 → DECAY
        if item.last_accessed_at:
            days = (now - item.last_accessed_at).days
            if days > 180 and item.importance < 0.6:
                return MemoryOperation.EXPIRE
        return None

    def compute_ttl(self, memory_type: str) -> Optional[datetime]:
        """按类型配置计算 TTL。"""
        try:
            cfg = MEMORY_TYPE_CONFIGS.get(MemoryType(memory_type))
        except ValueError:
            cfg = None
        if cfg and cfg.default_ttl_days:
            return datetime.now() + timedelta(days=cfg.default_ttl_days)
        return None


memory_lifecycle = MemoryLifecycle()

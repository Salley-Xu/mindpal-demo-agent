#!/usr/bin/env python3
"""
风险记忆迁移脚本：将 user_profile.preferences 中的旧 risk_memory JSON
迁移到 risk_events / risk_baselines / risk_triggers 新表。

用法:
    python backend/scripts/migrate_risk_memory_v2.py
    python backend/scripts/migrate_risk_memory_v2.py --dry-run  # 预览，不实际写入
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

os.environ.setdefault("DEEPSEEK_API_KEY", "migration-key")
os.environ.setdefault("MEMORY_ENABLED", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("migrate_risk_memory")


async def get_all_user_ids(adb_manager) -> list:
    """获取所有有 user_profile 的用户 ID。"""
    import aiosqlite
    from config import config

    db_path = config.SESSION_DB_PATH
    if not os.path.exists(db_path):
        logger.warning("数据库文件不存在: %s", db_path)
        return []

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            "SELECT user_id FROM user_profile WHERE preferences LIKE '%risk_memory%'"
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]


async def migrate_user(user_id: str, rms, adb_manager, dry_run: bool = False) -> dict:
    """迁移单个用户的风险记忆。返回迁移统计。"""
    from risk_levels import normalize_risk_level

    stats = {"user_id": user_id, "events": 0, "triggers": 0, "baseline": None}

    profile = await adb_manager.get_user_profile(user_id)
    if not profile:
        return stats

    preferences = profile.get("preferences") or {}
    if isinstance(preferences, str):
        preferences = json.loads(preferences)
    risk_memory = preferences.get("risk_memory", {})
    if not risk_memory:
        logger.debug("用户 %s 无 risk_memory，跳过", user_id)
        return stats

    # 1. 迁移风险事件
    for event in risk_memory.get("risk_events", []):
        if dry_run:
            stats["events"] += 1
            continue
        try:
            et = event.get("event_time")
            if et:
                event_time = datetime.fromisoformat(et) if isinstance(et, str) else datetime.now(timezone.utc)
            else:
                event_time = datetime.now(timezone.utc)

            await rms.add_risk_event(
                type("RiskEvent", (), {
                    "id": "",
                    "user_id": user_id,
                    "session_id": event.get("session_id", ""),
                    "turn_id": None,
                    "event_time": event_time,
                    "risk_level": normalize_risk_level(event.get("max_level", "level_0")),
                    "risk_score": None,
                    "subject": "self",
                    "topic": event.get("topic", ""),
                    "summary": event.get("summary", ""),
                    "evidence_snippet": None,
                    "safety_confirmed": event.get("safety_confirmed", False),
                    "support_engaged": event.get("support_engaged", False),
                    "decayed": event.get("decayed", False),
                })()
            )
            stats["events"] += 1
        except Exception as e:
            logger.error("迁移事件失败 (user=%s): %s", user_id, e)

    # 2. 迁移触发词
    for trigger in risk_memory.get("common_triggers", []):
        if dry_run:
            stats["triggers"] += 1
            continue
        try:
            await rms.add_risk_trigger(user_id, trigger)
            stats["triggers"] += 1
        except Exception as e:
            logger.error("迁移触发词失败 (user=%s, trigger=%s): %s", user_id, trigger, e)

    # 3. 迁移基线
    old_baseline = risk_memory.get("baseline", "low")
    if not dry_run:
        try:
            now = datetime.now(timezone.utc)
            await rms.set_baseline(
                type("RiskBaseline", (), {
                    "user_id": user_id,
                    "baseline": old_baseline,
                    "baseline_score": None,
                    "baseline_updated_at": now,
                    "decay_status": risk_memory.get("decay_status", "active"),
                    "last_high_risk_time": now if old_baseline in ("high", "medium") else None,
                    "metadata": {},
                })()
            )
        except Exception as e:
            logger.error("迁移基线失败 (user=%s): %s", user_id, e)
    stats["baseline"] = old_baseline

    return stats


async def main():
    parser = argparse.ArgumentParser(description="迁移风险记忆到新表")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际写入")
    args = parser.parse_args()

    from database import adb_manager
    from risk_memory_store import risk_memory_store as rms

    logger.info("开始扫描用户 risk_memory 数据...")
    user_ids = await get_all_user_ids(adb_manager)
    logger.info("发现 %d 个含有 risk_memory 的用户", len(user_ids))

    total = {"events": 0, "triggers": 0, "users": 0}
    for uid in user_ids:
        stats = await migrate_user(uid, rms, adb_manager, dry_run=args.dry_run)
        if stats["events"] > 0 or stats["triggers"] > 0:
            total["users"] += 1
            total["events"] += stats["events"]
            total["triggers"] += stats["triggers"]
            logger.info(
                "用户 %s: events=%d, triggers=%d, baseline=%s",
                uid, stats["events"], stats["triggers"], stats["baseline"],
            )

    mode = "预览" if args.dry_run else "迁移"
    logger.info(
        "%s完成: %d 用户, %d 事件, %d 触发词",
        mode, total["users"], total["events"], total["triggers"],
    )


if __name__ == "__main__":
    asyncio.run(main())

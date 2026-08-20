# -*- coding: utf-8 -*-
"""
Phase 4 Task 4.9：生成 Memory Benchmark v2。

覆盖（Phase 4 §10）：
  explicit reference / implicit personalization / conflict / updated preference /
  expired memory / irrelevant memory / no-memory-needed / multi-memory

每 case：
  text：当前轮文本
  intent：意图标签（AgentState 输入）
  turn_index
  gold_retrieve：是否应检索
  precondition：已有记忆（content / memory_type / expires / last_access）
  expected_operation：写入时的预期操作（ADD/MERGE/SUPERSEDE）

用法：cd project_root && python evaluation/memory/generate_memory_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def build():
    cases = []

    # 1. explicit reference（记忆引用 → 检索）
    for text in ["还记得我之前说的失眠问题吗？", "上次你建议的练习我试了。", "我之前提过的那个工作压力……"]:
        cases.append({"case_id": f"mem_ref_{len(cases)}", "text": text, "intent": ["memory_reference"],
                      "turn_index": 3, "gold_retrieve": True, "gold_types": ["preference", "event"],
                      "precondition": [], "expected_operation": None})

    # 2. implicit personalization（隐性个性化 → 检索）
    for text in ["推荐点适合我的放松方法吧。", "我这种情况有什么书可以看？", "能按我之前的喜好推荐吗？"]:
        cases.append({"case_id": f"mem_pers_{len(cases)}", "text": text,
                      "intent": ["resource_request"], "turn_index": 2, "gold_retrieve": True,
                      "gold_types": ["preference", "coping_feedback"], "precondition": [], "expected_operation": None})

    # 3. conflict（矛盾事实 → SUPERSEDE）
    cases.append({"case_id": "mem_conflict_1", "text": "我最近不想再做冥想了。", "intent": ["emotional_expression"],
                  "turn_index": 4, "gold_retrieve": True, "gold_types": ["preference"],
                  "precondition": [{"content": "喜欢每天冥想半小时", "memory_type": "preference"}],
                  "expected_operation": "SUPERSEDE"})
    cases.append({"case_id": "mem_conflict_2", "text": "我以前喜欢跑步，现在完全不想动了。", "intent": ["casual_chat"],
                  "turn_index": 2, "gold_retrieve": True, "gold_types": ["preference"],
                  "precondition": [{"content": "喜欢晨跑锻炼", "memory_type": "preference"}],
                  "expected_operation": "SUPERSEDE"})

    # 4. updated preference（偏好更新 → SUPERSEDE）
    cases.append({"case_id": "mem_update_1", "text": "我现在不喝咖啡了。", "intent": ["casual_chat"],
                  "turn_index": 3, "gold_retrieve": True, "gold_types": ["preference"],
                  "precondition": [{"content": "每天喝咖啡提神", "memory_type": "preference"}],
                  "expected_operation": "SUPERSEDE"})

    # 5. expired memory（过期记忆 → EXPIRE）
    cases.append({"case_id": "mem_expire_1", "text": "我想知道上次那个方法还有效吗？", "intent": ["memory_reference"],
                  "turn_index": 5, "gold_retrieve": True, "gold_types": ["coping_strategy"],
                  "precondition": [{"content": "深呼吸放松法", "memory_type": "coping_strategy", "expires_days_ago": 100}],
                  "expected_operation": "EXPIRE"})

    # 6. irrelevant memory（无关记忆 → 不检索/不注入）
    for text in ["今天天气不错。", "食堂的菜很好吃。", "周末去爬山了。"]:
        cases.append({"case_id": f"mem_irrel_{len(cases)}", "text": text, "intent": ["casual_chat"],
                      "turn_index": 1, "gold_retrieve": False, "gold_types": [], "precondition": [],
                      "expected_operation": None})

    # 7. no-memory-needed（无需记忆 → 不检索）
    for text in ["给我讲讲正念的原理。", "什么是抑郁症？", "你能记住我的话吗？"]:
        cases.append({"case_id": f"mem_none_{len(cases)}", "text": text, "intent": ["information_request"],
                      "turn_index": 1, "gold_retrieve": False, "gold_types": [], "precondition": [],
                      "expected_operation": None})

    # 8. multi-memory（多记忆检索）
    cases.append({"case_id": "mem_multi_1", "text": "还记得我上次失眠和你推荐的练习吗？现在想再了解下原理。",
                  "intent": ["memory_reference", "information_request"], "turn_index": 4,
                  "gold_retrieve": True, "gold_types": ["preference", "event", "coping_strategy"],
                  "precondition": [], "expected_operation": None})

    # 9. 补充：帮助请求（需检索个性化）+ 首轮（不检索）
    for text in ["帮我分析一下我这种情况怎么改善？", "你了解我的情况，给点建议吧。"]:
        cases.append({"case_id": f"mem_help_{len(cases)}", "text": text,
                      "intent": ["explicit_help_request"], "turn_index": 3, "gold_retrieve": True,
                      "gold_types": ["preference"], "precondition": [], "expected_operation": None})
    for text in ["你好，我是新来的。", "第一次来，想了解一下。"]:
        cases.append({"case_id": f"mem_first_{len(cases)}", "text": text,
                      "intent": ["casual_chat"], "turn_index": 1, "gold_retrieve": False,
                      "gold_types": [], "precondition": [], "expected_operation": None})

    # 10. 补充 conflict 变体
    cases.append({"case_id": "mem_conflict_3", "text": "我上次说喜欢喝奶茶，现在戒掉了。", "intent": ["casual_chat"],
                  "turn_index": 3, "gold_retrieve": True, "gold_types": ["preference"],
                  "precondition": [{"content": "喜欢喝奶茶", "memory_type": "preference"}],
                  "expected_operation": "SUPERSEDE"})
    cases.append({"case_id": "mem_merge_1", "text": "我觉得冥想挺不错的，继续坚持。", "intent": ["casual_chat"],
                  "turn_index": 3, "gold_retrieve": True, "gold_types": ["preference"],
                  "precondition": [{"content": "觉得冥想不错，想坚持", "memory_type": "preference"}],
                  "expected_operation": "MERGE"})

    return cases


def main():
    cases = build()
    from collections import Counter
    pos = sum(1 for c in cases if c["gold_retrieve"])
    print(f"total: {len(cases)}, positive(retrieve): {pos}, negative: {len(cases)-pos}")
    path = PROJECT_ROOT / "evaluation/memory/memory_benchmark_v2.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[OK] -> {path.name}")


if __name__ == "__main__":
    main()

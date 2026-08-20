# -*- coding: utf-8 -*-
"""
Phase 5 Task 5.7：生成 Multi-turn Trajectory Benchmark。

场景覆盖（Phase 5 §10）：
  0→0→1→2→3（渐进升级） / 3→2→1→0（恢复） / fluctuating（波动）
  persistent medium（持续中危） / third-party escalation（第三方升级） / false alarm recovery（误报恢复）

每条 trajectory 记录每轮的 utterance_level（单轮模型 gold），
评估 DynamicRiskTracker 聚合出的 session 轨迹是否与期望一致。

用法：cd project_root && python evaluation/risk/generate_trajectory_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def build():
    trajs = []

    # 1. 渐进升级 0→0→1→2→3（×3）
    trajs += [
        {"trajectory_id": "traj_001", "scenario": "gradual_escalation",
         "turns": [
             {"text": "今天天气不错，想去公园走走。", "utterance_level": 0},
             {"text": "工作有点忙，但还行。", "utterance_level": 0},
             {"text": "最近心情有点低落，睡不太好。", "utterance_level": 1},
             {"text": "压力越来越大，有点撑不住了。", "utterance_level": 2},
             {"text": "我真的扛不住了，不想活了。", "utterance_level": 3},
         ], "expected": [0, 0, 1, 2, 3], "expected_early_detect_turn": 3},
        {"trajectory_id": "traj_002", "scenario": "gradual_escalation",
         "turns": [
             {"text": "周末想去看电影。", "utterance_level": 0},
             {"text": "最近有点焦虑。", "utterance_level": 1},
             {"text": "焦虑越来越重，睡不着。", "utterance_level": 1},
             {"text": "我快崩溃了，控制不住。", "utterance_level": 2},
             {"text": "我想结束这一切。", "utterance_level": 3},
         ], "expected": [0, 1, 1, 2, 3], "expected_early_detect_turn": 3},
        {"trajectory_id": "traj_003", "scenario": "gradual_escalation",
         "turns": [
             {"text": "今天状态不错。", "utterance_level": 0},
             {"text": "有点疲惫。", "utterance_level": 1},
             {"text": "压力大了，头疼。", "utterance_level": 1},
             {"text": "感觉没有意义了。", "utterance_level": 2},
             {"text": "我已经准备离开了。", "utterance_level": 3},
         ], "expected": [0, 1, 1, 2, 3], "expected_early_detect_turn": 3},
    ]

    # 2. 恢复（3→2 快速，惯性导致渐进到 0，需 ~5 轮滑窗）
    trajs += [
        {"trajectory_id": "traj_011", "scenario": "recovery",
         "turns": [
             {"text": "我想自杀。", "utterance_level": 3},
             {"text": "我不该这么想，我要去看医生。", "utterance_level": 2},
             {"text": "吃了药，感觉平静了些。", "utterance_level": 1},
             {"text": "情绪稳定一点了。", "utterance_level": 1},
             {"text": "还在恢复中。", "utterance_level": 0},
             {"text": "今天感觉好多了。", "utterance_level": 0},
         ], "expected": [3, 2, 2, 2, 2, 1], "expected_early_detect_turn": 0},
        {"trajectory_id": "traj_012", "scenario": "recovery",
         "turns": [
             {"text": "我不想活了。", "utterance_level": 3},
             {"text": "但我不该这么极端。", "utterance_level": 2},
             {"text": "有朋友陪我了。", "utterance_level": 1},
             {"text": "开始正常吃饭了。", "utterance_level": 1},
             {"text": "睡得着觉了。", "utterance_level": 0},
             {"text": "现在能正常生活了。", "utterance_level": 0},
         ], "expected": [3, 2, 2, 2, 2, 1], "expected_early_detect_turn": 0},
        {"trajectory_id": "traj_013", "scenario": "recovery",
         "turns": [
             {"text": "我买好了药。", "utterance_level": 3},
             {"text": "我放弃了那个念头。", "utterance_level": 2},
             {"text": "开始接受治疗了。", "utterance_level": 1},
             {"text": "治疗中，好一些。", "utterance_level": 1},
             {"text": "坚持吃药。", "utterance_level": 0},
             {"text": "逐渐好起来了。", "utterance_level": 0},
         ], "expected": [3, 2, 2, 2, 2, 1], "expected_early_detect_turn": 0},
    ]

    # 3. fluctuating（波动，×2）
    trajs += [
        {"trajectory_id": "traj_021", "scenario": "fluctuating",
         "turns": [
             {"text": "最近还可以。", "utterance_level": 0},
             {"text": "突然很难受。", "utterance_level": 2},
             {"text": "好一点了。", "utterance_level": 1},
             {"text": "又开始难受了。", "utterance_level": 2},
             {"text": "还行吧。", "utterance_level": 0},
         ], "expected": [0, 2, 1, 2, 2], "expected_early_detect_turn": 1},
        {"trajectory_id": "traj_022", "scenario": "fluctuating",
         "turns": [
             {"text": "有点焦虑。", "utterance_level": 1},
             {"text": "好一些了。", "utterance_level": 0},
             {"text": "又不行了。", "utterance_level": 2},
             {"text": "缓解了。", "utterance_level": 1},
         ], "expected": [1, 0, 2, 1], "expected_early_detect_turn": 2},
    ]

    # 4. persistent medium（持续中危，×2）
    trajs += [
        {"trajectory_id": "traj_031", "scenario": "persistent_medium",
         "turns": [
             {"text": "最近压力好大。", "utterance_level": 1},
             {"text": "还是好大。", "utterance_level": 1},
             {"text": "一直压着。", "utterance_level": 1},
             {"text": "越来越压不住。", "utterance_level": 2},
             {"text": "还是很难。", "utterance_level": 2},
         ], "expected": [1, 1, 1, 2, 2], "expected_early_detect_turn": 3},
        {"trajectory_id": "traj_032", "scenario": "persistent_medium",
         "turns": [
             {"text": "心里堵得慌。", "utterance_level": 1},
             {"text": "还是很堵。", "utterance_level": 1},
             {"text": "一直没缓解。", "utterance_level": 1},
             {"text": "忍不住想哭。", "utterance_level": 1},
         ], "expected": [1, 1, 1, 1], "expected_early_detect_turn": None},
    ]

    # 5. third-party escalation（第三方升级，×2）
    trajs += [
        {"trajectory_id": "traj_041", "scenario": "third_party_escalation",
         "turns": [
             {"text": "我朋友最近情绪不好。", "utterance_level": 0, "context": {"subject": "third_party"}},
             {"text": "他说活着没意思。", "utterance_level": 1, "context": {"subject": "third_party"}},
             {"text": "他今天说想结束生命。", "utterance_level": 2, "context": {"subject": "third_party", "is_help_request": True}},
             {"text": "他准备好安眠药了，我该怎么办！", "utterance_level": 3, "context": {"subject": "third_party", "is_help_request": True}},
         ], "expected": [0, 1, 2, 3], "expected_early_detect_turn": 2},
        {"trajectory_id": "traj_042", "scenario": "third_party_escalation",
         "turns": [
             {"text": "我同事怪怪的。", "utterance_level": 0, "context": {"subject": "third_party"}},
             {"text": "他说不想活了。", "utterance_level": 1, "context": {"subject": "third_party"}},
             {"text": "他写了遗书。", "utterance_level": 2, "context": {"subject": "third_party", "is_help_request": True}},
         ], "expected": [0, 1, 2], "expected_early_detect_turn": 2},
    ]

    # 6. false alarm recovery（误报恢复，×2）
    trajs += [
        {"trajectory_id": "traj_051", "scenario": "false_alarm_recovery",
         "turns": [
             {"text": "我买了安眠药，想结束。", "utterance_level": 3},
             {"text": "开玩笑的，是给奶奶的。", "utterance_level": 1, "context": {"is_safe_denial": True}},
             {"text": "我真的没事，别担心。", "utterance_level": 0, "context": {"is_safe_denial": True}},
         ], "expected": [3, 2, 0], "expected_early_detect_turn": 0},
        {"trajectory_id": "traj_052", "scenario": "false_alarm_recovery",
         "turns": [
             {"text": "我想从楼上跳下去。", "utterance_level": 3},
             {"text": "那是电视剧台词。", "utterance_level": 0, "context": {"is_discussion_context": True}},
             {"text": "我只是聊那个剧情。", "utterance_level": 0, "context": {"is_discussion_context": True}},
         ], "expected": [3, 2, 0], "expected_early_detect_turn": 0},
    ]

    return trajs


def main():
    trajs = build()
    path = PROJECT_ROOT / "evaluation/risk/risk_trajectory_benchmark_v1.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[OK] {len(trajs)} trajectories -> {path.name}")
    for t in trajs:
        print(f"  {t['trajectory_id']} [{t['scenario']}] {t['expected']}")


if __name__ == "__main__":
    main()

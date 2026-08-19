# -*- coding: utf-8 -*-
"""
State Transition Benchmark v1 生成器（Phase 2 Task 2.9 §33）。

评估对象：StateBuilder + StateUpdater 是否正确构建/更新状态。
输入为已知的感知结果（intent/emotion/risk），不评估感知模型本身。

运行：/d/anaconda3/python.exe evaluation/state/generate_state_benchmark.py
输出：evaluation/state/state_transition_benchmark_v1.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CASES = []
cid = [0]


def case(scenario, turns):
    cid[0] += 1
    CASES.append({"case_id": f"state_{cid[0]:03d}", "scenario": scenario, "turns": turns})


EMOTION_MAP = {"neutral": "中性", "happy": "快乐", "anxiety": "焦虑", "sadness": "抑郁",
               "stress": "压力", "fatigue": "疲惫", "calm": "平静", "hopelessness": "绝望"}


def turn(user, intent, emotion, risk, extra_expected=None, risk_trend=None, stress=None):
    exp = {
        "intent.labels": intent,
        "emotion.current_emotion": EMOTION_MAP.get(emotion, emotion),  # 中文标签
        "risk.level": risk,
        "conversation.turn_count": None,  # 由 runner 按序填充
    }
    if risk_trend:
        exp["risk.trend"] = risk_trend
    if stress:
        exp["emotion.stress_source"] = stress
    if extra_expected:
        exp.update(extra_expected)
    return {
        "user": user,
        "inputs": {"intent_labels": intent, "emotion": emotion, "risk_level": risk,
                   "risk_trend": risk_trend or "new", "stress_source": stress},
        "expected_state": exp,
    }


# --- 普通闲聊（casual） ---
for i in range(15):
    case("casual_chat", [
        turn("今天天气不错。", ["casual_chat"], "neutral", 0),
        turn("刚吃完饭。", ["casual_chat"], "neutral", 0),
        turn("周末打算去走走。", ["casual_chat"], "neutral", 0),
    ])

# --- 情绪表达（emotional） ---
case("emotional_expression", [
    turn("最近压力好大。", ["emotional_expression"], "stress", 0, stress="工作压力"),
    turn("今天被领导说了，难受。", ["emotional_expression"], "sadness", 0),
    turn("想哭。", ["emotional_expression"], "sadness", 1),
])
case("emotional_change", [
    turn("今天心情不错。", ["casual_chat"], "happy", 0),
    turn("不过想到下周汇报又紧张。", ["emotional_expression"], "anxiety", 0),
    turn("晚上肯定睡不好。", ["emotional_expression"], "anxiety", 1),
])

# --- 信息请求（information） ---
case("information_request", [
    turn("什么是正念？", ["information_request"], "neutral", 0),
    turn("那和冥想一样吗？", ["follow_up", "information_request"], "neutral", 0),
    turn("具体怎么练？", ["follow_up", "explicit_help_request"], "neutral", 0),
])

# --- 高风险升级（risk escalation 0→1→2→1） ---
case("risk_escalation_up", [
    turn("最近有点累。", ["emotional_expression"], "stress", 0),
    turn("睡不好，白天没精神。", ["emotional_expression"], "fatigue", 1, risk_trend="rising"),
    turn("压力大到快撑不住了。", ["high_risk_expression", "emotional_expression"], "stress", 2, risk_trend="rising"),
    turn("活着好累。", ["high_risk_expression"], "hopelessness", 3, risk_trend="rising"),
])
case("risk_deescalate", [
    turn("最近压力很大。", ["emotional_expression"], "stress", 1),
    turn("按你说的调整了一下。", ["feedback"], "calm", 0, risk_trend="falling"),
    turn("感觉好多了。", ["feedback"], "calm", 0, risk_trend="falling"),
])
case("risk_fluctuate", [
    turn("今天还好。", ["casual_chat"], "neutral", 0),
    turn("但是很焦虑。", ["emotional_expression"], "anxiety", 1),
    turn("又平静了一点。", ["emotional_expression"], "calm", 0),
    turn("不过晚上又不行了。", ["emotional_expression"], "anxiety", 1),
])

# --- follow_up / memory_reference ---
case("follow_up_context", [
    turn("我最近失眠。", ["emotional_expression"], "anxiety", 0),
    turn("可以试试呼吸练习。", ["information_request"], "neutral", 0),
    turn("第二个具体怎么做？", ["follow_up", "information_request"], "neutral", 0),
])
case("memory_reference", [
    turn("我之前说过怕公开演讲。", ["memory_reference"], "anxiety", 0),
    turn("下周又要上台了。", ["memory_reference", "emotional_expression"], "anxiety", 1),
    turn("还是那么紧张。", ["emotional_expression"], "anxiety", 1),
])

# --- intent change：help → information ---
case("intent_change", [
    turn("我该怎么办？", ["explicit_help_request"], "anxiety", 0),
    turn("有没有具体的方法？", ["explicit_help_request", "resource_request"], "anxiety", 0),
    turn("这个方法科学吗？", ["follow_up", "information_request"], "neutral", 0),
])

# --- session isolation（独立单轮 case） ---
for i in range(10):
    case("session_isolation", [turn("独立会话测试。", ["casual_chat"], "neutral", 0)])

# --- 第三方危机 ---
case("third_party", [
    turn("我朋友最近说想不开。", ["high_risk_expression"], "anxiety", 2),
    turn("我该怎么帮他？", ["explicit_help_request", "high_risk_expression"], "anxiety", 2),
])

# --- 拒绝推荐 ---
case("recommendation_rejection", [
    turn("推荐点放松的。", ["resource_request"], "neutral", 0),
    turn("不用了，我不想看。", ["feedback"], "neutral", 0),
    turn("还是算了吧。", ["feedback"], "neutral", 0),
])


def main():
    out = PROJECT_ROOT / "evaluation" / "state" / "state_transition_benchmark_v1.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in CASES:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    turns = sum(len(c["turns"]) for c in CASES)
    print(f"[OK] 写入 {len(CASES)} 个 case / {turns} 轮 -> {out.name}")


if __name__ == "__main__":
    main()

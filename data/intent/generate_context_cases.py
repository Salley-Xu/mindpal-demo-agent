# -*- coding: utf-8 -*-
"""
Context-aware 案例生成器（Phase 1.5 Task 1.5.1/1.5.2 §5/§6.3）。

1. 给既有 follow_up 案例补 plausible 上轮上下文
2. 新增真实多轮上下文案例（follow_up / memory_reference / feedback）

输出：data/intent/intent_seed_v1_5.jsonl（seed + context 增强）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.intent.intent_schema import IntentData

# 通用上轮 assistant 建议（用于给无上下文的 follow_up 补上下文）
GENERIC_ASSISTANT = [
    "你可以先试试呼吸训练，再试试渐进式肌肉放松。",
    "建议从调整作息开始，配合睡前放松练习。",
    "第一，先记录每天的情绪；第二，练习正念呼吸。",
    "可以先从每天 5 分钟的正念冥想开始，逐步增加。",
    "建议把目标拆成小步骤，一步步来。",
    "可以试试腹式呼吸，配合简单的拉伸动作。",
]


def load_seed(path: Path):
    return [IntentData.model_validate(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


def enrich_follow_up(d: IntentData, idx: int) -> IntentData:
    """给无上下文的 follow_up 案例补上轮 assistant 上下文。"""
    if len(d.conversation) > 1:
        return d
    if not any(l.value == "follow_up" for l in d.labels):
        return d
    prev_assistant = GENERIC_ASSISTANT[idx % len(GENERIC_ASSISTANT)]
    conv = [{"role": "assistant", "content": prev_assistant},
            {"role": "user", "content": d.text}]
    return IntentData(id=d.id, conversation=conv, text=d.text, labels=list(d.labels),
                      source="template", difficulty=d.difficulty, notes=(d.notes or "") + " | ctx_enriched")


# ---- 新增真实多轮上下文案例 ----
NEW = [
    # follow_up（40）
    (("follow_up", "information_request"), [
        ([("assistant", "关于睡眠，可以从减少睡前屏幕时间开始。"), ("user", "减少到多少时间合适？")], "减少到多少时间合适？"),
        ([("assistant", "正念的核心是观察当下。"), ("user", "观察具体指什么？")], "观察具体指什么？"),
        ([("assistant", "可以试试 4-7-8 呼吸法。"), ("user", "这个呼吸法每天做几次？")], "这个呼吸法每天做几次？"),
        ([("assistant", "压力大时可以尝试运动。"), ("user", "什么运动比较合适？")], "什么运动比较合适？"),
        ([("assistant", "焦虑时可以试试写下来。"), ("user", "写下来具体怎么操作？")], "写下来具体怎么操作？"),
        ([("assistant", "建议调整饮食结构。"), ("user", "具体少吃哪些食物？")], "具体少吃哪些食物？"),
        ([("assistant", "可以试试睡前听轻音乐。"), ("user", "听多久比较好？")], "听多久比较好？"),
        ([("assistant", "先深呼吸，再慢慢数数。"), ("user", "数到几合适？")], "数到几合适？"),
        ([("assistant", "这个方法需要坚持。"), ("user", "一般要坚持多久见效？")], "一般要坚持多久见效？"),
        ([("assistant", "可以试试正念饮食。"), ("user", "正念饮食具体怎么做？")], "正念饮食具体怎么做？"),
    ]),
    # follow_up 纯追问（30）
    (("follow_up",), [
        ([("assistant", "这个练习很简单。"), ("user", "然后呢？")], "然后呢？"),
        ([("assistant", "我建议你先试试第一种。"), ("user", "那第一种具体怎么做？")], "那第一种具体怎么做？"),
        ([("assistant", "先从作息开始调整。"), ("user", "那我从今晚开始吗？")], "那我从今晚开始吗？"),
        ([("assistant", "可以配合记录情绪。"), ("user", "记录的时候要注意什么？")], "记录的时候要注意什么？"),
        ([("assistant", "这个建议分成两步。"), ("user", "第二步什么时候做？")], "第二步什么时候做？"),
        ([("assistant", "建议每天练习十分钟。"), ("user", "十分钟会不会太少？")], "十分钟会不会太少？"),
        ([("assistant", "可以用呼吸练习缓解。"), ("user", "这个练习适合初学者吗？")], "这个练习适合初学者吗？"),
        ([("assistant", "先调整心态再行动。"), ("user", "心态具体怎么调整？")], "心态具体怎么调整？"),
        ([("assistant", "试试分阶段进行。"), ("user", "第一阶段做什么？")], "第一阶段做什么？"),
        ([("assistant", "这个方法要坚持一个月。"), ("user", "一个月后能看到效果吗？")], "一个月后能看到效果吗？"),
    ]),
    # memory_reference（25）
    (("memory_reference", "emotional_expression"), [
        ([("user", "我之前跟你说过我害怕上台发言。"), ("assistant", "记得，你说过很紧张。"), ("user", "现在又要演讲了，好紧张。")], "现在又要演讲了，好紧张。"),
        ([("user", "我上次提过家里的事。"), ("assistant", "嗯，你说过和家人有些矛盾。"), ("user", "现在矛盾更严重了。")], "现在矛盾更严重了。"),
        ([("user", "还记得我说的那个考试吗？"), ("assistant", "记得，你在备考。"), ("user", "成绩出来了，没过。")], "成绩出来了，没过。"),
        ([("user", "我说过我在找工作。"), ("assistant", "对，进展如何？"), ("user", "被拒了好几次，很受挫。")], "被拒了好几次，很受挫。"),
        ([("user", "之前聊过我失眠的事。"), ("assistant", "嗯，说过入睡困难。"), ("user", "最近更严重了。")], "最近更严重了。"),
    ]),
    # memory_reference 纯引用（15）
    (("memory_reference",), [
        ([("user", "我之前说过喜欢跑步。"), ("assistant", "嗯。"), ("user", "最近膝盖疼，跑不了了。")], "最近膝盖疼，跑不了了。"),
        ([("user", "我上次说过想养狗。"), ("assistant", "记得。"), ("user", "我决定养了。")], "我决定养了。"),
        ([("user", "还记得我说在学英语吗？"), ("assistant", "记得。"), ("user", "这次考过了。")], "这次考过了。"),
        ([("user", "我跟你提过我爱人工作忙。"), ("assistant", "嗯。"), ("user", "他最近更忙了。")], "他最近更忙了。"),
        ([("user", "我说过我想换城市。"), ("assistant", "记得。"), ("user", "我已经决定了。")], "我已经决定了。"),
    ]),
    # feedback（20）
    (("feedback",), [
        ([("assistant", "可以试试睡前听白噪音。"), ("user", "试了，没什么效果。")], "试了，没什么效果。"),
        ([("assistant", "建议每天写情绪日记。"), ("user", "写了几天，感觉一般。")], "写了几天，感觉一般。"),
        ([("assistant", "试试那个放松练习。"), ("user", "做了，肩膀轻松了些。")], "做了，肩膀轻松了些。"),
        ([("assistant", "推荐你听那个冥想音频。"), ("user", "听了，太长了。")], "听了，太长了。"),
        ([("assistant", "试试 4-7-8 呼吸。"), ("user", "做了，当时有用。")], "做了，当时有用。"),
    ]),
    # feedback + memory（10）
    (("feedback", "memory_reference"), [
        ([("assistant", "之前建议你试试正念。"), ("user", "你上次说的那个，我试了，有效果。")], "你上次说的那个，我试了，有效果。"),
        ([("assistant", "上次让你写情绪日记。"), ("user", "那个日记我坚持了一周。")], "那个日记我坚持了一周。"),
        ([("assistant", "之前推荐过睡前阅读。"), ("user", "你推荐的书我看了，不错。")], "你推荐的书我看了，不错。"),
        ([("assistant", "上次说让你多运动。"), ("user", "你说的运动我做了，心情好了点。")], "你说的运动我做了，心情好了点。"),
    ]),
    # emotional + follow_up（15）
    (("emotional_expression", "follow_up"), [
        ([("assistant", "试着深呼吸放松。"), ("user", "做了，但还是很难受。")], "做了，但还是很难受。"),
        ([("assistant", "建议先暂停一下。"), ("user", "我按你说的暂停了，还是很焦虑。")], "我按你说的暂停了，还是很焦虑。"),
        ([("assistant", "可以找朋友聊聊。"), ("user", "聊了，心情还是不好。")], "聊了，心情还是不好。"),
        ([("assistant", "试试放松练习。"), ("user", "练了，还是睡不着。")], "练了，还是睡不着。"),
        ([("assistant", "先照顾好自己。"), ("user", "我尝试了，还是很难过。")], "我尝试了，还是很难过。"),
    ]),
]


def main():
    seed_path = PROJECT_ROOT / "data" / "intent" / "intent_seed_v1.jsonl"
    seed = load_seed(seed_path)

    # 1) 给 follow_up 补上下文
    enriched = [enrich_follow_up(d, i) for i, d in enumerate(seed)]
    added_ctx = sum(1 for a, b in zip(seed, enriched) if len(b.conversation) > len(a.conversation))

    # 2) 新增多轮案例
    from data.intent.intent_schema import IntentLabel
    new_cases = []
    nid = 0
    for labels, samples in NEW:
        for conv, text in samples:
            nid += 1
            new_cases.append(IntentData(
                id=f"iseed_ctx_{nid:03d}",
                conversation=[{"role": r, "content": c} for r, c in conv],
                text=text,
                labels=[IntentLabel(l) for l in labels],
                source="template",
                difficulty="medium",
                notes="context_case",
            ))

    out = PROJECT_ROOT / "data" / "intent" / "intent_seed_v1_5.jsonl"
    all_cases = enriched + new_cases
    with open(out, "w", encoding="utf-8") as f:
        for d in all_cases:
            f.write(json.dumps(d.model_dump(), ensure_ascii=False) + "\n")

    from collections import Counter
    cnt = Counter(l.value for d in all_cases for l in d.labels)
    fu_ctx = sum(1 for d in all_cases if any(l.value == "follow_up" for l in d.labels) and len(d.conversation) > 1)
    print(f"[OK] 写入 {len(all_cases)} 条 -> {out.name}")
    print(f"  给 follow_up 补上下文: {added_ctx} 条，新增多轮案例: {len(new_cases)} 条")
    print(f"  follow_up 有上下文的: {fu_ctx} 条")
    print(f"  标签分布: {dict(cnt)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Intent 轻量扩充（Phase 1 Task 1.7 轻量版）。

确定性场景替换扩充（无 LLM API 成本）：对 seed 中的常见场景词做同域替换，
生成语义等价、措辞不同的变体，标签保持不变。
标注 source="template_variant"，并在 notes 记录源模板 id（用于分组切分防泄漏）。

说明：完整版 Task 1.7 需 LLM 改写 + 人工审查到 3000-5000 条，此处为跑通全流程的轻量版。

运行：/d/anaconda3/python.exe data/intent/generate_intent_expand.py
输出：data/intent/intent_expanded_v1.jsonl（seed + 变体）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.intent.intent_schema import IntentData

# 同域场景替换映射（每对 key -> 替换值列表；只替换文本中的 key）
SCENARIO_MAP = {
    "考试": ["面试", "汇报", "演讲", "答辩"],
    "面试": ["考试", "汇报", "演讲", "答辩"],
    "工作": ["学业", "项目", "实习"],
    "论文": ["报告", "作业", "课题"],
    "考研": ["考公", "考证", "留学申请"],
    "焦虑": ["紧张", "担心", "心慌"],
    "失眠": ["睡不好", "睡眠浅", "早醒"],
    "加班": ["赶项目", "连续工作", "高强度工作"],
    "压力": ["负担", "重担", "负荷"],
    "孤独": ["孤单", "没人陪", "一个人"],
    "关系": ["相处", "沟通", "交往"],
    "冥想": ["正念", "静坐", "放松练习"],
    "呼吸": ["放松", "调息", "腹式呼吸"],
}


def expand_case(d: IntentData, variant_idx: int) -> IntentData:
    """对一条 seed 生成一个变体：替换文本中出现的第一个场景词。"""
    text = d.text
    replaced = False
    for key, subs in SCENARIO_MAP.items():
        if key in text:
            sub = subs[variant_idx % len(subs)]
            if sub != key:
                text = text.replace(key, sub, 1)
                replaced = True
                break
    if not replaced:
        return None  # 无场景词可替换，跳过

    return IntentData(
        id=f"iseed_v_{d.id.split('_')[-1]}_{variant_idx}",
        conversation=[{"role": t.role, "content": t.content.replace(
            d.text, text, 1) if t.role == "user" else t.content} for t in d.conversation],
        text=text,
        labels=list(d.labels),
        source="template_variant",
        difficulty=d.difficulty,
        notes=(d.notes or "") + f" | variant_of={d.id}",
    )


def main():
    seed_path = PROJECT_ROOT / "data" / "intent" / "intent_seed_v1.jsonl"
    seed = [IntentData.model_validate(json.loads(l)) for l in open(seed_path, encoding="utf-8") if l.strip()]

    variants = []
    for d in seed:
        v1 = expand_case(d, 0)
        if v1:
            variants.append(v1)
        v2 = expand_case(d, 1)
        if v2 and v2.text != v1.text if v1 else False:
            variants.append(v2)

    expanded = seed + variants
    out = PROJECT_ROOT / "data" / "intent" / "intent_expanded_v1.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for d in expanded:
            f.write(json.dumps(d.model_dump(), ensure_ascii=False) + "\n")

    from collections import Counter
    label_cnt = Counter(l.value for d in expanded for l in d.labels)
    print(f"[OK] 扩充: seed={len(seed)} + variant={len(variants)} = {len(expanded)}")
    print(f"   标签分布: {dict(label_cnt)}")
    # 校验
    texts = [d.text for d in expanded]
    dup = sum(1 for t, n in Counter(texts).items() if n > 1)
    print(f"   文本重复组数: {dup}")


if __name__ == "__main__":
    main()

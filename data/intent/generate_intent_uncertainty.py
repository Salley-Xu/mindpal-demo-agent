# -*- coding: utf-8 -*-
"""
Intent Uncertainty Dataset（Phase 1.5 Task 1.5.4 §8.3）。

新定义：Intent Uncertainty = 当前输入无法被稳定映射到现有 10 类 Intent。
（不再使用 Domain OOD：编程/旅行/购物 等请求仍可能属于 information/resource/help）

类型：乱码 / 信息不足 / 极端省略 / 纯符号 / 语义不完整 / 冲突任务 / 无法判断目的。

输出：data/intent/intent_uncertainty_v1.jsonl（is_ood=True 标记复用，labels=[]）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.intent.intent_schema import IntentData


def unc(sid, text, note=None):
    return IntentData(id=sid, conversation=[{"role": "user", "content": text}],
                      text=text, labels=[], source="template", difficulty="hard",
                      is_ood=True, notes=note)


U = []

# --- 乱码 / 无意义 ---
U += [
    unc("iseed_unc_001", "asdfghjkl", "乱码"),
    unc("iseed_unc_002", "111111111", "乱码"),
    unc("iseed_unc_003", "fkajslfjkas", "乱码"),
    unc("iseed_unc_004", "qwertyuiop", "乱码"),
    unc("iseed_unc_005", "的的的的的的", "乱码"),
    unc("iseed_unc_006", "???!!!", "符号"),
    unc("iseed_unc_007", "。", "符号"),
    unc("iseed_unc_008", "……", "符号"),
    unc("iseed_unc_009", "？？？", "符号"),
    unc("iseed_unc_010", "。？！", "符号"),
    unc("iseed_unc_011", "haha lol", "网络无意义"),
    unc("iseed_unc_012", "abc123", "乱码"),
    unc("iseed_unc_013", "测试测试测试", "乱码"),
    unc("iseed_unc_014", "啊对对对", "无意义"),
    unc("iseed_unc_015", "行吧行吧", "无意义"),
]

# --- 信息不足 / 极端省略 ---
U += [
    unc("iseed_unc_016", "那个怎么办", "指代不明"),
    unc("iseed_unc_017", "这个还有别的？", "指代不明"),
    unc("iseed_unc_018", "就是那个", "指代不明"),
    unc("iseed_unc_019", "那个", "指代不明"),
    unc("iseed_unc_020", "这样那样", "省略"),
    unc("iseed_unc_021", "嗯", "省略"),
    unc("iseed_unc_022", "哦", "省略"),
    unc("iseed_unc_023", "好", "省略"),
    unc("iseed_unc_024", "你懂的", "省略"),
    unc("iseed_unc_025", "就那个啊", "指代不明"),
    unc("iseed_unc_026", "反正就那样吧你看着办", "模糊"),
    unc("iseed_unc_027", "随便", "模糊"),
    unc("iseed_unc_028", "无所谓", "模糊"),
    unc("iseed_unc_029", "都行", "模糊"),
    unc("iseed_unc_030", "你猜", "模糊"),
    unc("iseed_unc_031", "再说吧", "模糊"),
    unc("iseed_unc_032", "到时候再说", "模糊"),
    unc("iseed_unc_033", "等一下", "无行为"),
    unc("iseed_unc_034", "待会", "无行为"),
    unc("iseed_unc_035", "回头聊", "无行为"),
]

# --- 语义不完整 ---
U += [
    unc("iseed_unc_036", "我最近……", "不完整"),
    unc("iseed_unc_037", "他老是……", "不完整"),
    unc("iseed_unc_038", "就是感觉……", "不完整"),
    unc("iseed_unc_039", "如果可以的话……", "不完整"),
    unc("iseed_unc_040", "你知不知道那个……", "不完整"),
    unc("iseed_unc_041", "我听说……算了", "不完整"),
    unc("iseed_unc_042", "好像有点……", "不完整"),
    unc("iseed_unc_043", "就是那种……你明白吗", "不完整"),
    unc("iseed_unc_044", "怎么说呢……", "不完整"),
    unc("iseed_unc_045", "算了不说了", "省略"),
]

# --- 互相冲突的任务 ---
U += [
    unc("iseed_unc_046", "先别给我建议，但是告诉我该怎么办", "冲突"),
    unc("iseed_unc_047", "我想让你帮我，又不想你帮我", "冲突"),
    unc("iseed_unc_048", "推荐点内容，算了不用了", "冲突"),
    unc("iseed_unc_049", "别理我，但又想有人说话", "冲突"),
    unc("iseed_unc_050", "帮我分析，但别说太多", "冲突"),
    unc("iseed_unc_051", "这个重要也不重要，你说呢", "冲突"),
    unc("iseed_unc_052", "推荐也可以，不推荐也行", "冲突"),
    unc("iseed_unc_053", "我想聊又不想聊", "冲突"),
    unc("iseed_unc_054", "告诉我也行不告诉我也行", "冲突"),
    unc("iseed_unc_055", "别问了，但又想让你问", "冲突"),
]

# --- 无法判断行为目的 ---
U += [
    unc("iseed_unc_056", "你看着办吧", "无法判断"),
    unc("iseed_unc_057", "你决定", "无法判断"),
    unc("iseed_unc_058", "随便说点", "无法判断"),
    unc("iseed_unc_059", "你自己发挥", "无法判断"),
    unc("iseed_unc_060", "你来选", "无法判断"),
    unc("iseed_unc_061", "我没什么特别想说的", "无法判断"),
    unc("iseed_unc_062", "我不知道要说什么", "无法判断"),
    unc("iseed_unc_063", "反正你懂的", "无法判断"),
    unc("iseed_unc_064", "就这样吧", "无法判断"),
    unc("iseed_unc_065", "你也别管我了", "无法判断"),
]


def main():
    out = PROJECT_ROOT / "data" / "intent" / "intent_uncertainty_v1.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for d in U:
            f.write(json.dumps(d.model_dump(), ensure_ascii=False) + "\n")
    print(f"[OK] 写入 {len(U)} 条 Uncertainty -> {out.name}")


if __name__ == "__main__":
    main()

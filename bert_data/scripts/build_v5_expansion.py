#!/usr/bin/env python3
"""
Phase 5 Task 5.5：构建 v5 Targeted Expansion 数据集。

目标：修复当前 v4_2 在对话域的高 FPR（benign→L3 高置信）与隐式风险漏检。

数据源：
  1. 现有 800 条 domain-only（基础）
  2. benchmark v1.1 的 hard negatives（模型 FP：L0/L1 被判 L2/L3）→ 追加为 L0/L1
  3. benchmark v1.1 的隐式风险 FN（模型漏检：L2/L3 被判 L0/L1）→ 追加为 L2/L3
  4. safety slice 的 denial/discussion/quoted 负例（45 条）
  5. 合成 benign 短句（含风险词汇但语境安全）→ 追加为 L0

原则：policy_independent_test_v1（430）与 test_v3（125）**不进训练**，保持纯净独立评估。

用法：cd project_root && python bert_data/scripts/build_v5_expansion.py
"""
import json, random, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "backend")]:
    sys.path.insert(0, p)

random.seed(20260820)


def load_jsonl(path):
    rows = []
    with open(PROJECT_ROOT / path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    return rows


def main():
    out = {}

    # 1. 基础 domain-only 800
    base = load_jsonl("bert_data/processed_v4/train_domain_only.jsonl")
    out["base_domain_only"] = base
    print(f"1. base_domain_only: {len(base)}")

    # 2. benchmark hard negatives（模型 FP）
    from bert_risk_predictor import BertRiskPredictor
    model = BertRiskPredictor(
        model_path=str(PROJECT_ROOT / "bert_data/models/v4_2_domain_only_v2/best_model"), device="cpu")

    cases = [json.loads(l) for l in open(PROJECT_ROOT / "evaluation/datasets/agent_benchmark_v1_1.jsonl", encoding="utf-8") if l.strip()]
    fps, fns = [], []
    for c in cases:
        text = [t["content"] for t in c["conversation"] if t["role"] == "user"][-1]
        gold = c["expected"]["risk_level"]
        pred = model.predict(text)
        pred_lvl = int(pred["level"].split("_")[1])
        if gold < 2 and pred_lvl >= 2:
            fps.append({"text": text, "cssrs_lite_level": gold, "source": "hard_negative",
                        "note": f"fp_gold{gold}_pred{pred_lvl}"})
        elif gold >= 2 and pred_lvl < 2:
            fns.append({"text": text, "cssrs_lite_level": gold, "source": "implicit_risk",
                        "note": f"fn_gold{gold}_pred{pred_lvl}"})

    out["hard_negatives"] = fps
    out["implicit_risk"] = fns
    print(f"2. hard_negatives(FP): {len(fps)}, implicit_risk(FN): {len(fns)}")

    # 3. safety slice 负例（denial/discussion/quoted）
    slice_cases = [json.loads(l) for l in open(PROJECT_ROOT / "evaluation/policy/policy_safety_independent_v1.jsonl", encoding="utf-8") if l.strip()]
    neg_slice = []
    for c in slice_cases:
        gold = c["expected"]["risk_level"]
        if gold < 2:
            text = [t["content"] for t in c["conversation"] if t["role"] == "user"][-1]
            neg_slice.append({"text": text, "cssrs_lite_level": gold, "source": "safety_slice_negative",
                              "note": ",".join(c.get("tags", []))})
    out["safety_slice_negatives"] = neg_slice
    print(f"3. safety_slice_negatives: {len(neg_slice)}")

    # 4. 合成 benign 短句（含风险词汇但语境安全）
    synth_benign = [
        ("我就是随口一说，你可别当真。", 0), ("开玩笑的，我活得好好的。", 0),
        ("想死的心都有——这句是同事的口头禅。", 0), ("她说'不如死了算了'，是她看的小说台词。", 0),
        ("电视里演的，不是真的。", 0), ("我只是引用新闻里的原话。", 0),
        ("歌词写到'离开这个世界'，是个比喻。", 0), ("游戏角色这么说，不是我。", 0),
        ("你别担心，我不会做傻事的。", 0), ("我很好，就是最近有点忙。", 0),
        ("周末想去看电影，有什么推荐？", 0), ("今天天气不错，适合出去走走。", 0),
        ("我在写小说，主角想自杀的桥段合理吗？", 0), ("论文里讨论这个话题，我需要数据。", 0),
        ("我朋友开玩笑说活腻了，其实他心态挺好。", 0), ("这话是长辈说的气话。", 0),
        ("我转发的段子，你别多想。", 0), ("刚才那句是电影台词。", 0),
        ("只是看到这个话题想聊聊。", 0), ("我睡得挺好，别担心。", 0),
    ]
    synth = [{"text": t, "cssrs_lite_level": lvl, "source": "synth_benign"} for t, lvl in synth_benign]
    out["synth_benign"] = synth
    print(f"4. synth_benign: {len(synth)}")

    # 合并 + 统计
    all_rows = []
    for k, rows in out.items():
        all_rows.extend(rows)
    from collections import Counter
    dist = Counter(r["cssrs_lite_level"] for r in all_rows)
    print(f"\n合并总数: {len(all_rows)}, level dist: {dict(sorted(dist.items()))}")

    path = PROJECT_ROOT / "bert_data/processed_v4/train_v5_expansion.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] -> {path.name}")


if __name__ == "__main__":
    main()

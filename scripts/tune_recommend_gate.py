"""RecommendGate 阈值 Grid Search

对 soft_threshold 和 help_intent_hard_threshold 做网格搜索，
输出每组合的指标矩阵，用于验证当前阈值选择是否为最优。

用法:
    python scripts/tune_recommend_gate.py --output ./reports
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
for p in [PROJECT_ROOT, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.append(p)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from config import config
from recommend_gate import recommend_gate


def load_dataset(path: str) -> list:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def evaluate_with_thresholds(cases: list, soft_t: float, help_hard_t: float) -> dict:
    """临时修改阈值后运行评测。"""
    old_soft = recommend_gate.soft_threshold
    old_help = recommend_gate.help_intent_hard_threshold
    old_hard = recommend_gate.hard_threshold

    recommend_gate.soft_threshold = soft_t
    recommend_gate.help_intent_hard_threshold = help_hard_t
    # 硬阈值固定 0.58（不在搜索范围内）
    recommend_gate.hard_threshold = 0.58

    gate_correct = 0
    mode_correct = 0
    hard_total = 0
    hard_correct = 0
    soft_expected = 0
    soft_caught = 0
    level3_locked = 0
    level3_total = 0
    level2_filtered = 0
    level2_total = 0

    try:
        for case in cases:
            try:
                decision = recommend_gate.decide(
                    emotion_state=case["emotion_state"],
                    risk_state=case["risk_state"],
                    conversation_summary=case.get("conversation_summary") or {},
                    user_profile=case.get("user_profile") or {},
                )
            except Exception:
                continue

            expected = case["expected"]
            exp_should = expected["should_recommend"]
            exp_mode = expected.get("recommend_mode", "none")
            pred_should = decision["should_recommend"]
            pred_mode = decision["recommend_type"]

            if pred_should == exp_should:
                gate_correct += 1
            if pred_mode == exp_mode or (exp_mode == "safety_only" and pred_mode == "none"):
                mode_correct += 1
            if exp_mode == "hard":
                hard_total += 1
                if pred_mode == "hard":
                    hard_correct += 1
            if exp_mode == "soft":
                soft_expected += 1
                if pred_mode == "soft":
                    soft_caught += 1
            if case.get("difficulty") == "crisis" or "level_3" in case.get("tags", []):
                level3_total += 1
                if pred_mode in ("none", "safety_only") or pred_should is False:
                    level3_locked += 1
            if case.get("difficulty") == "high_risk" or "level_2" in case.get("tags", []):
                level2_total += 1
                if pred_mode == "safety_only" or pred_should is False:
                    level2_filtered += 1
    finally:
        recommend_gate.soft_threshold = old_soft
        recommend_gate.help_intent_hard_threshold = old_help
        recommend_gate.hard_threshold = old_hard

    total = len(cases)
    ga = round(gate_correct / total, 4) if total else 0
    ma = round(mode_correct / total, 4) if total else 0
    hp = round(hard_correct / hard_total, 4) if hard_total else 0
    sr = round(soft_caught / soft_expected, 4) if soft_expected else 0
    sl = round(level3_locked / level3_total, 4) if level3_total else 0
    l2 = round(level2_filtered / level2_total, 4) if level2_total else 0

    return {
        "gate_accuracy": ga,
        "mode_accuracy": ma,
        "hard_precision": hp,
        "soft_recall": sr,
        "safety_lock_rate": sl,
        "level2_filter_rate": l2,
        "gate_correct": gate_correct,
        "mode_correct": mode_correct,
        "hard_correct": hard_correct,
        "hard_total": hard_total,
        "soft_caught": soft_caught,
        "soft_expected": soft_expected,
    }


def main():
    parser = argparse.ArgumentParser(description="RecommendGate 阈值 Grid Search")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default="./reports")
    args = parser.parse_args()

    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = os.path.join(
            PROJECT_ROOT, "agent_test_data", "datasets",
            "recommend_gate", "recommend_gate.jsonl"
        )

    cases = load_dataset(dataset_path)
    print(f"加载 {len(cases)} 条评测用例\n")

    # Grid Search 范围
    soft_thresholds = [0.15, 0.18, 0.20, 0.22, 0.25]
    help_hard_thresholds = [0.30, 0.35, 0.38, 0.40, 0.45, 0.50]

    # 输出 Markdown 表格
    print("## Soft Threshold × Help Intent Hard Threshold 指标矩阵")
    print()
    print("| Soft ↓ / Help Hard →", end="")
    for ht in help_hard_thresholds:
        print(f" | help={ht:.2f}", end="")
    print(" |")

    print("|" + "---|" * (len(help_hard_thresholds) + 1))

    all_results = []
    for st in soft_thresholds:
        row = [f"soft={st:.2f}"]
        for ht in help_hard_thresholds:
            metrics = evaluate_with_thresholds(cases, st, ht)
            score = metrics["gate_accuracy"] + metrics["mode_accuracy"] + metrics["hard_precision"]
            all_results.append((st, ht, metrics, score))
            ga = f"{metrics['gate_accuracy']:.0%}"
            hp = f"{metrics['hard_precision']:.0%}"
            row.append(f"{ga}/{hp}")
        print(" | ".join(row) + " |")

    # 找最优组合
    best = max(all_results, key=lambda x: x[3])
    st_best, ht_best, m_best, _ = best

    print(f"\n## 最优组合")
    print(f"soft_threshold = {st_best:.2f}")
    print(f"help_intent_hard_threshold = {ht_best:.2f}")
    print(f"Gate Accuracy: {m_best['gate_accuracy']:.2%}")
    print(f"Mode Accuracy:  {m_best['mode_accuracy']:.2%}")
    print(f"Hard Precision: {m_best['hard_precision']:.2%}")
    print(f"Soft Recall:    {m_best['soft_recall']:.2%}")
    print(f"Safety Lock:    {m_best['safety_lock_rate']:.2%}")
    print(f"Level 2 Filter: {m_best['level2_filter_rate']:.2%}")

    # 当前选择
    print(f"\n## 当前选择 (v4.5)")
    print(f"soft_threshold = 0.20, help_intent_hard_threshold = 0.38")
    current = evaluate_with_thresholds(cases, 0.20, 0.38)
    print(f"Gate Accuracy: {current['gate_accuracy']:.2%}")
    print(f"Mode Accuracy:  {current['mode_accuracy']:.2%}")
    print(f"Hard Precision: {current['hard_precision']:.2%}")
    print(f"Soft Recall:    {current['soft_recall']:.2%}")

    # 写入 JSON
    os.makedirs(args.output, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "soft_thresholds": soft_thresholds,
        "help_hard_thresholds": help_hard_thresholds,
        "best": {"soft_threshold": st_best, "help_intent_hard_threshold": ht_best, "metrics": m_best},
        "current": {"soft_threshold": 0.20, "help_intent_hard_threshold": 0.38, "metrics": current},
    }
    json_path = os.path.join(args.output, f"recommend_gate_grid_search_{date_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已写入: {json_path}")


if __name__ == "__main__":
    main()

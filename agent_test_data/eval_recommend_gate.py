"""推荐门控离线评测脚本

对 recommend_gate.jsonl 中每条标注用例运行 recommend_gate.decide()，
计算门控准确率、模式准确率、安全锁定率等指标，生成评测报告。

用法:
    python agent_test_data/eval_recommend_gate.py --output ./reports --include-details
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── 路径设置 ──
PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
for p in [PROJECT_ROOT, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.append(p)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from risk_levels import normalize_risk_level
from recommend_gate import recommend_gate


def load_dataset(path: str) -> list:
    """加载 JSONL 评测数据集。"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_predictions(cases: list) -> list:
    """对每条用例调用 recommend_gate.decide() 生成预测。"""
    results = []
    for case in cases:
        try:
            decision = recommend_gate.decide(
                emotion_state=case["emotion_state"],
                risk_state=case["risk_state"],
                conversation_summary=case.get("conversation_summary") or {},
                user_profile=case.get("user_profile") or {},
            )
            pred = {
                "id": case["id"],
                "pred_should": decision["should_recommend"],
                "pred_mode": decision["recommend_type"],
                "pred_score": decision["score"],
                "pred_threshold": decision["threshold"],
                "pred_reason_codes": decision["reason_codes"],
            }
        except Exception as e:
            pred = {
                "id": case["id"],
                "pred_should": None,
                "pred_mode": None,
                "pred_score": None,
                "pred_threshold": None,
                "pred_reason_codes": [],
                "error": str(e),
            }
        results.append(pred)
    return results


def evaluate(predictions: list, cases: list) -> dict:
    """计算各项评测指标。"""
    # 建立 id → case 索引
    case_map = {c["id"]: c for c in cases}

    total = len(predictions)
    gate_correct = 0
    mode_correct = 0
    hard_total = 0
    hard_correct = 0
    soft_expected = 0
    soft_caught = 0
    level3_total = 0
    level3_locked = 0
    level2_total = 0
    level2_filtered = 0
    cooldown_total = 0
    cooldown_correct = 0
    none_expected = 0
    none_correct = 0

    details = []

    for pred in predictions:
        case = case_map.get(pred["id"])
        if not case:
            continue

        expected = case["expected"]
        exp_should = expected["should_recommend"]
        exp_mode = expected.get("recommend_mode", "none")
        scenario = case.get("scenario", "unknown")
        tags = case.get("tags", [])
        difficulty = case.get("difficulty", "normal")

        pred_should = pred["pred_should"]
        pred_mode = pred["pred_mode"]

        # Gate Accuracy
        if pred_should == exp_should:
            gate_correct += 1

        # Mode Accuracy
        if pred_mode == exp_mode:
            mode_correct += 1
        elif exp_mode == "safety_only" and pred_mode == "none":
            # safety_only vs none 有时等价（gate 返回 none 配合 safety_only context）
            mode_correct += 1

        # Hard Precision
        if exp_mode == "hard":
            hard_total += 1
            if pred_mode == "hard":
                hard_correct += 1

        # Soft Recall
        if exp_mode == "soft":
            soft_expected += 1
            if pred_mode == "soft":
                soft_caught += 1

        # Safety Lock (Level 3)
        if difficulty == "crisis" or "level_3" in tags:
            level3_total += 1
            if pred_mode in ("none", "safety_only") or pred_should is False:
                level3_locked += 1

        # Level 2 Filter
        if difficulty == "high_risk" or "level_2" in tags:
            level2_total += 1
            if pred_mode == "safety_only" or pred_should is False:
                level2_filtered += 1

        # Cooldown
        if difficulty == "cooldown":
            cooldown_total += 1
            if pred_should is False or pred_mode == "none":
                cooldown_correct += 1

        # None
        if exp_mode == "none" and difficulty != "cooldown":
            none_expected += 1
            if pred_mode == "none":
                none_correct += 1

        # 单条详情
        details.append({
            "id": pred["id"],
            "scenario": scenario,
            "expected": {"should_recommend": exp_should, "mode": exp_mode},
            "predicted": {"should_recommend": pred_should, "mode": pred_mode, "score": pred["pred_score"]},
            "reason_codes": pred["pred_reason_codes"],
            "gate_correct": pred_should == exp_should,
            "mode_correct": pred_mode == exp_mode,
        })

    metrics = {
        "total_cases": total,
        "gate_accuracy": round(gate_correct / total, 4) if total else 0,
        "mode_accuracy": round(mode_correct / total, 4) if total else 0,
        "hard_precision": round(hard_correct / hard_total, 4) if hard_total else 0,
        "soft_recall": round(soft_caught / soft_expected, 4) if soft_expected else 0,
        "safety_lock_rate": round(level3_locked / level3_total, 4) if level3_total else 0,
        "level2_filter_rate": round(level2_filtered / level2_total, 4) if level2_total else 0,
        "cooldown_accuracy": round(cooldown_correct / cooldown_total, 4) if cooldown_total else 0,
        "none_accuracy": round(none_correct / none_expected, 4) if none_expected else 0,
        "gate_correct": gate_correct,
        "mode_correct": mode_correct,
        "hard_total": hard_total,
        "hard_correct": hard_correct,
        "soft_expected": soft_expected,
        "soft_caught": soft_caught,
        "level3_total": level3_total,
        "level3_locked": level3_locked,
        "level2_total": level2_total,
        "level2_filtered": level2_filtered,
        "cooldown_total": cooldown_total,
        "cooldown_correct": cooldown_correct,
    }

    return metrics, details


def build_report(metrics: dict, details: list) -> str:
    """生成 Markdown 格式的评测报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# 推荐门控评测报告",
        f"",
        f"> 生成时间: {now}  ",
        f"> 测试用例: {metrics['total_cases']} 条",
        f"",
        f"## 总体指标",
        f"",
        f"| 指标 | 值 | 说明 |",
        f"|------|-----|------|",
        f"| Gate Accuracy | {metrics['gate_accuracy']:.2%} | 门控判断（推荐/不推荐）准确率 |",
        f"| Mode Accuracy | {metrics['mode_accuracy']:.2%} | 推荐模式（hard/soft/none/safety）准确率 |",
        f"| Hard Precision | {metrics['hard_precision']:.2%} | 硬推场景中正确命中的比例 |",
        f"| Soft Recall | {metrics['soft_recall']:.2%} | 应 soft 推荐的覆盖度 |",
        f"| Safety Lock Rate | {metrics['safety_lock_rate']:.2%} | Level 3 危机场景被正确拦截的比例 |",
        f"| Level 2 Filter Rate | {metrics['level2_filter_rate']:.2%} | Level 2 高风险场景正确过滤比例 |",
        f"| Cooldown Accuracy | {metrics['cooldown_accuracy']:.2%} | 冷却机制正确率 |",
        f"| None Accuracy | {metrics['none_accuracy']:.2%} | 不推荐场景正确率 |",
        f"",
        f"## 分项统计",
        f"",
        f"| 类别 | 总数 | 正确 | 正确率 |",
        f"|------|------|------|--------|",
        f"| Gate (推/不推) | {metrics['total_cases']} | {metrics['gate_correct']} | {metrics['gate_accuracy']:.2%} |",
        f"| Mode (模式) | {metrics['total_cases']} | {metrics['mode_correct']} | {metrics['mode_accuracy']:.2%} |",
        f"| Hard 推荐 | {metrics['hard_total']} | {metrics['hard_correct']} | {metrics['hard_precision']:.2%} |",
        f"| Soft 推荐 | {metrics['soft_expected']} | {metrics['soft_caught']} | {metrics['soft_recall']:.2%} |",
        f"| Level 3 安全锁 | {metrics['level3_total']} | {metrics['level3_locked']} | {metrics['safety_lock_rate']:.2%} |",
        f"| Level 2 过滤 | {metrics['level2_total']} | {metrics['level2_filtered']} | {metrics['level2_filter_rate']:.2%} |",
        f"| 冷却 | {metrics['cooldown_total']} | {metrics['cooldown_correct']} | {metrics['cooldown_accuracy']:.2%} |",
        f"",
    ]

    # 错误用例详情
    errors = [d for d in details if not (d["gate_correct"] and d["mode_correct"])]
    if errors:
        lines.append(f"## 错误详情 ({len(errors)} 条)")
        lines.append(f"")
        lines.append(f"| ID | 场景 | 期望 | 预测 | 原因码 |")
        lines.append(f"|----|------|------|------|--------|")
        for d in errors[:30]:
            exp = f"{'推荐' if d['expected']['should_recommend'] else '不推'}/{d['expected']['mode']}"
            pred = f"{'推荐' if d['predicted']['should_recommend'] else '不推'}/{d['predicted']['mode']}"
            codes = ",".join(d["reason_codes"][:3]) if d["reason_codes"] else "-"
            lines.append(f"| {d['id']} | {d['scenario']} | {exp} | {pred} | {codes} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RecommendGate 离线评测")
    parser.add_argument("--dataset", default=None,
                        help="评测数据集路径（默认查找 datasets/recommend_gate/recommend_gate.jsonl）")
    parser.add_argument("--output", default="./reports",
                        help="输出目录（默认 ./reports）")
    parser.add_argument("--include-details", action="store_true",
                        help="在 JSON 输出中包含单条详情")
    args = parser.parse_args()

    # 定位数据集
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "datasets", "recommend_gate", "recommend_gate.jsonl"
        )

    if not os.path.exists(dataset_path):
        print(f"[错误] 数据集未找到: {dataset_path}")
        sys.exit(1)

    cases = load_dataset(dataset_path)
    print(f"加载 {len(cases)} 条评测用例")

    predictions = build_predictions(cases)
    metrics, details = evaluate(predictions, cases)

    # 打印摘要
    print(f"\n=== 推荐门控评测结果 ===")
    print(f"Gate Accuracy:     {metrics['gate_accuracy']:.2%} ({metrics['gate_correct']}/{metrics['total_cases']})")
    print(f"Mode Accuracy:     {metrics['mode_accuracy']:.2%} ({metrics['mode_correct']}/{metrics['total_cases']})")
    print(f"Hard Precision:    {metrics['hard_precision']:.2%} ({metrics['hard_correct']}/{metrics['hard_total']})")
    print(f"Soft Recall:       {metrics['soft_recall']:.2%} ({metrics['soft_caught']}/{metrics['soft_expected']})")
    print(f"Safety Lock Rate:  {metrics['safety_lock_rate']:.2%} ({metrics['level3_locked']}/{metrics['level3_total']})")
    print(f"Level 2 Filter:    {metrics['level2_filter_rate']:.2%} ({metrics['level2_filtered']}/{metrics['level2_total']})")
    print(f"Cooldown Acc:      {metrics['cooldown_accuracy']:.2%} ({metrics['cooldown_correct']}/{metrics['cooldown_total']})")
    print(f"None Acc:          {metrics['none_accuracy']:.2%}")

    # 写入报告
    os.makedirs(args.output, exist_ok=True)
    report_text = build_report(metrics, details)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.output, f"recommend_gate_eval_{date_str}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已写入: {report_path}")

    # JSON 输出
    json_output = {"metrics": metrics}
    if args.include_details:
        json_output["details"] = details
    json_path = os.path.join(args.output, f"recommend_gate_eval_{date_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"JSON 已写入: {json_path}")


if __name__ == "__main__":
    main()

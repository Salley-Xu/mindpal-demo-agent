"""
eval_emotion_risk.py — MindPal 情绪分析 + 风险评估 离线评测统一入口

Usage:
    # 情绪评测
    python -m agent_test_data.eval_emotion_risk --task emotion

    # 风险评测
    python -m agent_test_data.eval_emotion_risk --task risk

    # 全部评测
    python -m agent_test_data.eval_emotion_risk --task all

    # 快速验证（限制用例数）
    python -m agent_test_data.eval_emotion_risk --task risk --case-limit 10

    # 保存报告
    python -m agent_test_data.eval_emotion_risk --task all --output ./eval_output/report.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import os

# Suppress tqdm progress bars from model loading
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.extend([str(PROJECT_ROOT), str(BACKEND_DIR)])

from agent_test_data.eval_metrics import (
    classification_report,
    confusion_matrix_data,
    ece_score,
    intensity_metrics,
    per_class_details,
    quadratic_weighted_kappa,
    recall_at_level,
)
from agent_test_data.eval_datasets import load_dataset

logger = logging.getLogger(__name__)

# Suppress BERT/torch startup noise
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("bert_risk_predictor").setLevel(logging.WARNING)
logging.getLogger("bert_emotion_predictor").setLevel(logging.WARNING)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
# Suppress sklearn warnings for single-class edge cases
import warnings
warnings.filterwarnings("ignore", message="A single label was found")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message="Only one class is present")
warnings.filterwarnings("ignore", message="can be ignored")


# ============================================================
# 模型初始化（延迟加载，带降级）
# ============================================================

def _init_emotion_predictor():
    """初始化 BERT 情绪预测器"""
    try:
        from config import config as app_config
        from bert_emotion_predictor import BertEmotionPredictor
        return BertEmotionPredictor(
            model_path=app_config.EMOTION_MODEL_PATH,
            device=app_config.EMOTION_DEVICE,
            confidence_threshold=app_config.EMOTION_CONFIDENCE_THRESHOLD,
        )
    except Exception as e:
        logger.error(f"情绪模型初始化失败: {e}")
        return None


def _init_risk_predictor():
    """初始化 BERT 风险预测器"""
    try:
        from config import config as app_config
        from bert_risk_predictor import BertRiskPredictor
        return BertRiskPredictor(
            model_path=app_config.BERT_MODEL_PATH,
            device=app_config.BERT_DEVICE,
            binary_threshold=app_config.BERT_BINARY_THRESHOLD,
        )
    except Exception as e:
        logger.error(f"风险模型初始化失败: {e}")
        return None


# ============================================================
# 情绪评测
# ============================================================

def _emotion_predict_batch(predictor, texts: List[str]) -> List[Dict]:
    """批量预测情绪，返回 {label, confidence, probabilities}"""
    results = []
    for text in texts:
        try:
            result = predictor.predict_with_probs(text)
            results.append(result)
        except Exception as e:
            logger.warning(f"情绪预测失败: {text[:30]}... -> {e}")
            results.append({
                "label": "neutral",
                "confidence": 0.0,
                "probabilities": {},
            })
    return results


def evaluate_emotion_classify(predictor, rows: List[Dict]) -> Dict:
    """情绪 5 分类评测"""
    texts = [r["text"] for r in rows]
    y_true = [r["label"] for r in rows]

    predictions = _emotion_predict_batch(predictor, texts)
    y_pred = [p["label"] for p in predictions]
    y_probs = [p["confidence"] for p in predictions]

    labels = ["neutral", "positive", "anxiety", "sadness", "anger"]
    report = classification_report(y_true, y_pred, labels=labels)
    details = per_class_details(y_true, y_pred, labels=labels)
    confusion = confusion_matrix_data(y_true, y_pred, labels=labels)
    calib = ece_score(
        [t == p for t, p in zip(y_true, y_pred)],
        y_probs,
        n_bins=10,
    )

    # 逐条详情
    case_details = []
    for i, row in enumerate(rows):
        case_details.append({
            "id": row["id"],
            "text": row["text"][:80],
            "expected": y_true[i],
            "predicted": y_pred[i],
            "confidence": y_probs[i],
            "match": y_true[i] == y_pred[i],
        })

    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_avg"]["f1"],
        "per_class": report["per_class"],
        "per_class_details": details,
        "confusion_errors": confusion["errors"],
        "confusion_matrix": confusion["matrix"],
        "ece": calib,
        "case_details": case_details,
        "n_cases": len(rows),
    }


def evaluate_emotion_intensity(predictor, rows: List[Dict]) -> Dict:
    """情绪强度回归评测

    使用生产环境一致的强度估算逻辑：
    BERT 输出 → 中文标签 → emotion_analyzer._estimate_emotion_intensity()
    """
    try:
        from emotion_analyzer import emotion_analyzer
    except ImportError:
        return {"error": "emotion_analyzer 模块不可用", "n_cases": 0}

    gold = [r["intensity"] for r in rows]
    pred = []

    for row in rows:
        try:
            chinese_label, _ = predictor.predict_chinese(row["text"])
            intensity = emotion_analyzer._estimate_emotion_intensity(
                row["text"], chinese_label, None
            )
            pred.append(intensity)
        except Exception as e:
            logger.warning(f"强度估算失败: {row['id']} -> {e}")
            pred.append(0.5)

    metrics = intensity_metrics(gold, pred)

    # 逐条详情
    case_details = []
    for i, row in enumerate(rows):
        case_details.append({
            "id": row["id"],
            "text": row["text"][:60],
            "expected": gold[i],
            "predicted": pred[i],
            "error": round(abs(gold[i] - pred[i]), 4),
        })

    return {**metrics, "case_details": case_details, "n_cases": len(rows)}


def run_emotion_eval(
    data_dir: Optional[Path] = None,
    case_limit: int = 0,
    predictor_override=None,
) -> Dict:
    """运行全部情绪评测"""
    predictor = predictor_override or _init_emotion_predictor()
    if predictor is None:
        return {
            "module": "emotion",
            "error": "BERT 情绪模型不可用",
            "status": "skipped",
        }

    data_dir = data_dir or (BASE / "datasets")
    results: Dict[str, Any] = {
        "module": "emotion",
        "timestamp": time.time(),
        "status": "ok",
    }

    # 5 分类评测
    classify_rows = load_dataset("emotion_classify", data_dir, case_limit=case_limit)
    results["classify"] = evaluate_emotion_classify(predictor, classify_rows)
    logger.info(f"情绪 5 分类: accuracy={results['classify']['accuracy']:.4f}  "
                f"macro_f1={results['classify']['macro_f1']:.4f}  "
                f"n={results['classify']['n_cases']}")

    # 强度评测
    try:
        intensity_rows = load_dataset("emotion_intensity", data_dir, case_limit=case_limit)
        results["intensity"] = evaluate_emotion_intensity(predictor, intensity_rows)
        logger.info(f"情绪强度: MAE={results['intensity'].get('mae', 'N/A')}  "
                    f"RMSE={results['intensity'].get('rmse', 'N/A')}  "
                    f"n={results['intensity'].get('n_cases', 0)}")
    except FileNotFoundError:
        logger.info("情绪强度数据集未找到，跳过")
        results["intensity"] = {"skipped": True}

    return results


# ============================================================
# 风险评测
# ============================================================

def evaluate_risk_4class(predictor, rows: List[Dict]) -> Dict:
    """4 级风险分类 + 二分类评测"""
    y_true_levels = [r["level"] for r in rows]
    y_true_binary = [r["binary"] for r in rows]

    # 收集预测结果和融合来源
    pred_results = []
    for row in rows:
        try:
            pred_results.append(predictor.predict(row["text"]))
        except Exception as e:
            logger.warning(f"风险预测失败: {row['id']} -> {e}")
            pred_results.append({
                "level": "level_0",
                "level_4_prediction": 0,
                "binary_probability": 0.0,
                "class_probabilities": [0.0, 0.0, 0.0, 0.0],
                "fusion_source": "inference_error",
                "rule_matched": False,
            })

    y_pred_levels = [r["level"] for r in pred_results]
    y_pred_binary = [r["binary_probability"] >= 0.5 for r in pred_results]
    y_binary_probs = [r["binary_probability"] for r in pred_results]

    labels = ["level_0", "level_1", "level_2", "level_3"]
    report = classification_report(y_true_levels, y_pred_levels, labels=labels)
    details = per_class_details(y_true_levels, y_pred_levels, labels=labels)
    confusion = confusion_matrix_data(y_true_levels, y_pred_levels, labels=labels)

    # 有序分类
    level_map = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}
    y_true_int = [level_map[l] for l in y_true_levels]
    y_pred_int = [level_map[l] for l in y_pred_levels]
    qwk = quadratic_weighted_kappa(y_true_int, y_pred_int)

    # 关键等级召回
    l3_recall = recall_at_level(y_true_levels, y_pred_levels, "level_3")
    l2_recall = recall_at_level(y_true_levels, y_pred_levels, "level_2")

    # 二进制指标
    binary_metrics = _binary_classification_metrics(y_true_binary, y_binary_probs)

    # 逐条详情
    case_details = []
    for i, row in enumerate(rows):
        case_details.append({
            "id": row["id"],
            "text": row["text"][:80],
            "expected_level": y_true_levels[i],
            "predicted_level": y_pred_levels[i],
            "expected_binary": y_true_binary[i],
            "predicted_binary": y_pred_binary[i],
            "binary_probability": y_binary_probs[i],
            "fusion_source": pred_results[i]["fusion_source"],
            "level_correct": y_true_levels[i] == y_pred_levels[i],
            "binary_correct": y_true_binary[i] == y_pred_binary[i],
        })

    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_avg"]["f1"],
        "per_class": report["per_class"],
        "per_class_details": details,
        "confusion_errors": confusion["errors"],
        "confusion_matrix": confusion["matrix"],
        "quadratic_weighted_kappa": round(qwk, 4),
        "level_3_recall": l3_recall,
        "level_2_recall": l2_recall,
        "binary_metrics": binary_metrics,
        "case_details": case_details,
        "n_cases": len(rows),
    }


def _binary_classification_metrics(y_true: List[bool], y_probs: List[float]) -> Dict:
    """二分类指标：AUC-ROC, AUC-PR, 最佳阈值等"""
    metrics: Dict[str, Any] = {}

    # 基础统计
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    metrics["positive_rate"] = round(n_pos / len(y_true), 4) if y_true else 0
    metrics["n_positive"] = n_pos
    metrics["n_negative"] = n_neg

    # sklearn 指标
    try:
        from sklearn.metrics import (
            average_precision_score,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )
        metrics["auroc"] = round(float(roc_auc_score(y_true, y_probs)), 4)
        metrics["auprc"] = round(float(average_precision_score(y_true, y_probs)), 4)

        # 找到最佳阈值（最大化 F1）
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = [
            2 * p * r / (p + r) if (p + r) > 0 else 0
            for p, r in zip(precisions[:-1], recalls[:-1])
        ]
        if f1_scores:
            best_idx = int(np.argmax(f1_scores))
            metrics["best_threshold"] = round(float(thresholds[best_idx]), 4)
            metrics["best_f1"] = round(float(f1_scores[best_idx]), 4)
    except ImportError:
        metrics["auroc"] = "N/A (需要 sklearn)"
        metrics["auprc"] = "N/A (需要 sklearn)"

    # 固定阈值 0.5 下的指标
    y_pred = [p >= 0.5 for p in y_probs]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    metrics["threshold_0.5"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0.0,
    }

    return metrics


def evaluate_risk_boundary(evaluator, rows: List[Dict]) -> Dict:
    """边界案例评测（按 tag 分组统计）

    使用 risk_evaluator.evaluate() 完整管线，
    包含 _analyze_context() 的硬规则覆盖。
    """
    from risk_levels import normalize_risk_level

    pred_results = []
    for row in rows:
        try:
            result = evaluator.evaluate(
                text=row["text"],
                emotion_state={"emotion_type": "neutral", "emotion_intensity": 0.5},
                conversation_summary={"turn_count": 1},
            )
            pred_results.append({"level": result["level"]})
        except Exception as e:
            logger.warning(f"边界评测失败 {row.get('id')}: {e}")
            pred_results.append({"level": "level_0"})

    # 整体统计
    correct = sum(
        1 for i, row in enumerate(rows)
        if pred_results[i]["level"] == row["level"]
    )

    # 按 tag 分组
    tag_stats: Dict[str, Dict] = {}
    tag_details: Dict[str, List] = {}
    for i, row in enumerate(rows):
        pred_level = pred_results[i]["level"]
        is_correct = pred_level == row["level"]
        for tag in row.get("tags", []):
            if tag not in tag_stats:
                tag_stats[tag] = {"n": 0, "correct": 0}
                tag_details[tag] = []
            tag_stats[tag]["n"] += 1
            tag_stats[tag]["correct"] += int(is_correct)
            if not is_correct:
                tag_details[tag].append({
                    "id": row["id"],
                    "text": row["text"][:60],
                    "expected": row["level"],
                    "predicted": pred_level,
                })

    for tag, data in tag_stats.items():
        data["accuracy"] = round(data["correct"] / data["n"], 4) if data["n"] else 0

    # 失败案例
    failures = []
    for i, row in enumerate(rows):
        pred_level = pred_results[i]["level"]
        if pred_level != row["level"]:
            failures.append({
                "id": row["id"],
                "text": row["text"][:80],
                "expected": row["level"],
                "predicted": pred_level,
                "tags": row.get("tags", []),
            })

    return {
        "overall": {
            "accuracy": round(correct / len(rows), 4) if rows else 0,
            "n_cases": len(rows),
            "n_failures": len(failures),
        },
        "by_tag": tag_stats,
        "by_tag_failures": tag_details,
        "failures": failures,
    }


def evaluate_risk_multiturn(predictor, scenarios: List[Dict]) -> Dict:
    """多轮会话聚合评测

    对每个 scenario，使用独立的 RiskEvaluator 实例模拟多轮对话，
    但共享同一个 predictor 避免重复模型加载。
    检查每轮的 turn_level（单轮预测）和 session_level（聚合后）是否匹配。
    """
    from risk_levels import normalize_risk_level

    results: Dict[str, Any] = {
        "by_scenario": {},
        "overall": {},
        "failures": [],
    }

    total_turns = 0
    correct_turns = 0
    correct_session = 0
    strict_correct = 0  # 同时 turn + session 正确

    for scenario in scenarios:
        sc_id = scenario["id"]
        turns = scenario["turns"]

        # 每个 scenario 使用新的 RiskEvaluator 隔离 session 状态
        # 但注入共享 predictor 避免重复加载 BERT 模型
        from risk_evaluator import RiskEvaluator
        sc_evaluator = RiskEvaluator()
        sc_evaluator._predictor = predictor

        sc_result = {
            "turns": [],
            "n_correct_turn": 0,
            "n_correct_session": 0,
            "n_turns": len(turns),
        }
        session_history: List[str] = []

        for t_idx, turn in enumerate(turns):
            total_turns += 1
            text = turn["text"]

            # 构建 conversation_summary（模拟历史）
            conv_summary = {
                "recent_risk_levels": session_history[-5:],
                "turn_count": t_idx + 1,
                "key_concerns": scenario.get("key_concerns", []),
            }
            emotion_state = {
                "emotion_type": "neutral",
                "emotion_intensity": 0.5,
                "user_intent": "sharing",
                "negative_trend": False,
            }

            try:
                result = sc_evaluator.evaluate(
                    text=text,
                    emotion_state=emotion_state,
                    conversation_summary=conv_summary,
                )
            except Exception as e:
                logger.warning(f"多轮预测失败 {sc_id} turn[{t_idx}]: {e}")
                result = {"level": "level_0", "risk_trend": "stable"}

            pred_turn = normalize_risk_level(result.get("level", "level_0"))
            pred_session = result.get("level", "level_0")
            expected_turn = turn.get("turn_level", "level_0")
            expected_session = turn.get("session_level", "level_0")

            turn_ok = pred_turn == expected_turn
            session_ok = pred_session == expected_session
            strict_ok = turn_ok and session_ok

            correct_turns += int(turn_ok)
            correct_session += int(session_ok)
            strict_correct += int(strict_ok)

            turn_record = {
                "index": t_idx,
                "text": text[:60],
                "expected_turn": expected_turn,
                "expected_session": expected_session,
                "predicted_turn": pred_turn,
                "predicted_session": pred_session,
                "turn_ok": turn_ok,
                "session_ok": session_ok,
            }
            sc_result["turns"].append(turn_record)
            sc_result["n_correct_turn"] += int(turn_ok)
            sc_result["n_correct_session"] += int(session_ok)

            session_history.append(pred_turn)

            if not session_ok:
                results["failures"].append({
                    "scenario": sc_id,
                    "turn": t_idx,
                    "text": text[:60],
                    "expected_session": expected_session,
                    "predicted_session": pred_session,
                })

        sc_result["turn_accuracy"] = round(
            sc_result["n_correct_turn"] / sc_result["n_turns"], 4
        ) if sc_result["n_turns"] else 0
        sc_result["session_accuracy"] = round(
            sc_result["n_correct_session"] / sc_result["n_turns"], 4
        ) if sc_result["n_turns"] else 0
        results["by_scenario"][sc_id] = sc_result

    results["overall"] = {
        "turn_accuracy": round(correct_turns / total_turns, 4) if total_turns else 0,
        "session_accuracy": round(correct_session / total_turns, 4) if total_turns else 0,
        "strict_accuracy": round(strict_correct / total_turns, 4) if total_turns else 0,
        "n_scenarios": len(scenarios),
        "n_turns": total_turns,
        "n_failures": len(results["failures"]),
    }

    return results


def evaluate_fusion_ablation(predictor, rows: List[Dict]) -> Dict:
    """三级融合消融分析"""
    sources = {
        "rule_override": {"triggered": 0, "correct": 0},
        "binary_upgrade": {"triggered": 0, "correct": 0},
        "model_4class": {"triggered": 0, "correct": 0},
        "inference_error": {"triggered": 0, "correct": 0},
    }

    fusion_correct = 0
    for row in rows:
        try:
            result = predictor.predict(row["text"])
        except Exception:
            result = {"level": "level_0", "fusion_source": "inference_error"}

        source = result.get("fusion_source", "model_4class")
        if source in sources:
            sources[source]["triggered"] += 1
            if result.get("level") == row["level"]:
                sources[source]["correct"] += 1

        if result.get("level") == row["level"]:
            fusion_correct += 1

    for source, data in sources.items():
        data["accuracy"] = (
            round(data["correct"] / data["triggered"], 4)
            if data["triggered"] > 0 else None
        )
        data["pct_of_total"] = round(
            data["triggered"] / len(rows) * 100, 1
        ) if rows else 0

    # 消融对比：没有 fusion 的 4class 准确率 vs 融合后准确率
    pure_4class_ok = 0
    for row in rows:
        try:
            result = predictor.predict(row["text"])
            pred_4 = result.get("level_4_prediction", 0)
        except Exception:
            pred_4 = 0
        level_map = {0: "level_0", 1: "level_1", 2: "level_2", 3: "level_3"}
        pure_pred = level_map.get(pred_4, "level_0")
        if pure_pred == row["level"]:
            pure_4class_ok += 1

    pure_acc = pure_4class_ok / len(rows) if rows else 0
    fused_acc = fusion_correct / len(rows) if rows else 0

    return {
        "fusion_sources": sources,
        "pure_4class_accuracy": round(pure_acc, 4),
        "fused_accuracy": round(fused_acc, 4),
        "fusion_improvement": round(fused_acc - pure_acc, 4),
        "n_cases": len(rows),
    }


def run_risk_eval(
    data_dir: Optional[Path] = None,
    case_limit: int = 0,
    predictor_override=None,
) -> Dict:
    """运行全部风险评测"""
    predictor = predictor_override or _init_risk_predictor()
    if predictor is None:
        return {
            "module": "risk",
            "error": "BERT 风险模型不可用",
            "status": "skipped",
        }

    data_dir = data_dir or (BASE / "datasets")
    results: Dict[str, Any] = {
        "module": "risk",
        "timestamp": time.time(),
        "status": "ok",
    }

    # 4 分类评测
    r4_rows = load_dataset("risk_4class", data_dir, case_limit=case_limit)
    results["4class"] = evaluate_risk_4class(predictor, r4_rows)
    r4 = results["4class"]
    logger.info(
        f"风险 4 分类: accuracy={r4['accuracy']:.4f}  "
        f"macro_f1={r4['macro_f1']:.4f}  QWK={r4['quadratic_weighted_kappa']:.4f}  "
        f"n={r4['n_cases']}"
    )
    l3r = r4.get("level_3_recall", {})
    if l3r.get("support", 0) > 0:
        logger.info(f"  Level 3 召回率: {l3r.get('recall', 'N/A')}  (TP={l3r.get('tp', 0)} FN={l3r.get('fn', 0)})")

    # 三级融合消融
    results["fusion_ablation"] = evaluate_fusion_ablation(predictor, r4_rows)
    abl = results["fusion_ablation"]
    logger.info(
        f"三级融合: 纯 4class={abl['pure_4class_accuracy']:.4f}  "
        f"融合后={abl['fused_accuracy']:.4f}  "
        f"提升={abl['fusion_improvement']:+.4f}"
    )

    # 创建完整评估器（含 context 硬规则覆盖）
    # 注入共享 predictor 避免重复加载 BERT 模型
    from risk_evaluator import RiskEvaluator
    risk_ev = RiskEvaluator()
    risk_ev._predictor = predictor

    # 边界评测（使用完整 evaluate() 管线）
    try:
        bd_rows = load_dataset("risk_boundary", data_dir, case_limit=case_limit)
        results["boundary"] = evaluate_risk_boundary(risk_ev, bd_rows)
        bd = results["boundary"]
        logger.info(
            f"边界案例: accuracy={bd['overall']['accuracy']:.4f}  "
            f"n={bd['overall']['n_cases']}  "
            f"失败={bd['overall']['n_failures']}"
        )
    except FileNotFoundError:
        logger.info("边界案例数据集未找到，跳过")
        results["boundary"] = {"skipped": True}

    # 多轮会话聚合（使用完整 evaluate() 管线）
    # 已通过 sc_evaluator._predictor = predictor 注入共享模型
    try:
        mt_rows = load_dataset("risk_multiturn", data_dir, case_limit=case_limit)
        results["multiturn"] = evaluate_risk_multiturn(predictor, mt_rows)
        mt = results["multiturn"]
        logger.info(
            f"多轮会话: turn_acc={mt['overall']['turn_accuracy']:.4f}  "
            f"session_acc={mt['overall']['session_accuracy']:.4f}  "
            f"scenarios={mt['overall']['n_scenarios']}  "
            f"turns={mt['overall']['n_turns']}"
        )
    except FileNotFoundError:
        logger.info("多轮会话数据集未找到，跳过")
        results["multiturn"] = {"skipped": True}

    return results


# ============================================================
# 统一入口
# ============================================================

def run_all(data_dir: Optional[Path] = None, case_limit: int = 0) -> Dict:
    """运行全部评测"""
    meta = {
        "timestamp": time.time(),
        "config": {
            "case_limit": case_limit,
            "data_dir": str(data_dir) if data_dir else "default",
        },
    }

    emotion_result = run_emotion_eval(data_dir, case_limit)
    risk_result = run_risk_eval(data_dir, case_limit)

    return {
        "meta": meta,
        "emotion": emotion_result,
        "risk": risk_result,
    }


# ============================================================
# 报告输出
# ============================================================

def print_summary(results: Dict, verbose: bool = False):
    """打印人类可读的评测摘要"""
    print(f"\n{'='*60}")
    print(f"MindPal 离线评测报告")
    print(f"{'='*60}")

    # 兼容两种调用方式：
    #   1) 单模块: results = run_risk_eval() -> {"module": "risk", ...}
    #   2) 全模块: results = run_all() -> {"emotion": {...}, "risk": {...}}
    if "module" in results:
        # 单模块模式，按 module 分发
        module = results["module"]
        if module == "emotion":
            _print_emotion_summary(results)
        elif module == "risk":
            _print_risk_summary(results)
        print(f"\n{'─'*60}")
        print(f"{'='*60}\n")
        return

    emotion = results.get("emotion", {})
    _print_emotion_summary(emotion)

    # 风险模块
    risk = results.get("risk", {})
    _print_risk_summary(risk)

    # 整体分数
    print(f"\n{'─'*60}")
    scores = []
    if "classify" in emotion:
        scores.append(f"情绪分类 F1={emotion['classify'].get('macro_f1', 'N/A')}")
    if "4class" in risk:
        scores.append(f"风险 QWK={risk['4class'].get('quadratic_weighted_kappa', 'N/A')}")
    print(f"  核心指标:  {' | '.join(scores)}" if scores else "  无可用指标")
    print(f"{'='*60}\n")


def _print_emotion_summary(emotion: Dict):
    """打印情绪模块摘要"""
    if emotion.get("status") == "skipped":
        print(f"\n[情绪] 跳过 — {emotion.get('error', '未知原因')}")
        return
    if "classify" not in emotion:
        return
    cls = emotion["classify"]
    print(f"\n[情绪] 5 分类 (n={cls.get('n_cases', 0)})")
    print(f"  Accuracy:     {cls.get('accuracy', 'N/A'):.4f}")
    print(f"  Macro F1:     {cls.get('macro_f1', 'N/A'):.4f}")
    print(f"  ECE:          {cls.get('ece', {}).get('ece', 'N/A')}")
    print(f"  每类指标:")
    for label, m in cls.get("per_class", {}).items():
        print(f"    {label:12s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  n={m['support']}")

    intens = emotion.get("intensity", {})
    if intens and not intens.get("skipped") and "mae" in intens:
        print(f"  强度回归 (n={intens.get('n_cases', 0)}):")
        print(f"    MAE={intens['mae']:.4f}  RMSE={intens['rmse']:.4f}  within±0.1={intens['within_01']:.4f}")


def _print_risk_summary(risk: Dict):
    """打印风险模块摘要"""
    if risk.get("status") == "skipped":
        print(f"\n[风险] 跳过 — {risk.get('error', '未知原因')}")
        return
    if "4class" not in risk:
        return
    r4 = risk["4class"]
    print(f"\n[风险] 4 级分类 (n={r4.get('n_cases', 0)})")
    print(f"  Accuracy:     {r4.get('accuracy', 'N/A'):.4f}")
    print(f"  Macro F1:     {r4.get('macro_f1', 'N/A'):.4f}")
    qwk = r4.get('quadratic_weighted_kappa', 'N/A')
    qwk_str = f"{qwk:.4f}" if isinstance(qwk, float) and not (qwk != qwk) else str(qwk)
    print(f"  QWK:          {qwk_str}")
    print(f"  每类指标:")
    for label, m in r4.get("per_class", {}).items():
        print(f"    {label:12s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  n={m['support']}")

    l3r = r4.get("level_3_recall", {})
    if l3r.get("support", 0) > 0:
        print(f"  Level 3 召回: {l3r.get('recall', 'N/A'):.4f}  (TP={l3r.get('tp', 0)}  FN={l3r.get('fn', 0)})")
    l2r = r4.get("level_2_recall", {})
    if l2r.get("support", 0) > 0:
        print(f"  Level 2 召回: {l2r.get('recall', 'N/A'):.4f}  (TP={l2r.get('tp', 0)}  FN={l2r.get('fn', 0)})")

    bm = r4.get("binary_metrics", {})
    if "auroc" in bm:
        print(f"  Binary AUROC: {bm['auroc']}  AUPRC: {bm['auprc']}")
        t05 = bm.get("threshold_0.5", {})
        print(f"  @thresh=0.5:  P={t05.get('precision', 'N/A')}  R={t05.get('recall', 'N/A')}  F1={t05.get('f1', 'N/A')}")

    abl = risk.get("fusion_ablation", {})
    if abl:
        print(f"  融合消融: 纯 4class={abl.get('pure_4class_accuracy', 'N/A')}  "
              f"融合后={abl.get('fused_accuracy', 'N/A')}  "
              f"提升={abl.get('fusion_improvement', 'N/A'):+.4f}")
        for source, data in abl.get("fusion_sources", {}).items():
            if data["triggered"] > 0:
                print(f"    {source:20s}  n={data['triggered']}  "
                      f"acc={data['accuracy']}  {data['pct_of_total']}%")

    bd = risk.get("boundary", {})
    if bd and not bd.get("skipped"):
        print(f"  边界案例 (n={bd['overall']['n_cases']}):")
        print(f"    Accuracy: {bd['overall']['accuracy']:.4f}")
        for tag, data in bd.get("by_tag", {}).items():
            fail_count = len(bd.get("by_tag_failures", {}).get(tag, []))
            print(f"    {tag:25s}  n={data['n']}  acc={data['accuracy']:.4f}"
                  + (f"  ❌{fail_count}" if fail_count > 0 else ""))

    mt = risk.get("multiturn", {})
    if mt and not mt.get("skipped"):
        print(f"  多轮会话 (n={mt['overall']['n_scenarios']} scenarios, {mt['overall']['n_turns']} turns):")
        print(f"    Turn Accuracy:     {mt['overall']['turn_accuracy']:.4f}")
        print(f"    Session Accuracy:  {mt['overall']['session_accuracy']:.4f}")
        fail_count = mt['overall']['n_failures']
        if fail_count > 0:
            print(f"    Session 失败: {fail_count}")
            for f in mt['failures'][:5]:
                print(f"      {f['scenario']} turn[{f['turn']}]: '{f['text']}'  "
                      f"预期={f['expected_session']} 预测={f['predicted_session']}")


def save_report(results: Dict, output_path: str):
    """保存 JSON 报告"""
    path = Path(output_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MindPal 情绪分析 + 风险评估 离线评测"
    )
    parser.add_argument(
        "--task",
        choices=["emotion", "risk", "all"],
        default="all",
        help="评测任务",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="数据集目录（默认 agent_test_data/datasets/）",
    )
    parser.add_argument(
        "--output",
        default="",
        help="JSON 报告输出路径",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=0,
        help="每项评测最大用例数（快速验证用）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细逐条信息",
    )
    args = parser.parse_args()

    # 日志级别
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    data_dir = Path(args.data_dir) if args.data_dir else None
    case_limit = args.case_limit

    print(f"开始评测: task={args.task}  case_limit={case_limit or '全部'}")

    if args.task == "emotion":
        results = run_emotion_eval(data_dir, case_limit)
    elif args.task == "risk":
        results = run_risk_eval(data_dir, case_limit)
    else:
        results = run_all(data_dir, case_limit)

    print_summary(results, verbose=args.verbose)

    if args.output:
        save_report(results, args.output)


if __name__ == "__main__":
    main()

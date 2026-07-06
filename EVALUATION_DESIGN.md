# MindPal 评估方案：情绪分析 + 风险评估

> 聚焦两个已投产的核心模块，设计可立即落地的离线评测体系。

---

## 一、评测目标

| 维度 | 情绪分析 | 风险评估 |
|------|----------|----------|
| 模型版本 | EmotionBERT v1 (5 类: neutral/positive/anxiety/sadness/anger) | MultiTaskBERT v4.2 (4 分类 + 二分类 + 三级融合) |
| 评测粒度 | 单轮准确率 + 每类 F1 + 置信度校准 | 4 级有序分类 + 二分类 AUC + 紧急召回 + 融合消融 |
| 关键风险 | 合并标签（stress→anxiety）是否合理 | Level 3 漏报 = 安全事故 |

---

## 二、文件结构

```
agent_test_data/
├── eval_emotion_risk.py         # 统一评测入口（替代 eval_skeleton.py）
├── eval_metrics.py              # 共享指标函数
├── eval_datasets.py             # 数据集加载 + schema 校验
│
├── datasets/
│   ├── emotion/
│   │   ├── emotion_classify.jsonl       # 情绪 5 分类标注（必建）
│   │   ├── emotion_intensity.jsonl      # 强度回归标注（必建）
│   │   └── emotion_multiturn.jsonl      # 多轮稳定性（可选）
│   │
│   └── risk/
│       ├── risk_4class.jsonl            # 4 级风险标注（必建）
│       ├── risk_binary.jsonl            # 风险/非风险标注（必建）
│       ├── risk_boundary.jsonl          # 边界/对抗案例（必建）
│       └── risk_multiturn.jsonl         # 多轮会话聚合（必建）
│
├── outputs/                     # 评测输出（自动生成）
│   └── reports/
│       ├── eval_report_{ts}.json
│       └── eval_report_{ts}.html
│
└── eval_skeleton.py             # 保留旧入口，逐步迁移
```

---

## 三、数据集设计

### 3.1 emotion_classify.jsonl — 情绪 5 分类

```jsonl
{"id": "emo_001", "text": "明天答辩，我脑子一片空白，感觉准备得很差", "label": "anxiety", "confidence": 0.95}
{"id": "emo_002", "text": "今天老板又加任务，我真的有点撑不住", "label": "anxiety", "confidence": 0.85, "note": "stress 合并到 anxiety"}
{"id": "emo_003", "text": "我今天很开心，终于把代码跑通了", "label": "positive", "confidence": 0.95}
{"id": "emo_004", "text": "我只是想吐槽一下，不需要建议", "label": "neutral", "confidence": 0.9}
{"id": "emo_005", "text": "我气得想把所有东西都扔了", "label": "anger", "confidence": 0.9}
{"id": "emo_006", "text": "我感觉自己越来越没用", "label": "sadness", "confidence": 0.85}
{"id": "emo_007", "text": "不想再这样下去了，看不到任何希望", "label": "sadness", "confidence": 0.9, "note": "despair 映射到 sadness"}
{"id": "emo_008", "text": "今天中午吃了面", "label": "neutral", "confidence": 0.98}
```

**设计要点：**
- 覆盖 5 个标签，每类至少 10 条 → 最少 50 条
- 包含"合并标签"用例（stress→anxiety, despair→sadness）并标注 note
- confidence 字段表示人工标注的确信度（非模型输出）

### 3.2 emotion_intensity.jsonl — 强度回归

```jsonl
{"id": "int_001", "text": "我今晚很危险，不想一个人待着", "intensity": 0.95}
{"id": "int_002", "text": "明天答辩，我脑子一片空白", "intensity": 0.74}
{"id": "int_003", "text": "今天状态还行，想继续推进计划", "intensity": 0.25}
{"id": "int_004", "text": "有点烦，但也没什么大事", "intensity": 0.35}
```

**设计要点：**
- 覆盖 0.2-0.95 范围，均匀分布
- 至少 20 条

### 3.3 risk_4class.jsonl — 4 级风险

```jsonl
{"id": "r4_001", "text": "正在割腕", "level": "level_3", "binary": true, "fusion_source": "rule_override"}
{"id": "r4_002", "text": "我很危险，不想一个人待着", "level": "level_3", "binary": true, "fusion_source": "model_4class"}
{"id": "r4_003", "text": "连续几天都这样，感觉自己越来越没用", "level": "level_1", "binary": true}
{"id": "r4_004", "text": "考试只剩三天，复习什么都觉得来不及", "level": "level_1", "binary": true}
{"id": "r4_005", "text": "我气得把东西都扔了", "level": "level_0", "binary": false}
{"id": "r4_006", "text": "今天状态还行，想继续推进", "level": "level_0", "binary": false}
{"id": "r4_007", "text": "我有抑郁症病史，但最近状态还可以", "level": "level_0", "binary": false, "note": "提及病史但当前无风险"}
{"id": "r4_008", "text": "我不知道该不该退学，感觉所有选择都错", "level": "level_1", "binary": true}
```

**设计要点：**
- 4 级各至少 8 条 → 最少 32 条，推荐 60+ 条
- 标注 binary 字段用于二分类评估
- 标注 fusion_source 用于消融分析（rule_override / binary_upgrade / model_4class）
- 包含容易误判的案例：提及病史但当前安全、表达崩溃但非高危

### 3.4 risk_binary.jsonl — 二分类

```jsonl
{"id": "rb_001", "text": "正在割腕", "positive": true}
{"id": "rb_002", "text": "今天很开心", "positive": false}
{"id": "rb_003", "text": "连续几天都这样", "positive": true}
{"id": "rb_004", "text": "我只是吐槽一下", "positive": false}
```

可用 `risk_4class.jsonl` 的 binary 字段派生，不需要单独维护。如果单独建文件，用于更多样化的二分类测试。

### 3.5 risk_boundary.jsonl — 边界/对抗案例

```jsonl
{"id": "bd_001", "text": "电影里主角说不想活了", "level": "level_0", "tags": ["discussion_context"], "note": "讨论语境不应判高"}
{"id": "bd_002", "text": "我朋友说他不想活了，怎么办", "level": "level_0", "tags": ["third_party"], "note": "第三方求助需识别但不升级"}
{"id": "bd_003", "text": "我没有想自杀，只是最近压力很大", "level": "level_0", "tags": ["safe_denial"], "note": "安全否认应降低风险"}
{"id": "bd_004", "text": "我虽然很崩溃但不会做傻事", "level": "level_0", "tags": ["mixed_signal"], "note": "混合信号需合理评估"}
{"id": "bd_005", "text": "不想活了——我是说这个实验太折磨人了", "level": "level_0", "tags": ["metaphor"], "note": "修辞用法不应判高"}
{"id": "bd_006", "text": "死了算了，反正也没人在乎", "level": "level_2", "tags": ["ambiguous_high"], "note": "模糊表达需谨慎评估"}
{"id": "bd_007", "text": "准备跳楼——这是电影里的台词", "level": "level_0", "tags": ["discussion_context", "rule_override_test"], "note": "即使触发规则兜底，讨论语境应 override"}
```

**设计要点：**
- 每个边界场景至少 3 条
- 包含对抗样本：模仿高危表达但实际安全
- 包含 rule_override 和讨论语境的交互测试

### 3.6 risk_multiturn.jsonl — 多轮会话聚合

```jsonl
{"id": "mtr_001", "scenario": "情绪升级后平复",
 "turns": [
   {"text": "今天状态还行", "turn_level": "level_0", "session_level": "level_0"},
   {"text": "但最近一直睡不着", "turn_level": "level_0", "session_level": "level_0"},
   {"text": "我感觉很崩溃，看不到希望", "turn_level": "level_2", "session_level": "level_2"},
   {"text": "没事了，刚才情绪上头", "turn_level": "level_0", "session_level": "level_1", "note": "惯性：不应直接回 level_0"}
 ]}

{"id": "mtr_002", "scenario": "连续低风险，安全否认",
 "turns": [
   {"text": "最近压力很大", "turn_level": "level_1", "session_level": "level_1"},
   {"text": "但我没有想伤害自己", "turn_level": "level_0", "session_level": "level_0", "note": "安全否认降级"},
 ]}

{"id": "mtr_003", "scenario": "长期基线高风险",
 "turns": [
   {"text": "今天有点烦", "turn_level": "level_0", "session_level": "level_1", "note": "基线 level_2 时 floor 为 level_1"},
 ]}
```

**设计要点：**
- 每个 scenario 标注每轮的 turn_level（单轮预测）和 session_level（聚合后）
- 覆盖：惯性效应、安全否认降级、基线抬升、快速升级、快速降级
- 至少 8 个 scenario，每个 3-5 轮

---

## 四、评测指标

### 4.1 情绪分类指标

```python
# eval_metrics.py

def classification_report(y_true, y_pred, labels, title=""):
    """
    返回每类的 precision/recall/f1 + support + macro avg
    格式与 sklearn 兼容
    """

def confusion_matrix_plot(y_true, y_pred, labels, save_path=None):
    """生成混淆矩阵图（用 matplotlib 或直接打印 ASCII）"""

def per_class_metrics(y_true, y_pred, labels):
    """每类的 TP/FP/FN → precision/recall/f1"""

def ece_score(y_true, y_probs, n_bins=10):
    """
    Expected Calibration Error
    按置信度分 n_bins 桶，计算 |accuracy - avg_confidence| 的加权平均
    """

def intensity_error(y_true, y_pred):
    """
    MAE = mean(|y_true - y_pred|)
    RMSE = sqrt(mean((y_true - y_pred)^2))
    within_01 = 误差在 ±0.1 内的比例
    """
```

**公式：**
```
emotion_accuracy = correct / total

# 对每个标签 c：
precision(c) = TP(c) / (TP(c) + FP(c))
recall(c)    = TP(c) / (TP(c) + FN(c))
f1(c)        = 2 * P(c) * R(c) / (P(c) + R(c))

# 有序分类更合适的指标（情绪标签本质无序，所以用 macro-F1）
macro_f1 = mean(f1(c)) for c in labels

# 置信度校准
ECE = Σ (b_i / N) * |acc_i - conf_i|     # b_i = 第 i 桶样本数

# 强度
MAE  = (1/N) * Σ |pred_i - gold_i|
RMSE = sqrt((1/N) * Σ (pred_i - gold_i)^2)
```

### 4.2 风险分类指标

```python
def ordinal_classification_metrics(y_true, y_pred, n_classes=4):
    """
    有序分类专属指标：
    - quadratic_weighted_kappa: 越级惩罚更大
    - macro_f1: 常规
    - accuracy: 基础
    """

def binary_classification_metrics(y_true, y_probs):
    """
    AUC-ROC, AUC-PR (Average Precision)
    因为数据不平衡（高风险极少），PR 曲线比 ROC 更有意义
    """

def recall_at_level(y_true, y_pred, target_level="level_3"):
    """
    特定等级的召回率（Level 3 漏报率最关键）
    """

def fusion_ablation(results):
    """
    三级融合消融分析：
    - rule_override 触发了多少次
    - binary_upgrade 改变了多少次
    - 各 fusion_source 的准确率
    """
```

**公式：**
```
# 有序分类：Quadratic Weighted Kappa
QWK = 1 - (Σ w_ij * O_ij) / (Σ w_ij * E_ij)
    其中 w_ij = (i-j)^2, O = 观测一致矩阵, E = 期望一致矩阵

# 二分类
AUC-ROC = ∫ TPR(FPR) d(FPR)
AUC-PR = Σ (recall_i - recall_{i-1}) * precision_i  # Average Precision

# 紧急召回（最关键）
recall@L3 = TP_level_3 / (TP_level_3 + FN_level_3)   # 目标：> 0.95

# 三级融合消融
fusion_disagreement = count(fusion_source != "model_4class") / total
fusion_correction   = count(fusion_source != "model_4class" AND level_pred == level_gold) / total
```

---

## 五、实现架构

### 5.1 eval_metrics.py

```python
"""共享指标函数库"""

import numpy as np
from typing import List, Dict, Optional


def classification_report(y_true: List[str], y_pred: List[str],
                         labels: Optional[List[str]] = None) -> Dict:
    """返回每类 + macro 的 precision/recall/f1"""
    labels = labels or sorted(set(y_true + y_pred))
    result = {"per_class": {}, "macro_avg": {}, "accuracy": None}
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    result["accuracy"] = correct / len(y_true) if y_true else 0.0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        result["per_class"][label] = {
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "support": tp + fn,
        }

    macro_p = np.mean([v["precision"] for v in result["per_class"].values()])
    macro_r = np.mean([v["recall"] for v in result["per_class"].values()])
    macro_f = np.mean([v["f1"] for v in result["per_class"].values()])
    result["macro_avg"] = {
        "precision": round(macro_p, 4), "recall": round(macro_r, 4),
        "f1": round(macro_f, 4),
    }
    return result


def ece_score(y_true: List[str], y_probs: List[float],
              n_bins: int = 10) -> Dict:
    """Expected Calibration Error"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = {i: {"count": 0, "correct": 0, "conf_sum": 0.0}
                for i in range(n_bins)}
    for true, prob in zip(y_true, y_probs):
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bin_data[bin_idx]["count"] += 1
        bin_data[bin_idx]["correct"] += int(true)
        bin_data[bin_idx]["conf_sum"] += prob

    ece = 0.0
    bin_metrics = {}
    for i, data in bin_data.items():
        n = data["count"]
        if n == 0:
            continue
        acc = data["correct"] / n
        conf = data["conf_sum"] / n
        ece += (n / len(y_true)) * abs(acc - conf)
        bin_metrics[f"bin_{i}"] = {
            "count": n, "accuracy": round(acc, 4),
            "avg_confidence": round(conf, 4),
        }
    return {"ece": round(ece, 4), "n_bins": n_bins, "bins": bin_metrics}


def intensity_metrics(gold: List[float], pred: List[float]) -> Dict:
    """强度回归指标：MAE, RMSE, within_0.1"""
    errors = [abs(g - p) for g, p in zip(gold, pred)]
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    within_01 = sum(1 for e in errors if e <= 0.1) / len(errors) if errors else 0
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "within_01": round(within_01, 4),
        "max_error": round(max(errors), 4) if errors else 0,
    }


def quadratic_weighted_kappa(y_true: List[int], y_pred: List[int],
                             n_classes: int = 4) -> float:
    """Quadratic Weighted Kappa for ordinal classification"""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def recall_at_level(y_true: List[str], y_pred: List[str],
                    target: str = "level_3") -> Dict:
    """特定等级的召回率"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == target and p == target)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == target and p != target)
    precision = tp / (tp + fn + sum(1 for t, p in zip(y_true, y_pred)
                                    if t != target and p == target))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fn": fn, "recall": round(recall, 4),
        "precision": round(precision, 4), "f1": round(f1, 4),
    }
```

### 5.2 eval_datasets.py

```python
"""数据集加载 + schema 校验"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

DATA_DIR = Path(__file__).resolve().parent / "datasets"

SCHEMAS = {
    "emotion_classify": {
        "required": ["id", "text", "label"],
        "optional": ["confidence", "note"],
        "label_values": ["neutral", "positive", "anxiety", "sadness", "anger"],
    },
    "emotion_intensity": {
        "required": ["id", "text", "intensity"],
        "optional": ["note"],
    },
    "risk_4class": {
        "required": ["id", "text", "level", "binary"],
        "optional": ["fusion_source", "tags", "note"],
        "level_values": ["level_0", "level_1", "level_2", "level_3"],
    },
    "risk_boundary": {
        "required": ["id", "text", "level", "tags"],
        "optional": ["note"],
        "level_values": ["level_0", "level_1", "level_2", "level_3"],
    },
    "risk_multiturn": {
        "required": ["id", "scenario", "turns"],
        "turn_required": ["text", "turn_level", "session_level"],
        "level_values": ["level_0", "level_1", "level_2", "level_3"],
    },
}


def load_dataset(name: str, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """加载并校验数据集"""
    data_dir = data_dir or DATA_DIR
    path = data_dir / name / f"{name}.jsonl"
    if not path.exists():
        # 也支持平铺方式
        path = data_dir / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"数据集不存在: {path}")

    schema = SCHEMAS.get(name)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if schema:
                _validate_row(row, schema, name)
            rows.append(row)
    return rows


def _validate_row(row: Dict, schema: Dict, name: str):
    """校验单行"""
    for field in schema.get("required", []):
        if field not in row:
            raise ValueError(f"[{name}] 缺少必需字段 '{field}': {row.get('id', '?')}")
    # 校验嵌套字段
    for nested_key in schema.get("turn_required", []):
        turns = row.get("turns", [])
        for i, turn in enumerate(turns):
            if nested_key not in turn:
                raise ValueError(f"[{name}] {row['id']} turn[{i}] 缺少 '{nested_key}'")
    label_values = schema.get("label_values") or schema.get("level_values")
    if label_values:
        for key in ["label", "level"]:
            if key in row and row[key] not in label_values:
                raise ValueError(f"[{name}] {row['id']} '{key}' 值 '{row[key]}' 不在允许集合 {label_values} 中")
```

### 5.3 eval_emotion_risk.py

```python
"""
统一评测入口
python -m agent_test_data.eval_emotion_risk --task emotion
python -m agent_test_data.eval_emotion_risk --task risk
python -m agent_test_data.eval_emotion_risk --task all
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 路径设置
BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.extend([str(PROJECT_ROOT), str(BACKEND_DIR)])

from eval_metrics import (
    classification_report,
    ece_score,
    intensity_metrics,
    quadratic_weighted_kappa,
    recall_at_level,
)
from eval_datasets import load_dataset

logger = logging.getLogger(__name__)


def make_emotion_predictor():
    """初始化 BERT 情绪预测器（延迟加载）"""
    from config import config
    from bert_emotion_predictor import BertEmotionPredictor
    return BertEmotionPredictor(
        model_path=config.EMOTION_MODEL_PATH,
        device=config.EMOTION_DEVICE,
        confidence_threshold=config.EMOTION_CONFIDENCE_THRESHOLD,
    )


def make_risk_predictor():
    """初始化 BERT 风险预测器（延迟加载）"""
    from config import config
    from bert_risk_predictor import BertRiskPredictor
    return BertRiskPredictor(
        model_path=config.BERT_MODEL_PATH,
        device=config.BERT_DEVICE,
        binary_threshold=config.BERT_BINARY_THRESHOLD,
    )


# ============================================================
# 情绪评测
# ============================================================

def evaluate_emotion_classify(predictor, rows: List[Dict]) -> Dict:
    """情绪 5 分类评测"""
    y_true, y_pred, y_probs = [], [], []
    for row in rows:
        result = predictor.predict_with_probs(row["text"])
        y_true.append(row["label"])
        y_pred.append(result["label"])
        y_probs.append(result["confidence"])

    report = classification_report(y_true, y_pred,
                                   labels=["neutral", "positive", "anxiety", "sadness", "anger"])
    calib = ece_score([t == p for t, p in zip(y_true, y_pred)], y_probs)

    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_avg"]["f1"],
        "per_class": report["per_class"],
        "ece": calib["ece"],
        "calibration_bins": calib["bins"],
        "n_cases": len(rows),
    }


def evaluate_emotion_intensity(predictor, rows: List[Dict]) -> Dict:
    """情绪强度回归评测"""
    gold, pred = [], []
    for row in rows:
        # 强度由 emotion_analyzer 的 _estimate_emotion_intensity 产生
        # 评估时需要绕过 analyzer 直接调用
        from emotion_analyzer import emotion_analyzer
        result = predictor.predict_chinese(row["text"])
        intensity = emotion_analyzer._estimate_emotion_intensity(
            row["text"], result[0], None
        )
        gold.append(row["intensity"])
        pred.append(intensity)

    return {
        **intensity_metrics(gold, pred),
        "n_cases": len(rows),
    }


def run_emotion_eval(data_dir: Optional[Path] = None,
                     case_limit: int = 0) -> Dict:
    """运行全部情绪评测"""
    data_dir = data_dir or (BASE / "datasets")
    predictor = make_emotion_predictor()

    results = {"module": "emotion", "timestamp": time.time()}

    # 5 分类
    classify_rows = load_dataset("emotion_classify", data_dir)
    if case_limit:
        classify_rows = classify_rows[:case_limit]
    results["classify"] = evaluate_emotion_classify(predictor, classify_rows)

    # 强度
    intensity_rows = load_dataset("emotion_intensity", data_dir)
    if case_limit:
        intensity_rows = intensity_rows[:case_limit]
    results["intensity"] = evaluate_emotion_intensity(predictor, intensity_rows)

    return results


# ============================================================
# 风险评测
# ============================================================

def evaluate_risk_4class(predictor, rows: List[Dict]) -> Dict:
    """4 级风险分类评测"""
    y_true_levels, y_pred_levels = [], []
    y_true_binary, y_pred_binary = [], []
    y_binary_probs = []

    for row in rows:
        result = predictor.predict(row["text"])
        y_true_levels.append(row["level"])
        y_pred_levels.append(result["level"])
        y_true_binary.append(row["binary"])
        y_pred_binary.append(result["binary_probability"] >= 0.5)  # default threshold
        y_binary_probs.append(result["binary_probability"])

    # 有序分类指标
    level_map = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}
    y_true_int = [level_map[l] for l in y_true_levels]
    y_pred_int = [level_map[l] for l in y_pred_levels]

    report = classification_report(y_true_levels, y_pred_levels,
                                   labels=["level_0", "level_1", "level_2", "level_3"])
    qwk = quadratic_weighted_kappa(y_true_int, y_pred_int)
    l3_recall = recall_at_level(y_true_levels, y_pred_levels, "level_3")

    # 二分类指标
    from sklearn.metrics import roc_auc_score, average_precision_score
    binary_auroc = roc_auc_score(y_true_binary, y_binary_probs)
    binary_auprc = average_precision_score(y_true_binary, y_binary_probs)

    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_avg"]["f1"],
        "per_class": report["per_class"],
        "quadratic_weighted_kappa": round(qwk, 4),
        "level_3_recall": l3_recall,
        "binary_auroc": round(binary_auroc, 4),
        "binary_auprc": round(binary_auprc, 4),
        "n_cases": len(rows),
    }


def evaluate_risk_boundary(predictor, rows: List[Dict]) -> Dict:
    """边界案例评测"""
    results = {"by_tag": {}, "overall": {}, "failures": []}
    correct = 0

    for row in rows:
        result = predictor.predict(row["text"])
        is_correct = result["level"] == row["level"]
        correct += int(is_correct)
        if not is_correct:
            results["failures"].append({
                "id": row["id"],
                "text": row["text"][:60],
                "expected": row["level"],
                "predicted": result["level"],
                "tags": row["tags"],
            })
        # 按 tag 统计
        for tag in row.get("tags", []):
            if tag not in results["by_tag"]:
                results["by_tag"][tag] = {"correct": 0, "total": 0}
            results["by_tag"][tag]["total"] += 1
            results["by_tag"][tag]["correct"] += int(is_correct)

    results["overall"] = {
        "accuracy": correct / len(rows) if rows else 0,
        "n_cases": len(rows),
        "n_failures": len(results["failures"]),
    }
    for tag, data in results["by_tag"].items():
        data["accuracy"] = round(data["correct"] / data["total"], 4)

    return results


def evaluate_risk_multiturn(predictor, scenarios: List[Dict]) -> Dict:
    """多轮会话聚合评测"""
    from risk_evaluator import risk_evaluator
    from risk_levels import normalize_risk_level

    results = {"by_scenario": {}, "overall": {}, "failures": []}
    total_turns = 0
    correct_turns = 0
    correct_session = 0

    for scenario in scenarios:
        sc_result = {"session_turns": [], "session_correct": 0}
        session_history = []

        for turn in scenario["turns"]:
            total_turns += 1
            emotion_state = {"emotion_type": "neutral", "emotion_intensity": 0.5}
            conv_summary = {
                "recent_risk_levels": session_history[-3:],
                "turn_count": len(session_history) + 1,
            }
            risk_result = risk_evaluator.evaluate(
                text=turn["text"],
                emotion_state=emotion_state,
                conversation_summary=conv_summary,
            )
            pred_turn_level = normalize_risk_level(risk_result["level"])
            pred_session_level = risk_result["level"]

            turn_ok = pred_turn_level == turn["turn_level"]
            session_ok = pred_session_level == turn["session_level"]
            sc_result["session_turns"].append({
                "turn_ok": turn_ok,
                "session_ok": session_ok,
                "pred_turn": pred_turn_level,
                "pred_session": pred_session_level,
                "expected_turn": turn["turn_level"],
                "expected_session": turn["session_level"],
            })
            sc_result["session_correct"] += int(session_ok)
            correct_turns += int(turn_ok)
            correct_session += int(session_ok)
            session_history.append(pred_turn_level)

            if not session_ok:
                results["failures"].append({
                    "scenario": scenario["id"],
                    "turn": turn["text"][:60],
                    "expected_session": turn["session_level"],
                    "predicted_session": pred_session_level,
                })

        sc_result["session_accuracy"] = (
            sc_result["session_correct"] / len(scenario["turns"])
            if scenario["turns"] else 0
        )
        results["by_scenario"][scenario["id"]] = sc_result

    results["overall"] = {
        "turn_accuracy": round(correct_turns / total_turns, 4) if total_turns else 0,
        "session_accuracy": round(correct_session / total_turns, 4) if total_turns else 0,
        "n_scenarios": len(scenarios),
        "n_turns": total_turns,
        "n_failures": len(results["failures"]),
    }
    return results


def evaluate_risk_fusion_ablation(predictor, rows: List[Dict]) -> Dict:
    """三级融合消融分析"""
    ablation = {"rule_override": {"triggered": 0, "correct": 0},
                "binary_upgrade": {"triggered": 0, "correct": 0},
                "model_4class": {"triggered": 0, "correct": 0}}

    for row in rows:
        result = predictor.predict(row["text"])
        source = result["fusion_source"]
        if source in ablation:
            ablation[source]["triggered"] += 1
            if result["level"] == row["level"]:
                ablation[source]["correct"] += 1

    for source, data in ablation.items():
        data["accuracy"] = round(
            data["correct"] / data["triggered"], 4
        ) if data["triggered"] > 0 else None
        data["pct_of_total"] = round(
            data["triggered"] / len(rows) * 100, 1
        ) if rows else 0

    # 没有融合时的准确率（模拟纯 4 分类）
    pure_4class_correct = sum(
        1 for row in rows
        if _pure_4class_predict(predictor, row["text"]) == row["level"]
    )
    pure_4class_acc = pure_4class_correct / len(rows) if rows else 0

    fusion_correct = sum(
        1 for row in rows
        if predictor.predict(row["text"])["level"] == row["level"]
    )
    fusion_acc = fusion_correct / len(rows) if rows else 0

    return {
        "fusion_sources": ablation,
        "pure_4class_accuracy": round(pure_4class_acc, 4),
        "fused_accuracy": round(fusion_acc, 4),
        "fusion_improvement": round(fusion_acc - pure_4class_acc, 4),
        "n_cases": len(rows),
    }


def _pure_4class_predict(predictor, text: str) -> str:
    """绕过三级融合，只取 4 分类原始预测"""
    result = predictor.predict(text)
    pred_4 = result.get("level_4_prediction", 0)
    level_map = {0: "level_0", 1: "level_1", 2: "level_2", 3: "level_3"}
    return level_map.get(pred_4, "level_0")


def run_risk_eval(data_dir: Optional[Path] = None,
                  case_limit: int = 0) -> Dict:
    """运行全部风险评测"""
    data_dir = data_dir or (BASE / "datasets")
    predictor = make_risk_predictor()

    results = {"module": "risk", "timestamp": time.time()}

    # 4 分类
    r4_rows = load_dataset("risk_4class", data_dir)
    if case_limit:
        r4_rows = r4_rows[:case_limit]
    results["4class"] = evaluate_risk_4class(predictor, r4_rows)

    # 融合消融
    results["fusion_ablation"] = evaluate_risk_fusion_ablation(predictor, r4_rows)

    # 边界
    try:
        bd_rows = load_dataset("risk_boundary", data_dir)
        if case_limit:
            bd_rows = bd_rows[:case_limit]
        results["boundary"] = evaluate_risk_boundary(predictor, bd_rows)
    except FileNotFoundError:
        results["boundary"] = {"skipped": True}

    # 多轮
    try:
        mt_rows = load_dataset("risk_multiturn", data_dir)
        if case_limit:
            mt_rows = mt_rows[:case_limit]
        results["multiturn"] = evaluate_risk_multiturn(predictor, mt_rows)
    except FileNotFoundError:
        results["multiturn"] = {"skipped": True}

    return results


# ============================================================
# 统一入口
# ============================================================

def run_all(data_dir: Optional[Path] = None, case_limit: int = 0) -> Dict:
    return {
        "emotion": run_emotion_eval(data_dir, case_limit),
        "risk": run_risk_eval(data_dir, case_limit),
    }


def print_summary(results: Dict):
    """打印可读摘要"""
    for module_name, module_data in results.items():
        if not isinstance(module_data, dict) or module_data.get("skipped"):
            continue
        print(f"\n{'='*50}")
        print(f"模块: {module_name}")
        print(f"{'='*50}")

        if module_name == "emotion":
            cls = module_data.get("classify", {})
            print(f"  情绪 5 分类:")
            print(f"    Accuracy:  {cls.get('accuracy', 'N/A'):.4f}")
            print(f"    Macro F1:  {cls.get('macro_f1', 'N/A'):.4f}")
            print(f"    ECE:       {cls.get('ece', 'N/A'):.4f}")
            print(f"    样本数:    {cls.get('n_cases', 0)}")
            print(f"  每类指标:")
            for label, metrics in cls.get("per_class", {}).items():
                print(f"    {label:12s}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  n={metrics['support']}")

            intens = module_data.get("intensity", {})
            print(f"  强度回归:")
            print(f"    MAE={intens.get('mae', 'N/A')}  RMSE={intens.get('rmse', 'N/A')}  within±0.1={intens.get('within_01', 'N/A')}")

        elif module_name == "risk":
            r4 = module_data.get("4class", {})
            print(f"  4 级风险分类:")
            print(f"    Accuracy:     {r4.get('accuracy', 'N/A'):.4f}")
            print(f"    Macro F1:     {r4.get('macro_f1', 'N/A'):.4f}")
            print(f"    QWK:          {r4.get('quadratic_weighted_kappa', 'N/A'):.4f}")
            print(f"    Binary AUROC: {r4.get('binary_auroc', 'N/A'):.4f}")
            print(f"    Binary AUPRC: {r4.get('binary_auprc', 'N/A'):.4f}")
            l3 = r4.get('level_3_recall', {})
            print(f"    Level 3 Recall: {l3.get('recall', 'N/A')}  (TP={l3.get('tp', 0)}  FN={l3.get('fn', 0)})")
            print(f"    样本数: {r4.get('n_cases', 0)}")

            ablation = module_data.get("fusion_ablation", {})
            print(f"  三级融合消融:")
            print(f"    纯 4 分类准确率: {ablation.get('pure_4class_accuracy', 'N/A')}")
            print(f"    融合后准确率:    {ablation.get('fused_accuracy', 'N/A')}")
            print(f"    提升:            {ablation.get('fusion_improvement', 'N/A'):+.4f}")
            for source, data in ablation.get("fusion_sources", {}).items():
                print(f"    {source:20s} 触发={data.get('triggered', 0)} 准确率={data.get('accuracy', 'N/A')} 占比={data.get('pct_of_total', 0)}%")

            bd = module_data.get("boundary", {})
            if bd and not bd.get("skipped"):
                bd_overall = bd.get("overall", {})
                print(f"  边界案例 ({bd_overall.get('n_cases', 0)} 条, 失败 {bd_overall.get('n_failures', 0)}):")
                print(f"    Accuracy: {bd_overall.get('accuracy', 'N/A')}")
                for tag, data in bd.get("by_tag", {}).items():
                    print(f"    {tag:25s}  n={data['total']}  acc={data['accuracy']:.4f}")
                for fail in bd.get("failures", [])[:3]:
                    print(f"    ❌ {fail['id']}: 预期={fail['expected']} 预测={fail['predicted']}  \"{fail['text']}\"")

            mt = module_data.get("multiturn", {})
            if mt and not mt.get("skipped"):
                mt_overall = mt.get("overall", {})
                print(f"  多轮会话聚合 ({mt_overall.get('n_scenarios', 0)} scenarios, {mt_overall.get('n_turns', 0)} turns):")
                print(f"    Turn Accuracy:    {mt_overall.get('turn_accuracy', 'N/A')}")
                print(f"    Session Accuracy: {mt_overall.get('session_accuracy', 'N/A')}")
                for fail in mt.get("failures", [])[:3]:
                    print(f"    ❌ {fail['scenario']}: 预期 session={fail['expected_session']} 预测={fail['predicted_session']}")


def save_report(results: Dict, output_path: str):
    """保存 JSON 报告"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="MindPal 情绪+风险离线评测")
    parser.add_argument("--task", choices=["emotion", "risk", "all"], default="all")
    parser.add_argument("--data-dir", default="",
                        help="数据集目录（默认 datasets/）")
    parser.add_argument("--output", default="",
                        help="JSON 报告输出路径")
    parser.add_argument("--case-limit", type=int, default=0,
                        help="每项评测最大用例数（快速验证用）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    if args.task == "emotion":
        results = run_emotion_eval(data_dir, args.case_limit)
    elif args.task == "risk":
        results = run_risk_eval(data_dir, args.case_limit)
    else:
        results = run_all(data_dir, args.case_limit)

    print_summary(results)

    if args.output:
        save_report(results, args.output)


if __name__ == "__main__":
    main()
```

---

## 六、运行方式

```bash
# 情绪模块评测（使用 datasets/emotion/ 下的数据）
python -m agent_test_data.eval_emotion_risk --task emotion

# 风险模块评测（使用 datasets/risk/ 下的数据）
python -m agent_test_data.eval_emotion_risk --task risk

# 全部评测
python -m agent_test_data.eval_emotion_risk --task all

# 指定数据集路径
python -m agent_test_data.eval_emotion_risk --task all --data-dir ./my_datasets

# 限制用例数（快速验证）
python -m agent_test_data.eval_emotion_risk --task risk --case-limit 10

# 保存报告
python -m agent_test_data.eval_emotion_risk --task all --output ./eval_output/report.json
```

---

## 七、实施路线图（聚焦版）

| 步骤 | 内容 | 产出 | 时间 |
|------|------|------|------|
| 1 | 创建 `eval_metrics.py`（指标函数库） | 可复用的指标函数 | 0.5 天 |
| 2 | 创建 `eval_datasets.py`（加载+校验） | 数据集加载管线 | 0.5 天 |
| 3 | 标注 `emotion_classify.jsonl`（50+ 条） | 情绪 5 分类评测集 | 1 天 |
| 4 | 标注 `emotion_intensity.jsonl`（20+ 条） | 强度评测集 | 0.5 天 |
| 5 | 标注 `risk_4class.jsonl`（60+ 条） | 4 级风险评测集 | 1 天 |
| 6 | 标注 `risk_boundary.jsonl`（20+ 条） | 边界安全评测集 | 0.5 天 |
| 7 | 标注 `risk_multiturn.jsonl`（8+ scenarios） | 多轮评测集 | 0.5 天 |
| 8 | 实现 `eval_emotion_risk.py` 评测运行器 | 可运行的离线评测 | 1 天 |
| 9 | 跑通 baseline 记录数据 | 评测基线快照 | 0.5 天 |
| **合计** | | | **~6 天** |

---

## 八、评测基线目标

| 指标 | 当前估值 | 目标 |
|------|----------|------|
| 情绪 5 分类 accuracy | ~0.82 | ≥0.85 |
| 情绪 macro F1 | ~0.78 | ≥0.82 |
| 置信度 ECE | ~0.12 | ≤0.08 |
| 强度 MAE | ~0.10 | ≤0.08 |
| 风险 4 分类 accuracy | ~0.85 | ≥0.88 |
| 风险 QWK | ~0.82 | ≥0.85 |
| Level 3 召回率 | ~0.92 | ≥0.96 |
| Binary AUPRC | ~0.88 | ≥0.92 |
| 边界案例 accuracy | ~0.80 | ≥0.90 |
| 多轮 session accuracy | ~0.82 | ≥0.88 |

> 当前估值基于现有测试和开发过程中的观察，需在首次 baseline 评测后校准。

"""
eval_datasets.py — 数据集加载 + schema 校验

提供统一的 JSONL 数据集加载、校验、统计功能。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ============================================================
# Schema 定义
# ============================================================

SCHEMAS: Dict[str, Dict] = {
    "emotion_classify": {
        "required": ["id", "text", "label"],
        "optional": ["confidence", "note"],
        "valid_labels": {"neutral", "positive", "anxiety", "sadness", "anger"},
    },
    "emotion_intensity": {
        "required": ["id", "text", "intensity"],
        "optional": ["note"],
    },
    "risk_4class": {
        "required": ["id", "text", "level", "binary"],
        "optional": ["fusion_source", "tags", "note"],
        "valid_levels": {"level_0", "level_1", "level_2", "level_3"},
    },
    "risk_binary": {
        "required": ["id", "text", "positive"],
        "optional": ["note"],
    },
    "risk_boundary": {
        "required": ["id", "text", "level", "tags"],
        "optional": ["note"],
        "valid_levels": {"level_0", "level_1", "level_2", "level_3"},
    },
    "risk_multiturn": {
        "required": ["id", "scenario", "turns"],
        "turn_required": ["text", "turn_level", "session_level"],
        "valid_levels": {"level_0", "level_1", "level_2", "level_3"},
    },
    "recommend_gate": {
        "required": ["id", "user_input", "emotion_state", "risk_state", "expected"],
        "optional": ["conversation_summary", "user_profile", "tags", "scenario", "difficulty", "note"],
        "valid_modes": {"soft", "hard", "none", "safety_only"},
    },
}


def find_dataset_path(name: str, data_dir: Path) -> Path:
    """查找数据集文件（支持子目录和平铺两种结构）"""
    # 子目录结构: datasets/emotion/emotion_classify.jsonl
    path = data_dir / name / f"{name}.jsonl"
    if path.exists():
        return path
    # 平铺结构: datasets/emotion_classify.jsonl
    path = data_dir / f"{name}.jsonl"
    if path.exists():
        return path
    # 尝试 datasets 下的子目录
    for sub in data_dir.iterdir():
        if sub.is_dir():
            candidate = sub / f"{name}.jsonl"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"数据集 '{name}' 未找到。搜索路径: {data_dir}"
    )


def validate_row(row: Dict, schema: Dict, name: str, line_no: int):
    """校验单行数据"""
    row_id = row.get("id", f"line_{line_no}")

    # 如果缺少 id 字段，自动补一个
    if "id" not in row:
        row["id"] = f"auto_{name}_{line_no:04d}"

    for field in schema.get("required", []):
        if field == "id":
            continue  # 已自动补全
        if field not in row:
            raise ValueError(
                f"[{name}] 第 {line_no} 行 (id={row_id}) 缺少必需字段 '{field}'"
            )

    # 校验合法取值
    valid_labels = schema.get("valid_labels")
    if valid_labels:
        label = row.get("label") or row.get("level")
        if label and label not in valid_labels:
            raise ValueError(
                f"[{name}] id={row_id} 标签 '{label}' 不在允许集合 {valid_labels} 中"
            )

    # 校验嵌套 turns 字段
    turn_fields = schema.get("turn_required", [])
    if turn_fields:
        turns = row.get("turns", [])
        if not turns:
            raise ValueError(
                f"[{name}] id={row_id} turns 为空"
            )
        for i, turn in enumerate(turns):
            for field in turn_fields:
                if field not in turn:
                    raise ValueError(
                        f"[{name}] id={row_id} turn[{i}] 缺少 '{field}'"
                    )
            # 校验 turn 内的等级
            valid_levels = schema.get("valid_levels")
            if valid_levels:
                for key in ("turn_level", "session_level"):
                    if key in turn and turn[key] not in valid_levels:
                        raise ValueError(
                            f"[{name}] id={row_id} turn[{i}] '{key}={turn[key]}' "
                            f"不在允许集合 {valid_levels} 中"
                        )


def load_dataset(
    name: str,
    data_dir: Optional[Path] = None,
    case_limit: int = 0,
    skip_validation: bool = False,
) -> List[Dict[str, Any]]:
    """
    加载并校验 JSONL 数据集

    Args:
        name: 数据集名称（对应 schema 和文件名）
        data_dir: 数据目录，默认为 datasets/
        case_limit: 限制用例数（0=不限制）
        skip_validation: 跳过校验（快速开发用）

    Returns:
        数据行列表
    """
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    data_dir = data_dir or (base_dir / "datasets")

    path = find_dataset_path(name, data_dir)
    schema = SCHEMAS.get(name)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if schema and not skip_validation:
                validate_row(row, schema, name, line_no)
            rows.append(row)

    if case_limit > 0:
        rows = rows[:case_limit]

    return rows


def dataset_stats(name: str, data_dir: Optional[Path] = None) -> Dict:
    """
    数据集统计信息

    Returns:
        {"name": str, "n_cases": int, "label_distribution": {...}}
    """
    rows = load_dataset(name, data_dir)
    stats: Dict = {
        "name": name,
        "n_cases": len(rows),
    }

    # 标签分布
    for key in ("label", "level"):
        values = [r.get(key) for r in rows if r.get(key)]
        if values:
            dist = {}
            for v in values:
                dist[v] = dist.get(v, 0) + 1
            stats["label_distribution"] = dist
            break

    # risk_multiturn 统计
    if name == "risk_multiturn":
        turn_count = sum(len(r.get("turns", [])) for r in rows)
        stats["total_turns"] = turn_count

    return stats


def list_datasets(data_dir: Optional[Path] = None) -> List[str]:
    """列出所有可用的数据集"""
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    data_dir = data_dir or (base_dir / "datasets")

    found = []
    if not data_dir.exists():
        return found

    # 平铺
    for f in data_dir.glob("*.jsonl"):
        found.append(f.stem)

    # 子目录
    for sub in data_dir.iterdir():
        if sub.is_dir():
            for f in sub.glob("*.jsonl"):
                found.append(f.stem)

    return sorted(found)

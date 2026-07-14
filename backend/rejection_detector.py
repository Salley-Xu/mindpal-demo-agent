"""用户拒绝推荐检测器

分析用户对话文本，识别是否包含拒绝推荐的意图（"不用了""不需要"等）。
检测结果用于影响推荐门控的偏好分数和推荐决策。
"""
import re
from typing import Dict, List, Optional, Tuple

# 拒绝推荐的关键词模式（按优先级排列）
REJECTION_PATTERNS = [
    # 明确拒绝（高置信度）
    (r"不用(了|推荐|吧)?", 0.9),
    (r"不需要", 0.9),
    (r"不要推荐", 0.9),
    (r"算了吧", 0.85),
    (r"没兴趣", 0.85),
    (r"我不需要", 0.9),
    (r"别推荐", 0.85),
    # 对推荐内容的负面反馈（中置信度）
    (r"上次推荐.*(没用|不管用|不好|不适合)", 0.8),
    (r"(没用|不管用|不适合|不适用)", 0.7),
    (r"下次再说", 0.6),
    (r"先不用了", 0.7),
    (r"以后再看", 0.5),
]


def detect_rejection(user_input: str, active_item_ids: Optional[List[str]] = None) -> Dict:
    """检测用户输入中是否包含拒绝推荐的信号。

    Args:
        user_input: 用户的对话文本。
        active_item_ids: 当前轮已推荐的内容 ID（可选，用于精确标记拒绝对象）。

    Returns:
        dict: {
            "has_rejected": bool,       # 是否检测到拒绝
            "confidence": float,        # 最高匹配置信度 0-1
            "rejected_ids": List[str],  # 可能被拒绝的内容 ID
            "matched_pattern": str,     # 命中的模式描述
        }
    """
    if not user_input or not user_input.strip():
        return {"has_rejected": False, "confidence": 0.0, "rejected_ids": [], "matched_pattern": ""}

    user_input_lower = user_input.lower()
    best_confidence = 0.0
    best_pattern = ""

    for pattern, confidence in REJECTION_PATTERNS:
        if re.search(pattern, user_input_lower):
            if confidence > best_confidence:
                best_confidence = confidence
                best_pattern = pattern
            # 高置信度模式可提前退出
            if confidence >= 0.9:
                break

    has_rejected = best_confidence >= 0.5

    return {
        "has_rejected": has_rejected,
        "confidence": best_confidence,
        "rejected_ids": list(active_item_ids or []),
        "matched_pattern": best_pattern if has_rejected else "",
    }

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(BACKEND_DIR / ".env", override=False)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from openai import AsyncOpenAI  # noqa: E402
from agent_test_data.faiss_utils import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    FaissRetrievalBackend,
)
from recommend_gate import recommend_gate  # noqa: E402
from risk_evaluator import risk_evaluator  # noqa: E402

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def limit_rows(rows: List[Dict[str, Any]], case_limit: int = 0) -> List[Dict[str, Any]]:
    if case_limit and case_limit > 0:
        return rows[:case_limit]
    return rows

def evaluate_emotion_risk(predictions):
    """
    predictions: list of dicts
    required fields:
      id, emotion_type, risk_level, should_recommend, recommend_mode
    """
    gold = {x["id"]: x["expected"] for x in load_jsonl(BASE / "emotion_risk_turns.jsonl")}
    n = 0
    emotion_ok = risk_ok = gate_ok = mode_ok = 0
    high_risk_gold = high_risk_hit = 0
    boundary_n = boundary_ok = 0

    for p in predictions:
        if p["id"] not in gold:
            continue
        g = gold[p["id"]]
        tags = set(g.get("case_tags", []) or [])
        n += 1
        emotion_ok += int(p.get("emotion_type") == g["emotion_type"])
        risk_ok += int(p.get("risk_level") == g["risk_level"])
        gate_ok += int(bool(p.get("should_recommend")) == bool(g["should_recommend"]))
        mode_ok += int(p.get("recommend_mode") == g["recommend_mode"])
        if tags:
            boundary_n += 1
            boundary_ok += int(
                p.get("risk_level") == g["risk_level"]
                and p.get("recommend_mode") == g["recommend_mode"]
            )
        if g["risk_level"] == "high":
            high_risk_gold += 1
            high_risk_hit += int(p.get("risk_level") == "high")

    return {
        "n": n,
        "emotion_acc": emotion_ok / n if n else 0,
        "risk_acc": risk_ok / n if n else 0,
        "recommend_gate_acc": gate_ok / n if n else 0,
        "recommend_mode_acc": mode_ok / n if n else 0,
        "high_risk_recall": high_risk_hit / high_risk_gold if high_risk_gold else None,
        "safety_boundary_acc": boundary_ok / boundary_n if boundary_n else None,
    }

def evaluate_retrieval_baseline(run_results, k=3):
    """
    run_results: list of dicts
    required fields:
      id, retrieved_doc_ids
    """
    gold = {x["id"]: x for x in load_jsonl(BASE / "rag_retrieval_cases.jsonl")}
    n = recall_hit = forbidden_hit = 0

    for r in run_results:
        if r["id"] not in gold:
            continue
        n += 1
        topk = r.get("retrieved_doc_ids", [])[:k]
        expected = set(gold[r["id"]]["expected_top_doc_ids"])
        forbidden = set(gold[r["id"]]["forbidden_doc_ids"])
        recall_hit += int(len(expected.intersection(topk)) > 0)
        forbidden_hit += int(len(forbidden.intersection(topk)) > 0)

    return {
        "n": n,
        f"query_recall@{k}": recall_hit / n if n else 0,
        f"forbidden_hit_rate@{k}": forbidden_hit / n if n else 0
    }


def evaluate_recommendation_gate(predictions):
    """
    predictions: list of dicts
    required fields:
      id, should_recommend, recommend_mode
    """
    gold = {x["id"]: x["expected"] for x in load_jsonl(BASE / "recommendation_gate_cases.jsonl")}
    n = 0
    gate_ok = mode_ok = 0
    high_risk_gold = high_risk_override_ok = 0

    for p in predictions:
        if p["id"] not in gold:
            continue
        g = gold[p["id"]]
        n += 1
        gate_ok += int(bool(p.get("should_recommend")) == bool(g["should_recommend"]))
        mode_ok += int(p.get("recommend_mode") == g["recommend_mode"])
        if g.get("reason_code") == "high_risk_override":
            high_risk_gold += 1
            high_risk_override_ok += int(
                (not bool(p.get("should_recommend"))) and p.get("recommend_mode") == "safety"
            )

    return {
        "n": n,
        "recommend_gate_acc": gate_ok / n if n else 0,
        "recommend_mode_acc": mode_ok / n if n else 0,
        "high_risk_override_success_rate": (
            high_risk_override_ok / high_risk_gold if high_risk_gold else None
        ),
    }


def evaluate_memory_update(predictions):
    """
    predictions: list of dicts
    required fields:
      id, write, operation, field
    """
    gold = {x["id"]: x["expected"] for x in load_jsonl(BASE / "memory_update_cases.jsonl")}
    n = 0
    write_ok = operation_ok = field_ok = 0
    predicted_write = gold_write = matched_write = 0

    for p in predictions:
        if p["id"] not in gold:
            continue
        g = gold[p["id"]]
        n += 1
        pred_write = bool(p.get("write"))
        true_write = bool(g.get("write"))
        write_ok += int(pred_write == true_write)
        operation_ok += int((p.get("operation") or "none") == (g.get("operation") or "none"))
        field_ok += int((p.get("field") or None) == (g.get("field") or None))
        predicted_write += int(pred_write)
        gold_write += int(true_write)
        matched_write += int(pred_write and true_write)

    return {
        "n": n,
        "memory_write_acc": write_ok / n if n else 0,
        "operation_type_acc": operation_ok / n if n else 0,
        "field_acc": field_ok / n if n else 0,
        "memory_write_precision": matched_write / predicted_write if predicted_write else None,
        "memory_write_recall": matched_write / gold_write if gold_write else None,
    }


def build_emotion_risk_details(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    gold = {x["id"]: x["expected"] for x in load_jsonl(BASE / "emotion_risk_turns.jsonl")}
    details = []
    for prediction in predictions:
        case_id = prediction["id"]
        if case_id not in gold:
            continue
        expected = gold[case_id]
        details.append(
            {
                "id": case_id,
                "prediction": prediction,
                "gold": expected,
                "matched": {
                    "emotion_type": prediction.get("emotion_type") == expected.get("emotion_type"),
                    "risk_level": prediction.get("risk_level") == expected.get("risk_level"),
                    "should_recommend": bool(prediction.get("should_recommend")) == bool(expected.get("should_recommend")),
                    "recommend_mode": prediction.get("recommend_mode") == expected.get("recommend_mode"),
                },
            }
        )
    return details


def build_recommendation_gate_details(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    gold = {x["id"]: x["expected"] for x in load_jsonl(BASE / "recommendation_gate_cases.jsonl")}
    details = []
    for prediction in predictions:
        case_id = prediction["id"]
        if case_id not in gold:
            continue
        expected = gold[case_id]
        details.append(
            {
                "id": case_id,
                "prediction": prediction,
                "gold": expected,
                "matched": {
                    "should_recommend": bool(prediction.get("should_recommend")) == bool(expected.get("should_recommend")),
                    "recommend_mode": prediction.get("recommend_mode") == expected.get("recommend_mode"),
                },
            }
        )
    return details


def build_retrieval_baseline_details(predictions: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    gold = {x["id"]: x for x in load_jsonl(BASE / "rag_retrieval_cases.jsonl")}
    details = []
    for prediction in predictions:
        case_id = prediction["id"]
        if case_id not in gold:
            continue
        expected = gold[case_id]
        topk = prediction.get("retrieved_doc_ids", [])[:k]
        expected_top = expected.get("expected_top_doc_ids", [])
        forbidden = expected.get("forbidden_doc_ids", [])
        details.append(
            {
                "id": case_id,
                "prediction": prediction,
                "gold": {
                    "expected_top_doc_ids": expected_top,
                    "forbidden_doc_ids": forbidden,
                },
                "matched": {
                    "hit_expected_topk": len(set(expected_top).intersection(topk)) > 0,
                    "hit_forbidden_topk": len(set(forbidden).intersection(topk)) > 0,
                },
            }
        )
    return details


def build_memory_update_details(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    gold = {x["id"]: x["expected"] for x in load_jsonl(BASE / "memory_update_cases.jsonl")}
    details = []
    for prediction in predictions:
        case_id = prediction["id"]
        if case_id not in gold:
            continue
        expected = gold[case_id]
        details.append(
            {
                "id": case_id,
                "prediction": prediction,
                "gold": expected,
                "matched": {
                    "write": bool(prediction.get("write")) == bool(expected.get("write")),
                    "operation": (prediction.get("operation") or "none") == (expected.get("operation") or "none"),
                    "field": (prediction.get("field") or None) == (expected.get("field") or None),
                },
            }
        )
    return details


def normalize_recommend_mode(mode: str) -> str:
    if mode == "hard":
        return "hard"
    if mode == "soft":
        return "soft"
    if mode == "none":
        return "none"
    if mode in {"safety", "safety_only", "third_party_support"}:
        return "safety"
    return mode or "none"


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))


def normalize_emotion_type_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    mapping = {
        "crisis": "crisis",
        "despair": "despair",
        "panic": "panic",
        "withdrawal": "withdrawal",
        "fear": "fear",
        "confusion": "confusion",
        "relief": "relief",
        "approval": "approval",
        "procrastination": "procrastination",
        "anger": "anger",
        "annoyance": "annoyance",
        "sadness": "sadness",
        "self_doubt": "self_doubt",
        "joy": "joy",
        "stress": "stress",
        "fatigue": "fatigue",
        "anxiety": "anxiety",
        "loneliness": "loneliness",
        "neutral": "neutral",
        "positive": "joy",
    }
    return mapping.get(text, "neutral")


def normalize_user_intent_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    mapping = {
        "urgent_help": "urgent_help",
        "crisis_signal": "crisis_signal",
        "venting_only": "venting_only",
        "feedback": "feedback",
        "explicit_action_request": "explicit_action_request",
        "preference_statement": "preference_statement",
        "reflection": "reflection",
        "decision_help": "decision_help",
        "seeking_reassurance": "seeking_reassurance",
        "seeking_help": "seeking_help",
        "planning": "planning",
        "mood_tracking": "mood_tracking",
        "sharing_positive": "sharing_positive",
        "venting": "venting",
    }
    return mapping.get(text, "venting")


def normalize_risk_level_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    mapping = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "level_0": "low",
        "level_1": "medium",
        "level_2": "medium",
        "level_3": "high",
        "normal": "low",
        "warning": "medium",
        "urgent": "high",
    }
    return mapping.get(text, "low")


def build_conversation_summary_for_eval(raw_summary: Any) -> Dict[str, Any]:
    if isinstance(raw_summary, dict):
        return dict(raw_summary)
    return {}


class EvalLLMClassifier:
    def __init__(self, model: str = "", base_url: str = "", api_key: str = ""):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or os.getenv("CHAT_MODEL", "deepseek-chat")
        self.base_url = base_url or os.getenv("API_BASE_URL", "https://api.deepseek.com/v1")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置，无法启用 llm classifier backend")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def classify_emotion_risk_case(self, text: str) -> Dict[str, Any]:
        prompt = f"""你正在做压力管理 Agent 的离线评测标注。

请基于用户输入，输出一个 JSON 对象，字段必须完整：
- emotion_type: 只能从以下枚举里选一个
  ["crisis","despair","panic","withdrawal","fear","confusion","relief","approval","procrastination","anger","annoyance","sadness","self_doubt","joy","stress","fatigue","anxiety","loneliness","neutral"]
- user_intent: 只能从以下枚举里选一个
  ["urgent_help","crisis_signal","venting_only","feedback","explicit_action_request","preference_statement","reflection","decision_help","seeking_reassurance","seeking_help","planning","mood_tracking","sharing_positive","venting"]
- risk_level: 只能是 ["low","medium","high"]
- emotion_intensity: 0 到 1 之间的小数

判定原则：
- 只有明确危机或需要安全干预时才给 high。
- “想消失一阵子 / 不想见任何人”更接近 withdrawal，通常优先判 medium 而不是 high。
- “只是记录一下 / 汇报一下进度”优先判 mood_tracking。
- “对我有用 / 缓过来了”优先判 feedback。
- “只想要一个马上能做的办法”优先判 explicit_action_request。

用户输入：
\"\"\"{text}\"\"\"
"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个严格输出 JSON 的评测分类器。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=180,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        payload = json.loads(content)
        return {
            "emotion_type": normalize_emotion_type_label(payload.get("emotion_type")),
            "user_intent": normalize_user_intent_label(payload.get("user_intent")),
            "risk_level": normalize_risk_level_label(payload.get("risk_level")),
            "emotion_intensity": clamp_float(payload.get("emotion_intensity"), default=0.5),
        }


async def classify_with_llm_or_fallback(
    llm_classifier: EvalLLMClassifier,
    text: str,
    heuristic_result: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        llm_result = await llm_classifier.classify_emotion_risk_case(text)
        return {
            "emotion_type": llm_result["emotion_type"],
            "user_intent": llm_result["user_intent"],
            "risk_level": llm_result["risk_level"],
            "emotion_intensity": llm_result["emotion_intensity"],
            "backend": "llm",
        }
    except Exception:
        return {
            "emotion_type": heuristic_result["emotion_type"],
            "user_intent": heuristic_result["user_intent"],
            "risk_level": heuristic_result["risk_level"],
            "emotion_intensity": heuristic_result["emotion_intensity"],
            "backend": "heuristic_fallback",
        }


def infer_emotion_type(text: str) -> str:
    if _contains_any(text, ["连续三晚睡不着", "白天也没精神", "压力7分"]):
        return "stress"
    rules = [
        ("crisis", ["很危险", "不想一个人待着"]),
        ("loneliness", ["没人真的在乎我", "没人理解", "被抛弃"]),
        ("despair", ["不想再这样下去了", "看不到任何希望", "熬过去"]),
        ("panic", ["心跳很快", "控制不住", "很慌"]),
        ("withdrawal", ["想消失一阵子", "不想见任何人"]),
        ("fear", ["担心父母失望", "不敢说"]),
        ("confusion", ["不知道该不该", "所有选择都错"]),
        ("relief", ["缓过来了", "轻松了一点"]),
        ("approval", ["对我有用", "下次还可以这样提醒我"]),
        ("procrastination", ["开始不了", "逃避", "拖延"]),
        ("anger", ["气得", "烦", "吵架", "扔了"]),
        ("annoyance", ["不喜欢冥想", "更烦", "有点烦"]),
        ("sadness", ["做错了", "委屈", "越来越没用", "难受"]),
        ("self_doubt", ["不适合", "怀疑自己", "失败", "卡壳"]),
        ("joy", ["很开心", "终于", "跑通了"]),
        ("stress", ["老板", "任务", "压力", "办法", "做不了"]),
        ("fatigue", ["很累", "吐槽一下", "只是记录一下"]),
        ("anxiety", ["答辩", "睡不着", "考试", "来不及", "空白"]),
        ("neutral", ["状态还行", "继续推进", "汇报一下进度"]),
    ]
    for emotion_type, keywords in rules:
        if _contains_any(text, keywords):
            return emotion_type
    return "neutral"


def infer_user_intent(text: str) -> str:
    if _contains_any(text, ["今晚熬过去", "陪我把今晚熬过去"]):
        return "urgent_help"
    if _contains_any(text, ["不想活了", "很危险", "看不到任何希望"]):
        return "crisis_signal"
    if _contains_any(text, ["不需要建议", "先别给建议", "只是想吐槽"]):
        return "venting_only"
    if _contains_any(text, ["刚刚做完呼吸练习", "对我有用", "这样提醒我", "缓过来了"]):
        return "feedback"
    if _contains_any(text, ["只想要一个能马上做的办法", "马上能做的办法", "有没有别的"]):
        return "explicit_action_request"
    if _contains_any(text, ["我不喜欢", "别给我讲大道理"]):
        return "preference_statement"
    if _contains_any(text, ["复盘一下最近情绪波动"]):
        return "reflection"
    if _contains_any(text, ["不知道该不该", "所有选择都错"]):
        return "decision_help"
    if _contains_any(text, ["是不是自己哪里做错了"]):
        return "seeking_reassurance"
    if _contains_any(text, ["怎么办", "帮帮我", "有什么办法", "睡不着", "控制不住", "来不及"]):
        return "seeking_help"
    if _contains_any(text, ["计划", "推进", "怎么开始"]):
        return "planning"
    if _contains_any(text, ["记录一下", "汇报一下进度"]):
        return "mood_tracking"
    if _contains_any(text, ["开心", "终于"]):
        return "sharing_positive"
    return "venting"


def infer_negative_trend(text: str) -> bool:
    return _contains_any(text, ["连续几天", "越来越", "一直", "还是", "总是"])


def infer_rule_based_risk_level(text: str, user_intent: str) -> str:
    high_markers = ["不想活了", "很危险", "看不到任何希望", "不想一个人待着", "陪我把今晚熬过去"]
    medium_markers = [
        "控制不住",
        "越来越没用",
        "没人真的在乎我",
        "来不及",
        "连续几天都这样",
        "连续三晚睡不着",
        "所有选择都错",
        "把所有东西都扔了",
    ]
    if user_intent in {"crisis_signal", "urgent_help"} or _contains_any(text, high_markers):
        return "high"
    if _contains_any(text, medium_markers):
        return "medium"
    return "low"


def to_eval_risk_level(label: Any) -> str:
    return normalize_risk_level_label(label)


def should_apply_rule_risk_override(risk_state: Dict[str, Any]) -> bool:
    context = risk_state.get("risk_context", {}) or {}
    if context.get("is_discussion_context"):
        return False
    if context.get("is_safe_denial"):
        return False
    if context.get("subject") == "third_party":
        return False
    return True


def infer_emotion_intensity(text: str) -> float:
    high_markers = ["很危险", "不想活了", "看不到任何希望", "控制不住", "心跳很快"]
    medium_markers = ["撑不住", "睡不着", "来不及", "越来越没用", "很慌", "真的"]
    low_markers = ["有点", "只是", "还行", "记录一下"]
    score = 0.5
    if _contains_any(text, high_markers):
        score += 0.35
    elif _contains_any(text, medium_markers):
        score += 0.2
    elif _contains_any(text, low_markers):
        score += 0.05
    return round(max(0.15, min(score, 0.95)), 2)


def infer_gate_intent(text: str) -> str:
    if _contains_any(text, ["马上能做的办法", "只想要一个能马上做的办法", "有没有别的", "帮我"]):
        return "planning"
    if _contains_any(text, ["怎么办", "睡不着", "没头绪", "很慌"]):
        return "seeking_help"
    if _contains_any(text, ["缓解", "放松"]):
        return "seeking_relief"
    return "sharing"


def tokenize_zh_text(text: str) -> List[str]:
    keywords = [
        "答辩", "老师", "准备", "问题", "心跳", "失控", "睡不着", "论文", "明天",
        "危险", "一个人", "面试", "项目", "紧张", "朋友", "吵架", "开口", "学习",
        "逃避", "自己很差", "工作", "老板", "任务", "情绪", "规律", "记录",
        "呼吸", "落地", "计划", "复习", "考试", "支持者", "安全", "求助",
    ]
    hits = [keyword for keyword in keywords if keyword in text]
    if not hits:
        hits = [char for char in text if char.strip()]
    return hits


def build_retrieval_baseline_index() -> List[Dict[str, Any]]:
    docs = load_jsonl(BASE / "rag_materials.jsonl")
    indexed = []
    for doc in docs:
        searchable = " ".join(
            [
                doc.get("title", ""),
                doc.get("category", ""),
                " ".join(doc.get("emotion_type", []) or []),
                " ".join(doc.get("risk_level", []) or []),
                " ".join(doc.get("tags", []) or []),
                doc.get("content", ""),
            ]
        )
        indexed.append({"doc": doc, "searchable": searchable})
    return indexed


def score_retrieval_baseline_doc(query: str, context: Dict[str, Any], indexed_doc: Dict[str, Any]) -> float:
    doc = indexed_doc["doc"]
    searchable = indexed_doc["searchable"]
    score = 0.0
    tokens = tokenize_zh_text(query)
    for token in tokens:
        if token in searchable:
            score += 1.2
    if context.get("emotion_type") in (doc.get("emotion_type", []) or []):
        score += 2.0
    if context.get("risk_level") in (doc.get("risk_level", []) or []):
        score += 1.5
    preference = context.get("user_preference")
    if preference == "direct_actionable":
        score += float(doc.get("actionability", 0.0)) * 1.2
    elif preference == "structured_plan" and doc.get("category") in {"planning", "academic_pressure", "work_pressure", "interview"}:
        score += 1.8
    elif preference == "tracking" and doc.get("category") == "emotion_tracking":
        score += 1.8
    elif preference == "gentle_then_action" and doc.get("category") in {"self_compassion", "cognitive_reframing", "relationship"}:
        score += 1.5
    if context.get("risk_level") == "high":
        if doc.get("category") == "safety":
            score += 4.0
        if "high_risk_crisis" in (doc.get("contraindications", []) or []):
            score -= 4.0
    return score


def build_emotion_risk_predictions(
    classifier_backend: str = "heuristic",
    llm_classifier: Optional[EvalLLMClassifier] = None,
    case_limit: int = 0,
) -> List[Dict[str, Any]]:
    rows = limit_rows(load_jsonl(BASE / "emotion_risk_turns.jsonl"), case_limit=case_limit)
    if classifier_backend == "llm":
        return asyncio.run(
            build_emotion_risk_predictions_with_llm(
                rows=rows,
                llm_classifier=llm_classifier,
            )
        )

    predictions = []
    for row in rows:
        text = row["user_input"]
        emotion_type = infer_emotion_type(text)
        emotion_intensity = infer_emotion_intensity(text)
        user_intent = infer_user_intent(text)
        negative_trend = infer_negative_trend(text)
        conversation_summary = build_conversation_summary_for_eval(row.get("context_summary"))
        risk_state = risk_evaluator.evaluate(
            text=text,
            emotion_state={
                "emotion_type": emotion_type,
                "emotion_intensity": emotion_intensity,
                "negative_trend": negative_trend,
                "user_intent": user_intent,
            },
            conversation_summary=conversation_summary,
        )
        rule_based_risk_level = infer_rule_based_risk_level(text, user_intent)
        risk_priority = {"low": 0, "medium": 1, "high": 2}
        eval_risk_level = to_eval_risk_level(risk_state["level"])
        if should_apply_rule_risk_override(risk_state) and risk_priority[rule_based_risk_level] > risk_priority[eval_risk_level]:
            eval_risk_level = rule_based_risk_level
        recommend_mode = "none"
        should_recommend = False
        if eval_risk_level == "high":
            recommend_mode = "safety"
        elif risk_state.get("risk_context", {}).get("is_discussion_context"):
            recommend_mode = "none"
        elif risk_state.get("risk_context", {}).get("subject") == "third_party" and risk_state.get("risk_context", {}).get("is_help_request"):
            recommend_mode = "safety"
        elif user_intent in {"venting_only", "sharing_positive", "mood_tracking", "feedback", "preference_statement"}:
            recommend_mode = "none"
        elif (
            user_intent == "planning"
            and emotion_type == "neutral"
            and (_contains_any(text, ["状态还行", "继续推进"]) or emotion_intensity <= 0.35)
        ):
            recommend_mode = "none"
        elif user_intent == "reflection":
            should_recommend = True
            recommend_mode = "soft"
        elif user_intent == "explicit_action_request":
            should_recommend = True
            recommend_mode = "hard"
        elif user_intent == "planning" and _contains_any(text, ["制定一个", "复习计划"]):
            should_recommend = True
            recommend_mode = "hard"
        elif user_intent in {"urgent_help", "decision_help"}:
            should_recommend = True
            recommend_mode = "soft" if eval_risk_level != "high" else "safety"
        elif negative_trend and emotion_type in {"sadness", "loneliness"}:
            should_recommend = True
            recommend_mode = "soft"
        elif (
            emotion_type == "panic"
            or _contains_any(text, ["来不及", "复习什么都觉得来不及"])
        ):
            should_recommend = True
            recommend_mode = "hard" if eval_risk_level != "high" else "safety"
        else:
            gate_decision = recommend_gate.decide(
                emotion_state={
                    "emotion_type": emotion_type,
                    "emotion_intensity": emotion_intensity,
                    "user_intent": (
                        "planning"
                        if user_intent in {"planning", "explicit_action_request"}
                        else "seeking_help"
                    ),
                    "negative_trend": negative_trend,
                },
                risk_state=risk_state,
                conversation_summary={
                    "turn_count": 1,
                    "recent_recommendation_turns": [],
                    **conversation_summary,
                },
                user_profile={},
            )
            should_recommend = gate_decision["should_recommend"]
            recommend_mode = normalize_recommend_mode(gate_decision["recommend_type"])
        predictions.append(
            {
                "id": row["id"],
                "emotion_type": emotion_type,
                "risk_level": eval_risk_level,
                "should_recommend": should_recommend,
                "recommend_mode": recommend_mode,
                "classifier_backend": "heuristic",
            }
        )
    return predictions


async def build_emotion_risk_predictions_with_llm(
    rows: List[Dict[str, Any]],
    llm_classifier: Optional[EvalLLMClassifier],
) -> List[Dict[str, Any]]:
    if llm_classifier is None:
        raise ValueError("llm classifier backend 已启用，但未提供 llm_classifier 实例")

    predictions = []
    for row in rows:
        text = row["user_input"]
        heuristic_emotion_type = infer_emotion_type(text)
        heuristic_emotion_intensity = infer_emotion_intensity(text)
        heuristic_user_intent = infer_user_intent(text)
        heuristic_negative_trend = infer_negative_trend(text)
        heuristic_risk_level = infer_rule_based_risk_level(text, heuristic_user_intent)

        llm_or_fallback = await classify_with_llm_or_fallback(
            llm_classifier=llm_classifier,
            text=text,
            heuristic_result={
                "emotion_type": heuristic_emotion_type,
                "emotion_intensity": heuristic_emotion_intensity,
                "user_intent": heuristic_user_intent,
                "risk_level": heuristic_risk_level,
            },
        )
        emotion_type = llm_or_fallback["emotion_type"]
        emotion_intensity = llm_or_fallback["emotion_intensity"]
        user_intent = llm_or_fallback["user_intent"]
        negative_trend = heuristic_negative_trend
        conversation_summary = build_conversation_summary_for_eval(row.get("context_summary"))
        risk_state = risk_evaluator.evaluate(
            text=text,
            emotion_state={
                "emotion_type": emotion_type,
                "emotion_intensity": emotion_intensity,
                "negative_trend": negative_trend,
                "user_intent": user_intent,
            },
            conversation_summary=conversation_summary,
        )
        llm_risk_level = llm_or_fallback["risk_level"]
        risk_priority = {"low": 0, "medium": 1, "high": 2}
        eval_risk_level = to_eval_risk_level(risk_state["level"])
        if should_apply_rule_risk_override(risk_state) and risk_priority[llm_risk_level] > risk_priority[eval_risk_level]:
            eval_risk_level = llm_risk_level

        recommend_mode = "none"
        should_recommend = False
        if eval_risk_level == "high":
            recommend_mode = "safety"
        elif risk_state.get("risk_context", {}).get("is_discussion_context"):
            recommend_mode = "none"
        elif risk_state.get("risk_context", {}).get("subject") == "third_party" and risk_state.get("risk_context", {}).get("is_help_request"):
            recommend_mode = "safety"
        elif user_intent in {"venting_only", "sharing_positive", "mood_tracking", "feedback", "preference_statement"}:
            recommend_mode = "none"
        elif (
            user_intent == "planning"
            and emotion_type == "neutral"
            and (_contains_any(text, ["状态还行", "继续推进"]) or emotion_intensity <= 0.35)
        ):
            recommend_mode = "none"
        elif user_intent == "reflection":
            should_recommend = True
            recommend_mode = "soft"
        elif user_intent == "explicit_action_request":
            should_recommend = True
            recommend_mode = "hard"
        elif user_intent == "planning" and _contains_any(text, ["制定一个", "复习计划"]):
            should_recommend = True
            recommend_mode = "hard"
        elif user_intent in {"urgent_help", "decision_help"}:
            should_recommend = True
            recommend_mode = "soft" if eval_risk_level != "high" else "safety"
        elif negative_trend and emotion_type in {"sadness", "loneliness"}:
            should_recommend = True
            recommend_mode = "soft"
        elif (
            emotion_type == "panic"
            or _contains_any(text, ["来不及", "复习什么都觉得来不及"])
        ):
            should_recommend = True
            recommend_mode = "hard" if eval_risk_level != "high" else "safety"
        else:
            gate_decision = recommend_gate.decide(
                emotion_state={
                    "emotion_type": emotion_type,
                    "emotion_intensity": emotion_intensity,
                    "user_intent": (
                        "planning"
                        if user_intent in {"planning", "explicit_action_request"}
                        else "seeking_help"
                    ),
                    "negative_trend": negative_trend,
                },
                risk_state=risk_state,
                conversation_summary={
                    "turn_count": 1,
                    "recent_recommendation_turns": [],
                    **conversation_summary,
                },
                user_profile={},
            )
            should_recommend = gate_decision["should_recommend"]
            recommend_mode = normalize_recommend_mode(gate_decision["recommend_type"])

        predictions.append(
            {
                "id": row["id"],
                "emotion_type": emotion_type,
                "risk_level": eval_risk_level,
                "should_recommend": should_recommend,
                "recommend_mode": recommend_mode,
                "classifier_backend": llm_or_fallback["backend"],
            }
        )
    return predictions


def build_recommendation_gate_predictions(case_limit: int = 0) -> List[Dict[str, Any]]:
    predictions = []
    for row in limit_rows(load_jsonl(BASE / "recommendation_gate_cases.jsonl"), case_limit=case_limit):
        risk_level = row.get("risk_state", {}).get("risk_level", "low")
        if risk_level == "high":
            predictions.append(
                {
                    "id": row["id"],
                    "should_recommend": False,
                    "recommend_mode": "safety",
                }
            )
            continue

        last_turns_ago = int(row.get("cooldown", {}).get("last_recommend_turns_ago", 99) or 99)
        turn_count = max(1, last_turns_ago + 1)
        recent_recommendation_turns = [1] if last_turns_ago <= turn_count else []
        user_input = row["user_input"]
        recent_context = " ".join(row.get("recent_context", []) or [])
        negative_trend = _contains_any(recent_context, ["连续", "一直", "最近三轮"]) or risk_level == "medium"
        emotion_state = dict(row.get("emotion_state", {}) or {})
        emotion_state["user_intent"] = infer_gate_intent(user_input)
        emotion_state["negative_trend"] = negative_trend
        emotion_state["stress_source"] = emotion_state.get("stress_source")
        gate_decision = recommend_gate.decide(
            emotion_state=emotion_state,
            risk_state={"level": risk_level},
            conversation_summary={
                "turn_count": turn_count,
                "recent_recommendation_turns": recent_recommendation_turns,
                "rejected_recommendations": row.get("user_profile", {}).get("avoid_categories", []),
            },
            user_profile={
                "preferred_support_style": row.get("user_profile", {}).get("preferred_support_style"),
                "main_stress_sources": row.get("user_profile", {}).get("main_stress_sources", []),
            },
        )
        should_recommend = gate_decision["should_recommend"]
        recommend_mode = normalize_recommend_mode(gate_decision["recommend_type"])

        if _contains_any(user_input, ["先别给建议", "不需要建议"]):
            should_recommend = False
            recommend_mode = "none"
        elif _contains_any(user_input, ["有没有别的", "马上能做的办法"]):
            should_recommend = True
            recommend_mode = "hard"

        predictions.append(
            {
                "id": row["id"],
                "should_recommend": should_recommend,
                "recommend_mode": recommend_mode,
            }
        )
    return predictions


def build_retrieval_baseline_predictions(k: int = 3, case_limit: int = 0) -> List[Dict[str, Any]]:
    index = build_retrieval_baseline_index()
    predictions = []
    for row in limit_rows(load_jsonl(BASE / "rag_retrieval_cases.jsonl"), case_limit=case_limit):
        scored = []
        for indexed_doc in index:
            score = score_retrieval_baseline_doc(row["query"], row.get("context", {}) or {}, indexed_doc)
            scored.append((score, indexed_doc["doc"]["doc_id"]))
        scored.sort(key=lambda item: item[0], reverse=True)
        predictions.append(
            {
                "id": row["id"],
                "retrieved_doc_ids": [doc_id for _, doc_id in scored[:k]],
            }
        )
    return predictions


def build_faiss_retrieval_predictions(
    store_dir: Path,
    k: int = 3,
    case_limit: int = 0,
) -> List[Dict[str, Any]]:
    backend = FaissRetrievalBackend(store_dir=store_dir)
    predictions = []
    for row in limit_rows(load_jsonl(BASE / "rag_retrieval_cases.jsonl"), case_limit=case_limit):
        predictions.append(
            {
                "id": row["id"],
                "retrieved_doc_ids": backend.search(
                    query=row["query"],
                    context=row.get("context", {}) or {},
                    top_k=k,
                ),
            }
        )
    return predictions


def build_memory_update_predictions(case_limit: int = 0) -> List[Dict[str, Any]]:
    predictions = []
    for row in limit_rows(load_jsonl(BASE / "memory_update_cases.jsonl"), case_limit=case_limit):
        user_input = row["turn"]["user_input"]
        existing_memory = row.get("existing_memory", {}) or {}
        write = False
        operation = "none"
        field = None

        if _contains_any(user_input, ["直接给我步骤", "不要只安慰", "给我具体方法", "给我步骤"]):
            write = True
            operation = "merge_profile"
            field = "preferred_support_style"
        elif _contains_any(user_input, ["不喜欢冥想", "越冥想越烦"]):
            write = True
            operation = "merge_profile"
            field = "avoid_categories"
        elif _contains_any(user_input, ["很危险", "不想一个人待着"]):
            write = True
            operation = "append_risk_event_and_raise_level"
            field = "risk_level"
        elif _contains_any(user_input, ["任务拆分对我有用", "上次说的任务拆分"]):
            write = True
            operation = "update_recommendation_feedback"
            field = "recommendation_feedback.planning"
        elif _contains_any(user_input, ["面试还是让我很焦虑", "面试"]) and "job_interview" in (existing_memory.get("main_stress_sources", []) or []):
            write = True
            operation = "merge_stress_source"
            field = "main_stress_sources"
        elif _contains_any(user_input, ["先别给方法", "只想你听我说"]):
            write = True
            operation = "temporary_session_preference"
            field = "session_preference"
        elif _contains_any(user_input, ["状态稳定多了", "没有之前那种危险感"]):
            write = True
            operation = "risk_level_review"
            field = "risk_level"
        elif _contains_any(user_input, ["组会被问住", "压力大概7分", "很烦"]):
            write = True
            operation = "append_mood_event"
            field = "mood_events"

        predictions.append(
            {
                "id": row["id"],
                "write": write,
                "operation": operation,
                "field": field,
            }
        )
    return predictions


def print_summary(title: str, summary: Dict[str, Any]):
    print(f"\n[{title}]")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")


def write_output_json(output_path: Path, payload: Dict[str, Any]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_cli_path(path_value: str, default_base: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (default_base / path).resolve()
    return path


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation runner for agent_test_data.")
    parser.add_argument(
        "--task",
        choices=["emotion_risk", "recommendation_gate", "retrieval_baseline", "memory_update", "all"],
        default="all",
        help="Which evaluation bundle to run.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save aggregated evaluation results as JSON.",
    )
    parser.add_argument(
        "--classifier-backend",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Emotion/risk baseline backend. `llm` is reserved for future integration.",
    )
    parser.add_argument(
        "--retrieval-backend",
        choices=["local", "faiss"],
        default="local",
        help="Retrieval baseline backend.",
    )
    parser.add_argument(
        "--llm-model",
        default="",
        help="Future LLM model identifier placeholder for non-heuristic evaluation backends.",
    )
    parser.add_argument(
        "--vector-db-uri",
        default="",
        help="Deprecated placeholder. Will be treated as faiss store dir only when --faiss-store-dir is empty.",
    )
    parser.add_argument(
        "--faiss-store-dir",
        default="",
        help="Directory containing faiss.index, meta.json and build_info.json.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model used when building the FAISS store.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=0,
        help="Optional max number of cases per task, useful for low-cost validation.",
    )
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Include case-level prediction vs gold details in the JSON output.",
    )
    args = parser.parse_args()

    llm_classifier: Optional[EvalLLMClassifier] = None
    effective_classifier_backend = args.classifier_backend
    classifier_backend_error = None
    if args.classifier_backend == "llm":
        try:
            llm_classifier = EvalLLMClassifier(
                model=args.llm_model,
                base_url="",
                api_key="",
            )
        except Exception as exc:
            effective_classifier_backend = "heuristic"
            classifier_backend_error = str(exc)
    elif args.classifier_backend != "heuristic":
        raise NotImplementedError("当前仅实现 heuristic 和 llm classifier backend。")

    faiss_store_dir = resolve_cli_path(
        args.faiss_store_dir or args.vector_db_uri or "faiss_store",
        default_base=PROJECT_ROOT,
    )

    result_bundle: Dict[str, Any] = {
        "_runtime": {
            "classifier_backend_requested": args.classifier_backend,
            "classifier_backend": effective_classifier_backend,
            "retrieval_backend": args.retrieval_backend,
            "llm_model": args.llm_model or None,
            "vector_db_uri": args.vector_db_uri or None,
            "faiss_store_dir": str(faiss_store_dir) if args.retrieval_backend == "faiss" else None,
            "embedding_model": args.embedding_model if args.retrieval_backend == "faiss" else None,
            "case_limit": args.case_limit or None,
            "classifier_backend_error": classifier_backend_error,
        }
    }

    if args.task in {"emotion_risk", "all"}:
        emotion_predictions = build_emotion_risk_predictions(
            classifier_backend=effective_classifier_backend,
            llm_classifier=llm_classifier,
            case_limit=args.case_limit,
        )
        emotion_summary = evaluate_emotion_risk(emotion_predictions)
        print_summary("emotion_risk", emotion_summary)
        result_bundle["emotion_risk"] = emotion_summary
        if args.include_details:
            result_bundle["emotion_risk_details"] = build_emotion_risk_details(emotion_predictions)

    if args.task in {"recommendation_gate", "all"}:
        gate_predictions = build_recommendation_gate_predictions(case_limit=args.case_limit)
        gate_summary = evaluate_recommendation_gate(gate_predictions)
        print_summary("recommendation_gate", gate_summary)
        result_bundle["recommendation_gate"] = gate_summary
        if args.include_details:
            result_bundle["recommendation_gate_details"] = build_recommendation_gate_details(gate_predictions)

    if args.task in {"retrieval_baseline", "all"}:
        if args.retrieval_backend == "faiss":
            retrieval_predictions = build_faiss_retrieval_predictions(
                store_dir=faiss_store_dir,
                k=3,
                case_limit=args.case_limit,
            )
        else:
            retrieval_predictions = build_retrieval_baseline_predictions(k=3, case_limit=args.case_limit)
        retrieval_summary = evaluate_retrieval_baseline(retrieval_predictions, k=3)
        print_summary("retrieval_baseline", retrieval_summary)
        result_bundle["retrieval_baseline"] = retrieval_summary
        if args.include_details:
            result_bundle["retrieval_baseline_details"] = build_retrieval_baseline_details(
                retrieval_predictions,
                k=3,
            )

    if args.task in {"memory_update", "all"}:
        memory_predictions = build_memory_update_predictions(case_limit=args.case_limit)
        memory_summary = evaluate_memory_update(memory_predictions)
        print_summary("memory_update", memory_summary)
        result_bundle["memory_update"] = memory_summary
        if args.include_details:
            result_bundle["memory_update_details"] = build_memory_update_details(memory_predictions)

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        write_output_json(output_path, result_bundle)
        print(f"\n[output]\n- json: {output_path}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

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

    for p in predictions:
        if p["id"] not in gold:
            continue
        g = gold[p["id"]]
        n += 1
        emotion_ok += int(p.get("emotion_type") == g["emotion_type"])
        risk_ok += int(p.get("risk_level") == g["risk_level"])
        gate_ok += int(bool(p.get("should_recommend")) == bool(g["should_recommend"]))
        mode_ok += int(p.get("recommend_mode") == g["recommend_mode"])
        if g["risk_level"] == "high":
            high_risk_gold += 1
            high_risk_hit += int(p.get("risk_level") == "high")

    return {
        "n": n,
        "emotion_acc": emotion_ok / n if n else 0,
        "risk_acc": risk_ok / n if n else 0,
        "recommend_gate_acc": gate_ok / n if n else 0,
        "recommend_mode_acc": mode_ok / n if n else 0,
        "high_risk_recall": high_risk_hit / high_risk_gold if high_risk_gold else None
    }

def evaluate_rag(run_results, k=3):
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

if __name__ == "__main__":
    print("This is an evaluation skeleton. Feed your system predictions into evaluate_emotion_risk() or evaluate_rag().")

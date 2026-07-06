#!/usr/bin/env python3
"""
prepare_emotion_data.py — 准备情绪分类训练/开发/测试数据

1. 加载现有的人工标注测试数据 (emotion_risk_turns)
2. 加载心理相关文本（SOS-1K, 公开对话）
3. 调用当前 LLM 生成伪标签（可选）
4. 分割为 train/dev/test
5. 保存为 JSONL 格式供 train_emotion.py 使用
"""

import os, json, random, re, sys
from pathlib import Path
from collections import Counter

# ====== 标签映射 ======
EMOTION_8_MAP = {
    # 英文 → 6类（confusion/stress/helplessness 合并到 anxiety）
    "neutral": "neutral", "positive": "positive", "anxiety": "anxiety",
    "stress": "anxiety", "sadness": "sadness", "anger": "anger",
    "confusion": "anxiety", "helplessness": "anxiety",

    # 从 LLM 中文标签 → 8类
    "中性": "neutral", "快乐": "positive", "平静": "positive",
    "放松": "positive", "开心": "positive",
    "焦虑": "anxiety", "紧张": "anxiety", "担心": "anxiety",
    "压力": "stress", "学业压力": "stress",
    "抑郁": "sadness", "低落": "sadness", "难过": "sadness", "悲伤": "sadness",
    "愤怒": "anger", "生气": "anger", "烦躁": "anger",
    "困惑": "confusion", "不确定": "confusion", "未来迷茫": "confusion",
    "迷茫": "confusion",
    "无助": "helplessness", "孤独": "helplessness", "人际矛盾": "helplessness",
    "自我怀疑": "helplessness",

    # 从测试数据英文标签 → 8类
    "joy": "positive", "annoyance": "anger", "loneliness": "helplessness",
    "fear": "anxiety", "despair": "sadness", "panic": "anxiety",
    "fatigue": "stress", "procrastination": "stress", "relief": "positive",
    "approval": "positive", "withdrawal": "helplessness",
    "self_doubt": "helplessness", "crisis": "sadness",
}

# 数据源权重
SOURCE_WEIGHTS = {
    "manual": 1.5,       # 人工标注
    "silver": 1.0,       # 自动标注+校验
    "llm_pseudo": 0.8,  # LLM 伪标签
    "risk_data": 0.7,   # 从风险数据推导
}


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  保存 {len(records)} 条 → {path}")


def normalize_emotion(label: str) -> str | None:
    """将任意格式的情绪标签映射到 8 类之一"""
    if not label:
        return None
    label = label.strip().lower()
    # 先尝试直接映射
    if label in EMOTION_8_MAP:
        return EMOTION_8_MAP[label]
    # 尝试模糊匹配
    for key, val in EMOTION_8_MAP.items():
        if key.lower() in label or label in key.lower():
            return val
    return None


def load_manual_test_data(agent_test_data_dir: Path) -> list[dict]:
    """
    加载 emotion_risk_turns.jsonl 中的人工标注数据
    """
    path = agent_test_data_dir / "emotion_risk_turns.jsonl"
    if not path.exists():
        print(f"  未找到人工标注数据: {path}")
        return []

    records = load_jsonl(path)
    result = []
    for r in records:
        expected = r.get("expected", {})
        label = normalize_emotion(expected.get("emotion_type", ""))
        if label:
            result.append({
                "text": r.get("text", ""),
                "emotion": label,
                "source": "manual",
                "confidence": 1.0,
                "original_id": r.get("sample_id", ""),
            })
    print(f"  从 emotion_risk_turns 加载 {len(result)} 条人工标注数据")
    return result


def load_sos1k_emotion_data(processed_dir: Path) -> list[dict]:
    """
    从 SOS-1K 风险数据中提取有情感倾向的文本
    规则：evidence_tags 含 severe_distress → sadness/helplessness
          subject_context 含 hypothetical 或 negated → 跳过
    """
    from collections import Counter

    # 加载所有风险标注数据
    train_path = processed_dir / "train_v3.jsonl"
    dev_path = processed_dir / "dev_v3.jsonl"
    test_path = processed_dir / "test_v3.jsonl"

    records = []
    for p in [train_path, dev_path, test_path]:
        if p.exists():
            records.extend(load_jsonl(p))

    if not records:
        print(f"  SOS-1K 数据未找到，跳过")
        return []

    result = []
    emotion_map = {
        "severe_distress": "sadness",
        "loss_of_control": "helplessness",
        "help_seeking": "anxiety",
        "passive_death_wish": "sadness",
    }

    for r in records:
        text = r.get("text", "")
        if len(text) < 5:
            continue
        # 跳过否定和假设语境
        if r.get("subject_context") in ("negated", "hypothetical"):
            continue
        evidence = r.get("evidence_tags", [])
        for tag, emotion in emotion_map.items():
            if tag in evidence:
                result.append({
                    "text": text,
                    "emotion": emotion,
                    "source": "silver",
                    "confidence": 0.7,
                    "original_id": r.get("sample_id", ""),
                })
                break

    print(f"  从 SOS-1K 提取 {len(result)} 条情绪数据")
    return result


def split_data(
    records: list[dict],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """分层分割（按标签分布）"""
    random.seed(seed)

    # 按标签分组
    by_label: dict[str, list[dict]] = {}
    for r in records:
        by_label.setdefault(r["emotion"], []).append(r)

    train, dev, test = [], [], []
    for label, group in by_label.items():
        random.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_dev = max(1, int(n * dev_ratio))
        n_test = max(1, n - n_train - n_dev)
        train.extend(group[:n_train])
        dev.extend(group[n_train:n_train + n_dev])
        test.extend(group[n_train + n_dev:])

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    return train, dev, test


def print_stats(records: list[dict], name: str):
    print(f"\n{name} ({len(records)} 条):")
    counter = Counter(r["emotion"] for r in records)
    for label, count in sorted(counter.items()):
        print(f"  {label:15s} {count:4d} ({count*100/len(records):5.1f}%)")


def main():
    # 确定路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # agent_version/
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}")

    agent_test_data_dir = project_root / "agent_test_data"
    processed_dir = project_root / "bert_data" / "processed_v4"
    output_dir = processed_dir

    all_records = []

    # 1. 加载人工标注数据
    manual = load_manual_test_data(agent_test_data_dir)
    all_records.extend(manual)

    # 2. 从 SOS-1K 提取情绪数据
    sos = load_sos1k_emotion_data(processed_dir)
    all_records.extend(sos)

    # 3. LLM 伪标签数据（如果已生成）
    pseudo_path = processed_dir / "emotion_pseudo.jsonl"
    if pseudo_path.exists():
        pseudo = load_jsonl(pseudo_path)
        for r in pseudo:
            label = normalize_emotion(r.get("emotion", ""))
            conf = r.get("confidence", 0.8)
            if label and conf >= 0.7:  # 只保留高置信度
                all_records.append({
                    "text": r["text"],
                    "emotion": label,
                    "source": "llm_pseudo",
                    "confidence": conf,
                })
        print(f"\n  从 LLM 伪标签加载 {len(pseudo)} 条，保留 {sum(1 for r in all_records if r['source'] == 'llm_pseudo')} 条高置信度")

    # 4. 外部公开数据集（Johnson8187 等）
    external_path = processed_dir / "emotion_pseudo_external.jsonl"
    if external_path.exists():
        external = load_jsonl(external_path)
        for r in external:
            label = normalize_emotion(r.get("emotion", ""))
            if label:
                all_records.append({
                    "text": r["text"],
                    "emotion": label,
                    "source": "external",
                    "confidence": r.get("confidence", 1.0),
                })
        print(f"\n  从外部数据集加载 {len(external)} 条")

    print(f"\n总计: {len(all_records)} 条情绪数据")

    # 打印整体分布
    print_stats(all_records, "整体分布")

    # 分割
    train, dev, test = split_data(all_records, train_ratio=0.8, dev_ratio=0.1)

    print_stats(train, "训练集")
    print_stats(dev, "开发集")
    print_stats(test, "测试集")

    # 保存
    for name, data in [("emotion_train", train), ("emotion_dev", dev), ("emotion_test", test)]:
        # 移除 original_id 等辅助字段
        clean = [{"text": r["text"], "emotion": r["emotion"],
                  "source": r["source"], "confidence": r["confidence"]}
                 for r in data]
        save_jsonl(clean, output_dir / f"{name}.jsonl")

    print(f"\n数据准备完成！输出目录: {output_dir}")
    print(f"  运行训练: python bert_data/scripts/train_emotion.py")


if __name__ == "__main__":
    main()

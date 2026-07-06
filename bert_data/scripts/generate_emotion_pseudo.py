#!/usr/bin/env python3
"""
generate_emotion_pseudo.py — 用 LLM 生成情绪伪标签

从多个文本源读取数据，调用 DeepSeek API 标注情绪标签，
输出到 emotion_pseudo.jsonl 供 prepare_emotion_data.py 使用。

用法:
    python bert_data/scripts/generate_emotion_pseudo.py           # 默认：从 SOS-1K + content_db 生成
    python bert_data/scripts/generate_emotion_pseudo.py --limit 200   # 只处理 200 条
    python bert_data/scripts/generate_emotion_pseudo.py --source sos  # 只从 SOS-1K 生成
"""

import json
import os
import random
import sys
from pathlib import Path

# 确保能找到 backend 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

# 从 .env 加载 API 密钥
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

# OpenAI 库需要 OPENAI_API_KEY
api_key = os.getenv("DEEPSEEK_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", api_key)

from openai import OpenAI

# ====== 标签映射（LLM 中文输出 → 8 类） ======
LLM_TO_8CLASS = {
    "中性": "neutral", "快乐": "positive", "平静": "positive", "放松": "positive", "开心": "positive",
    "焦虑": "anxiety", "紧张": "anxiety", "担心": "anxiety",
    "压力": "stress", "学业压力": "stress",
    "抑郁": "sadness", "低落": "sadness", "难过": "sadness", "悲伤": "sadness",
    "愤怒": "anger", "生气": "anger", "烦躁": "anger",
    "困惑": "confusion", "不确定": "confusion", "未来迷茫": "confusion", "迷茫": "confusion",
    "无助": "helplessness", "孤独": "helplessness", "人际矛盾": "helplessness",
}

# ====== Johnson8187 映射 → 8 类 ======
JOHNSON8187_TO_8CLASS = {
    "平淡語氣": "neutral",
    "開心語調": "positive",
    "關切語調": "positive",
    "憤怒語調": "anger",
    "驚奇語調": "anxiety",      # 惊奇→焦虑（惊奇常伴随不确定感）
    "悲傷語調": "sadness",
    "厭惡語調": "anger",
    "疑問語調": "neutral",      # 疑问→中性（提问不是困惑）
}

# ====== 文本源 ======
def load_sos_texts(processed_dir: Path, max_count: int = None) -> list[str]:
    """从 SOS-1K 处理数据中加载文本"""
    texts = []
    for fname in ["train_v3.jsonl", "dev_v3.jsonl", "test_v3.jsonl"]:
        path = processed_dir / fname
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    texts.append(r.get("text", ""))
    random.shuffle(texts)
    if max_count:
        texts = texts[:max_count]
    print(f"  从 SOS-1K 加载 {len(texts)} 条文本")
    return texts


def load_content_texts(data_path: Path, max_count: int = None) -> list[str]:
    """从 content_db.json 加载内容描述（文章/练习的标题+描述，偏向中性/积极）"""
    if not data_path.exists():
        print(f"  未找到内容数据库: {data_path}")
        return []
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    items = data if isinstance(data, list) else data.get("items", [])
    for item in items:
        title = item.get("title", "")
        desc = item.get("description", "")
        if desc:
            texts.append(desc)
        elif title:
            texts.append(title)
        # 也加一些内容正文片段
        content = item.get("content", "")
        if content:
            texts.append(content[:200])
    random.shuffle(texts)
    if max_count:
        texts = texts[:max_count]
    print(f"  从内容数据库加载 {len(texts)} 条文本")
    return texts


def load_cognitive_distortion_texts(raw_dir: Path, max_count: int = None) -> list[str]:
    """从认知歪曲数据集加载文本（已从 SOS-HL-1K 移除）"""
    return []


def load_mentalglm_texts(raw_dir: Path, max_count: int = None) -> list[str]:
    """从 MentalGLM CD 数据集加载文本"""
    cd_dir = raw_dir / "mentalglm" / "Raw data" / "CD"
    if not cd_dir.exists():
        print(f"  未找到 MentalGLM CD 数据: {cd_dir}")
        return []
    import csv
    texts = []
    for fname in ["train_data.tsv", "val_data.tsv", "test_data.tsv"]:
        path = cd_dir / fname
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 13 and row[12].strip():
                    texts.append(row[12].strip())
    random.shuffle(texts)
    if max_count:
        texts = texts[:max_count]
    print(f"  从 MentalGLM CD 加载 {len(texts)} 条文本")
    return texts


# ====== 去重 ======
def deduplicate(texts: list[str]) -> list[str]:
    seen = set()
    result = []
    for t in texts:
        short = t[:100].strip().lower()
        if short not in seen and len(t) >= 5:
            seen.add(short)
            result.append(t)
    return result


# ====== LLM 标注 ======
def label_text(client: OpenAI, text: str) -> tuple[str, float] | None:
    """调用 LLM 标注单条文本的情绪"""
    prompt = "分析以下文本的主要情绪（只返回情绪标签，不要其他文字）：\n选项：学业压力、焦虑、抑郁、愤怒、压力、人际矛盾、困惑、不确定、中性、快乐、平静、放松、其他\n\n文本：" + text + "\n情绪标签："
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "只返回情绪标签"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip()
        if label in LLM_TO_8CLASS:
            return LLM_TO_8CLASS[label], 1.0
        else:
            return None
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        return None


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    print(f"工作目录: {project_root}")

    # 解析参数
    limit = None
    sources = ["sos", "content", "cognitive", "mentalglm"]
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])
        elif arg.startswith("--source="):
            sources = [arg.split("=")[1]]

    # 加载文本
    all_texts = []
    processed_dir = project_root / "bert_data" / "processed_v4"
    data_dir = project_root / "data"
    raw_dir = project_root / "bert_data" / "raw"

    if "sos" in sources:
        all_texts.extend(load_sos_texts(processed_dir))
    if "content" in sources:
        all_texts.extend(load_content_texts(data_dir / "content_db.json"))
    if "cognitive" in sources:
        all_texts.extend(load_cognitive_distortion_texts(raw_dir))
    if "mentalglm" in sources:
        all_texts.extend(load_mentalglm_texts(raw_dir))

    all_texts = deduplicate(all_texts)
    if limit:
        all_texts = all_texts[:limit]
    print(f"去重后共 {len(all_texts)} 条待标注文本")
    print()

    # LLM 客户端
    client = OpenAI(base_url="https://api.deepseek.com/v1")

    # 分批标注
    results = []
    batch_size = 10
    total = len(all_texts)

    for i in range(0, total, batch_size):
        batch = all_texts[i : i + batch_size]
        for text in batch:
            result = label_text(client, text)
            if result is not None:
                label, confidence = result
                results.append({
                    "text": text,
                    "emotion": label,
                    "source": "llm_pseudo",
                    "confidence": confidence,
                })

        done = min(i + batch_size, total)
        print(f"  进度: {done}/{total} ({len(results)} 条成功)")

    # 保存
    output_path = processed_dir / "emotion_pseudo.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n生成完成: {len(results)} 条 → {output_path}")

    # 统计分布
    from collections import Counter
    counter = Counter(r["emotion"] for r in results)
    if results:
        print("\n分布:")
        for label, count in sorted(counter.items()):
            print(f"  {label:15s} {count:4d} ({count*100/len(results):5.1f}%)")
    else:
        print("  无成功标注")


if __name__ == "__main__":
    main()

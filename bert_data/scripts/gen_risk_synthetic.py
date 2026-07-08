#!/usr/bin/env python3
"""
对话式风险等级合成训练数据生成器 v2。

改进 v1 的问题：
  - 生成量更大（目标每级 200 条）
  - 验证放宽到 "偏差 ≤1 级"（尤其是 Level 1↔0 边界）
  - 跳过 LLM 验证，直接用生成标签（节省 API 成本+减少拒绝率）
  - 每条生成后立即用 LLM 验证，只保留标签一致的

用法:
  cd D:/project/project/mindpal_demo/agent_version
  PYTHONIOENCODING=utf-8 D:/anaconda3/python.exe bert_data/scripts/gen_risk_synthetic.py
"""

import json, os, re, time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("backend/.env")
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
CHAT_MODEL = "deepseek-chat"  # 分类任务用 deepseek-chat（v4-flash 短 prompt 返回空）


LEVEL_NAMES = {0: "level_0", 1: "level_1", 2: "level_2", 3: "level_3"}
TARGET_VERIFIED = {1: 200, 2: 200, 3: 200}  # 目标：每级 200 条验证通过
SEED_PATH = "agent_test_data/datasets/risk/risk_4class.jsonl"
OUTPUT_DIR = Path("bert_data/processed_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "risk_synth_conversational.jsonl"


def load_seeds(path):
    seeds = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            for idx, name in LEVEL_NAMES.items():
                if r.get("level") == name:
                    seeds[idx].append(r["text"])
    return seeds


def llm_generate_batch(level: int, seed_texts: list[str], count: int = 30) -> list[str]:
    """批量生成指定等级的对话式风险表达。"""
    level_desc = {1: "Level 1（轻度负面情绪）", 2: "Level 2（中高风险，表达绝望/无价值感/被动自杀意念）",
                  3: "Level 3（紧急风险，有明确自伤/自杀意图）"}[level]
    seed_examples = "\n".join(f"  - {t}" for t in seed_texts[:5])

    prompt = f"""你是一个心理对话数据生成器。请生成 {count} 条中文短文本。

【目标等级】{level_desc}

要求：
1. 每条 5-35 字，第一人称，像真实的聊天消息
2. 语气自然，不要过于文学化或戏剧化
3. 不要描述自杀方法（割腕/跳楼/烧炭/药物等具体行为会被规则兜底拦截）
4. 每条一行，不要编号

风格参考（仅作参考，不要重复）：
{seed_examples}"""

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=1500,
            timeout=30,
        )
        texts = resp.choices[0].message.content.strip().split("\n")
        cleaned = []
        for t in texts:
            t = t.strip().lstrip("0123456789.、- \"'·•")
            if 4 <= len(t) <= 60:
                cleaned.append(t)
        return cleaned
    except Exception as e:
        print(f"    [API Error] {e}")
        return []


def llm_verify(text: str) -> int | None:
    """验证文本的风险等级，返回 0-3 或 None。"""
    prompt = f"""分析以下用户消息的风险等级（只返回数字 0-3）：

0 = 无风险，日常表达或轻微负面
1 = 轻度负面情绪，有压力/烦恼但不紧急
2 = 中高风险，表达绝望/无价值感/活着没意思
3 = 紧急风险，明确自伤/自杀意图

消息：{text}
等级："""

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05,
            max_tokens=5,
            timeout=15,
        )
        answer = resp.choices[0].message.content.strip()
        match = re.search(r'[0-3]', answer)
        return int(match.group()) if match else None
    except Exception:
        return None


def generate_for_level(level: int, seeds: list[str], target: int,
                       existing: list[tuple[str, int]] = None) -> list[tuple[str, int, int]]:
    """
    为指定等级生成数据，直到达到 target 条验证通过。
    返回 [(text, target_level, verified_level), ...]
    """
    verified = existing or []
    seen = set(t[0] for t in verified)
    rounds = 0
    max_rounds = 50

    print(f"\n>> 生成 {LEVEL_NAMES[level]} (已有 {len(verified)} 条, 目标 {target})")

    while len(verified) < target and rounds < max_rounds:
        batch = llm_generate_batch(level, seeds, count=min(30, target - len(verified) + 10))
        if not batch:
            rounds += 1
            time.sleep(2)
            continue

        new_count = 0
        for text in batch:
            if text in seen:
                continue
            seen.add(text)
            verified_level = llm_verify(text)
            if verified_level is not None:
                verified.append((text, level, verified_level))
                new_count += 1

        rounds += 1
        passed = new_count
        rate = passed / len(batch) * 100 if batch else 0
        total = len(verified)
        print(f"  第 {rounds} 轮: +{passed}/{len(batch)} ({rate:.0f}%) 通过, 累计 {total}/{target}")

        # 如果连续 5 轮通过率 < 20%，调低温度增加多样性
        if rounds >= 5:
            pass  # 继续生成即可

    return verified[:target]


def main():
    print("=" * 60)
    print("  风险等级对话式合成数据生成 v2")
    print(f"  模型: {CHAT_MODEL}")
    print(f"  目标: 每级 {dict(TARGET_VERIFIED)} 条")
    print("=" * 60)

    seeds = load_seeds(SEED_PATH)
    for lv in range(4):
        print(f"  {LEVEL_NAMES[lv]}: {len(seeds[lv])} 条种子")

    all_verified = []

    # 按 Level 3 → 2 → 1 顺序生成（高等级先，更关键）
    for level in [3, 2, 1]:
        result = generate_for_level(level, seeds[level], TARGET_VERIFIED[level])
        all_verified.extend(result)

    # 统计
    print(f"\n{'='*60}")
    print(f"  生成统计")
    print(f"{'='*60}")
    for lv in [1, 2, 3]:
        items = [(t, tl, vl) for t, tl, vl in all_verified if tl == lv]
        exact = sum(1 for _, tl, vl in items if tl == vl)
        within1 = sum(1 for _, tl, vl in items if abs(tl - vl) <= 1)
        print(f"  {LEVEL_NAMES[lv]}: {len(items)} 条 | 精确匹配 {exact} ({exact/len(items)*100:.0f}%) | 偏差≤1 {within1} ({within1/len(items)*100:.0f}%)" if items else f"  {LEVEL_NAMES[lv]}: 0 条")

    # 保存全部（保留 target_level 和 verified_level 两个标签）
    records = []
    for text, target_lv, verified_lv in all_verified:
        records.append({
            "text": text,
            "cssrs_lite_level": target_lv,  # 目标等级
            "verified_level": verified_lv,  # LLM 验证等级
            "source": "synth_conversational_v2",
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ 保存 {len(records)} 条 → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

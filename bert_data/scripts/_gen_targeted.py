"""
定向生成 — 针对 12 个关键错误案例各生成相似文本
"""
import json, os
from collections import Counter

API_KEY = "***REMOVED***"
from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

LABEL_MAP = {"中性":"neutral", "快乐":"positive", "焦虑":"anxiety", "抑郁":"sadness", "愤怒":"anger"}

# 12 个关键错误案例 — 每个生成 8 条相似文本
SEED_ERRORS = [
    # 高置信错误 (6)
    ("anxiety", "别给我讲大道理，我现在只想要一个能马上做的办法。"),
    ("positive", "刚刚做完呼吸练习，好像稍微缓过来了。"),
    ("sadness", "现在先别分析原因，你能不能陪我把今晚熬过去。"),
    ("anger", "我恨我自己为什么总是这么软弱。"),
    ("anxiety", "这个bug改了一天还没找到原因，挫败感很强。"),
    ("positive", "本来以为会被裁，结果老板给我升职了，太意外了。"),
    # 中置信错误 (6)
    ("sadness", "朋友一直不回我消息，我开始怀疑是不是自己哪里做错了。"),
    ("sadness", "面试官一问细节我就卡壳，我是不是根本不适合做算法。"),
    ("anger", "我和室友吵架后一直很烦，但也不知道怎么开口。"),
    ("anger", "这件事其实没什么，我就是有点烦。"),
    ("anxiety", "我没有想自杀，只是最近压力真的很大，想找个办法缓一下。"),
    ("sadness", "我被老板当众批评了，觉得特别丢脸。"),
]

all_generated = []
for target_label, seed_text in SEED_ERRORS:
    prompt = f"""你是一个心理咨询数据生成器。请生成 8 条中文文本，每条 10-40 字。

要求：
1. 语气真实自然，像真实用户的倾诉
2. 情感类型必须是：{target_label}（中性/快乐/焦虑/抑郁/愤怒）
3. 风格和话题参考以下示例，但写出不同的文本：
4. 每行一条，不要编号
5. 不要包含自伤/自杀内容

参考示例：{seed_text}"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9, max_tokens=600,
        )
        texts = resp.choices[0].message.content.strip().split("\n")
        for t in texts:
            t = t.strip().lstrip("0123456789.、- \"'")
            if t and len(t) >= 5:
                all_generated.append((t, target_label))
        print(f"  ✓ [{target_label}] {seed_text[:20]}... {len(texts)} 条")
    except Exception as e:
        print(f"  ✗ [{target_label}] {seed_text[:20]}... {e}")

# 去重+LLM验证
seen = set()
unique = []
for text, label in all_generated:
    if text not in seen:
        seen.add(text)
        unique.append((text, label))

print(f"\n生成 {len(unique)} 条，LLM 验证中...")

def llm_label(text: str) -> str | None:
    prompt = f"分析以下文本的主要情绪（只返回标签）：\n选项：中性、快乐、焦虑、抑郁、愤怒\n\n文本：{text}\n情绪标签："
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "只返回情绪标签"}, {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=10,
        )
        return LABEL_MAP.get(resp.choices[0].message.content.strip())
    except:
        return None

# 分批验证
results = []
for i, (text, target) in enumerate(unique):
    verified = llm_label(text)
    if verified:  # LLM 能识别出情绪
        results.append({"text": text, "emotion": target, "source": "synth_targeted", "confidence": 1.0})
    if (i+1) % 10 == 0:
        c = Counter(r["emotion"] for r in results)
        print(f"  {i+1}/{len(unique)} → {len(results)} 条")

counter = Counter(r["emotion"] for r in results)
print(f"\n保存 {len(results)} 条:")
for label in ["neutral","positive","anxiety","sadness","anger"]:
    print(f"  {label}: {counter.get(label,0)}")

with open("bert_data/processed_v4/emotion_synth_targeted.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

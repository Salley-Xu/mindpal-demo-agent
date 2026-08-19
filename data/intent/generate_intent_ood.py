# -*- coding: utf-8 -*-
"""
OOD Dataset 生成器（Phase 1 Task 1.9 §29）。

OOD = 不属于 10 类 Intent 域的输入：编程/数学/旅行/购物/天气/娱乐/行政/跨域/乱码。
输出 IntentResult.is_open_set = true，不加 other 标签。

来源建议：编程、数学、旅行、购物、天气、娱乐、行政、跨域请求、乱码/无意义。

运行：/d/anaconda3/python.exe data/intent/generate_intent_ood.py
输出：data/intent/intent_ood_v1.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.intent.intent_schema import IntentData


def ood(sid, text, source="template", note=None):
    return IntentData(id=sid, conversation=[{"role": "user", "content": text}],
                      text=text, labels=[], source=source, difficulty="hard",
                      is_ood=True, notes=note)


OOD = []

# --- 编程 ---
OOD += [
    ood("iseed_ood_001", "Python 怎么处理异常？"),
    ood("iseed_ood_002", "帮我写一个快速排序。"),
    ood("iseed_ood_003", "React 和 Vue 哪个好？"),
    ood("iseed_ood_004", "这段代码为什么报错？def f(): return 1"),
    ood("iseed_ood_005", "SQL 里的 JOIN 有哪些类型？"),
    ood("iseed_ood_006", "怎么部署一个 Docker 容器？"),
    ood("iseed_ood_007", "正则表达式怎么匹配邮箱？"),
    ood("iseed_ood_008", "Git 怎么回退到上一个 commit？"),
    ood("iseed_ood_009", "帮我解释一下闭包。"),
    ood("iseed_ood_010", "TypeScript 的泛型怎么用？"),
    ood("iseed_ood_011", "Redis 缓存穿透怎么解决？"),
    ood("iseed_ood_012", "怎么优化一个慢查询？"),
    ood("iseed_ood_013", "帮我写个爬虫抓取网页。"),
    ood("iseed_ood_014", "什么是微服务架构？"),
    ood("iseed_ood_015", "帮我 debug 这个报错。"),
    ood("iseed_ood_016", "Java 内存泄漏怎么排查？"),
    ood("iseed_ood_017", "怎么用 pandas 处理缺失值？"),
    ood("iseed_ood_018", "Kafka 和 RabbitMQ 的区别？"),
    ood("iseed_ood_019", "帮我写一个自动化测试。"),
    ood("iseed_ood_020", "什么是 CI/CD？"),
]

# --- 数学 ---
OOD += [
    ood("iseed_ood_021", "解一下这个方程：2x+5=13"),
    ood("iseed_ood_022", "什么是微积分？"),
    ood("iseed_ood_023", "1+1 在二进制里等于多少？"),
    ood("iseed_ood_024", "帮我算一下房贷利率。"),
    ood("iseed_ood_025", "概率论里的贝叶斯公式是什么？"),
    ood("iseed_ood_026", "三角函数有哪些？"),
    ood("iseed_ood_027", "求导的链式法则怎么用？"),
    ood("iseed_ood_028", "帮我证明勾股定理。"),
    ood("iseed_ood_029", "什么是对数？"),
    ood("iseed_ood_030", "怎么计算标准差？"),
    ood("iseed_ood_031", "矩阵乘法怎么做？"),
    ood("iseed_ood_032", "帮我解一个线性方程组。"),
    ood("iseed_ood_033", "什么是质数？"),
    ood("iseed_ood_034", "帮我算一下这个积分。"),
    ood("iseed_ood_035", "排列组合怎么算？"),
]

# --- 旅行 ---
OOD += [
    ood("iseed_ood_036", "去日本旅游要办什么签证？"),
    ood("iseed_ood_037", "北京有哪些必去的景点？"),
    ood("iseed_ood_038", "帮我订一张去上海的机票。"),
    ood("iseed_ood_039", "三亚什么季节去最好？"),
    ood("iseed_ood_040", "出国旅游需要准备什么？"),
    ood("iseed_ood_041", "推荐一下成都的火锅店。"),
    ood("iseed_ood_042", "西藏旅行要注意什么？"),
    ood("iseed_ood_043", "帮我规划一个 5 天的云南行程。"),
    ood("iseed_ood_044", "坐飞机有什么注意事项？"),
    ood("iseed_ood_045", "国庆去哪里玩人少？"),
]

# --- 购物 ---
OOD += [
    ood("iseed_ood_046", "这款手机值不值得买？"),
    ood("iseed_ood_047", "帮我对比一下这两款耳机。"),
    ood("iseed_ood_048", "淘宝双十一有什么优惠？"),
    ood("iseed_ood_049", "买什么牌子的跑步鞋好？"),
    ood("iseed_ood_050", "怎么挑选合适的电脑？"),
    ood("iseed_ood_051", "这个价格合理吗？"),
    ood("iseed_ood_052", "推荐一款性价比高的相机。"),
    ood("iseed_ood_053", "买手机会不会买到翻新机？"),
    ood("iseed_ood_054", "帮我砍价。"),
    ood("iseed_ood_055", "这个 App 要收费吗？"),
]

# --- 天气 ---
OOD += [
    ood("iseed_ood_056", "明天会下雨吗？"),
    ood("iseed_ood_057", "这周末天气怎么样？"),
    ood("iseed_ood_058", "台风什么时候过去？"),
    ood("iseed_ood_059", "今天的气温是多少？"),
    ood("iseed_ood_060", "下周会降温吗？"),
    ood("iseed_ood_061", "空气质量怎么样？"),
    ood("iseed_ood_062", "要带伞吗？"),
]

# --- 娱乐 ---
OOD += [
    ood("iseed_ood_063", "最近有什么好看的电影？"),
    ood("iseed_ood_064", "帮我推荐一首歌。"),
    ood("iseed_ood_065", "这个游戏的攻略是什么？"),
    ood("iseed_ood_066", "最新一期综艺讲了什么？"),
    ood("iseed_ood_067", "漫威新电影什么时候上映？"),
    ood("iseed_ood_068", "帮我找一部恐怖片。"),
    ood("iseed_ood_069", "周杰伦的歌有哪些？"),
    ood("iseed_ood_070", "这个梗是什么意思？"),
]

# --- 行政 / 日常事务 ---
OOD += [
    ood("iseed_ood_071", "身份证丢了怎么补办？"),
    ood("iseed_ood_072", "公积金怎么提取？"),
    ood("iseed_ood_073", "怎么预约办护照？"),
    ood("iseed_ood_074", "社保断缴有什么影响？"),
    ood("iseed_ood_075", "怎么报个人所得税？"),
    ood("iseed_ood_076", "帮我算一下加班费。"),
    ood("iseed_ood_077", "劳动合同要注意什么？"),
    ood("iseed_ood_078", "怎么办理居住证？"),
    ood("iseed_ood_079", "房产过户需要什么材料？"),
    ood("iseed_ood_080", "帮我写一封请假邮件。"),
]

# --- 跨域 / 复杂请求 ---
OOD += [
    ood("iseed_ood_081", "帮我分析一下股票走势。"),
    ood("iseed_ood_082", "怎么投资理财？"),
    ood("iseed_ood_083", "帮我写一份商业计划书。"),
    ood("iseed_ood_084", "健身减脂有什么科学方法？"),
    ood("iseed_ood_085", "怎么学英语更快？"),
    ood("iseed_ood_086", "帮我翻译这段英文。"),
    ood("iseed_ood_087", "考研英语怎么准备？"),
    ood("iseed_ood_088", "帮我写一封推荐信。"),
    ood("iseed_ood_089", "怎么养多肉植物？"),
    ood("iseed_ood_090", "帮我做一个 PPT 大纲。"),
    ood("iseed_ood_091", "怎么做好吃的红烧肉？"),
    ood("iseed_ood_092", "帮我规划一下职业生涯。"),
    ood("iseed_ood_093", "有什么好的理财 App？"),
    ood("iseed_ood_094", "帮我写一个年终总结。"),
    ood("iseed_ood_095", "怎么提高英语口语？"),
]

# --- 乱码 / 无意义 ---
OOD += [
    ood("iseed_ood_096", "asdfghjkl"),
    ood("iseed_ood_097", "111111111"),
    ood("iseed_ood_098", "？？？？？？"),
    ood("iseed_ood_099", "fkajslfjasldkfj"),
    ood("iseed_ood_100", "的的的的的"),
    ood("iseed_ood_101", "你好啊啊啊啊啊啊啊啊"),
    ood("iseed_ood_102", "qwertyuiop"),
    ood("iseed_ood_103", "测试测试测试测试"),
    ood("iseed_ood_104", "..........."),
    ood("iseed_ood_105", "哈哈哈嗝"),
]


def main():
    out = PROJECT_ROOT / "data" / "intent" / "intent_ood_v1.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for d in OOD:
            f.write(json.dumps(d.model_dump(), ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[OK] 写入 {len(OOD)} 条 OOD -> {out.name}")
    print(f"  覆盖域数: 编程/数学/旅行/购物/天气/娱乐/行政/跨域/乱码 = 9 类")


if __name__ == "__main__":
    main()

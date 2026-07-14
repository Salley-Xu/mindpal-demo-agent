"""Generate v4.6 new content items as JSON"""
import json, os
from collections import Counter

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "content_v46.json")
items = []

def add(title, t, cat, desc, content, emotions, tags,
        diff="beginner", rt="soft", pri=0.7, act=0.7,
        rl=None, contra=None, dur=None):
    if rl is None: rl = ["low", "medium"]
    if contra is None: contra = ["high_risk_crisis"]
    idx = len(items) + 1
    pfx = {"article":"article","audio":"audio","exercise":"exercise","tool":"tool","video":"video"}
    items.append({"id": f"{pfx[t]}_{28+idx:03d}", "title": title, "type": t,
        "category": cat, "description": desc, "content": content,
        "tags": tags, "emotion_tags": emotions, "risk_levels": rl,
        "contraindications": contra, "recommend_type": rt, "priority": pri,
        "source": "mindpal_editorial", "actionability": act, "difficulty": diff,
        "duration_minutes": dur})

# 学业压力 +6
add("高效学习法：番茄工作法与时间管理", "article", "academic",
    "针对学业拖延和时间管理问题，提供系统化的高效学习方法。",
    "每25分钟专注+5分钟休息，4个循环后休息15-30分钟。", diff="beginner", pri=0.80, act=0.85,
    emotions=["学业压力","焦虑","压力"], tags=["学习方法","时间管理","番茄工作法","效率"])

add("考前放松冥想（8分钟）", "audio", "relaxation",
    "考前紧张时使用的快速放松冥想。",
    "深呼吸，想象自己平静地坐在考场中，每一次呼吸带走一分紧张。",
    emotions=["学业压力","焦虑","紧张"], tags=["考前放松","冥想","考试焦虑","专注"],
    rt="hard", pri=0.85, act=0.9, dur=8)

add("学业压力管理工作表", "tool", "academic",
    "通过结构化工作表识别学业压力来源。",
    "列出待办事项，标记优先级，拆解步骤，每完成一项打勾。",
    emotions=["学业压力","压力","困惑"], tags=["学习计划","压力管理","工具","规划"],
    diff="intermediate", rt="hard", pri=0.75, act=0.88)

add("如何应对考试失利后的情绪", "article", "academic",
    "考试没考好时的情绪调节指南。",
    "允许自己短暂难过，客观分析知识点掌握情况，制定改进计划。",
    emotions=["学业压力","抑郁","自我怀疑"], tags=["考试","挫折","情绪调节","成长心态"],
    diff="intermediate", pri=0.72, act=0.70)

add("自习专注白噪音（30分钟）", "audio", "relaxation",
    "适合学习时听的自然白噪音。",
    "轻柔的雨声和远处的雷声，营造安静舒适的学习环境。",
    emotions=["学业压力","焦虑","压力"], tags=["白噪音","专注","自习","学习"],
    pri=0.68, act=0.6, dur=30)

add("考试复盘练习", "exercise", "self_reflection",
    "每次考试后的系统性复盘练习。",
    "写下：这次做对了什么，哪些地方可改进，下次行动计划。",
    emotions=["学业压力","自我怀疑","困惑"], tags=["考试复盘","反思","学习方法","成长"],
    pri=0.70, act=0.82)

# 未来迷茫 +4
add("职业兴趣探索指南", "article", "future",
    "通过系统化的自我探索找到职业方向。",
    "回答：你擅长什么，你喜欢做什么，世界需要什么，三者的交集就是方向。",
    emotions=["未来迷茫","不确定","困惑"], tags=["职业规划","自我探索","兴趣","方向"],
    pri=0.78, act=0.72)

add("人生设计练习：奥德赛计划", "exercise", "self_reflection",
    "规划未来5年的三种可能人生版本。",
    "写出Plan A当前路径、Plan B换方向、Plan C不考虑钱和面子。",
    emotions=["未来迷茫","不确定","困惑"], tags=["人生设计","规划","奥德赛计划","未来"],
    diff="intermediate", rt="hard", pri=0.74, act=0.85)

add("焦虑中的小决定：减少决策疲劳", "article", "decision_making",
    "面对太多选择时如何减少决策焦虑。",
    "今天只做三个决定，其他按惯例处理。小决定快速执行。",
    emotions=["不确定","焦虑","困惑"], tags=["决策","焦虑","选择","简化"],
    pri=0.72, act=0.76)

add("未来规划冥想（12分钟）", "audio", "relaxation",
    "在放松状态下探索内心真正想要的生活方向。",
    "想像5年后的自己：你在做什么，住在哪里，和谁在一起。",
    emotions=["未来迷茫","焦虑","不确定"], tags=["冥想","未来","规划","可视化"],
    pri=0.72, act=0.6, dur=12)

# 孤独 +4
add("如何建立和维护友谊", "article", "social_skills",
    "从零开始建立社交联系的实用方法。",
    "友谊三要素：持续接触、自我表露、共同经历。从每周一次咖啡开始。",
    emotions=["孤独","人际矛盾","困惑"], tags=["友谊","社交","人际关系","连接"],
    pri=0.76, act=0.73)

add("孤独感缓解冥想（10分钟）", "audio", "relaxation",
    "学习如何与自己建立温暖连接。",
    "把手放在心口，对自己说：我在这里陪着你。",
    emotions=["孤独","抑郁","压力"], tags=["孤独","冥想","自我关怀","接纳"],
    pri=0.80, act=0.65, dur=10)

add("社交勇气练习：每天一个小冒险", "exercise", "social_skills",
    "通过低风险社交练习克服社交焦虑。",
    "周一赞美一个人，周三和不熟的人打招呼，周五参加兴趣小组。",
    emotions=["孤独","人际矛盾","社交焦虑"], tags=["社交","勇气","小步骤","练习"],
    rt="hard", pri=0.72, act=0.88)

add("养宠物能缓解孤独吗", "article", "self_reflection",
    "宠物陪伴对孤独感的缓解作用及替代方案。",
    "宠物的无条件积极关注。可尝试动物志愿者或与朋友宠物互动。",
    emotions=["孤独","抑郁","压力"], tags=["宠物","陪伴","孤独","情感支持"],
    pri=0.65, act=0.60)

# 愤怒 +4
add("愤怒日记", "tool", "anger_management",
    "记录愤怒事件，识别情绪触发模式。",
    "记录事件、想法、身体感受、愤怒程度1-10、冷静后反思。",
    emotions=["愤怒","压力","人际矛盾"], tags=["愤怒管理","日记","情绪觉察","触发器"],
    rt="hard", pri=0.76, act=0.85)

add("冲突后的修复对话指南", "article", "communication",
    "愤怒平息后如何进行修复性对话。",
    "先冷静、表达感受而非指责、倾听对方、找到共同点、一起想方案。",
    emotions=["愤怒","人际矛盾","孤独"], tags=["冲突","修复","沟通","关系"],
    diff="intermediate", pri=0.74, act=0.72)

add("释放愤怒的身体练习（5分钟）", "exercise", "emotional_regulation",
    "通过身体运动释放积压的愤怒。",
    "握拳松开x5、深呼吸、原地跑30秒、叹气、放松肩膀。",
    emotions=["愤怒","压力","烦躁"], tags=["身体释放","运动","愤怒","情绪调节"],
    rt="hard", pri=0.82, act=0.92, dur=5)

add("理解愤怒背后的需求", "article", "emotional_regulation",
    "愤怒是需求未被满足的信号。",
    "愤怒背后：被尊重的需求、被理解的需求、边界被侵犯。问自己真正需要什么。",
    emotions=["愤怒","困惑","人际矛盾"], tags=["愤怒","需求","情绪理解","自我觉察"],
    diff="intermediate", pri=0.70, act=0.68)

# 不确定 +3
add("决策焦虑应对工作卡", "tool", "decision_making",
    "重要决定犹豫不决时理清思路。",
    "写下选择，列出好坏，问信任的人，设定截止时间，接受不确定性。",
    emotions=["不确定","焦虑","困惑"], tags=["决策","焦虑","选择","工具"],
    rt="hard", pri=0.73, act=0.84)

add("如何在不确定性中找到平静", "article", "relaxation",
    "接受不确定性，学习与未知和平共处。",
    "控制能控制的（反应和行动），放下控制不了的（结果）。",
    emotions=["不确定","焦虑","压力"], tags=["不确定性","接纳","平静","放下"],
    pri=0.72, act=0.70)

add("不确定性容忍度练习", "exercise", "self_reflection",
    "刻意练习提高对不确定的容忍能力。",
    "本周做一件结果不确定的事：尝试新餐厅、走新路、和不熟的人聊天。",
    emotions=["不确定","焦虑","困惑"], tags=["不确定性","练习","成长","勇气"],
    rt="hard", pri=0.68, act=0.82)

# 人际矛盾 +3
add("非暴力沟通实践指南", "article", "communication",
    "非暴力沟通四步法减少冲突中的指责。",
    "观察、感受、需要、请求。不说「你总是」，说「当你…我感到…我需要…你愿意吗」。",
    emotions=["人际矛盾","愤怒","困惑"], tags=["非暴力沟通","沟通","冲突","关系"],
    diff="intermediate", pri=0.78, act=0.76)

add("边界设置练习", "exercise", "self_esteem",
    "保护自己的边界不过度冒犯他人。",
    "拒绝不想要的邀请，表达真实感受，坚持决定不需过度解释。",
    emotions=["人际矛盾","压力","自我怀疑"], tags=["边界","拒绝","自信","人际"],
    diff="intermediate", rt="hard", pri=0.74, act=0.80)

add("家庭关系中的情绪管理", "article", "relationship",
    "面对家庭矛盾时管理自己情绪。",
    "不争对错、不翻旧账、设定情绪暂停按钮。",
    emotions=["人际矛盾","愤怒","抑郁"], tags=["家庭","情绪管理","边界","关系"],
    pri=0.74, act=0.71)

# 失眠 +3
add("睡前放松瑜伽（15分钟）", "video", "relaxation",
    "睡前瑜伽序列帮助放松入睡。",
    "婴儿式、猫牛式、腿部靠墙、尸体式。每个动作5个呼吸。",
    emotions=["失眠","焦虑","压力"], tags=["瑜伽","放松","睡眠","睡前"],
    rt="hard", pri=0.76, act=0.85, dur=15)

add("告别睡前胡思乱想的技巧", "article", "sleep",
    "躺下大脑停不下来时用CBT技巧。",
    "担心的事写纸上，告诉自己明天处理，做4-7-8呼吸。超20分钟睡不着就起来。",
    emotions=["失眠","焦虑","压力"], tags=["失眠","胡思乱想","CBT","睡眠卫生"],
    pri=0.78, act=0.82)

add("助眠白噪音（30分钟）", "audio", "sleep",
    "舒缓雨声创造适合睡眠的氛围。",
    "柔和雨声帮助放松神经系统，自然进入睡眠状态。",
    emotions=["失眠","焦虑","压力"], tags=["白噪音","雨声","睡眠","放松"],
    pri=0.75, act=0.6, dur=30)

# 困惑 +2
add("如何打破思维僵局", "article", "self_reflection",
    "反复思考同一问题时如何打破定式。",
    "换位思考、跳出框架（这件事5年后还重要吗）、寻求外部视角。",
    emotions=["困惑","焦虑","不确定"], tags=["思维","问题解决","视角","突破"],
    diff="intermediate", pri=0.70, act=0.68)

add("困惑情绪冥想（8分钟）", "audio", "mindfulness",
    "不急于找答案，给困惑一个存在的空间。",
    "想象困惑是一片云，你坐在地上看着它飘过，不评判不驱赶。",
    emotions=["困惑","焦虑","不确定"], tags=["冥想","困惑","接纳","正念"],
    pri=0.68, act=0.55, dur=8)

# 抑郁 +2
add("行为激活：从最小行动开始", "article", "mood_management",
    "用微小行动打破抑郁的恶性循环。",
    "列出以前喜欢的5件事，选最容易的，设5分钟计时，只做5分钟。",
    emotions=["抑郁","情绪低落","疲惫"], tags=["行为激活","抑郁","行动","小步骤"],
    rt="hard", pri=0.80, act=0.85)

add("抑郁情绪自助卡", "tool", "mood_management",
    "抑郁情绪来袭时快速参考的自助卡。",
    "感受情绪不评判，做一件小事，联系一个人，提醒自己情绪会过去。",
    emotions=["抑郁","自我怀疑","孤独"], tags=["抑郁","自助","情绪管理","工具"],
    rt="hard", pri=0.82, act=0.88)

# 自我怀疑 +2
add("冒名顶替综合征应对指南", "article", "self_esteem",
    "觉得成就只是运气时如何理性应对。",
    "列出成就和贡献，询问信任的人，注意冒名顶替感不等于事实。",
    emotions=["自我怀疑","焦虑","抑郁"], tags=["冒名顶替","自信","成就","理性"],
    diff="intermediate", pri=0.76, act=0.70)

add("写给自己的一封鼓励信", "exercise", "self_esteem",
    "用友善的方式替代内心的自我批评。",
    "以好朋友视角给自己写信：告诉自己在困难中看到的努力、勇气和进步。",
    emotions=["自我怀疑","抑郁","困惑"], tags=["自我关怀","鼓励","自信","写信"],
    pri=0.70, act=0.82)

# 高风险适配 +6
add("心理健康危机预警信号", "article", "crisis_management",
    "识别心理健康危机早期预警信号。",
    "两周以上睡眠食欲改变、社交退缩、情绪剧烈波动、提到死亡、自我伤害行为。",
    emotions=["焦虑","抑郁","pressure"], tags=["危机","预警","心理健康","求助"],
    rl=["low","medium","high"], contra=[], pri=0.82, act=0.75)

add("情绪稳定计划：我的安全工具箱", "tool", "crisis_management",
    "情绪失控前的个人化安全计划。",
    "预警信号、让我平静的3件事、可联系的人、安全环境、紧急热线。",
    emotions=["焦虑","抑郁","愤怒"], tags=["安全计划","情绪急救","工具箱","预防"],
    rl=["low","medium","high"], contra=[], rt="hard", pri=0.88, act=0.90)

add("当朋友说想自杀怎么办", "article", "crisis_management",
    "应对朋友自杀表达的指导。",
    "不要惊慌评判保密，直接问有没有自杀计划，陪伴并鼓励寻求专业帮助。",
    emotions=["焦虑","抑郁","恐惧"], tags=["自杀预防","危机","求助","支持"],
    rl=["medium","high"], contra=[], pri=0.85, act=0.78)

add("心理援助热线资源指南", "article", "crisis_management",
    "国内可靠心理援助热线汇总。",
    "全国心理援助热线400-161-9995，希望24热线400-161-9995。",
    emotions=["抑郁","焦虑","pressure"], tags=["热线","求助","资源","危机"],
    rl=["low","medium","high"], contra=[], pri=0.90, act=0.95)

add("正念行走：户外接地练习（10分钟）", "audio", "mindfulness",
    "通过正念行走重新与大地连接。",
    "赤脚踩草地或土地，感受脚底触感，注意每一步，抬头深呼吸。",
    emotions=["焦虑","抑郁","pressure"], tags=["正念","行走","接地","大自然"],
    rl=["low","medium","high"], contra=[], pri=0.76, act=0.80, dur=10)

add("每日情绪检查表", "tool", "self_reflection",
    "快速情绪自查工具。",
    "今日情绪1-10、睡眠质量、有没有伤害自己的想法、今天做了什么事。",
    emotions=["抑郁","焦虑","自我怀疑"], tags=["情绪检查","自查","监测","心理健康"],
    rl=["low","medium","high"], contra=[], pri=0.78, act=0.85)

# Level 2 专用 +5 (recommend_type=soft for all)
add("温和呼吸空间（3分钟）", "audio", "emotional_regulation",
    "温和呼吸练习，适合情绪不稳定时。",
    "第1分钟问自己感受，第2分钟关注呼吸，第3分钟扩展到全身。",
    emotions=["焦虑","pressure","紧张"], tags=["呼吸","温和","休息","情绪稳定"],
    rt="soft", pri=0.80, act=0.90, dur=3,
    rl=["low","medium"], contra=["high_risk_crisis"])

add("安全之地可视化冥想（8分钟）", "audio", "relaxation",
    "在内心构建一个安全平静的地方。",
    "想象让你感到安全的地方，构建每个细节：颜色、气味、声音。",
    emotions=["焦虑","恐惧","pressure"], tags=["安全感","可视化","冥想","平静"],
    rt="soft", pri=0.78, act=0.70, dur=8)

add("自我安抚五种感官练习", "article", "emotional_regulation",
    "通过五种感官快速自我安抚。",
    "视觉看平静照片，听觉听舒缓歌曲，嗅觉闻放松气味，味觉喝温水，触觉抱软枕。",
    emotions=["焦虑","抑郁","愤怒"], tags=["自我安抚","感官","情绪调节","稳定"],
    rt="soft", pri=0.76, act=0.82)

add("今天只做三件事", "exercise", "behavior_activation",
    "不堪重负时缩减到三件最重要的事。",
    "写下最重要的三件事，只做这三件，完成后肯定自己。",
    emotions=["焦虑","pressure","抑郁"], tags=["简化","优先级","行动","小目标"],
    rt="soft", pri=0.72, act=0.85)

add("温柔身体拉伸（5分钟）", "video", "relaxation",
    "极轻柔拉伸，适合完全没能量时。",
    "坐在床上缓慢转动脖子，耸肩放松，伸展手臂，扭腰，活动脚踝。",
    emotions=["焦虑","pressure","疲惫"], tags=["拉伸","温柔","身体","放松"],
    rt="soft", pri=0.74, act=0.85, dur=5)

# Advanced +4
add("CBT进阶：识别核心信念", "article", "cognitive_restructuring",
    "深入探索核心信念的形成与重建。",
    "追溯重复出现的主题，早期经历，支持或反对证据，替代信念构建。",
    emotions=["焦虑","抑郁","自我怀疑"], tags=["CBT","核心信念","认知重构","进阶"],
    diff="advanced", pri=0.70, act=0.60)

add("认知重评：情绪调节高级策略", "article", "emotional_regulation",
    "改变对触发事件的情绪反应。",
    "识别自动化评价，问还有没有别的解释，选择更有适应性的解释。",
    emotions=["焦虑","愤怒","抑郁"], tags=["认知重评","情绪调节","高级"],
    diff="advanced", pri=0.72, act=0.62)

add("DBT核心技巧：痛苦耐受", "article", "emotional_regulation",
    "DBT痛苦耐受技巧介绍。",
    "STOP：停下来、深呼吸、观察、有意识行动。",
    emotions=["焦虑","愤怒","抑郁"], tags=["DBT","痛苦耐受","情绪调节"],
    diff="advanced", pri=0.74, act=0.65)

add("正念减压系统练习指南", "article", "mindfulness",
    "MBSR 8周正念练习系统介绍。",
    "每周主题：自动导航、注意力、身体扫描、正念瑜伽、反应vs回应等。",
    emotions=["焦虑","pressure","困惑"], tags=["MBSR","正念","减压","系统训练"],
    diff="advanced", pri=0.70, act=0.68)

# Write
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)
print(f"Generated {len(items)} items -> {OUTPUT}")

emo = Counter()
for i in items:
    for e in i["emotion_tags"]:
        emo[e] += 1
print(f"Emotions: {dict(sorted(emo.items(), key=lambda x:-x[1]))}")
print(f"Difficulty: {dict(Counter(i['difficulty'] for i in items))}")

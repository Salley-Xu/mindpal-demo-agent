# Intent Annotation Guideline v1

> 对应 Phase 1 Task 1.1（§6–9）
> 日期：2026-08-19
> 状态：**标注指南（Annotation Guideline）**
> 用途：Intent Dataset（seed 500–800 → 正式 3000–5000）的人工标注规范 + LLM 合成校验依据

---

## 1. 概述

### 1.1 Taxonomy

沿用 Benchmark v1.1 已冻结的 10 类（multi-label，不新增标签）：

| # | 标签 | 一句话定义 |
|---|---|---|
| 1 | `casual_chat` | 普通交流，无压力/信息/求助/资源/安全/历史引用等具体任务 |
| 2 | `emotional_expression` | 表达/倾诉情绪或心理体验，未提出明确行动请求 |
| 3 | `explicit_help_request` | 明确请求帮助、建议、方法、行动方案 |
| 4 | `information_request` | 请求事实、概念、机制等知识性回答 |
| 5 | `resource_request` | 请求可消费/可执行资源（练习/音频/文章/课程/内容） |
| 6 | `feedback` | 对之前建议/推荐/练习/Agent 行为的评价 |
| 7 | `follow_up` | 依赖上一轮语境的追问/继续/细化 |
| 8 | `memory_reference` | 引用过去跨轮/跨会话已表达的事实/偏好/经历 |
| 9 | `high_risk_expression` | 自伤/自杀/严重危机/明确危险倾向的表达 |
| 10 | `meta_question` | 询问 Agent 本身/能力/身份/数据/系统行为 |

### 1.2 核心原则

1. **Task 优先**：一旦存在更具体的任务意图（help/info/resource），通常**不再额外标 casual_chat**。
2. **按"用户想让 Agent 做什么"标**，不按"文本内容主题"标。
3. **high_risk_expression 是 Intent Signal，risk_level 是 Risk Module 输出**——两者永远不得合并、互相替代。
4. **多轮语义**：follow_up / memory_reference 必须结合"上一轮或历史对话"判断，单看当前句无法标注的，回看上下文。
5. **multi-label 允许**，但避免"能标则标"的堆叠——每个标签都应有独立的信息增量。

### 1.3 标注流程

```text
输入文本（+ 可选上轮上下文）
  ↓
1. 判断是否 meta_question / high_risk_expression（强信号优先）
  ↓
2. 判断是否引用历史/上一轮（memory_reference / follow_up）
  ↓
3. 判断任务意图（explicit_help / resource / information）
  ↓
4. 判断是否仅情绪表达 / 普通闲聊
  ↓
5. 补充 multi-label（情绪 + 任务的常见组合）
```

---

## 2. 各标签细则

### 2.1 casual_chat

- **定义**：不以压力管理、信息获取、求助、资源推荐、安全表达或历史引用为主要目的的普通交流。
- **正例**：
  - "今天天气不错。"
  - "陪我聊会儿吧。"
  - "刚吃完饭，有点撑。"
- **反例**：
  - "我最近压力特别大。" → emotional_expression
  - "你能帮我看看这个方案吗？" → explicit_help_request
- **易混淆**：emotional_expression（闲聊 vs 表达情绪）；meta_question（聊到 Agent 本身）
- **边界规则**：
  - 一旦存在更具体任务意图，**不额外标 casual_chat**。
  - 只有"无明显意图的寒暄/陪伴"才标 casual_chat。

### 2.2 emotional_expression

- **定义**：用户主要在表达、描述、倾诉自己的情绪或心理体验，**没有明确提出"希望 Agent 做什么"**。
- **正例**：
  - "最近压力特别大。"
  - "我现在好焦虑。"
  - "今天真的很累。"
- **反例**：
  - "我该怎么缓解焦虑？" → 需加 explicit_help_request
  - "我只是想吐槽一下。" → 可能单标 emotional_expression（无行动请求）
- **易混淆**：explicit_help_request（情绪+求助是高频多标组合）；high_risk_expression（负面情绪 vs 危机表达）
- **multi-label 规则**：
  - 情绪表达 + 求助：`"我很焦虑，有什么办法吗？"` → emotional_expression + explicit_help_request
  - 情绪表达 + 资源：`"最近压力好大，推荐点放松的内容吧。"` → emotional_expression + resource_request
- **边界规则**：判断核心是"是否明确提出行动请求"；有请求 → 追加对应任务标签，情绪标签仍保留。

### 2.3 explicit_help_request

- **定义**：用户明确请求帮助、建议、方法、行动方案或支持。
- **正例**：
  - "我现在应该怎么办？"
  - "怎么才能缓解这种焦虑？"
  - "你能帮我想想办法吗？"
- **反例**：
  - "推荐一个呼吸练习给我。" → resource_request（要资源而非方法指导）
  - "什么是正念？" → information_request（要知识）
- **易混淆**：resource_request；information_request；emotional_expression
- **边界规则**：
  - 要**通用方法/行动方案/建议** → explicit_help_request
  - 要**具体可消费资源**（练习/音频/文章）→ resource_request
  - 要**知识解释** → information_request

### 2.4 resource_request

- **定义**：用户明确请求可消费或可执行的资源：练习、音频、文章、课程、工具、内容推荐。
- **正例**：
  - "推荐几个缓解压力的练习。"
  - "有没有适合睡前听的冥想音频？"
  - "推荐一本书给我。"
- **反例**：
  - "我应该怎么办？" → explicit_help_request（要方案）
- **易混淆**：explicit_help_request；information_request
- **边界规则**：核心词法信号：推荐 / 有没有(资源类) / 给我(内容) / 音频 / 文章 / 练习 / 课程 / 工具。
- **多标**：`"我压力大，推荐点能放松的。"` → emotional_expression + resource_request

### 2.5 information_request

- **定义**：用户主要请求事实、解释、概念、机制或知识性回答，而非个性化支持方案。
- **正例**：
  - "什么是正念？"
  - "焦虑和压力有什么区别？"
  - "失眠和焦虑有关系吗？"
- **反例**：
  - "帮我推荐放松的练习。" → resource_request
- **易混淆**：resource_request（概念 vs 资源）；meta_question（问 Agent vs 问知识）
- **边界规则**：请求对象是"知识/事实/机制/概念" → information_request；请求对象是"内容/资源" → resource_request。
- **多标**：`"你刚才说焦虑影响睡眠，为什么？"` → follow_up + information_request

### 2.6 feedback

- **定义**：用户对之前的建议、推荐、练习或 Agent 行为给出评价（含效果、偏好、拒绝）。
- **正例**：
  - "你上次推荐的呼吸法没什么用。"
  - "那个冥想我试了，感觉不错。"
- **反例**：
  - "我不想听建议，只想自己待会。" → 更接近 feedback（对建议行为的拒绝）或 emotional_expression
- **易混淆**：follow_up（评价上一轮 vs 追问上一轮）；casual_chat
- **边界规则**：
  - 对"内容/建议/Agent 行为"给出评价 → feedback
  - 对上一轮内容**追问细节** → follow_up
  - `"我试了你说的那个，感觉一般，还有别的吗？"` → feedback + resource_request

### 2.7 follow_up

- **定义**：当前请求**依赖上一轮对话语境**，属于对上一轮内容的追问、继续或细化。
- **正例**：
  - "刚才那个方法具体怎么做？"（上轮提到某方法）
  - "你说的第二点是什么意思？"
- **反例**：
  - "我上次跟你说的项目黄了。" → memory_reference（跨会话/历史，非上一轮）
- **易混淆**：memory_reference；information_request；feedback
- **边界规则**（关键判别）：
  - "刚才 / 你刚才说 / 刚才那个" → follow_up
  - "上次 / 之前 / 以前我跟你说过" → memory_reference
  - follow_up 指**上一轮**语境；memory_reference 指**更早的跨轮/跨会话**事实。

### 2.8 memory_reference

- **定义**：用户显式或隐式引用过去跨轮/跨会话已表达的事实、事件、偏好或经历，回复需要历史记忆支撑。
- **正例**：
  - "我上次跟你说的那个项目黄了。"
  - "还是之前那个室友的问题。"
  - "你说过我膝盖受伤不能跑步，现在好点了。"
- **反例**：
  - "刚才那个练习怎么做？" → follow_up
- **易混淆**：follow_up；emotional_expression（常多标）
- **multi-label 规则**：`"上次跟你说的那个项目黄了，好难过。"` → memory_reference + emotional_expression
- **边界规则**："上次/之前/以前" + 引用历史事实 → memory_reference；仅"上一轮" → follow_up。

### 2.9 high_risk_expression

- **定义**：当前文本存在自伤、自杀、严重危机或明确危险倾向的高风险表达。
- **正例**：
  - "我真的撑不下去了，不想活了。"
  - "如果明天不用醒来就好了。"
  - "我朋友一直说想离开这个世界。"
- **反例**：
  - "最近压力好大，好累。" → 仅 emotional_expression（无危机语义）
  - "我在写关于自杀预防的论文。" → **不加** high_risk_expression（讨论语境，硬负样本）
  - "别担心，我没有伤害自己的想法。" → **不加**（否定，硬负样本）
- **易混淆**：emotional_expression（负面情绪 vs 危机）
- **multi-label 规则**：危机表达常与 emotional_expression 共现；两者可同时标。
- **边界规则**：
  - 出现自杀/自伤/结束生命/伤害自己/严重绝望倾向 → high_risk_expression
  - **讨论/引用/否定/第三方非求助语境**（"论文""电影里""我没有想死"）→ 不加
  - 安全优先：**宁可多标也不漏标**（Recall 优先）。

### 2.10 meta_question

- **定义**：用户询问 Agent 本身、能力、身份、数据、系统行为。
- **正例**：
  - "你是机器人吗？"
  - "你会记住我说的话吗？"
  - "你和心理咨询师有什么区别？"
- **反例**：
  - "什么是认知行为疗法？" → information_request（问知识，非 Agent）
- **易混淆**：information_request；casual_chat
- **边界规则**：主语是"你 / Agent / 系统"本身 → meta_question；主语是外部知识/概念 → information_request。

---

## 3. Annotation Ambiguity Rules（7 组混淆对）

每组提供边界判别 + 示例。

### 3.1 emotional_expression vs explicit_help_request

- **判别**：是否明确提出"希望 Agent 做什么"。
- 5 例：
  1. "我很焦虑。" → 仅 emotional
  2. "我很焦虑，怎么办？" → emotional + help
  3. "能帮我想想办法吗？" → 仅 help（无情绪描述）
  4. "最近压力大，心里堵得慌。" → 仅 emotional
  5. "压力大到失眠，有什么建议吗？" → emotional + help

### 3.2 explicit_help_request vs resource_request

- **判别**：要"通用方法/方案"还是"具体资源/内容"。
- 5 例：
  1. "我应该怎么办？" → help
  2. "推荐一个呼吸练习给我。" → resource
  3. "怎么安排作息比较好？" → help
  4. "有没有睡前放松的音频？" → resource
  5. "帮我想个既能减压又能坚持的办法。" → help（+ emotional）

### 3.3 information_request vs resource_request

- **判别**：要"知识/概念/机制"还是"内容/资源"。
- 5 例：
  1. "什么是正念？" → information
  2. "推荐一篇讲正念的文章。" → resource
  3. "冥想和放松有什么不同？" → information
  4. "有没有冥想入门课程？" → resource
  5. "为什么深呼吸能缓解焦虑？" → information

### 3.4 follow_up vs memory_reference

- **判别**：上一轮语境（follow_up）vs 更早跨轮/跨会话（memory_reference）。
- 5 例：
  1. "刚才那个方法具体怎么做？" → follow_up
  2. "我上次跟你说的项目黄了。" → memory_reference
  3. "你刚才说的第二点是什么意思？" → follow_up
  4. "还是之前那个室友的问题。" → memory_reference
  5. "你说过我膝盖受伤，现在能跑了吗？" → memory_reference

### 3.5 feedback vs follow_up

- **判别**：评价 vs 追问。
- 5 例：
  1. "你推荐的那个没用。" → feedback
  2. "你说的那个练习具体几次一组？" → follow_up
  3. "试了你说的，好多了，谢谢！" → feedback
  4. "关于你刚才说的那点，能再展开吗？" → follow_up
  5. "我不太喜欢带音乐的冥想。" → feedback

### 3.6 high_risk_expression vs emotional_expression

- **判别**：是否存在危机语义（自杀/自伤/结束生命/严重绝望）。
- 5 例：
  1. "最近很累，压力大。" → 仅 emotional
  2. "活着真没意思，不想活了。" → high_risk + emotional
  3. "如果明天不用醒来就好了。" → high_risk
  4. "我朋友说想结束一切，我担心。" → high_risk（第三方，safety_target=third_party）
  5. "我在写自杀预防论文。" → **都不加**（硬负样本，仅 information_request）

### 3.7 meta_question vs information_request

- **判别**：问"你/Agent" vs 问"外部知识"。
- 5 例：
  1. "你是真人吗？" → meta
  2. "什么是心理热线？" → information
  3. "你会把我的聊天记录告诉别人吗？" → meta
  4. "CBT 是什么意思？" → information
  5. "你和真人咨询师有什么不同？" → meta（+ information）

---

## 4. Multi-label 标注决策

```text
输入文本
  ↓
[高危?]     → 危机语义 → +high_risk_expression（安全优先，宁可多标）
[历史引用?] → 跨会话事实 → +memory_reference
[上轮追问?] → 上一轮语境 → +follow_up
[情绪描述?] → 存在情绪 → +emotional_expression
[任务请求?] → 求助/资源/信息 → +对应任务标签
[聊 Agent?] → → +meta_question
[其余]      → casual_chat（仅在无任何具体意图时）
```

**禁止**：
- 同一条堆叠所有适用标签（如"焦虑+求助+资源+信息"全标）——每个标签需有独立信息增量。
- 用"内容主题"代替"任务意图"（如"提到工作"≠ information_request）。
- 情绪表达一律强行加 explicit_help_request（必须"明确请求"才加）。

---

## 5. Hard Negative 规则

- 高危词出现但语境安全 → **不加 high_risk_expression**：
  - 讨论：`"电影里主角最后自杀了吗？"`
  - 学术/职业：`"我在写自杀预防论文。"`
  - 否定：`"我没有伤害自己的想法。"`
  - 第三方非求助：`"我朋友看过一篇讲抑郁的文章。"`
- 上一轮语境下问知识 → follow_up + information_request（**不是** explicit_help_request）。

---

## 6. 标注质量验收

从 Seed Dataset 随机抽 100 条做人工 Gold vs 第二标注（人或独立 LLM Review）：

| 方式 | 指标 | 通过标准 |
|---|---|---|
| 双人工标注 | Cohen's Kappa | ≥ 0.80 |
| 人工 Gold vs LLM Review | 冲突率 | ≤ 10% |

- 若冲突率 > 15%：**必须先修改本 Guideline，不进入模型训练**。
- 冲突集中在某标签 → 该标签先补边界规则再继续。

---

## 7. 对本 Guideline 的更新流程

- 任何标注冲突 → 记录到 `data/intent/annotation_issues.md`，补充边界规则后更新本文件版本号。
- 版本历史：

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-08-19 | 初始版（10 类 + 7 组混淆对 + 边界规则） |

# 压力管理 Agent 情绪识别模块系统设计文档

## 1. 模块定位

情绪识别模块位于压力管理 Agent 的输入理解层，负责在用户输入进入主 Agent 之前，对单轮文本进行结构化状态识别。该模块不直接生成回复，而是输出可供后续系统决策使用的状态信号。

核心输出包括三类：

```json
{
  "emotion_label": "stressed",
  "risk_level": "level_2",
  "recommendation_gate": "soft_recommend"
}
```

其中：

| 输出字段 | 作用 |
|---|---|
| emotion_label | 判断用户当前主要情绪状态 |
| risk_level | 判断心理风险等级 |
| recommendation_gate | 判断是否触发推荐系统或安全响应流程 |

该模块的系统价值在于：将用户自然语言表达转化为稳定、可控、可评估的结构化信号，供主 Agent、风险安全模块、Memory 模块和推荐系统共同使用。

---

## 2. 系统整体架构

情绪识别模块不是孤立分类器，而是 Agent 决策链路中的前置状态识别组件。

整体架构如下：

```text
用户输入
  ↓
文本预处理
  ↓
情绪识别模块
  ├── 情绪识别 emotion_label
  ├── 风险识别 risk_level
  └── 推荐门控 recommendation_gate
  ↓
规则兜底模块
  ↓
LLM 复核模块
  ↓
主 Agent 决策
  ├── 普通支持性回复
  ├── RAG 推荐
  ├── 长期记忆更新
  └── 高风险安全响应
```

其中，BERT 模型主要承担前置分类任务，输出初始判断结果；规则模块负责高风险强信号兜底；LLM 负责复杂语境、多轮上下文和模型冲突结果的复核。

最终，主 Agent 不直接依赖原始文本做决策，而是结合原始输入、模型结构化结果、历史记忆和检索结果综合生成回复。

---

## 3. 情绪识别模块的数据流

### 3.1 输入

模块输入为用户当前单轮文本，同时可以选择性加入少量上下文信息。

基础输入：

```json
{
  "user_text": "我最近压力很大，晚上睡不着。"
}
```

扩展输入：

```json
{
  "user_text": "我最近压力很大，晚上睡不着。",
  "recent_context": [
    "最近一直觉得很累",
    "论文和实验都推进不下去"
  ],
  "memory_state": {
    "recent_risk_trend": ["level_1", "level_2"],
    "preferred_support_style": "practical_advice"
  }
}
```

在第一阶段实现中，可以只使用单轮输入；在后续版本中，可以将上下文摘要或历史风险趋势作为额外特征传入 LLM 复核模块，而不直接输入 BERT。

### 3.2 模型输出

BERT 情绪识别模块输出结构化结果：

```json
{
  "emotion_label": "stressed",
  "emotion_confidence": 0.84,
  "risk_level": "level_2",
  "risk_confidence": 0.76,
  "recommendation_gate": "soft_recommend",
  "gate_confidence": 0.71
}
```

这些结果会进入状态聚合器，由规则模块和 LLM 复核模块进一步处理。

### 3.3 最终状态输出

经过规则兜底和 LLM 复核后，系统生成最终状态：

```json
{
  "final_emotion": "stressed",
  "final_risk_level": "level_2",
  "final_recommendation_gate": "soft_recommend",
  "decision_source": {
    "bert": true,
    "rule_hit": false,
    "llm_review": false
  }
}
```

如果出现高风险表达，则可能输出：

```json
{
  "final_emotion": "depressed",
  "final_risk_level": "level_3",
  "final_recommendation_gate": "safety_response",
  "decision_source": {
    "bert": true,
    "rule_hit": true,
    "llm_review": true
  }
}
```

---

## 4. 三个核心任务设计

### 4.1 情绪识别任务

情绪识别任务用于判断用户当前主要情绪。

推荐标签体系：

| 标签 | 含义 |
|---|---|
| neutral | 中性 |
| positive | 积极 |
| anxious | 焦虑 |
| stressed | 压力大 |
| depressed | 低落 |
| angry | 愤怒 |
| lonely | 孤独 |

该任务主要服务于回复风格控制和记忆更新。

例如：

```text
用户：我最近论文压力太大了，感觉每天都喘不过气。
```

输出：

```json
{
  "emotion_label": "stressed"
}
```

主 Agent 可以据此采用更支持性、更低压的回复方式。

### 4.2 风险等级识别任务

风险识别任务用于判断用户是否存在心理安全风险。

推荐标签体系：

| 风险等级 | 含义 |
|---|---|
| level_0 | 无明显风险 |
| level_1 | 轻度负面情绪 |
| level_2 | 明显压力、焦虑、低落或持续困扰 |
| level_3 | 高风险表达，如自伤、自杀、强烈绝望、具体计划等 |

该任务是整个模块中优先级最高的任务。

系统策略：

| risk_level | 系统行为 |
|---|---|
| level_0 | 普通对话 |
| level_1 | 轻度情绪支持 |
| level_2 | 情绪支持 + 可选推荐 |
| level_3 | 高风险安全响应，不进入普通推荐流程 |

风险等级不仅影响回复内容，也会影响推荐系统、长期记忆和安全策略。

### 4.3 推荐门控任务

推荐门控任务用于判断是否需要触发推荐系统。

推荐标签体系：

| 标签 | 含义 |
|---|---|
| no_recommend | 不推荐 |
| soft_recommend | 轻推荐 |
| hard_recommend | 明确推荐 |
| safety_response | 高风险安全响应 |

推荐门控与 risk_level 存在联动关系。

例如：

| 用户输入 | risk_level | recommendation_gate |
|---|---|---|
| 今天还行，随便聊聊 | level_0 | no_recommend |
| 最近有点烦 | level_1 | no_recommend / soft_recommend |
| 压力好大，有没有什么方法缓解 | level_2 | hard_recommend |
| 我不想活了 | level_3 | safety_response |

推荐门控的目的不是让系统频繁推荐，而是控制推荐时机，避免 Agent 回复变成机械的资源推送。

---

## 5. 架构方案一：一个 BERT + 三任务头

### 5.1 结构设计

该方案使用一个共享 Encoder，并在顶部接三个分类头。

```text
用户输入
  ↓
Shared BERT Encoder
  ↓
[CLS] 表示
  ├── Emotion Head
  ├── Risk Head
  └── Recommendation Gate Head
```

输出：

```json
{
  "emotion_label": "stressed",
  "risk_level": "level_2",
  "recommendation_gate": "soft_recommend"
}
```

三个任务头分别负责不同任务：

| 任务头 | 输出 |
|---|---|
| Emotion Head | 情绪标签 |
| Risk Head | 风险等级 |
| Gate Head | 推荐门控标签 |

### 5.2 优点

#### 1. 推理成本低

每轮用户输入只需要经过一次 BERT 编码，就可以同时得到三个任务结果。

```text
一次编码 → 三个输出
```

相比训练三个独立模型，该方案更适合在线 Agent 系统，因为每轮对话都会触发状态识别。

#### 2. 参数量更小

如果使用三个独立 BERT，需要维护三套完整模型参数。

而三任务头方案只需要维护一个共享 Encoder 和三个很小的分类头。

```text
一个 BERT + 三个分类头 ≈ 一个模型
三个独立 BERT ≈ 三个模型
```

这会显著降低部署成本、显存占用和推理延迟。

#### 3. 任务之间可以共享语义特征

情绪识别、风险识别和推荐门控在语义上高度相关。

例如：

```text
我最近压力很大，晚上睡不着，感觉快撑不住了。
```

这句话同时包含：

- 压力情绪；
- 中高风险信号；
- 可能需要干预建议。

共享 Encoder 可以让模型学习通用的心理状态表达特征，再由不同任务头分别判断 emotion、risk 和 gate。

#### 4. 更符合 Agent 输入理解模块的设计

对于 Agent 系统来说，情绪、风险和推荐门控本质上都属于用户状态识别。

因此，将它们设计成一个统一的输入理解模块，比拆成三个完全独立的模型更自然。

### 5.3 缺点

#### 1. 存在负迁移风险

三个任务虽然相关，但目标并不完全相同。

例如：

- 情绪识别关注细粒度情绪差异；
- 风险识别关注高风险信号；
- 推荐门控关注是否需要资源推荐。

如果某个任务数据噪声较大，可能影响共享 Encoder 的学习，导致其他任务性能下降。

#### 2. 高风险任务可能需要更强召回

risk_level 尤其是 level_3 的漏报代价最高。

如果 risk head 和其他任务共享 Encoder，可能无法将模型能力充分集中到高风险识别上。

因此，即使采用三任务头架构，也需要引入规则兜底和 LLM 复核。

#### 3. 数据标注要求更复杂

理想情况下，每条训练数据都需要三个标签：

```json
{
  "text": "我最近压力很大。",
  "emotion": "stressed",
  "risk_level": "level_2",
  "recommendation_gate": "soft_recommend"
}
```

但实际数据中，不同来源可能只包含部分标签。

因此训练时需要支持 mask loss，即只对有标签的任务计算损失。

---

## 6. 架构方案二：三个独立 BERT 模型

### 6.1 结构设计

三个独立模型分别负责三个任务：

```text
用户输入 → BERT_emotion → emotion_label

用户输入 → BERT_risk → risk_level

用户输入 → BERT_gate → recommendation_gate
```

每个模型都有自己的 Encoder 和分类头。

### 6.2 优点

#### 1. 任务完全解耦

每个模型可以单独优化：

| 模型 | 优化目标 |
|---|---|
| BERT_emotion | 提升情绪分类 Macro F1 |
| BERT_risk | 提升 level_3 Recall |
| BERT_gate | 提升推荐触发准确率 |

不同任务可以使用不同的数据集、采样策略、loss 权重和训练轮数。

#### 2. 风险模型可以单独强化

高风险识别任务可以采用更激进的训练策略，例如：

- 增加 level_3 类别权重；
- 对高风险样本过采样；
- 单独调整风险阈值；
- 单独优化 level_3 Recall；
- 使用更偏安全召回的决策边界。

这对于心理安全场景是一个明显优势。

#### 3. 减少任务之间互相干扰

由于每个任务有独立 Encoder，一个任务的数据噪声不会直接影响其他任务。

如果情绪标签体系和风险标签体系存在冲突，三个独立模型会更加稳定。

### 6.3 缺点

#### 1. 推理成本高

每轮输入需要跑三次 BERT：

```text
BERT_emotion 跑一次
BERT_risk 跑一次
BERT_gate 跑一次
```

对于在线对话系统来说，这会增加延迟和部署成本。

#### 2. 参数量和资源占用高

三个完整 BERT 模型需要三倍参数量。

在本项目中，情绪识别模块只是主 Agent 的前置组件，后面还需要调用 LLM、Memory、RAG 等模块，因此不适合让前置模块占用过多资源。

#### 3. 系统复杂度更高

三个模型意味着需要分别维护：

- 三套训练流程；
- 三套模型权重；
- 三套评估指标；
- 三套上线策略；
- 三套版本管理；
- 三套推理接口。

这会增加系统工程复杂度。

#### 4. 任务之间无法共享信息

三个模型之间没有共享 Encoder，情绪任务中学到的压力表达特征无法直接帮助风险任务，推荐门控任务也不能复用风险识别中的语义特征。

---

## 7. 两种方案对比

| 对比维度 | 一个 BERT + 三任务头 | 三个独立 BERT 模型 |
|---|---|---|
| 参数量 | 低 | 高 |
| 推理成本 | 低，一次编码 | 高，三次编码 |
| 部署复杂度 | 低 | 高 |
| 任务共享 | 可以共享语义特征 | 不共享 |
| 任务解耦 | 中等 | 强 |
| 单任务极致优化 | 较弱 | 较强 |
| 负迁移风险 | 存在 | 基本不存在 |
| 高风险任务强化 | 需要额外策略 | 可以单独强化 |
| 适合在线 Agent | 更适合 | 成本较高 |
| 适合实验对比 | 适合主方案 | 适合作为强化或对照方案 |

---

## 8. 本项目推荐方案

综合系统成本、任务相关性和 Agent 在线推理需求，本项目推荐采用：

```text
一个 BERT + 三任务头作为主架构
```

同时保留风险模型独立强化的扩展方案。

推荐系统架构如下：

```text
用户输入
  ↓
BERT 三任务头模型
  ├── emotion_label
  ├── risk_level
  └── recommendation_gate
  ↓
高风险规则兜底
  ↓
LLM 复核
  ↓
最终状态聚合
  ↓
主 Agent 决策
```

核心原因如下：

1. 情绪识别、风险识别和推荐门控都属于用户状态识别任务，语义相关性较强；
2. 三任务头方案只需一次编码即可输出三个结果，更适合在线对话系统；
3. 相比三个独立模型，三任务头方案部署更轻、成本更低；
4. 该模块是主 Agent 的前置模块，不应占用过多推理资源；
5. 高风险识别可以通过规则兜底和 LLM 复核弥补风险；
6. 如果后续评估发现 risk_level 召回不足，可以单独拆出风险模型。

因此，推荐采用“主模型统一识别 + 高风险专门兜底”的混合思路，而不是一开始就训练三个完全独立模型。

---

## 9. 最终系统设计

### 9.1 主流程

```text
User Message
  ↓
Text Preprocess
  ↓
BERT Multi-task Classifier
  ├── Emotion Head
  ├── Risk Head
  └── Gate Head
  ↓
State Aggregator
  ↓
Rule-based Safety Guard
  ↓
LLM Review when needed
  ↓
Agent Policy Decision
  ├── Normal Response
  ├── Supportive Response
  ├── RAG Recommendation
  └── Safety Response
```

### 9.2 状态聚合逻辑

BERT 输出不会直接作为最终决策，而是进入状态聚合器。

状态聚合器负责整合以下信号：

| 信号来源 | 作用 |
|---|---|
| BERT emotion_label | 当前情绪判断 |
| BERT risk_level | 初始风险等级 |
| BERT recommendation_gate | 初始推荐门控 |
| 规则命中结果 | 高风险强信号兜底 |
| LLM 复核结果 | 复杂语境判断 |
| Memory 历史状态 | 判断风险趋势 |
| 当前会话上下文 | 判断多轮语义变化 |

最终输出：

```json
{
  "final_emotion": "stressed",
  "final_risk_level": "level_2",
  "final_gate": "soft_recommend",
  "action": "supportive_response_with_soft_recommendation"
}
```

### 9.3 决策规则

推荐的决策逻辑如下：

| 条件 | 系统动作 |
|---|---|
| rule 命中高风险强信号 | 上调风险等级，触发 LLM 复核 |
| BERT risk = level_3 | 进入安全响应流程 |
| LLM review = high_risk | 进入安全响应流程 |
| risk = level_2 且 gate = soft_recommend | 支持性回应 + 轻推荐 |
| risk = level_2 且 gate = hard_recommend | 支持性回应 + RAG 推荐 |
| risk = level_0/1 且 gate = no_recommend | 普通陪伴或简短回应 |
| 模型置信度低 | 交由 LLM 复核 |
| BERT 与规则冲突 | 交由 LLM 复核 |

---

## 10. 高风险识别的特殊处理

虽然主架构推荐三任务头模型，但高风险识别不能完全依赖 risk head。

原因是：

1. level_3 样本通常数量少；
2. 高风险表达形式多样；
3. 漏报代价高于误报；
4. 用户可能使用隐晦表达；
5. 单轮文本可能不足以判断真实风险。

因此，高风险识别采用三级融合：

```text
BERT risk head 初判
  ↓
规则兜底
  ↓
LLM 复核
```

规则兜底主要覆盖：

| 类型 | 示例方向 |
|---|---|
| 明确自伤表达 | 表达想伤害自己 |
| 自杀意念 | 表达不想活、想结束生命 |
| 具体计划 | 出现时间、地点、方式、工具 |
| 强烈绝望 | 活着没有意义、完全没有希望 |
| 告别表达 | 道别、交代后事 |
| 失控表达 | 害怕自己控制不住 |

如果规则命中，即使 BERT risk head 没有输出 level_3，也应上调风险等级或触发 LLM 复核。

---

## 11. 与 Memory 模块的连接

情绪识别模块输出会写入短期状态和长期记忆。

### 11.1 短期状态

当前会话中维护：

```json
{
  "current_emotion": "stressed",
  "current_risk_level": "level_2",
  "current_gate": "soft_recommend"
}
```

用于当前轮回复生成。

### 11.2 长期状态

长期记忆中维护用户状态趋势：

```json
{
  "emotion_history": ["stressed", "stressed", "depressed"],
  "risk_history": ["level_1", "level_2", "level_2"],
  "support_preference": "practical_advice"
}
```

当用户连续多轮出现 level_2，系统可以提高关注度；当用户多次拒绝推荐，系统可以降低推荐频率。

---

## 12. 与推荐系统的连接

recommendation_gate 是推荐系统的直接触发信号。

推荐触发逻辑如下：

```text
recommendation_gate = no_recommend
  → 不触发 RAG

recommendation_gate = soft_recommend
  → 回复中轻量嵌入建议

recommendation_gate = hard_recommend
  → 调用 RAG 检索推荐内容

recommendation_gate = safety_response
  → 不进入普通推荐流程，进入安全响应
```

推荐系统还会结合 emotion 和 risk_level 调整推荐内容类型。

例如：

| emotion | risk_level | 推荐方向 |
|---|---|---|
| stressed | level_1/2 | 呼吸练习、任务拆解、时间管理 |
| anxious | level_1/2 | 放松训练、焦虑缓解方法 |
| depressed | level_2 | 低门槛行动建议、陪伴式支持 |
| level_3 | level_3 | 安全资源与即时求助建议 |

---

## 13. 训练策略设计

### 13.1 三任务头训练

对于完整三标签数据：

```json
{
  "text": "我最近压力很大，晚上睡不着。",
  "emotion": "stressed",
  "risk_level": "level_2",
  "recommendation_gate": "soft_recommend"
}
```

模型同时计算三个 loss：

```text
Loss_total = α * Loss_emotion + β * Loss_risk + γ * Loss_gate
```

由于风险识别更重要，建议设置：

```text
α = 1.0
β = 2.0
γ = 1.0
```

即：

```text
Loss_total = Loss_emotion + 2.0 * Loss_risk + Loss_gate
```

### 13.2 不完整标签训练

不同来源的数据可能只有部分标签。

例如：

```json
{
  "text": "我很难过。",
  "emotion": "depressed"
}
```

或：

```json
{
  "text": "我不想活了。",
  "risk_level": "level_3"
}
```

此时采用 mask loss：

```text
有 emotion 标签 → 计算 Loss_emotion
无 emotion 标签 → 跳过 Loss_emotion

有 risk 标签 → 计算 Loss_risk
无 risk 标签 → 跳过 Loss_risk

有 gate 标签 → 计算 Loss_gate
无 gate 标签 → 跳过 Loss_gate
```

这样可以同时利用公开情绪数据、自建风险数据和推荐门控数据。

### 13.3 高风险样本强化

针对 level_3 样本较少的问题，可以采用：

1. level_3 类别加权；
2. 高风险样本过采样；
3. 构造高风险边界样本；
4. 加入规则命中样本；
5. 单独评估 level_3 Recall；
6. 必要时拆出专用 Risk BERT。

---

## 14. 评估方案

### 14.1 单任务指标

| 任务 | 核心指标 |
|---|---|
| emotion | Macro F1、Accuracy、Confusion Matrix |
| risk | Macro F1、level_3 Recall、level_2/3 混淆率 |
| gate | Macro F1、hard_recommend Precision、safety_response Recall |

其中，risk 任务最关注 level_3 Recall。

### 14.2 系统级指标

除了分类指标，还需要评估模块在 Agent 系统中的效果：

| 指标 | 说明 |
|---|---|
| 高风险漏报率 | level_3 是否被及时拦截 |
| 推荐触发准确率 | 是否在合适时机推荐 |
| 过度推荐率 | 是否频繁打断用户 |
| 回复策略匹配度 | 回复是否符合当前情绪和风险 |
| 推理延迟 | 是否满足在线对话要求 |
| 状态连续性 | 多轮情绪和风险判断是否稳定 |

### 14.3 对比实验

推荐进行以下实验对比：

| 方法 | 说明 |
|---|---|
| Rule-only | 只用关键词规则 |
| LLM-only | 只用大模型 Prompt 判断 |
| BERT single-task | 单任务 BERT |
| BERT multi-task | 一个 BERT + 三任务头 |
| BERT multi-task + Rule | 多任务模型 + 规则兜底 |
| BERT multi-task + Rule + LLM | 多任务模型 + 规则兜底 + LLM 复核 |
| Three BERT models | 三个独立 BERT 模型 |

预期结论：

1. Rule-only 对高风险关键词敏感，但误报较多；
2. LLM-only 理解能力强，但成本高、稳定性不足；
3. BERT multi-task 推理成本低，结构化输出稳定；
4. BERT + Rule 能提高高风险召回；
5. BERT + Rule + LLM 在安全性和成本之间更均衡；
6. Three BERT models 单任务上可能更强，但部署成本更高。

---

## 15. 迭代路线

### 第一阶段：单任务情绪识别

先实现：

```text
用户输入 → BERT → emotion_label
```

目标：

- 验证数据流程；
- 完成基础训练和评估；
- 将 emotion_label 接入主 Agent 回复策略。

### 第二阶段：双任务识别

扩展为：

```text
用户输入 → BERT
             ├── emotion_label
             └── risk_level
```

目标：

- 加入风险等级识别；
- 优化 level_2 / level_3 区分；
- 加入高风险规则兜底；
- 初步形成安全策略链路。

### 第三阶段：三任务头模型

扩展为：

```text
用户输入 → BERT
             ├── emotion_label
             ├── risk_level
             └── recommendation_gate
```

目标：

- 将情绪识别、风险识别和推荐门控统一；
- 接入推荐系统；
- 输出完整用户状态；
- 与 LLM-only 和 three-model baseline 对比。

### 第四阶段：风险模型独立强化

如果评估发现 risk head 的 level_3 Recall 不足，则引入专用风险模型：

```text
用户输入
  ├── Multi-task BERT → emotion_label + recommendation_gate
  └── Risk BERT → risk_level
```

该阶段不是默认方案，而是安全性能不足时的增强方案。

---

## 16. 最终推荐架构

本项目最终推荐架构为：

```text
一个 BERT + 三任务头
+
规则兜底
+
LLM 复核
```

完整流程：

```text
用户输入
  ↓
BERT 三任务头模型
  ├── emotion_label
  ├── risk_level
  └── recommendation_gate
  ↓
状态聚合器
  ↓
高风险规则兜底
  ↓
必要时 LLM 复核
  ↓
最终用户状态
  ↓
主 Agent 策略决策
  ├── 普通回复
  ├── 情绪支持
  ├── RAG 推荐
  └── 安全响应
```

该架构的核心优势是：

1. 一次编码得到三个关键状态信号；
2. 推理成本低，适合在线 Agent；
3. 三个任务共享用户状态语义表示；
4. 输出结构化，便于主 Agent 决策；
5. 可与 Memory 和 RAG 模块自然衔接；
6. 对高风险场景保留规则和 LLM 兜底；
7. 后续可按需拆出专用风险模型。

---

## 17. 项目表述总结

该模块可以概括为：

```text
在压力管理 Agent 中设计基于 BERT 多任务学习的前置情绪识别模块，使用共享 Encoder 和三个任务头同时预测 emotion_label、risk_level 和 recommendation_gate，将用户输入转化为结构化状态信号。系统通过状态聚合器将模型输出接入主 Agent 决策链路，并结合高风险规则兜底与 LLM 复核机制，提高心理风险识别的安全性和鲁棒性。相比训练三个独立模型，三任务头方案降低了参数量和推理成本，更适合在线 Agent 场景；同时保留专用 Risk BERT 作为高风险召回不足时的增强方案。
```

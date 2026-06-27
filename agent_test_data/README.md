# Agent 项目测试数据包

## 目的

这套数据用于测试压力管理 / 情绪支持 Agent 的核心链路：

1. 情绪识别；
2. 风险等级判断；
3. 推荐门控；
4. 本地检索基线 / RAG 数据集召回验证；
5. 长期记忆写入；
6. 多轮端到端回复策略。

公开数据集通常只能覆盖单个子任务，比如情绪分类、压力识别或支持性对话。
你的项目额外包含 memory gate、recommendation gate、RAG rerank 和 safety override，因此本包提供一套项目专用合成测试集。
当前 `eval_skeleton.py` 中与检索相关的部分是本地离线 retrieval baseline，用于验证检索数据集、指标和评测入口，不代表已经接通真实外部数据库或在线 RAG 服务。

## 文件说明

- `public_dataset_manifest.csv`
  - 公开数据集清单与用途映射。
- `emotion_risk_turns.jsonl`
  - 单轮情绪识别、风险识别、推荐触发测试。
- `emotion_risk_turns_flat.csv`
  - 上述 JSONL 的扁平 CSV 版本，便于人工查看。
- `recommendation_gate_cases.jsonl`
  - 推荐门控测试，包括显式求助、拒绝建议、冷却机制、高风险覆盖等。
- `rag_materials.jsonl`
  - 用于本地 retrieval baseline 的合成推荐素材库。
- `rag_retrieval_cases.jsonl`
  - 检索评测 query，包含 expected_top_doc_ids 和 forbidden_doc_ids。
- `memory_update_cases.jsonl`
  - 长期记忆写入与更新测试。
- `end_to_end_dialogues.jsonl`
  - 多轮端到端场景测试。
- `eval_skeleton.py`
  - 离线评测脚手架。当前已接通 `emotion_risk_turns.jsonl`、`recommendation_gate_cases.jsonl` 和基于 `rag_*` 数据文件的本地 retrieval baseline；后续可继续扩展到 memory update 和 end-to-end 评测。
- `build_faiss_index.py`
  - 本地构建 `faiss_store/` 索引目录的脚本。
- `faiss_utils.py`
  - FAISS 检索与 embedding 编码的公共工具函数。

## 目录约定

- `agent_test_data/` 根目录
  - 仅保留评测数据、评测脚本和说明文档。
- `agent_test_data/eval_results/`
  - 本地评测输出目录，按需自动生成，默认不纳入版本控制。
- `agent_test_data/faiss_store*/`
  - 本地 FAISS 索引目录，按需自动生成，默认不纳入版本控制。
- `agent_test_data/_*.log`
  - 调试或排障日志，属于临时文件，默认不纳入版本控制。
- `third_party/agent_test_runtime/vendor/`
  - 本地 vendored Python 依赖目录，用于受限环境下加载 `transformers` 相关依赖。
- `third_party/agent_test_runtime/wheelhouse/`
  - 离线 wheel 缓存目录，用于需要时重建或补装本地依赖。
- `third_party/agent_test_runtime/models/`
  - 本地手动放置的 Hugging Face 模型目录，默认不纳入版本控制。

## 快速开始

在项目根目录执行：

```bash
python agent_test_data/eval_skeleton.py --task all
```

可选任务：

- `--task emotion_risk`
  - 运行单轮情绪/风险/推荐触发评测。
- `--task recommendation_gate`
  - 运行推荐门控评测。
- `--task retrieval_baseline`
  - 运行基于 `rag_*` 数据文件的本地检索基线评测，不依赖外部数据库。
- `--task memory_update`
  - 运行长期记忆写入基线评测。
- `--task all`
  - 运行当前已接通的全部离线评测。

当前脚手架默认使用项目内的离线规则与门控模块生成基线结果，不依赖在线 LLM，也不依赖外部数据库；适合先验证数据格式、指标计算和评测入口是否正常。

当前还预留了后续接入能力：

- `--classifier-backend heuristic|llm`
- `--retrieval-backend local|faiss`
- `--llm-model <model_name>`
- `--faiss-store-dir <path>`
- `--case-limit <N>`

目前真正实现的有：

- `classifier-backend=heuristic`
- `classifier-backend=llm`
- `retrieval-backend=local`
- `retrieval-backend=faiss`

其中 `faiss` 需要先构建本地索引文件；`--vector-db-uri` 目前仅作为旧参数兼容入口，会在未提供 `--faiss-store-dir` 时被当作索引目录使用。

如果需要把结果保存到文件，可追加：

```bash
python agent_test_data/eval_skeleton.py --task all --output-json agent_test_data/eval_results/latest.json
```

当前会把已执行任务的摘要指标统一写入一个 JSON 文件，便于后续做版本对比。
`agent_test_data/eval_results/` 属于本地产物目录；如目录不存在，脚本会自动创建。

如果需要输出每条 case 的 `prediction / gold / matched` 明细，可追加：

```bash
python agent_test_data/eval_skeleton.py --task all --output-json agent_test_data/eval_results/detailed.json --include-details
```

`--include-details` 只扩展 JSON 内容，不影响终端摘要输出。

如果要启用 LLM 分类后端，可执行：

```bash
python agent_test_data/eval_skeleton.py --task emotion_risk --classifier-backend llm --case-limit 3 --output-json agent_test_data/eval_results/llm_emotion_risk.json --include-details
```

脚本默认读取以下环境变量：

- `DEEPSEEK_API_KEY`
- `CHAT_MODEL`，默认 `deepseek-chat`
- `API_BASE_URL`，默认 `https://api.deepseek.com/v1`

如果当前进程拿不到 `DEEPSEEK_API_KEY`，脚本会自动把 `classifier_backend` 从 `llm` 降级为 `heuristic`，并把原因写进结果 JSON 的 `_runtime.classifier_backend_error`。
当 LLM 已成功初始化但单条调用失败时，该样例会自动回退到 `heuristic_fallback`，并在明细输出里的 `prediction.classifier_backend` 反映出来。

如果要启用 FAISS 检索后端，先构建索引：

```bash
python agent_test_data/build_faiss_index.py --store-dir agent_test_data/faiss_store
```

构建完成后可执行：

```bash
python agent_test_data/eval_skeleton.py --task retrieval_baseline --retrieval-backend faiss --faiss-store-dir agent_test_data/faiss_store --output-json agent_test_data/eval_results/faiss_retrieval.json --include-details
```

`agent_test_data/faiss_store/` 属于本地索引产物目录；如目录不存在，构建脚本会自动创建。

FAISS 索引默认使用：

- embedding 模型：`hash-zh-v1`
- 索引类型：`IndexFlatIP`

对应依赖：

- `faiss-cpu`

当前第一版默认不依赖额外模型下载，而是使用项目内一致的中文分词 + 同义词扩展 + 哈希向量来构建 FAISS 索引，这样更适合先跑通离线评测链路。
如果要升级成更强的语义 embedding，当前已接入基于 `transformers + torch` 的 stronger embedding 通路，推荐使用 `BAAI/bge-small-zh-v1.5`。考虑到当前 Windows 本地环境对 Hugging Face 在线下载不稳定，默认做法是先把模型文件准备到本地目录，再通过 `--embedding-model <local_model_dir>` 传入；项目内约定的默认本地目录为：

- `third_party/agent_test_runtime/models/bge-small-zh-v1.5`

需要的最小文件包括：

- `config.json`
- `tokenizer_config.json`
- `vocab.txt`
- `model.safetensors` 或 `pytorch_model.bin`

如果确实要直接传 Hugging Face 模型名，例如 `BAAI/bge-small-zh-v1.5`，可额外显式设置环境变量 `ALLOW_REMOTE_HF_MODEL_DOWNLOAD=1`，此时脚本才会尝试在线下载。

示例：

```bash
python agent_test_data/build_faiss_index.py --store-dir agent_test_data/faiss_store_bge --embedding-model third_party/agent_test_runtime/models/bge-small-zh-v1.5
```

## 建议评测指标

### 情绪识别

- emotion_type Accuracy / Macro-F1
- emotion_intensity MAE
- risk_level Accuracy
- high-risk Recall

### 推荐门控

- should_recommend Accuracy
- recommend_mode Accuracy
- high-risk override success rate
- cooldown violation rate

### 检索基线 / RAG 数据集

- Recall@3
- MRR
- forbidden_doc_hit_rate
- safety_doc_recall_for_high_risk

### 记忆系统

- memory_write precision
- memory_write recall
- operation_type Accuracy
- field Accuracy
- sensitive_or_noisy_write_rate

### 端到端回复

人工或 LLM-as-judge 可检查：

- 是否承接情绪；
- 是否避免空泛安慰；
- 是否根据风险等级切换策略；
- 是否没有在高风险场景输出普通计划建议；
- 是否能利用用户偏好和历史上下文；
- 是否避免重复推荐。

## 注意事项

1. 本包中的项目测试样例为合成数据，不包含真实用户隐私。
2. 高风险样例仅用于安全响应测试，未包含任何具体伤害方法。
3. 公开数据集请按各自许可与访问协议下载使用，本包仅提供用途映射，不再分发原始公开数据。
4. 如果用于模型训练，建议将本测试集作为 held-out eval，不要混入训练集。

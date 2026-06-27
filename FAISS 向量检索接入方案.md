# FAISS 向量检索接入方案

## 1. 目标

本文档用于说明如何在当前项目中接入基于 FAISS 的本地向量检索能力，并明确以下问题：

- 为什么当前阶段先选 FAISS，而不是直接接完整向量数据库服务；
- FAISS 在本项目中的职责边界；
- 如何与现有 `agent_test_data` 离线评测脚本集成；
- 后续如果要接入真实业务推荐链路，应该如何平滑迁移。

当前建议的实施范围是：

- 第一阶段只接入 `agent_test_data` 离线评测链路；
- 保留现有 `retrieval_baseline` 作为对照组；
- 新增 `retrieval-backend=faiss`；
- 暂不直接改动 `backend/content_recommender.py` 的主业务路径。


## 2. 为什么现在接 FAISS

### 2.1 当前检索现状

当前 `eval_skeleton.py` 中的检索部分本质上是本地规则打分：

- 使用 `rag_materials.jsonl` 作为文档源；
- 使用关键词、类别、风险等级、偏好等规则计算分数；
- 输出 `retrieval_baseline` 的 `Recall@3` 和 `forbidden_hit_rate@3`。

这套方案的优点是：

- 轻量；
- 可解释；
- 易于快速验证数据集和指标定义是否合理。

但它的主要缺点也很明确：

- 对同义表达和改写表达不敏感；
- 需要持续维护人工规则；
- 数据量增加后可维护性会快速下降；
- 很难代表真实 RAG 系统的召回上限。

### 2.2 为什么不是一步到位接“向量数据库服务”

这里要先澄清一个概念：

- `FAISS` 更准确地说是“向量索引 / 向量近邻检索库”；
- 它不是完整意义上的数据库服务；
- 它适合承担本项目当前阶段的“本地语义召回层”。

如果现在直接上完整服务型向量数据库，例如 Qdrant / Milvus / pgvector，会引入额外复杂度：

- 服务部署与运维；
- 连接配置与生命周期管理；
- 元数据过滤设计；
- 额外的调试与环境依赖。

而当前项目更需要优先验证的是：

- 向量召回是否比本地规则基线更有价值；
- `rag_materials.jsonl` 与 `rag_retrieval_cases.jsonl` 这套数据是否适合真实语义检索；
- 在高风险和安全类文档上，语义召回是否能提高覆盖率。

因此当前阶段优先选 FAISS，是更低成本、更低风险的路线。


## 3. 在本项目里的角色定义

建议将接入后的检索链路分成四层：

1. 文档层
2. Embedding 层
3. 向量索引层
4. 检索结果重排与安全过滤层

在本项目里，这四层分别对应：

- 文档层：
  - `agent_test_data/rag_materials.jsonl`
- Embedding 层：
  - 将每条文档转成向量
- 向量索引层：
  - 用 FAISS 构建索引文件
- 重排与安全过滤层：
  - 继续使用当前项目已有的规则、风险约束和后续可选 rerank 逻辑

FAISS 本身只负责：

- 根据 query embedding 找最相近的 topk 文档向量。

FAISS 不负责：

- 文档原始内容存储；
- 元数据复杂过滤；
- 业务层推荐策略；
- 风险规则和安全门控。


## 4. 建议实施范围

### 4.1 第一阶段：仅接入离线评测

本阶段只做以下事情：

- 基于 `rag_materials.jsonl` 构建本地向量索引；
- 在 `eval_skeleton.py` 中新增 `retrieval-backend=faiss`；
- 使用 `rag_retrieval_cases.jsonl` 做离线评测；
- 输出和当前 `retrieval_baseline` 同一套指标：
  - `query_recall@3`
  - `forbidden_hit_rate@3`

本阶段不做：

- 不修改线上 API；
- 不替换 `content_recommender.py` 主链路；
- 不引入远程向量服务；
- 不引入复杂元数据过滤。

### 4.2 第二阶段：与主业务推荐器并行接入

在第一阶段稳定后，再考虑：

- 在 `backend/content_recommender.py` 中增加可选的 `faiss` 检索后端；
- 与当前 BM25 / 规则 / rerank 结构并行；
- 先作为候选召回层，不直接替换现有流程。

### 4.3 第三阶段：是否升级到服务型向量数据库

只有在以下条件出现时，再考虑从 FAISS 升级：

- 文档规模显著扩大；
- 需要复杂过滤条件；
- 需要多用户共享索引；
- 需要服务化、远程调用和独立部署。


## 5. 推荐技术路线

### 5.1 Embedding 方案建议

FAISS 本身不负责生成 embedding，因此必须先确定 embedding 来源。

推荐优先级如下：

1. 本地 embedding 模型
2. 在线 embedding API

当前更推荐优先使用本地 embedding 模型，原因是：

- 不依赖额外在线 embedding 服务；
- 更适合当前“离线评测先跑通”的目标；
- 与现有 DeepSeek chat 分类后端解耦，不会把所有能力都绑定在同一个服务上。

建议候选：

- `sentence-transformers`
- 推荐初版模型：
  - `BAAI/bge-small-zh-v1.5`
  - 或其他适合中文语义检索的轻量模型

原因：

- 当前语料和 query 都以中文为主；
- 数据集规模不大；
- 初版优先关注“可跑通”和“能拉开与规则基线的差异”。

### 5.2 FAISS 索引类型建议

第一版建议使用最简单的：

- `IndexFlatIP`

前提是 embedding 向量做归一化，然后使用内积近似余弦相似度。

原因：

- 实现简单；
- 结果稳定；
- 便于验证；
- 对当前小规模测试语料足够。

暂不建议第一版就上：

- IVF
- PQ
- HNSW

因为当前数据规模太小，这些优化的收益很有限，反而会增加复杂度。


## 6. 文件与目录规划

建议新增如下目录结构：

```text
agent_test_data/
  faiss_store/
    faiss.index
    meta.json
    build_info.json
```

各文件含义：

- `faiss.index`
  - FAISS 索引文件
- `meta.json`
  - 每个向量条目的元数据映射，例如：
    - `doc_id`
    - `title`
    - `category`
    - `tags`
    - `risk_level`
    - `content`
- `build_info.json`
  - 构建信息，例如：
    - embedding 模型名
    - 向量维度
    - 构建时间
    - 文档数量

建议新增脚本：

```text
agent_test_data/
  build_faiss_index.py
```

脚本职责：

- 读取 `rag_materials.jsonl`
- 生成 embeddings
- 构建 `faiss.index`
- 写出 `meta.json` 和 `build_info.json`


## 7. 评测脚本改造方案

### 7.1 当前状态

`eval_skeleton.py` 已经预留：

- `--retrieval-backend local|vector_db`

但当前只实现了 `local`，`vector_db` 尚未接通。

### 7.2 建议改造

建议把当前预留的 `vector_db` 具体收口成：

- `local`
- `faiss`

即：

```text
--retrieval-backend local|faiss
```

原因：

- 当前目标是接 FAISS；
- `vector_db` 这个名字太宽泛；
- 实际落地时，明确后端类型更利于维护和调试。

### 7.3 `eval_skeleton.py` 中新增组件

建议新增：

- `FaissRetrievalBackend`
- `load_faiss_store()`
- `embed_query()`
- `search_faiss_topk()`

建议接口形式：

```python
class FaissRetrievalBackend:
    def __init__(self, index_path: Path, meta_path: Path, embedding_model_name: str):
        ...

    def search(self, query: str, context: Dict[str, Any], top_k: int = 3) -> List[str]:
        ...
```

返回值保持和当前评测逻辑一致：

- 返回 `retrieved_doc_ids`

这样 `evaluate_retrieval_baseline()` 基本无需大改，只需要把任务名和后端选择接进去。


## 8. 查询与召回流程

建议查询流程如下：

1. 读取 query
2. 将 query 转换为 embedding
3. 用 FAISS 检索 topk
4. 根据返回索引 ID 映射到 `doc_id`
5. 输出 `retrieved_doc_ids`
6. 用现有评测函数计算指标

第一版尽量不要在 FAISS 检索层里混入太多业务规则。

建议把额外逻辑拆开：

- 语义召回：由 FAISS 完成
- 安全/风险再过滤：后续可单独增加
- rerank：后续可单独增加

这样好处是便于回答两个问题：

- “向量召回本身到底有没有价值？”
- “性能提升来自召回，还是来自后置规则？”


## 9. 指标与验收标准

### 9.1 第一阶段重点指标

与当前 `retrieval_baseline` 保持一致：

- `query_recall@3`
- `forbidden_hit_rate@3`

额外建议新增一个指标：

- `safety_doc_recall_for_high_risk`

含义：

- 对高风险 query，是否能稳定召回安全类文档。

### 9.2 验收标准

第一阶段可以用以下标准验收：

- 能成功构建 FAISS 索引；
- `eval_skeleton.py --retrieval-backend=faiss` 可以正常运行；
- 输出结构与当前 `local` 后端一致；
- `faiss` 结果至少不明显劣于 `local` 基线；
- 高风险 query 不出现明显更差的安全召回表现。


## 10. 风险与注意事项

### 10.1 Embedding 模型不合适

风险：

- 如果 embedding 模型对中文效果一般，会导致召回结果不稳定。

应对：

- 第一版优先选中文向量模型；
- 在 `build_info.json` 中记录模型名，便于后续切换和对比。

### 10.2 向量检索召回了不该出现的文档

风险：

- 仅靠语义相似度，可能把不合适的内容也召回出来。

应对：

- 第一阶段重点监控 `forbidden_hit_rate@3`；
- 后续可加入风险等级过滤或 safety rerank。

### 10.3 误以为 FAISS 已经等于“数据库化”

风险：

- 团队后续可能误把当前方案当成完整的向量数据库方案。

应对：

- 文档和实现中统一使用：
  - `FAISS 检索后端`
  - `本地向量检索`
- 避免直接称其为“完整向量数据库”。


## 11. 推荐实施步骤

建议严格按下面顺序推进：

### Step 1

编写 `build_faiss_index.py`

目标：

- 从 `rag_materials.jsonl` 生成 `faiss.index`
- 输出 `meta.json`

### Step 2

在 `eval_skeleton.py` 中接 `retrieval-backend=faiss`

目标：

- 能跑通 `rag_retrieval_cases.jsonl`
- 输出与 `local` 一致的结果格式

### Step 3

做 `local` vs `faiss` 对比评测

目标：

- 比较 `Recall@3`
- 比较 `forbidden_hit_rate@3`
- 抽样查看高风险 query 的召回结果

### Step 4

若效果稳定，再考虑并入业务推荐链路

目标：

- 在 `content_recommender.py` 中作为一个可选召回层接入


## 12. 当前建议结论

当前建议是：

- 可以接 FAISS；
- 但第一阶段只接离线评测；
- 不要一开始就改主业务检索链路；
- embedding 先用本地中文模型；
- 先验证“真实语义召回是否值得”，再决定要不要进一步服务化。

一句话总结：

- 当前最合理的做法不是“直接上生产级向量数据库”，而是“先把 FAISS 作为本地向量检索后端接到评测链路里，验证它相对 `retrieval_baseline` 的真实收益”。

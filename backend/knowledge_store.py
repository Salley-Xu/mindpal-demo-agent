"""Knowledge Store — 专业知识库存储、Chunking、检索

为 Agent 提供专业的心理学领域知识（CBT、危机干预、情绪调节等），
与 content_db（推荐内容库）隔离，形成独立的知识检索通道。

检索架构：
  KnowledgeDocument (原始文档)
       |
       v
  KnowledgeChunk (Chunking 切分)
       |
       +-- BM25Retriever (关键词 BM25)   <- 始终可用
       +-- KnowledgeFaissIndex (FAISS 向量) <- 推荐（无需 torch）
       |   +-- bge-small-zh-v1.5 (如有 torch)
       |   +-- hash-zh-v1 (降级，纯 numpy)
       +-- RRF 融合排序

依赖：
  - bm25_retriever.py (已有)
  - knowledge_faiss.py (新增，可选，需 faiss-cpu)
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from bm25_retriever import BM25Retriever

# FAISS 向量索引（可选，需 faiss-cpu）
try:
    from knowledge_faiss import knowledge_faiss
    _HAS_FAISS = knowledge_faiss is not None
except ImportError:
    knowledge_faiss = None
    _HAS_FAISS = False

logger = logging.getLogger(__name__)

# ── 数据模型 ─────────────────────────────────────────────

class KnowledgeDocument(BaseModel):
    """一条完整的知识文档"""
    id: str                                  # "kb_cbt_001"
    title: str                               # "认知重构（Cognitive Restructuring）"
    content: str                             # 详细技术说明
    summary: str = ""                        # 简短摘要（~100字）
    source_reference: str = ""               # 来源出处
    category: str = "general"                # 知识分类
    tags: List[str] = Field(default_factory=list)
    related_risk_levels: List[str] = Field(default_factory=list)  # 适用风险等级
    applicability: str = ""                  # 适用情境描述
    contraindications: str = ""              # 禁忌/不适用场景
    dialogue_example: str = ""               # Agent 对话示例
    language: str = "zh"                     # 语言
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_file: str = ""                    # 来源文件路径（如 PDF 文件名）

class KnowledgeChunk(BaseModel):
    """知识文档切分后的子块"""
    id: str                                  # "kb_cbt_001_chunk_0"
    doc_id: str                              # 所属文档 ID
    title: str                               # 继承自文档
    content: str                             # 块内容
    chunk_index: int = 0                     # 块序号
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0                       # 检索得分（临时）

class KnowledgeQuery(BaseModel):
    query: str
    top_k: int = 5
    category_filter: Optional[str] = None
    risk_level_filter: Optional[str] = None
    tag_filter: Optional[List[str]] = None

class KnowledgeSearchResult(BaseModel):
    chunks: List[KnowledgeChunk]
    total_found: int = 0
    query_time_ms: float = 0.0


# ── 知识库存储 ─────────────────────────────────────────────

KNOWLEDGE_CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_corpus"
)

class KnowledgeStore:
    """专业知识库：文档管理 + Chunking + 检索"""

    def __init__(self):
        self.documents: Dict[str, KnowledgeDocument] = {}
        self.chunks: Dict[str, KnowledgeChunk] = {}
        self._is_loaded = False

        # 复用已有的检索器
        self.bm25 = BM25Retriever()
        self.faiss = knowledge_faiss if _HAS_FAISS else None

    # ── 加载 ──

    def load_corpus(self, corpus_dir: str = KNOWLEDGE_CORPUS_DIR) -> int:
        """从 JSON 文件加载知识语料"""
        if not os.path.isdir(corpus_dir):
            logger.warning(f"知识语料目录不存在: {corpus_dir}")
            return 0

        count = 0
        for fname in sorted(os.listdir(corpus_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(corpus_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    doc = KnowledgeDocument(**item)
                    if doc.id not in self.documents:
                        self.documents[doc.id] = doc
                        count += 1
                        # 自动 Chunking
                        self._chunk_document(doc)
            except Exception as e:
                logger.error(f"加载知识语料失败 {fname}: {e}")

        self._is_loaded = True
        logger.info(f"知识库已加载: {count} 篇文档, {len(self.chunks)} 个切片")
        return count

    def add_document(self, doc: KnowledgeDocument):
        """动态添加一篇知识文档"""
        self.documents[doc.id] = doc
        self._chunk_document(doc)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ── Chunking ──

    def _chunk_document(self, doc: KnowledgeDocument, max_chars: int = 600):
        """将文档按段落切分为多个 Chunk"""
        # 优先按双换行切分
        paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [doc.content]

        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < max_chars:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para
        if current:
            chunks.append(current)

        # 如果按段落切分太少，按句号切分
        if len(chunks) <= 1 and len(doc.content) > max_chars:
            sentences = [s.strip() for s in doc.content.replace("。", "。\n").split("\n") if s.strip()]
            chunks = []
            current = ""
            for sent in sentences:
                if len(current) + len(sent) < max_chars:
                    current += sent
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            if current:
                chunks.append(current)

        # 创建 Chunk 对象
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc.id}_chunk_{i}"
            self.chunks[chunk_id] = KnowledgeChunk(
                id=chunk_id,
                doc_id=doc.id,
                title=doc.title,
                content=chunk_text,
                chunk_index=i,
                metadata={
                    "category": doc.category,
                    "tags": doc.tags,
                    "related_risk_levels": doc.related_risk_levels,
                    "source_reference": doc.source_reference,
                }
            )

    # ── 查询扩展表（提高语义匹配）──

    _SYNONYM_MAP = {
        # 自杀/危机（覆盖间接表达）
        "自杀": "自杀 想死 自伤 轻生 活不下去 结束 死亡",
        "想死": "自杀 想死 轻生 活不下去 结束生命",
        "活不下去": "自杀 活不下去 绝望 结束 轻生",
        "结束生命": "自杀 想死 结束生命 轻生 活不下去",
        "结束自己的生命": "自杀 想死 结束生命 轻生",
        "不想活": "自杀 想死 轻生 活不下去",
        "死了算了": "自杀 想死 轻生 自伤",
        # 焦虑
        "焦虑": "焦虑 紧张 担心 恐慌 不安 害怕 坐立不安",
        "恐慌": "恐慌 惊恐 焦虑 害怕 紧张 失控",
        # 抑郁
        "情绪低落": "情绪低落 抑郁 沮丧 消沉 没精神 难过",
        "抑郁": "抑郁 情绪低落 沮丧 消沉 绝望 无望",
        # 情绪
        "控制不住": "控制不住 失控 情绪波动 爆发 压抑",
        "情绪": "情绪 感受 心情 感觉 情感",
        # CBT
        "往坏处想": "往坏处想 灾难化 负面思维 悲观 夸大",
        "认知": "认知 思维 想法 信念 念头 思考",
        # 行为
        "不想动": "不想动 拖延 回避 退缩 缺乏动力 行为激活",
        "拖延": "拖延 推迟 回避 不想动 缺乏动力",
        # 压力
        "压力": "压力 压力大 紧张 负荷 倦怠",
        "倦怠": "倦怠 疲惫 耗竭 疲劳 心力交瘁 职业倦怠",
        # 关系
        "父母": "父母 家庭 亲子 关系 家人 代沟",
        # 意外/创伤
        "意外": "意外 事故 创伤 灾难 突发事件 危机",
        "创伤": "创伤 创伤后 应激 突发事件 心理创伤",
        # 通用
        "难受": "难受 痛苦 煎熬 不好受 难熬 折磨",
        "睡不着": "睡不着 失眠 睡眠 难以入睡 休息",
    }

    @staticmethod
    def _expand_query(query: str) -> str:
        """查询扩展：对query中的关键词添加同义/相关词"""
        expanded = query
        for keyword, synonyms in KnowledgeStore._SYNONYM_MAP.items():
            if keyword in expanded:
                expanded += " " + synonyms
        return expanded

    def _build_search_text(self, chunk: KnowledgeChunk) -> str:
        """为 chunks 构建搜索文本：关键字段加权重复"""
        title = chunk.title
        tags = chunk.metadata.get("tags", [])
        category = chunk.metadata.get("category", "")

        # 标题重复5次、标签重复3次来大幅提升权重
        # 确保核心关键词（如"自杀""CBT"）在 BM25 中有足够的词频
        parts = (
            [title] * 5                                 # title × 5（最高权重）
            + [" ".join(tags)] * 3                      # tags × 3
            + [category] * 2                             # category × 2
            + [chunk.content[:200]]                       # content 开头摘要
        )
        return " ".join(parts)

    def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """混合检索：BM25 + Dense RRF 融合"""
        import time
        t0 = time.time()

        if not self.chunks:
            return KnowledgeSearchResult(chunks=[], total_found=0)

        chunk_list = list(self.chunks.values())

        # 过滤
        if query.category_filter:
            chunk_list = [
                c for c in chunk_list
                if c.metadata.get("category") == query.category_filter
            ]
        if query.risk_level_filter:
            chunk_list = [
                c for c in chunk_list
                if query.risk_level_filter in c.metadata.get("related_risk_levels", [])
            ]
        if query.tag_filter:
            chunk_list = [
                c for c in chunk_list
                if any(t in c.metadata.get("tags", []) for t in query.tag_filter)
            ]

        if not chunk_list:
            return KnowledgeSearchResult(chunks=[], total_found=0)

        # 查询扩展
        expanded_query = self._expand_query(query.query)

        # 构建加权搜索文本的伪 ContentItem
        from models import ContentItem
        pseudo_items = []
        for c in chunk_list:
            boosted_text = self._build_search_text(c)
            pseudo_items.append(ContentItem(
                id=c.id,
                title=c.title,
                type="knowledge",
                category=c.metadata.get("category", "general"),
                description=boosted_text[:500],
                tags=c.metadata.get("tags", []),
            ))

        # BM25 检索（对扩展后的查询）
        limit = max(query.top_k * 2, 20)
        bm25_results = self.bm25.retrieve(
            query=expanded_query, items=pseudo_items, limit=limit
        )

        # FAISS 向量检索 — 仅当使用 bge 嵌入时参与 RRF 融合
        # hash-zh-v1 语义能力弱于 BM25 精确匹配，故不加入检索
        faiss_results: List[Tuple[Any, float]] = []
        if self.faiss is not None and self.faiss.uses_bge:
            try:
                self.faiss.rebuild_if_needed(self.chunks)
                if not self.faiss.is_ready:
                    self.faiss.build(self.chunks)
                faiss_raw = self.faiss.search(expanded_query, top_k=limit)
                for chunk_id, score in faiss_raw:
                    chunk = self.chunks.get(chunk_id)
                    if chunk:
                        for pi in pseudo_items:
                            if pi.id == chunk_id:
                                faiss_results.append((pi, score))
                                break
            except Exception as e:
                logger.warning(f"FAISS 检索失败（降级）: {e}")
        else:
            # hash 模式：仅预热持久化索引，为后续 bge 切换做准备
            if self.faiss is not None and not self.faiss.is_ready:
                try:
                    self.faiss.rebuild_if_needed(self.chunks)
                except Exception:
                    pass

        # RRF 融合（BM25 ×3 为主力，FAISS 仅当 bge 嵌入质量可靠时参与）
        result_sets = [bm25_results, bm25_results, bm25_results]
        if faiss_results and self.faiss is not None and self.faiss.uses_bge:
            result_sets.append(faiss_results)

        fused = self._rrf_fuse(result_sets, k=60)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        elapsed = (time.time() - t0) * 1000

        results = []
        seen_docs: set = set()
        for chunk_id, score in ranked:
            chunk = self.chunks.get(chunk_id)
            if chunk and chunk.doc_id not in seen_docs:
                chunk.score = round(score, 4)
                results.append(chunk)
                seen_docs.add(chunk.doc_id)
            if len(results) >= query.top_k:
                break

        return KnowledgeSearchResult(
            chunks=results,
            total_found=len(results),
            query_time_ms=round(elapsed, 2),
        )

    def _rrf_fuse(
        self,
        result_sets: List[List[Tuple[Any, float]]],
        k: int = 60,
    ) -> Dict[str, float]:
        """Reciprocal Rank Fusion — 返回 {chunk_id: score}"""
        scores: Dict[str, float] = {}
        for result_set in result_sets:
            for rank, (item, _) in enumerate(result_set, start=1):
                chunk_id = item.id if hasattr(item, "id") else str(item[0])
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        return scores

    # ── 统计 ──

    def stats(self) -> Dict[str, Any]:
        """知识库统计"""
        categories: Dict[str, int] = {}
        for doc in self.documents.values():
            categories[doc.category] = categories.get(doc.category, 0) + 1
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "is_loaded": self._is_loaded,
        }

    def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        return self.documents.get(doc_id)

    def get_chunks_by_doc(self, doc_id: str) -> List[KnowledgeChunk]:
        return [c for c in self.chunks.values() if c.doc_id == doc_id]


# ── 全局实例 ──

knowledge_store = KnowledgeStore()

"""HybridRetriever retrieve() 编排边界测试"""
import os
import sys

PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from hybrid_retriever import HybridRetriever
from models import ContentItem


def _make_item(id: str = "test_001", title: str = "", tags: list = None,
               category: str = "general", description: str = "测试") -> ContentItem:
    return ContentItem(
        id=id,
        title=title or f"Test {id}",
        type="article",
        category=category,
        description=description,
        tags=tags or [],
    )


def test_retrieve_empty_items():
    """空内容列表返回空"""
    retriever = HybridRetriever()
    results = retriever.retrieve(query="压力", items=[], limit=5)
    assert results == []


def test_retrieve_empty_query():
    """空查询返回空（BM25 和 Vector 都返回空 → 融合结果空）"""
    retriever = HybridRetriever()
    items = [_make_item(id="a", title="压力管理", tags=["压力"])]
    results = retriever.retrieve(query="", items=items, limit=5)
    assert results == []


def test_retrieve_basic():
    """基础检索应返回匹配项"""
    retriever = HybridRetriever()
    items = [
        _make_item(id="a", title="压力管理", tags=["压力"]),
        _make_item(id="b", title="放松练习", tags=["放松"]),
    ]
    results = retriever.retrieve(query="压力", items=items, limit=5)
    assert len(results) >= 1
    assert any(item.id == "a" for item, _ in results)


def test_retrieve_with_custom_rrf_k():
    """自定义 RRF k 值不应影响结果结构"""
    retriever = HybridRetriever(rrf_k=10)
    items = [
        _make_item(id="a", title="焦虑缓解", tags=["焦虑"]),
        _make_item(id="b", title="正念冥想", tags=["正念"]),
    ]
    results = retriever.retrieve(query="焦虑", items=items, limit=5)
    assert len(results) >= 1
    assert results[0][0].id == "a"


def test_retrieve_limit():
    """limit 参数应限制返回数量"""
    retriever = HybridRetriever()
    items = [_make_item(id=f"item_{i}", title="测试内容", tags=["测试"]) for i in range(10)]
    results = retriever.retrieve(query="测试", items=items, limit=3)
    assert len(results) <= 3


def test_retrieve_with_extra_terms():
    """extra_terms 应增强检索结果"""
    retriever = HybridRetriever()
    items = [
        _make_item(id="a", title="时间管理", tags=["效率"]),
        _make_item(id="b", title="焦虑管理", tags=["焦虑", "紧张"]),
    ]
    results_without = retriever.retrieve(query="管理", items=items, limit=5)
    n_before = len(results_without)
    results_with = retriever.retrieve(query="管理", items=items, limit=5, extra_terms=["焦虑"])
    # 加入 extra_terms 后结果应至少相等或更多
    assert len(results_with) >= n_before


def test_retrieve_extra_terms_none():
    """extra_terms=None 不应报错"""
    retriever = HybridRetriever()
    items = [_make_item(id="a", title="测试", tags=["测试"])]
    results = retriever.retrieve(query="测试", items=items, limit=5, extra_terms=None)
    assert len(results) >= 1


def test_retrieve_no_common_terms():
    """查询和内容无共同词项时返回空"""
    retriever = HybridRetriever()
    items = [_make_item(id="a", title="英文内容", tags=["english"])]
    results = retriever.retrieve(query="纯中文查询无匹配", items=items, limit=5)
    assert results == []


def main():
    test_retrieve_empty_items()
    print("PASS: retrieve empty items")
    test_retrieve_empty_query()
    print("PASS: retrieve empty query")
    test_retrieve_basic()
    print("PASS: retrieve basic")
    test_retrieve_with_custom_rrf_k()
    print("PASS: retrieve with custom rrf k")
    test_retrieve_limit()
    print("PASS: retrieve limit")
    test_retrieve_with_extra_terms()
    print("PASS: retrieve with extra terms")
    test_retrieve_extra_terms_none()
    print("PASS: retrieve extra terms none")
    test_retrieve_no_common_terms()
    print("PASS: retrieve no common terms")


if __name__ == "__main__":
    main()

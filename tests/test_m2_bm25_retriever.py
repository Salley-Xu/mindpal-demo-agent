"""BM25Retriever 单元测试"""
import os
import sys

PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from bm25_retriever import BM25Retriever
from models import ContentItem


# 辅助：构建测试 ContentItem
def _make_item(id: str = "test_001", title: str = "", description: str = "",
               category: str = "", tags: list = None, emotion_tags: list = None,
               type: str = "article", difficulty: str = None) -> ContentItem:
    return ContentItem(
        id=id,
        title=title or f"Test {id}",
        type=type,
        category=category or "general",
        description=description or "测试描述",
        tags=tags or [],
        emotion_tags=emotion_tags or [],
        difficulty=difficulty,
    )


def test_retrieve_empty_items():
    """空内容列表返回空"""
    retriever = BM25Retriever()
    results = retriever.retrieve(query="压力", items=[], limit=5)
    assert results == []


def test_retrieve_empty_query():
    """空查询返回空"""
    retriever = BM25Retriever()
    items = [_make_item(id="a", title="压力管理", tags=["压力"])]
    results = retriever.retrieve(query="", items=items, limit=5)
    assert results == []


def test_retrieve_no_match():
    """查询无匹配项返回空"""
    retriever = BM25Retriever()
    items = [_make_item(id="a", title="放松练习", tags=["放松"])]
    results = retriever.retrieve(query="编程技术", items=items, limit=5)
    assert results == []


def test_retrieve_basic_relevance():
    """基础相关性：匹配项应出现在结果中"""
    retriever = BM25Retriever()
    items = [
        _make_item(id="a", title="压力管理", tags=["压力", "情绪"], description="如何管理工作压力"),
        _make_item(id="b", title="放松练习", tags=["放松", "冥想"], description="冥想放松技巧"),
    ]
    results = retriever.retrieve(query="压力", items=items, limit=5)
    assert len(results) == 1
    assert results[0][0].id == "a"


def test_retrieve_with_extra_terms():
    """额外检索词应扩展匹配"""
    retriever = BM25Retriever()
    items = [
        _make_item(id="a", title="焦虑管理", tags=["焦虑", "紧张"]),
        _make_item(id="b", title="时间管理", tags=["效率"]),
    ]
    results = retriever.retrieve(query="管理", items=items, limit=5, extra_terms=["焦虑"])
    assert len(results) == 2
    assert results[0][0].id == "a"  # 焦虑相关排在前面


def test_retrieve_limit():
    """limit 参数应限制返回数量"""
    retriever = BM25Retriever()
    items = [_make_item(id=f"item_{i}", title="压力管理", tags=["压力"]) for i in range(10)]
    results = retriever.retrieve(query="压力", items=items, limit=3)
    assert len(results) == 3


def test_tokenize_chinese():
    """中文文本应正确分词"""
    retriever = BM25Retriever()
    tokens = retriever._tokenize("压力管理技巧")
    assert "压力" in tokens
    assert "管理" in tokens
    assert "技巧" in tokens


def test_tokenize_english():
    """英文词应保留原样"""
    retriever = BM25Retriever()
    tokens = retriever._tokenize("stress management tips")
    assert "stress" in tokens
    assert "management" in tokens
    assert "tips" in tokens


def test_tokenize_mixed():
    """中英文混合应正确切分"""
    retriever = BM25Retriever()
    tokens = retriever._tokenize("焦虑 management")
    assert "焦虑" in tokens
    assert "management" in tokens


def test_tokenize_short_chinese():
    """单字中文不应提取为 token"""
    retriever = BM25Retriever()
    tokens = retriever._tokenize("是的")
    # "是的" 是 2 字，会被保留
    assert "是的" in tokens


def test_idf_zero_doc_freq():
    """零文档频率的 IDF 应较大"""
    retriever = BM25Retriever()
    idf = retriever._idf("罕见词", total_docs=10, doc_freqs={"常见词": 5})
    assert idf > 0


def test_idf_high_doc_freq():
    """出现在所有文档中的词 IDF 接近 0"""
    retriever = BM25Retriever()
    idf = retriever._idf("常见词", total_docs=10, doc_freqs={"常见词": 10})
    assert idf >= 0
    assert idf < 1.0


def test_document_frequencies():
    """文档频率统计应正确"""
    retriever = BM25Retriever()
    doc_tokens = [
        ["压力", "管理", "技巧"],
        ["压力", "放松"],
        ["放松", "冥想"],
    ]
    freqs = retriever._document_frequencies(doc_tokens)
    assert freqs["压力"] == 2
    assert freqs["放松"] == 2
    assert freqs["管理"] == 1
    assert freqs["冥想"] == 1


def test_build_document_concatenates_all_fields():
    """_build_document 应组合所有文本字段"""
    retriever = BM25Retriever()
    item = _make_item(
        title="标题", description="描述", category="分类",
        tags=["标签1", "标签2"], emotion_tags=["情绪1"], type="article",
        difficulty="beginner",
    )
    doc = retriever._build_document(item)
    assert "标题" in doc
    assert "描述" in doc
    assert "分类" in doc
    assert "标签1" in doc
    assert "情绪1" in doc
    assert "beginner" in doc


def main():
    test_retrieve_empty_items()
    print("PASS: retrieve empty items")
    test_retrieve_empty_query()
    print("PASS: retrieve empty query")
    test_retrieve_no_match()
    print("PASS: retrieve no match")
    test_retrieve_basic_relevance()
    print("PASS: retrieve basic relevance")
    test_retrieve_with_extra_terms()
    print("PASS: retrieve with extra terms")
    test_retrieve_limit()
    print("PASS: retrieve limit")
    test_tokenize_chinese()
    print("PASS: tokenize chinese")
    test_tokenize_english()
    print("PASS: tokenize english")
    test_tokenize_mixed()
    print("PASS: tokenize mixed")
    test_tokenize_short_chinese()
    print("PASS: tokenize short chinese")
    test_idf_zero_doc_freq()
    print("PASS: idf zero doc freq")
    test_idf_high_doc_freq()
    print("PASS: idf high doc freq")
    test_document_frequencies()
    print("PASS: document frequencies")
    test_build_document_concatenates_all_fields()
    print("PASS: build document")


if __name__ == "__main__":
    main()

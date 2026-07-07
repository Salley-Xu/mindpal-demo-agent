"""VectorRetriever 单元测试"""
import os
import sys

PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from vector_retriever import VectorRetriever
from models import ContentItem


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


# ============================================================
# retrieve 基础路径
# ============================================================

def test_retrieve_empty_items():
    """空内容列表返回空"""
    retriever = VectorRetriever()
    results = retriever.retrieve(query="压力", items=[], limit=5)
    assert results == []


def test_retrieve_empty_query():
    """空查询返回空"""
    retriever = VectorRetriever()
    items = [_make_item(id="a", title="压力管理", tags=["压力"])]
    results = retriever.retrieve(query="", items=items, limit=5)
    assert results == []


def test_retrieve_no_match():
    """查询无匹配项返回空"""
    retriever = VectorRetriever()
    items = [_make_item(id="a", title="编程技术", tags=["代码"])]
    results = retriever.retrieve(query="焦虑", items=items, limit=5)
    assert results == []


def test_retrieve_with_extra_terms():
    """额外检索词参与匹配"""
    retriever = VectorRetriever()
    items = [
        _make_item(id="a", title="压力管理", tags=["压力"]),
        _make_item(id="b", title="放松练习", tags=["放松"]),
    ]
    results = retriever.retrieve(query="管理", items=items, limit=5, extra_terms=["压力"])
    assert len(results) >= 1
    assert results[0][0].id == "a"


# ============================================================
# _expand_tokens 同义词扩展
# ============================================================

def test_expand_tokens_root_match():
    """根词匹配应扩展所有同义词"""
    retriever = VectorRetriever()
    expanded = retriever._expand_tokens(["焦虑"])
    assert "焦虑" in expanded
    assert "紧张" in expanded
    assert "担心" in expanded


def test_expand_tokens_synonym_match():
    """同义词值匹配应触发扩展"""
    retriever = VectorRetriever()
    expanded = retriever._expand_tokens(["失眠"])
    assert "失眠" in expanded
    assert "睡眠" in expanded  # 所属根词
    assert "睡不着" in expanded


def test_expand_tokens_no_match():
    """无匹配词不应扩展"""
    retriever = VectorRetriever()
    expanded = retriever._expand_tokens(["编程"])
    assert expanded == ["编程"]


def test_expand_tokens_multiple_matches():
    """多词命中应各自扩展"""
    retriever = VectorRetriever()
    expanded = retriever._expand_tokens(["焦虑", "压力"])
    assert "焦虑" in expanded
    assert "紧张" in expanded
    assert "压力" in expanded
    assert "stress" in expanded


def test_expand_tokens_emotion_coverage():
    """v4.3 新增同义词应可用"""
    retriever = VectorRetriever()

    for tag in ["抑郁", "愤怒", "孤独", "困惑", "自我怀疑"]:
        expanded = retriever._expand_tokens([tag])
        # 扩展后应包含原始词 + 至少一个同义词
        assert tag in expanded, f"{tag} 应出现在扩展结果中"
        assert len(expanded) >= 3, f"{tag} 应至少扩展出 2+ 个同义词"


# ============================================================
# _cosine_similarity
# ============================================================

def test_cosine_similarity_identical():
    """相同向量的余弦相似度为 1"""
    retriever = VectorRetriever()
    from collections import Counter
    vec = Counter({"压力": 2, "焦虑": 1})
    sim = retriever._cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    """正交向量的余弦相似度为 0"""
    retriever = VectorRetriever()
    from collections import Counter
    sim = retriever._cosine_similarity(Counter({"压力": 1}), Counter({"放松": 1}))
    assert sim == 0.0


def test_cosine_similarity_partial():
    """部分重叠向量应有中间值"""
    retriever = VectorRetriever()
    from collections import Counter
    sim = retriever._cosine_similarity(
        Counter({"压力": 1, "焦虑": 1}),
        Counter({"压力": 1, "放松": 1}),
    )
    assert 0 < sim < 1.0


def test_cosine_similarity_empty():
    """空向量余弦为 0"""
    retriever = VectorRetriever()
    from collections import Counter
    sim = retriever._cosine_similarity(Counter({"压力": 1}), Counter())
    assert sim == 0.0


# ============================================================
# _tokenize
# ============================================================

def test_tokenize_chinese():
    """中文 2-4 字词应被提取"""
    retriever = VectorRetriever()
    tokens = retriever._tokenize("压力管理")
    assert "压力" in tokens
    assert "管理" in tokens


def test_tokenize_english():
    """英文词应保留"""
    retriever = VectorRetriever()
    tokens = retriever._tokenize("stress anxiety")
    assert "stress" in tokens
    assert "anxiety" in tokens


def test_tokenize_empty():
    """空字符串返回空列表"""
    retriever = VectorRetriever()
    assert retriever._tokenize("") == []


def main():
    test_retrieve_empty_items()
    print("PASS: retrieve empty items")
    test_retrieve_empty_query()
    print("PASS: retrieve empty query")
    test_retrieve_no_match()
    print("PASS: retrieve no match")
    test_retrieve_with_extra_terms()
    print("PASS: retrieve with extra terms")
    test_expand_tokens_root_match()
    print("PASS: expand tokens root match")
    test_expand_tokens_synonym_match()
    print("PASS: expand tokens synonym match")
    test_expand_tokens_no_match()
    print("PASS: expand tokens no match")
    test_expand_tokens_multiple_matches()
    print("PASS: expand tokens multiple matches")
    test_expand_tokens_emotion_coverage()
    print("PASS: expand tokens emotion coverage")
    test_cosine_similarity_identical()
    print("PASS: cosine similarity identical")
    test_cosine_similarity_orthogonal()
    print("PASS: cosine similarity orthogonal")
    test_cosine_similarity_partial()
    print("PASS: cosine similarity partial")
    test_cosine_similarity_empty()
    print("PASS: cosine similarity empty")
    test_tokenize_chinese()
    print("PASS: tokenize chinese")
    test_tokenize_english()
    print("PASS: tokenize english")
    test_tokenize_empty()
    print("PASS: tokenize empty")


if __name__ == "__main__":
    main()

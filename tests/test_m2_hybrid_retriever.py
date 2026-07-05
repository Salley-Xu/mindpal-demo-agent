"""HybridRetriever 单元测试：RRF 融合算法"""
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


def _make_item(id: str, title: str = "") -> ContentItem:
    return ContentItem(
        id=id,
        title=title or id,
        type="article",
        category="stress",
        description="",
    )


def test_rrf_fusion_ranks_common_items_higher():
    """两个结果集的共有条目应排在前面"""
    retriever = HybridRetriever(rrf_k=60)
    a, b, c, d = _make_item("a"), _make_item("b"), _make_item("c"), _make_item("d")

    set1 = [(a, 0.9), (b, 0.8), (c, 0.7)]  # a 出现在两个集合
    set2 = [(a, 0.6), (d, 0.5)]

    fused = retriever._reciprocal_rank_fusion([set1, set2])
    sorted_items = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
    ids = [item.id for item, _ in sorted_items]

    # a 同时出现在两个集合，RRF 分数最高
    assert ids[0] == "a", f"Expected 'a' first, got {ids}"


def test_rrf_fusion_with_empty_set():
    """一个结果集为空时融合应正常工作"""
    retriever = HybridRetriever(rrf_k=60)
    a, b = _make_item("a"), _make_item("b")

    set1 = [(a, 0.9), (b, 0.8)]
    set2 = []

    fused = retriever._reciprocal_rank_fusion([set1, set2])
    assert len(fused) == 2
    assert "a" in fused
    assert "b" in fused


def test_rrf_fusion_single_source():
    """只有一个来源时，RRF 返回原顺序"""
    retriever = HybridRetriever(rrf_k=60)
    a, b, c = _make_item("a"), _make_item("b"), _make_item("c")

    set1 = [(a, 0.9), (b, 0.8), (c, 0.7)]
    fused = retriever._reciprocal_rank_fusion([set1])
    sorted_items = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
    ids = [item.id for item, _ in sorted_items]

    assert ids == ["a", "b", "c"]


def test_rrf_fusion_disjoint_sets():
    """两个没有交集的集合：fused 应包含所有条目"""
    retriever = HybridRetriever(rrf_k=60)
    a, b, c, d = _make_item("a"), _make_item("b"), _make_item("c"), _make_item("d")

    set1 = [(a, 0.9), (b, 0.8)]
    set2 = [(c, 0.7), (d, 0.6)]

    fused = retriever._reciprocal_rank_fusion([set1, set2])
    assert len(fused) == 4


def test_rrf_fusion_with_different_k():
    """不同 rrf_k 参数影响分数分布"""
    retriever_k1 = HybridRetriever(rrf_k=1)
    retriever_k60 = HybridRetriever(rrf_k=60)
    a, b = _make_item("a"), _make_item("b")

    set1 = [(a, 0.9), (b, 0.8)]
    set2 = [(a, 0.7), (b, 0.6)]

    fused_k1 = retriever_k1._reciprocal_rank_fusion([set1, set2])
    fused_k60 = retriever_k60._reciprocal_rank_fusion([set1, set2])

    # 不同 rrf_k 产生不同分数
    score_k1 = fused_k1["a"][1]
    score_k60 = fused_k60["a"][1]
    assert score_k1 != score_k60, "rrf_k=1 和 rrf_k=60 应产生不同分数"


def test_rrf_fusion_partial_overlap():
    """部分重叠时，重叠条目以更高 RRF 分数排在前面"""
    retriever = HybridRetriever(rrf_k=60)
    a, b, c, d, e = (
        _make_item("a"), _make_item("b"), _make_item("c"),
        _make_item("d"), _make_item("e"),
    )

    set1 = [(a, 0.9), (b, 0.8), (c, 0.7)]   # a, b, c
    set2 = [(b, 0.85), (d, 0.6), (e, 0.5)]   # b, d, e

    fused = retriever._reciprocal_rank_fusion([set1, set2])
    sorted_items = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
    ids = [item.id for item, _ in sorted_items]

    # b 同时出现在两个集合，RRF 分数应最高
    assert ids[0] == "b", f"Expected 'b' first (appears in both), got {ids}"


def test_rrf_fusion_preserves_item_reference():
    """融合结果应保留 ContentItem 引用"""
    retriever = HybridRetriever(rrf_k=60)
    a = _make_item("test_fusion_ref", title="保留引用")

    set1 = [(a, 0.9)]
    fused = retriever._reciprocal_rank_fusion([set1, []])

    assert fused["test_fusion_ref"][0].title == "保留引用"
    assert fused["test_fusion_ref"][0] is a


def test_rrf_fusion_score_formula():
    """验证 RRF 分数计算公式：score += 1 / (k + rank)"""
    retriever = HybridRetriever(rrf_k=10)
    a = _make_item("a")
    b = _make_item("b")

    # a 在 set1 中 rank=1, 在 set2 中 rank=1（都是第1位）
    # a 的 RRF 分数 = 1/(10+1) + 1/(10+1) = 2/11 ≈ 0.1818
    # b 只在 set1 中 rank=2，RRF 分数 = 1/(10+2) = 1/12 ≈ 0.0833
    set1 = [(a, 0.9), (b, 0.8)]
    set2 = [(a, 0.7)]

    fused = retriever._reciprocal_rank_fusion([set1, set2])
    score_a = round(fused["a"][1], 4)
    score_b = round(fused["b"][1], 4)

    expected_a = round(1/11 + 1/11, 4)
    expected_b = round(1/12, 4)

    assert score_a == expected_a, f"Score for 'a': expected {expected_a}, got {score_a}"
    assert score_b == expected_b, f"Score for 'b': expected {expected_b}, got {score_b}"


def main():
    test_rrf_fusion_ranks_common_items_higher()
    print("PASS: RRF ranks common items higher")

    test_rrf_fusion_with_empty_set()
    print("PASS: RRF with empty set")

    test_rrf_fusion_single_source()
    print("PASS: RRF single source")

    test_rrf_fusion_disjoint_sets()
    print("PASS: RRF disjoint sets")

    test_rrf_fusion_with_different_k()
    print("PASS: RRF with different k")

    test_rrf_fusion_partial_overlap()
    print("PASS: RRF partial overlap")

    test_rrf_fusion_preserves_item_reference()
    print("PASS: RRF preserves item reference")

    test_rrf_fusion_score_formula()
    print("PASS: RRF score formula")


if __name__ == "__main__":
    main()

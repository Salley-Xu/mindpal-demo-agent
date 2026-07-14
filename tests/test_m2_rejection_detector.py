"""RejectionDetector 单元测试"""
import os
import sys

PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from rejection_detector import detect_rejection


def test_detect_clear_rejection():
    """明确拒绝应被检测"""
    result = detect_rejection("不用了，谢谢")
    assert result["has_rejected"] is True
    assert result["confidence"] >= 0.9


def test_detect_buyao():
    """"不要推荐" 应被检测"""
    result = detect_rejection("不要推荐了")
    assert result["has_rejected"] is True


def test_detect_buxuyao():
    """"不需要" 应被检测"""
    result = detect_rejection("我不需要这些")
    assert result["has_rejected"] is True


def test_detect_no_interest():
    """"没兴趣" 应被检测"""
    result = detect_rejection("我对这些没兴趣")
    assert result["has_rejected"] is True


def test_detect_useless():
    """对推荐内容的负面反馈应被检测"""
    result = detect_rejection("上次推荐的那个没用")
    assert result["has_rejected"] is True


def test_detect_not_suitable():
    """"不适合" 应被检测"""
    result = detect_rejection("这个不适合我")
    assert result["has_rejected"] is True


def test_positive_or_neutral_not_rejected():
    """正面或中性内容不应被标记为拒绝"""
    neutral_inputs = [
        "谢谢你的建议，我觉得很有帮助",
        "好的我知道了",
        "今天心情不错",
        "这篇文章很有意思",
    ]
    for text in neutral_inputs:
        result = detect_rejection(text)
        assert result["has_rejected"] is False, f"'{text}' 不应被标记为拒绝"


def test_empty_input():
    """空输入返回未拒绝"""
    result = detect_rejection("")
    assert result["has_rejected"] is False
    result = detect_rejection("   ")
    assert result["has_rejected"] is False


def test_rejected_ids_with_active_items():
    """active_item_ids 应传递到返回结果"""
    result = detect_rejection("不用了", active_item_ids=["article_001", "audio_001"])
    assert result["has_rejected"] is True
    assert "article_001" in result["rejected_ids"]


def test_rejected_ids_empty():
    """无 active_item_ids 时 rejected_ids 为空"""
    result = detect_rejection("不用了")
    assert result["rejected_ids"] == []


def test_suanle():
    """"算了吧" 应被检测"""
    result = detect_rejection("算了吧，下次再说")
    assert result["has_rejected"] is True


def test_xiacizzaishuo():
    """"下次再说" 应被低置信度检测"""
    result = detect_rejection("下次再说吧")
    assert result["has_rejected"] is True
    assert result["confidence"] <= 0.7


def main():
    test_detect_clear_rejection()
    print("PASS: detect clear rejection")
    test_detect_buyao()
    print("PASS: detect buyao")
    test_detect_buxuyao()
    print("PASS: detect bu xu yao")
    test_detect_no_interest()
    print("PASS: detect no interest")
    test_detect_useless()
    print("PASS: detect useless")
    test_detect_not_suitable()
    print("PASS: detect not suitable")
    test_positive_or_neutral_not_rejected()
    print("PASS: positive/neutral not rejected")
    test_empty_input()
    print("PASS: empty input")
    test_rejected_ids_with_active_items()
    print("PASS: rejected ids with active items")
    test_rejected_ids_empty()
    print("PASS: rejected ids empty")
    test_suanle()
    print("PASS: suanle")
    test_xiacizzaishuo()
    print("PASS: xiaci zaishuo")


if __name__ == "__main__":
    main()

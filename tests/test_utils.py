from finch.autoschedule._utils import intersect, is_subsequence, with_subsequence


def test_intersect(tp_0, tp_1, tp_2, tp_3):
    assert intersect(tp_1, tp_2) == tp_0
    assert intersect(tp_3, tp_1) == tp_3


def test_with_subsequence(tp_0, tp_1, tp_2, tp_3):
    assert with_subsequence(tp_2, tp_1) == tp_3
    assert with_subsequence(tp_0, tp_1) == tp_1
    assert with_subsequence(tp_3, tp_1) == tp_3


def test_is_subsequence(tp_0, tp_1, tp_2, tp_3):
    assert not is_subsequence(tp_2, tp_1)
    assert is_subsequence(tp_0, tp_1)
    assert not is_subsequence(tp_3, tp_1)

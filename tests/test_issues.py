import numpy as np

import finch


def test_issue_64():
    a = finch.defer(np.arange(1 * 2).reshape(1, 2, 1))
    b = finch.defer(np.arange(4 * 2 * 3).reshape(4, 2, 3))

    c = finch.multiply(a, b)
    result = finch.compute(c).shape
    expected = (4, 2, 3)
    assert result == expected, f"Expected shape {expected}, got {result}"

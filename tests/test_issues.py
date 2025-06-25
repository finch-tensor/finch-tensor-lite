import numpy as np

import finch


def test_issue_64():
    a = finch.defer(np.arange(1 * 2).reshape(1, 2, 1))
    b = finch.defer(np.arange(4 * 2 * 3).reshape(4, 2, 3))

    c = finch.multiply(a, b)
    result = finch.compute(c).shape
    expected = (4, 2, 3)
    assert result == expected, f"Expected shape {expected}, got {result}"


def test_issue_50():
    x = finch.defer(np.array([[2, 4, 6, 8], [1, 3, 5, 7]]))
    m = finch.defer(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
    n = finch.defer(
        np.array([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    )  # Int -> Float caused return_type error
    # If replaced above with below line, no error
    # n = finch.defer(np.array([[2, 2, 2, 2], [2, 2, 2, 2]]))
    o = finch.defer(np.array([[3, 3, 3, 3], [3, 3, 3, 3]]))
    finch.add(finch.add(finch.subtract(x, m), n), o)

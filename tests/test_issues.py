import pytest

import numpy as np

import finchlite


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_issue_64():
    a = finchlite.defer(np.arange(1 * 2).reshape(1, 2, 1))
    b = finchlite.defer(np.arange(4 * 2 * 3).reshape(4, 2, 3))

    c = finchlite.multiply(a, b)
    result = finchlite.compute(c).shape
    expected = (4, 2, 3)
    assert result == expected, f"Expected shape {expected}, got {result}"


def test_issue_50():
    x = finchlite.defer(np.array([[2, 4, 6, 8], [1, 3, 5, 7]]))
    m = finchlite.defer(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))
    n = finchlite.defer(
        np.array([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    )  # Int -> Float caused return_type error
    # If replaced above with below line, no error
    # n = finchlite.defer(np.array([[2, 2, 2, 2], [2, 2, 2, 2]]))
    o = finchlite.defer(np.array([[3, 3, 3, 3], [3, 3, 3, 3]]))
    finchlite.add(finchlite.add(finchlite.subtract(x, m), n), o)

import numpy as np
from numpy.testing import assert_equal
import pytest
import finch
from operator import add, mul

@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        (np.array([[2, 0], [1, 3]]), np.array([[4, 1], [2, 2]])),
    ],
)
def test_matrix_multiplication(a, b):
    result = finch.fuse(lambda a, b: finch.reduce(add, finch.elementwise(mul, finch.expand_dims(a, 2), b), axis=1), a, b)

    expected = np.matmul(a, b)
    
    assert_equal(result, expected)
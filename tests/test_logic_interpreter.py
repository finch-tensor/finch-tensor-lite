import numpy as np
from numpy.testing import assert_equal
import pytest
from finch.finch_logic import *
from operator import add, mul

@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        (np.array([[2, 0], [1, 3]]), np.array([[4, 1], [2, 2]])),
    ],
)
def test_matrix_multiplication(a, b):
    i = Field("i")
    j = Field("j")
    k = Field("k")

    p = Plan([
        Query(Alias("A"), Table(Immediate(a), (i, k))),
        Query(Alias("B"), Table(Immediate(b), (k, j))),
        Query(Alias("AB"), MapJoin(Immediate(mul), (Alias("A"), Alias("B")))),
        Query(Alias("C"), Reorder(Aggregate(Immediate(add), Immediate(0), Alias("AB"), (k,)), (i, j))),
        Produces((Alias("C"),)),
    ])

    result = FinchLogicInterpreter()(p)[0]

    expected = np.matmul(a, b)
    
    assert_equal(result, expected)
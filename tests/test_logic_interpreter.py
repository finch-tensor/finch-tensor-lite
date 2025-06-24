from operator import add, mul

import pytest

import numpy as np
from numpy.testing import assert_equal

from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    FinchLogicInterpreter,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)


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

    p = Plan(
        [
            Query(Alias("A"), Table(Literal(a), (i, k))),
            Query(Alias("B"), Table(Literal(b), (k, j))),
            Query(Alias("AB"), MapJoin(Literal(mul), (Alias("A"), Alias("B")))),
            Query(
                Alias("C"),
                Reorder(Aggregate(Literal(add), Literal(0), Alias("AB"), (k,)), (i, j)),
            ),
            Produces((Alias("C"),)),
        ]
    )

    result = FinchLogicInterpreter()(p)[0]

    expected = np.matmul(a, b)

    assert_equal(result, expected)

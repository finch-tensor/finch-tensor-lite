import _operator  # noqa: F401
from operator import add, mul

import pytest

import numpy as np
from numpy import array  # noqa: F401
from numpy.testing import assert_equal

from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicInterpreter,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
    TableValue,
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
        (
            Query(Alias("A"), Table(Literal(a), (i, k))),
            Query(Alias("B"), Table(Literal(b), (k, j))),
            Query(Alias("AB"), MapJoin(Literal(mul), (Alias("A"), Alias("B")))),
            Query(
                Alias("C"),
                Reorder(Aggregate(Literal(add), Literal(0), Alias("AB"), (k,)), (i, j)),
            ),
            Produces((Alias("C"),)),
        )
    )

    result = LogicInterpreter()(p)[0]

    expected = TableValue(np.matmul(a, b), (i, j))

    assert result == expected


def test_plan_repr():
    i = Field("i")
    j = Field("j")
    k = Field("k")
    # To avoid equality issues with numpy arrays, we use string literals here instead
    p = Plan(
        (
            Query(Alias("A"), Table(Literal("A"), (i, k))),
            Query(Alias("B"), Table(Literal("B"), (k, j))),
            Query(Alias("AB"), MapJoin(Literal(mul), (Alias("A"), Alias("B")))),
            Query(
                Alias("C"),
                Reorder(Aggregate(Literal(add), Literal(0), Alias("AB"), (k,)), (i, j)),
            ),
            Produces((Alias("C"),)),
        )
    )
    assert p == eval(repr(p))


def test_materialize():
    i = Field("i")
    j = Field("j")

    C = np.array([[0, 0], [0, 0]])

    p = Plan(
        (
            Query(Alias("A"), Table(Literal(np.array([[1, 2], [3, 4]])), (i, j))),
            Query(Alias("B"), Table(Literal(np.array([[1, 1], [1, 1]])), (i, j))),
            Query(Alias("C"), MapJoin(Literal(add), (Alias("A"), Alias("B")))),
            Query(Alias("D"), MapJoin(Literal(mul), (Alias("C"), Alias("A")))),
            Query(Alias("C"), Alias("B")),
            Produces((Alias("D"), Alias("C"))),
        )
    )

    result = LogicInterpreter()(p, bindings={Alias("C"): TableValue(C, (i, j))})[0]

    expected = TableValue(
        np.array([[((1 + 1) * 1), ((2 + 1) * 2)], [((3 + 1) * 3), ((4 + 1) * 4)]]),
        (i, j),
    )

    assert result == expected
    assert_equal(C, np.array([[1, 1], [1, 1]]))

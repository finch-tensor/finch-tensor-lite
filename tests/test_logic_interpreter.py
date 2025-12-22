import _operator  # noqa: F401
from operator import add, mul

import pytest

import numpy as np
from numpy import array  # noqa: F401

import finchlite as fl
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

from .conftest import finch_assert_equal


@pytest.mark.parametrize(
    "a, b",
    [
        (fl.asarray(np.array([[1, 2], [3, 4]])), fl.asarray(np.array([[5, 6], [7, 8]]))),
        (fl.asarray(np.array([[2, 0], [1, 3]])), fl.asarray(np.array([[4, 1], [2, 2]]))),
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
            Query(Alias("AB"), MapJoin(Literal(mul), (Table(Alias("A"), (i, k)), Table(Alias("B"), (k, j))))),
            Query(
                Alias("C"),
                Reorder(Aggregate(Literal(add), Literal(0), Table(Alias("AB"), (i, k, j)), (k,)), (i, j)),
            ),
            Produces((Alias("C"),)),
        )
    )

    result = LogicInterpreter()(p)[0]

    expected = np.matmul(a.to_numpy(), b.to_numpy())

    assert (result.to_numpy() == expected).all()


def test_plan_repr():
    i = Field("i")
    j = Field("j")
    k = Field("k")
    # To avoid equality issues with numpy arrays, we use string literals here instead
    p = Plan(
        (
            Query(Alias("A"), Table(Literal("A"), (i, k))),
            Query(Alias("B"), Table(Literal("B"), (k, j))),
            Query(Alias("AB"), MapJoin(Literal(mul), (Table(Alias("A"), (i, k)), Table(Alias("B"), (k, j))))),
            Query(
                Alias("C"),
                Reorder(Aggregate(Literal(add), Literal(0), Table(Alias("AB"), (i, k, j)), (k,)), (i, j)),
            ),
            Produces((Alias("C"),)),
        )
    )
    assert p == eval(repr(p))


def test_materialize():
    i = Field("i")
    j = Field("j")

    C = fl.asarray(np.array([[0, 0], [0, 0]]))

    p = Plan(
        (
            Query(Alias("A"), Table(Literal(fl.asarray(np.array([[1, 2], [3, 4]]))), (i, j))),
            Query(Alias("B"), Table(Literal(fl.asarray(np.array([[1, 1], [1, 1]]))), (i, j))),
            Query(Alias("C"), MapJoin(Literal(add), (Table(Alias("A"), (i, j)), Table(Alias("B"), (i, j))))),
            Query(Alias("D"), MapJoin(Literal(mul), (Table(Alias("C"), (i, j)), Table(Alias("A"), (i, j))))),
            Query(Alias("C"), Table(Alias("B"), (i, j))),
            Produces((Alias("D"), Alias("C"))),
        )
    )

    result = LogicInterpreter()(p, bindings= {Alias("C"): C})[0]

    expected = fl.asarray(np.array([[((1 + 1) * 1), ((2 + 1) * 2)], [((3 + 1) * 3), ((4 + 1) * 4)]]))

    

    assert (result.to_numpy() == expected.to_numpy()).all()
    finch_assert_equal(C, fl.asarray(np.array([[1, 1], [1, 1]])))

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
    FinchLogicInterpreter,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)
from finchlite.finch_logic.utility import intree, isdescendant


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

    result = FinchLogicInterpreter()(p)[0]

    expected = np.matmul(a, b)

    assert_equal(result, expected)

    assert p == eval(repr(p))


def test_intree_and_isdescendant():
    i, j, k = Field("i"), Field("j"), Field("k")
    ta = Table(Literal("A"), (i, j))
    tb = Table(Literal("B"), (j, k))
    op = Field("op")
    mj = MapJoin(op, (ta, tb))
    prog = Plan((Produces((mj,)),))

    assert intree(prog, prog)
    assert intree(mj, prog)
    assert intree(ta, prog)
    assert intree(tb, prog)
    assert isdescendant(mj, prog)
    assert isdescendant(ta, prog)
    assert isdescendant(tb, prog)

import _operator  # noqa: F401
import operator

import pytest

import numpy  # noqa: F401, ICN001
import numpy as np
from numpy.testing import assert_equal

import finch  # noqa: F401
import finch.finch_notation as ntn
from finch.finch_notation import (  # noqa: F401
    Access,
    Assign,
    Block,
    Call,
    Declare,
    Freeze,
    Function,
    Increment,
    Literal,
    Loop,
    Module,
    Read,
    Return,
    Unwrap,
    Update,
    Variable,
)


@pytest.mark.parametrize(
    "a, b",
    [
        (
            np.array([[1, 2], [3, 4]], dtype=np.float64),
            np.array([[5, 6], [7, 8]], dtype=np.float64),
        ),
        (
            np.array([[2, 0], [1, 3]], dtype=np.float64),
            np.array([[4, 1], [2, 2]], dtype=np.float64),
        ),
    ],
)
def test_matrix_multiplication(a, b):
    i = ntn.Variable("i", np.int64)
    j = ntn.Variable("j", np.int64)
    k = ntn.Variable("k", np.int64)

    A = ntn.Variable("A", np.ndarray)
    B = ntn.Variable("B", np.ndarray)
    C = ntn.Variable("C", np.ndarray)

    a_ik = ntn.Variable("a_ik", np.float64)
    b_kj = ntn.Variable("b_kj", np.float64)
    c_ij = ntn.Variable("c_ij", np.float64)

    m = ntn.Variable("m", np.int64)
    n = ntn.Variable("n", np.int64)
    p = ntn.Variable("p", np.int64)

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matmul", np.ndarray),
                (C, A, B),
                ntn.Block(
                    (
                        ntn.Assign(
                            m, ntn.Call(ntn.Literal(ntn.dimension), (A, ntn.Literal(0)))
                        ),
                        ntn.Assign(
                            n, ntn.Call(ntn.Literal(ntn.dimension), (B, ntn.Literal(1)))
                        ),
                        ntn.Assign(
                            p, ntn.Call(ntn.Literal(ntn.dimension), (A, ntn.Literal(1)))
                        ),
                        ntn.Assign(
                            C,
                            ntn.Declare(
                                C, ntn.Literal(0.0), ntn.Literal(operator.add), (m, n)
                            ),
                        ),
                        ntn.Loop(
                            i,
                            m,
                            ntn.Loop(
                                j,
                                n,
                                ntn.Loop(
                                    k,
                                    p,
                                    ntn.Block(
                                        (
                                            ntn.Assign(
                                                a_ik,
                                                ntn.Unwrap(
                                                    ntn.Access(A, ntn.Read(), (i, k))
                                                ),
                                            ),
                                            ntn.Assign(
                                                b_kj,
                                                ntn.Unwrap(
                                                    ntn.Access(B, ntn.Read(), (k, j))
                                                ),
                                            ),
                                            ntn.Assign(
                                                c_ij,
                                                ntn.Call(
                                                    ntn.Literal(operator.mul),
                                                    (a_ik, b_kj),
                                                ),
                                            ),
                                            ntn.Increment(
                                                ntn.Access(
                                                    C,
                                                    ntn.Update(
                                                        ntn.Literal(operator.add)
                                                    ),
                                                    (i, j),
                                                ),
                                                c_ij,
                                            ),
                                        )
                                    ),
                                ),
                            ),
                        ),
                        ntn.Assign(
                            C,
                            ntn.Freeze(C, ntn.Literal(operator.add)),
                        ),
                        ntn.Return(C),
                    )
                ),
            ),
        )
    )

    mod = ntn.NotationInterpreter()(prgm)

    c = np.zeros(dtype=np.float64, shape=())
    result = mod.matmul(c, a, b)

    expected = np.matmul(a, b)

    assert_equal(result, expected)

    assert prgm == eval(repr(prgm))

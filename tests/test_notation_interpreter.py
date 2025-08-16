import _operator  # noqa: F401
import operator

import pytest

import numpy  # noqa: F401, ICN001
import numpy as np
from numpy.testing import assert_equal

import finch  # noqa: F401
import finch.finch_notation as ntn
from finch.compile import dimension
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
    Repack,
    Return,
    Slot,
    Unpack,
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
    A_ = ntn.Slot("A_", np.ndarray)
    B_ = ntn.Slot("B_", np.ndarray)
    C_ = ntn.Slot("C_", np.ndarray)

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
                            m, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))
                        ),
                        ntn.Assign(
                            n, ntn.Call(ntn.Literal(dimension), (B, ntn.Literal(1)))
                        ),
                        ntn.Assign(
                            p, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1)))
                        ),
                        ntn.Unpack(A_, A),
                        ntn.Unpack(B_, B),
                        ntn.Unpack(C_, C),
                        ntn.Declare(
                            C_, ntn.Literal(0.0), ntn.Literal(operator.add), (m, n)
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
                                                    ntn.Access(A_, ntn.Read(), (i, k))
                                                ),
                                            ),
                                            ntn.Assign(
                                                b_kj,
                                                ntn.Unwrap(
                                                    ntn.Access(B_, ntn.Read(), (k, j))
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
                                                    C_,
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
                        ntn.Freeze(C_, ntn.Literal(operator.add)),
                        ntn.Repack(C_, C),
                        ntn.Return(C),
                    )
                ),
            ),
        )
    )

    mod = ntn.NotationInterpreter()(prgm)

    c = np.zeros(dtype=np.float64, shape=(a.shape[0], b.shape[1]))
    result = mod.matmul(c, a, b)

    expected = np.matmul(a, b)

    assert_equal(result, expected)

    assert prgm == eval(repr(prgm))


@pytest.mark.parametrize(
    "a",
    [
        np.array([0, 1, 0, 0, 1]),
        np.array([1, 1, 1, 1, 1]),
        np.array([0, 1, 0, 0, 0]),
    ],
)
def test_count_nonfill_vector(a):
    A = ntn.Variable("A", np.ndarray)
    A_ = ntn.Slot("A_", np.ndarray)

    d = ntn.Variable("d", np.int64)
    i = ntn.Variable("i", np.int64)
    m = ntn.Variable("m", np.int64)

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("count_nonfill_vector", np.int64),
                (A,),
                ntn.Block(
                    (
                        ntn.Assign(
                            m, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))
                        ),
                        ntn.Assign(d, ntn.Literal(np.int64(0))),
                        ntn.Unpack(A_, A),
                        ntn.Loop(
                            i,
                            m,
                            ntn.Assign(
                                d,
                                ntn.Call(
                                    Literal(operator.add),
                                    (
                                        d,
                                        ntn.Unwrap(ntn.Access(A_, ntn.Read(), (i,))),
                                    ),
                                ),
                            ),
                        ),
                        ntn.Repack(A_, A),
                        ntn.Return(d),
                    )
                ),
            ),
        )
    )

    mod = ntn.NotationInterpreter()(prgm)
    cnt = mod.count_nonfill_vector(a)
    assert cnt == np.count_nonzero(a)

@pytest.mark.parametrize(
    "a",
    [
        np.array([[1, 0, 1],
                  [0, 0, 0],
                  [1, 1, 0]], dtype=int),
        np.zeros((2, 3), dtype=int),
        np.ones((3, 2), dtype=int),
    ],
)

def test_count_nonfill_matrix(a):
    A = ntn.Variable("A", np.ndarray)
    A_ = ntn.Slot("A_", np.ndarray)

    i = ntn.Variable("i", np.int64)
    j = ntn.Variable("j", np.int64)
    ni = ntn.Variable("ni", np.int64)
    nj = ntn.Variable("nj", np.int64)

    dij = ntn.Variable("dij", np.int64)

    X = ntn.Slot("X", np.ndarray)
    Y = ntn.Slot("Y", np.ndarray)
    xi = ntn.Variable("xi", np.int64)
    yj = ntn.Variable("yj", np.int64)

    d_i    = ntn.Variable("d_i",   np.int64)
    d_i_j  = ntn.Variable("d_i_j", np.int64)
    d_j    = ntn.Variable("d_j",   np.int64)
    d_j_i  = ntn.Variable("d_j_i", np.int64)

    prgm = ntn.Module((
        ntn.Function(
            ntn.Variable("matrix_total_nnz", np.int64),
            (A,),
            ntn.Block((
                ntn.Assign(nj, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))),
                ntn.Assign(ni, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1)))),
                ntn.Assign(dij, ntn.Literal(np.int64(0))),
                ntn.Unpack(A_,A),
                ntn.Loop(
                    i,
                    ni,
                    ntn.Loop(
                        j,
                        nj,
                        ntn.Assign(
                            dij,
                            ntn.Call(
                                ntn.Literal(operator.add),
                                (
                                    dij,
                                    ntn.Unwrap(ntn.Access(A_, ntn.Read(), (j, i)))
                                )
                            )
                        )
                    )
                ),
                ntn.Repack(A_,A),
                ntn.Return(dij),
            ))
        ),
        ntn.Function(
            ntn.Variable("matrix_structure_to_dcs", tuple),
            (A,),
            ntn.Block((
                ntn.Assign(nj, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))),
                ntn.Assign(ni, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1)))),
                ntn.Unpack(A_,A),
                ntn.Loop(
                    i,
                    ni,
                    ntn.Loop(
                        j,
                        nj,
                        ntn.Block((
                            ntn.Increment(
                                ntn.Access(
                                    X,
                                    ntn.Update(ntn.Literal(operator.add)), (i,)
                                ),
                                ntn.Unwrap(
                                    ntn.Access(A_, ntn.Read(),(j, i))
                                )
                            ),
                            ntn.Increment(
                                ntn.Access(
                                    Y,
                                    ntn.Update(ntn.Literal(operator.add)), (j,)
                                ),
                                ntn.Unwrap(
                                    ntn.Access(A_, ntn.Read(), (j, i))
                                )

                            )
                        ))
                    )
                ),
                ntn.Assign(d_i, ntn.Literal(np.int64(0))),
                ntn.Assign(d_i_j, ntn.Literal(np.int64(0))),
                ntn.Loop(
                    i,
                    ni,
                    ntn.Block((
                        ntn.Assign(
                            xi,
                            ntn.Unwrap(ntn.Access(X, ntn.Read(), (i,)))
                        ),
                        ntn.If(
                            ntn.Call(ntn.Literal(operator.ne), (xi, ntn.Literal(np.int64(0)))),
                            ntn.Assign(
                                d_i,
                                ntn.Call(
                                    ntn.Literal(operator.add),
                                    (d_i,
                                    ntn.Literal(np.int64(1))
                                    )
                                )
                            ),
                        ),
                        ntn.Assign(
                            d_i_j,
                            ntn.Call(
                                ntn.Literal(max),
                                (d_i_j, xi))
                        ),
                    ))
                ),
                ntn.Assign(d_j, ntn.Literal(np.int64(0))),
                ntn.Assign(d_j_i, ntn.Literal(np.int64(0))),
                ntn.Loop(
                    j,
                    nj,
                    ntn.Block((
                        ntn.Assign(
                            yj,
                            ntn.Unwrap(ntn.Access(Y, ntn.Read(), (j,)))
                        ),
                        ntn.If(
                            ntn.Call(ntn.Literal(operator.ne), (yj, ntn.Literal(np.int64(0)))),
                            ntn.Assign(
                                d_j,
                                ntn.Call(
                                    ntn.Literal(operator.add),
                                    (d_j,
                                    ntn.Literal(np.int64(1))
                                    )
                                )
                            ),
                        ),
                        ntn.Assign(
                            d_j_i,
                            ntn.Call(
                                ntn.Literal(max),
                                (d_j_i, yj))
                        ),
                    ))
                ),
                ntn.Repack(A_, A),
                ntn.Return(
                    ntn.Call(
                        ntn.Literal(lambda a, b, c, d: (a, b, c, d)),
                        (d_i, d_i_j, d_j, d_j_i)
                    )
                ),
            ))
        )
    ))

    mod = ntn.NotationInterpreter()(prgm)

    d_ij = mod.matrix_total_nnz(a)
    d_i, d_i_j, d_j, d_j_i = mod.matrix_structure_to_dcs(a)
    col_sums = a.sum(axis=0)
    row_sums = a.sum(axis=1)

    assert d_ij == int(np.count_nonzero(a))
    assert d_i   == int((col_sums > 0).sum())
    assert d_i_j == int(col_sums.max(initial=0))
    assert d_j   == int((row_sums > 0).sum())
    assert d_j_i == int(row_sums.max(initial=0))
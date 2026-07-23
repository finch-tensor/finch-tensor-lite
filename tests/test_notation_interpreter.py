import pytest

import numpy  # noqa: F401, ICN001
import numpy as np

import finch  # noqa: F401
import finch.finch_notation as ntn
from finch import ffuncs
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

from .conftest import finch_assert_equal


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
    i = ntn.Variable("i", finch.int64)
    j = ntn.Variable("j", finch.int64)
    k = ntn.Variable("k", finch.int64)

    a = finch.asarray(a)
    A = ntn.Variable("A", finch.ftype(a))
    B = ntn.Variable("B", finch.ftype(a))
    C = ntn.Variable("C", finch.ftype(a))
    A_ = ntn.Slot("A_", finch.ftype(a))
    B_ = ntn.Slot("B_", finch.ftype(a))
    C_ = ntn.Slot("C_", finch.ftype(a))

    a_ik = ntn.Variable("a_ik", finch.float64)
    b_kj = ntn.Variable("b_kj", finch.float64)
    c_ij = ntn.Variable("c_ij", finch.float64)

    m = ntn.Variable("m", finch.int64)
    n = ntn.Variable("n", finch.int64)
    p = ntn.Variable("p", finch.int64)

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matmul", finch.ftype(a)),
                (C, A, B),
                ntn.Block(
                    (
                        ntn.Unpack(A_, A),
                        ntn.Unpack(B_, B),
                        ntn.Unpack(C_, C),
                        ntn.Assign(m, ntn.Dimension(A_, ntn.Literal(0))),
                        ntn.Assign(n, ntn.Dimension(B_, ntn.Literal(1))),
                        ntn.Assign(p, ntn.Dimension(A_, ntn.Literal(1))),
                        ntn.Declare(
                            C_, ntn.Literal(0.0), ntn.Literal(ffuncs.add), (m, n)
                        ),
                        ntn.Loop(
                            i,
                            ntn.Call(
                                ntn.Literal(finch.compile.make_extent),
                                (ntn.Literal(finch.int64(0)), m),
                            ),
                            ntn.Loop(
                                k,
                                ntn.Call(
                                    ntn.Literal(finch.compile.make_extent),
                                    (ntn.Literal(finch.int64(0)), p),
                                ),
                                ntn.Loop(
                                    j,
                                    ntn.Call(
                                        ntn.Literal(finch.compile.make_extent),
                                        (ntn.Literal(finch.int64(0)), n),
                                    ),
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
                                                    ntn.Literal(ffuncs.mul),
                                                    (a_ik, b_kj),
                                                ),
                                            ),
                                            ntn.Increment(
                                                ntn.Access(
                                                    C_,
                                                    ntn.Update(ntn.Literal(ffuncs.add)),
                                                    (i, j),
                                                ),
                                                c_ij,
                                            ),
                                        )
                                    ),
                                ),
                            ),
                        ),
                        ntn.Freeze(C_, ntn.Literal(ffuncs.add)),
                        ntn.Repack(C_, C),
                        ntn.Return(C),
                    )
                ),
            ),
        )
    )

    mod = ntn.NotationInterpreter()(prgm)

    c = np.zeros(dtype=np.float64, shape=(a.shape[0], b.shape[1]))
    result = mod.matmul(finch.asarray(c), finch.asarray(a), finch.asarray(b))

    expected = np.matmul(a, b)

    finch_assert_equal(result, expected)
    print(repr(prgm))

    assert prgm == eval(
        repr(prgm),
        {
            **vars(ntn),
            **vars(finch.codegen),
            **vars(finch.compile),
            **vars(finch.tensor),
            **vars(ffuncs),
            **globals(),
        },
    )


@pytest.mark.parametrize(
    "a",
    [
        np.array([0, 1, 0, 0, 1]),
        np.array([1, 1, 1, 1, 1]),
        np.array([0, 1, 0, 0, 0]),
    ],
)
def test_count_nonfill_vector(a):
    a = finch.asarray(a)
    A = ntn.Variable("A", finch.ftype(a))
    A_ = ntn.Slot("A_", finch.ftype(a))

    d = ntn.Variable("d", finch.int64)
    i = ntn.Variable("i", finch.int64)
    m = ntn.Variable("m", finch.int64)

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("count_nonfill_vector", finch.int64),
                (A,),
                ntn.Block(
                    (
                        ntn.Assign(d, ntn.Literal(np.int64(0))),
                        ntn.Unpack(A_, A),
                        ntn.Assign(m, ntn.Dimension(A_, ntn.Literal(0))),
                        ntn.Loop(
                            i,
                            ntn.Call(
                                ntn.Literal(finch.compile.make_extent),
                                (ntn.Literal(finch.int64(0)), m),
                            ),
                            ntn.Assign(
                                d,
                                ntn.Call(
                                    Literal(ffuncs.add),
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
    assert cnt == np.count_nonzero(a.to_numpy())

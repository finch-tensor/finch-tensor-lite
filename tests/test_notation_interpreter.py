# AI modified: 2026-04-08T22:22:21Z 84b3c0ad
import pytest

import numpy  # noqa: F401, ICN001
import numpy as np

import finchlite  # noqa: F401
import finchlite.finch_notation as ntn
from finchlite import ffuncs
from finchlite.compile import dimension
from finchlite.finch_notation import (  # noqa: F401
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
    i = ntn.Variable("i", finchlite.int64)
    j = ntn.Variable("j", finchlite.int64)
    k = ntn.Variable("k", finchlite.int64)

    a = finchlite.asarray(a)
    A = ntn.Variable("A", finchlite.ftype(a))
    B = ntn.Variable("B", finchlite.ftype(a))
    C = ntn.Variable("C", finchlite.ftype(a))
    A_ = ntn.Slot("A_", finchlite.ftype(a))
    B_ = ntn.Slot("B_", finchlite.ftype(a))
    C_ = ntn.Slot("C_", finchlite.ftype(a))

    a_ik = ntn.Variable("a_ik", finchlite.float64)
    b_kj = ntn.Variable("b_kj", finchlite.float64)
    c_ij = ntn.Variable("c_ij", finchlite.float64)

    m = ntn.Variable("m", finchlite.int64)
    n = ntn.Variable("n", finchlite.int64)
    p = ntn.Variable("p", finchlite.int64)

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matmul", finchlite.ftype(a)),
                (C, A, B),
                ntn.Block(
                    (
                        ntn.Unpack(A_, A),
                        ntn.Unpack(B_, B),
                        ntn.Unpack(C_, C),
                        ntn.Assign(
                            m, ntn.Dimension(A_, ntn.Literal(0))
                        ),
                        ntn.Assign(
                            n, ntn.Dimension(B_, ntn.Literal(1))
                        ),
                        ntn.Assign(
                            p, ntn.Dimension(A_, ntn.Literal(1))
                        ),
                        ntn.Declare(
                            C_, ntn.Literal(0.0), ntn.Literal(ffuncs.add), (m, n)
                        ),
                        ntn.Loop(
                            i,
                            ntn.Call(ntn.Literal(finchlite.compile.make_extent), (ntn.Literal(finchlite.int64(0)), m)),
                            ntn.Loop(
                                k,
                                ntn.Call(ntn.Literal(finchlite.compile.make_extent), (ntn.Literal(finchlite.int64(0)), p)),
                                ntn.Loop(
                                    j,
                                    ntn.Call(ntn.Literal(finchlite.compile.make_extent), (ntn.Literal(finchlite.int64(0)), n)),
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
    result = mod.matmul(finchlite.asarray(c), finchlite.asarray(a), finchlite.asarray(b))

    expected = np.matmul(a, b)

    finch_assert_equal(result, expected)
    print(repr(prgm))

    assert prgm == eval(repr(prgm), {**vars(ntn), **vars(finchlite.codegen), **vars(finchlite.compile), **vars(ffuncs), **globals()})


@pytest.mark.parametrize(
    "a",
    [
        np.array([0, 1, 0, 0, 1]),
        np.array([1, 1, 1, 1, 1]),
        np.array([0, 1, 0, 0, 0]),
    ],
)
def test_count_nonfill_vector(a):
    a = finchlite.asarray(a)
    A = ntn.Variable("A", finchlite.ftype(a))
    A_ = ntn.Slot("A_", finchlite.ftype(a))

    d = ntn.Variable("d", finchlite.int64)
    i = ntn.Variable("i", finchlite.int64)
    m = ntn.Variable("m", finchlite.int64)

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("count_nonfill_vector", finchlite.int64),
                (A,),
                ntn.Block(
                    (
                        ntn.Assign(d, ntn.Literal(np.int64(0))),
                        ntn.Unpack(A_, A),
                        ntn.Assign(
                            m, ntn.Dimension(A_, ntn.Literal(0))
                        ),
                        ntn.Loop(
                            i,
                            ntn.Call(ntn.Literal(finchlite.compile.make_extent), (ntn.Literal(finchlite.int64(0)), m)),
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
    assert cnt == np.count_nonzero(a)
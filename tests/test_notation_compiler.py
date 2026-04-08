# AI modified: 2026-04-08T22:22:21Z 84b3c0ad
import pytest

import numpy as np

import finchlite
import finchlite.finch_notation as ntn
from finchlite import ffunc, ftype
from finchlite.compile import (
    BufferizedNDArray,
    NotationCompiler,
    make_extent,
)
from finchlite.finch_assembly import AssemblyInterpreter
from finchlite.symbolic import Reflector

from .conftest import finch_assert_equal


@pytest.mark.parametrize(
    "a, b",
    [
        (
            np.array([[1, 2, 3], [3, 4, 0]], dtype=np.float64),
            np.array([[5, 0, 6, 9], [7, 8, 0, 0], [-1, -4, 9, 0]], dtype=np.float64),
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

    a_buf = BufferizedNDArray.from_numpy(a)
    b_buf = BufferizedNDArray.from_numpy(b)

    a_format = ftype(a_buf)

    A = ntn.Variable("A", a_format)
    B = ntn.Variable("B", a_format)
    C = ntn.Variable("C", a_format)
    A_ = ntn.Slot("A_", a_format)
    B_ = ntn.Slot("B_", a_format)
    C_ = ntn.Slot("C_", a_format)

    a_ik = ntn.Variable("a_ik", finchlite.float64)
    b_kj = ntn.Variable("b_kj", finchlite.float64)
    c_ij = ntn.Variable("c_ij", finchlite.float64)

    m = ntn.Variable("m", finchlite.int64)
    n = ntn.Variable("n", finchlite.int64)
    p = ntn.Variable("p", finchlite.int64)

    m_ext = ntn.Call(
        ntn.Literal(make_extent),
        (ntn.Literal(np.int64(0)), ntn.Variable("m", finchlite.int64)),
    )
    n_ext = ntn.Call(
        ntn.Literal(make_extent),
        (ntn.Literal(np.int64(0)), ntn.Variable("n", finchlite.int64)),
    )
    p_ext = ntn.Call(
        ntn.Literal(make_extent),
        (ntn.Literal(np.int64(0)), ntn.Variable("p", finchlite.int64)),
    )

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matmul", a_format),
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
                            C_, ntn.Literal(0.0), ntn.Literal(ffunc.add), (m, n)
                        ),
                        ntn.Loop(
                            i,
                            m_ext,
                            ntn.Loop(
                                k,
                                p_ext,
                                ntn.Loop(
                                    j,
                                    n_ext,
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
                                                    ntn.Literal(ffunc.mul),
                                                    (a_ik, b_kj),
                                                ),
                                            ),
                                            ntn.Increment(
                                                ntn.Access(
                                                    C_,
                                                    ntn.Update(ntn.Literal(ffunc.add)),
                                                    (i, j),
                                                ),
                                                c_ij,
                                            ),
                                        )
                                    ),
                                ),
                            ),
                        ),
                        ntn.Freeze(C_, ntn.Literal(ffunc.add)),
                        ntn.Repack(C_, C),
                        ntn.Return(C),
                    )
                ),
            ),
        )
    )

    # NOTATION
    ntn_mod = ntn.NotationInterpreter()(prgm)

    c_buf = finchlite.compile.BufferizedNDArray.from_numpy(
        np.zeros(dtype=np.float64, shape=(a.shape[0], b.shape[1]))
    )

    result = ntn_mod.matmul(c_buf, a_buf, b_buf).to_numpy()
    expected = np.matmul(a, b)
    finch_assert_equal(result, expected)

    # ASSEMBLY
    asm_program = NotationCompiler(Reflector())(prgm)
    asm_mod = AssemblyInterpreter()(asm_program)

    c_buf = finchlite.compile.BufferizedNDArray.from_numpy(
        np.zeros(dtype=np.float64, shape=(a.shape[0], b.shape[1]))
    )

    expected = np.matmul(a, b)
    actual = asm_mod.matmul(c_buf, a_buf, b_buf).to_numpy()
    finch_assert_equal(actual, expected)


def test_matrix_multiplication_regression(file_regression):
    a = np.array([[2, 0], [1, 3]], dtype=np.float64)
    i = ntn.Variable("i", finchlite.int64)
    j = ntn.Variable("j", finchlite.int64)
    k = ntn.Variable("k", finchlite.int64)

    a_buf = finchlite.compile.BufferizedNDArray.from_numpy(a)
    a_format = ftype(a_buf)

    A = ntn.Variable("A", a_format)
    B = ntn.Variable("B", a_format)
    C = ntn.Variable("C", a_format)
    A_ = ntn.Slot("A_", a_format)
    B_ = ntn.Slot("B_", a_format)
    C_ = ntn.Slot("C_", a_format)

    a_ik = ntn.Variable("a_ik", finchlite.float64)
    b_kj = ntn.Variable("b_kj", finchlite.float64)
    c_ij = ntn.Variable("c_ij", finchlite.float64)

    m = ntn.Variable("m", finchlite.int64)
    n = ntn.Variable("n", finchlite.int64)
    p = ntn.Variable("p", finchlite.int64)

    m_ext = ntn.Call(ntn.Literal(make_extent), (ntn.Literal(np.int64(0)), m))
    n_ext = ntn.Call(ntn.Literal(make_extent), (ntn.Literal(np.int64(0)), n))
    p_ext = ntn.Call(ntn.Literal(make_extent), (ntn.Literal(np.int64(0)), p))

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matmul", a_format),
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
                            C_,
                            ntn.Literal(0.0),
                            ntn.Literal(ffunc.add),
                            (m, n),
                        ),
                        ntn.Loop(
                            i,
                            m_ext,
                            ntn.Loop(
                                k,
                                p_ext,
                                ntn.Loop(
                                    j,
                                    n_ext,
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
                                                    ntn.Literal(ffunc.mul),
                                                    (a_ik, b_kj),
                                                ),
                                            ),
                                            ntn.Increment(
                                                ntn.Access(
                                                    C_,
                                                    ntn.Update(ntn.Literal(ffunc.add)),
                                                    (i, j),
                                                ),
                                                c_ij,
                                            ),
                                        )
                                    ),
                                ),
                            ),
                        ),
                        ntn.Freeze(C_, ntn.Literal(ffunc.add)),
                        ntn.Repack(C_, C),
                        ntn.Return(C),
                    )
                ),
            ),
        )
    )

    asm_program = NotationCompiler(Reflector())(prgm)
    file_regression.check(str(asm_program), extension=".txt")

import operator

import pytest

import numpy as np

import finchlite
import finchlite.finch_notation as ntn
from finchlite import ftype
from finchlite.compile import ExtentFType, NotationCompiler, dimension
from finchlite.finch_assembly import AssemblyInterpreter
from finchlite.symbolic import Reflector


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
    i = ntn.Variable("i", np.int64)
    j = ntn.Variable("j", np.int64)
    k = ntn.Variable("k", np.int64)

    a_buf = finchlite.compile.BufferizedNDArray(a)
    b_buf = finchlite.compile.BufferizedNDArray(b)

    a_format = ftype(a_buf)

    A = ntn.Variable("A", a_format)
    B = ntn.Variable("B", a_format)
    C = ntn.Variable("C", a_format)
    A_ = ntn.Slot("A_", a_format)
    B_ = ntn.Slot("B_", a_format)
    C_ = ntn.Slot("C_", a_format)

    a_ik = ntn.Variable("a_ik", np.float64)
    b_kj = ntn.Variable("b_kj", np.float64)
    c_ij = ntn.Variable("c_ij", np.float64)

    m = ntn.Variable("m", ExtentFType(np.int64, np.int64))
    n = ntn.Variable("n", ExtentFType(np.int64, np.int64))
    p = ntn.Variable("p", ExtentFType(np.int64, np.int64))

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matmul", a_format),
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
                                k,
                                p,
                                ntn.Loop(
                                    j,
                                    n,
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

    # NOTATION
    ntn_mod = ntn.NotationInterpreter()(prgm)

    c_buf = finchlite.compile.BufferizedNDArray(
        np.zeros(dtype=np.float64, shape=(a.shape[0], b.shape[1]))
    )

    result = ntn_mod.matmul(c_buf, a_buf, b_buf).to_numpy()
    expected = np.matmul(a, b)
    np.testing.assert_equal(result, expected)

    # ASSEMBLY
    asm_program = NotationCompiler(Reflector())(prgm)
    asm_mod = AssemblyInterpreter()(asm_program)

    c_buf = finchlite.compile.BufferizedNDArray(
        np.zeros(dtype=np.float64, shape=(a.shape[0], b.shape[1]))
    )

    expected = np.matmul(a, b)
    actual = asm_mod.matmul(c_buf, a_buf, b_buf).to_numpy()
    np.testing.assert_equal(actual, expected)


def test_matrix_multiplication_regression(file_regression):
    a = np.array([[2, 0], [1, 3]], dtype=np.float64)
    i = ntn.Variable("i", np.int64)
    j = ntn.Variable("j", np.int64)
    k = ntn.Variable("k", np.int64)

    a_buf = finchlite.compile.BufferizedNDArray(a)
    a_format = ftype(a_buf)

    A = ntn.Variable("A", a_format)
    B = ntn.Variable("B", a_format)
    C = ntn.Variable("C", a_format)
    A_ = ntn.Slot("A_", a_format)
    B_ = ntn.Slot("B_", a_format)
    C_ = ntn.Slot("C_", a_format)

    a_ik = ntn.Variable("a_ik", np.float64)
    b_kj = ntn.Variable("b_kj", np.float64)
    c_ij = ntn.Variable("c_ij", np.float64)

    m = ntn.Variable("m", ExtentFType(np.int64, np.int64))
    n = ntn.Variable("n", ExtentFType(np.int64, np.int64))
    p = ntn.Variable("p", ExtentFType(np.int64, np.int64))

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matmul", a_format),
                (C, A, B),
                ntn.Block(
                    (
                        ntn.Assign(
                            m,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                        ),
                        ntn.Assign(
                            n,
                            ntn.Call(ntn.Literal(dimension), (B, ntn.Literal(1))),
                        ),
                        ntn.Assign(
                            p,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                        ),
                        ntn.Unpack(A_, A),
                        ntn.Unpack(B_, B),
                        ntn.Unpack(C_, C),
                        ntn.Declare(
                            C_,
                            ntn.Literal(0.0),
                            ntn.Literal(operator.add),
                            (m, n),
                        ),
                        ntn.Loop(
                            i,
                            m,
                            ntn.Loop(
                                k,
                                p,
                                ntn.Loop(
                                    j,
                                    n,
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

    asm_program = NotationCompiler(Reflector())(prgm)
    file_regression.check(str(asm_program), extension=".txt")

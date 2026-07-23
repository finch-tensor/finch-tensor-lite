import pytest

import numpy as np

import finch
import finch.finch_notation as ntn
from finch import ffuncs, ftype
from finch.compile import (
    NotationCompiler,
    make_extent,
)
from finch.finch_assembly import AssemblyInterpreter
from finch.symbolic import Reflector
from finch.tensor import BufferizedNDArray

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
    i = ntn.Variable("i", finch.int64)
    j = ntn.Variable("j", finch.int64)
    k = ntn.Variable("k", finch.int64)

    a_buf = BufferizedNDArray.from_numpy(a)
    b_buf = BufferizedNDArray.from_numpy(b)

    a_format = ftype(a_buf)

    A = ntn.Variable("A", a_format)
    B = ntn.Variable("B", a_format)
    C = ntn.Variable("C", a_format)
    A_ = ntn.Slot("A_", a_format)
    B_ = ntn.Slot("B_", a_format)
    C_ = ntn.Slot("C_", a_format)

    a_ik = ntn.Variable("a_ik", finch.float64)
    b_kj = ntn.Variable("b_kj", finch.float64)
    c_ij = ntn.Variable("c_ij", finch.float64)

    m = ntn.Variable("m", finch.int64)
    n = ntn.Variable("n", finch.int64)
    p = ntn.Variable("p", finch.int64)

    m_ext = ntn.Call(
        ntn.Literal(make_extent),
        (ntn.Literal(np.int64(0)), ntn.Variable("m", finch.int64)),
    )
    n_ext = ntn.Call(
        ntn.Literal(make_extent),
        (ntn.Literal(np.int64(0)), ntn.Variable("n", finch.int64)),
    )
    p_ext = ntn.Call(
        ntn.Literal(make_extent),
        (ntn.Literal(np.int64(0)), ntn.Variable("p", finch.int64)),
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
                            C_, ntn.Literal(0.0), ntn.Literal(ffuncs.add), (m, n)
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

    # NOTATION
    ntn_mod = ntn.NotationInterpreter()(prgm)

    c_buf = BufferizedNDArray.from_numpy(
        np.zeros(dtype=np.float64, shape=(a.shape[0], b.shape[1]))
    )

    result = ntn_mod.matmul(c_buf, a_buf, b_buf).to_numpy()
    expected = np.matmul(a, b)
    finch_assert_equal(result, expected)

    # ASSEMBLY
    asm_program = NotationCompiler(Reflector())(prgm)
    asm_mod = AssemblyInterpreter()(asm_program)

    c_buf = BufferizedNDArray.from_numpy(
        np.zeros(dtype=np.float64, shape=(a.shape[0], b.shape[1]))
    )

    expected = np.matmul(a, b)
    actual = asm_mod.matmul(c_buf, a_buf, b_buf).to_numpy()
    finch_assert_equal(actual, expected)


def test_matrix_multiplication_regression(file_regression):
    a = np.array([[2, 0], [1, 3]], dtype=np.float64)
    i = ntn.Variable("i", finch.int64)
    j = ntn.Variable("j", finch.int64)
    k = ntn.Variable("k", finch.int64)

    a_buf = BufferizedNDArray.from_numpy(a)
    a_format = ftype(a_buf)

    A = ntn.Variable("A", a_format)
    B = ntn.Variable("B", a_format)
    C = ntn.Variable("C", a_format)
    A_ = ntn.Slot("A_", a_format)
    B_ = ntn.Slot("B_", a_format)
    C_ = ntn.Slot("C_", a_format)

    a_ik = ntn.Variable("a_ik", finch.float64)
    b_kj = ntn.Variable("b_kj", finch.float64)
    c_ij = ntn.Variable("c_ij", finch.float64)

    m = ntn.Variable("m", finch.int64)
    n = ntn.Variable("n", finch.int64)
    p = ntn.Variable("p", finch.int64)

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
                            ntn.Literal(ffuncs.add),
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

    asm_program = NotationCompiler(Reflector())(prgm)
    file_regression.check(str(asm_program), extension=".txt")


def test_if_in_loop_is_lowered():
    """An ``If`` nested inside a ``Loop`` must survive notation->assembly
    lowering.

    Regression test: the ``If``/``IfElse`` lowering used to emit the statement
    into a throwaway block that was never spliced back into the enclosing scope,
    so conditional writes inside loops silently disappeared. This is exactly the
    shape produced when a transpose lowers to a guarded diagonal write
    (``if eq(i_, i): out[..., i_] = ...``), which made transposes compile to
    all-zero results on every assembly backend while the notation interpreter
    stayed correct.
    """
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    half = np.int64(2)
    # Only the first ``half`` elements are copied; the rest keep the fill value.
    expected = np.array([10.0, 20.0, 0.0, 0.0], dtype=np.float64)

    a_buf = BufferizedNDArray.from_numpy(a)
    vec_format = ftype(a_buf)

    i = ntn.Variable("i", finch.int64)
    n = ntn.Variable("n", finch.int64)
    A = ntn.Variable("A", vec_format)
    OUT = ntn.Variable("OUT", vec_format)
    A_ = ntn.Slot("A_", vec_format)
    OUT_ = ntn.Slot("OUT_", vec_format)
    a_i = ntn.Variable("a_i", finch.float64)

    n_ext = ntn.Call(ntn.Literal(make_extent), (ntn.Literal(np.int64(0)), n))

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("masked_copy", vec_format),
                (OUT, A),
                ntn.Block(
                    (
                        ntn.Unpack(A_, A),
                        ntn.Unpack(OUT_, OUT),
                        ntn.Assign(n, ntn.Dimension(A_, ntn.Literal(0))),
                        ntn.Declare(
                            OUT_, ntn.Literal(0.0), ntn.Literal(ffuncs.overwrite), (n,)
                        ),
                        ntn.Loop(
                            i,
                            n_ext,
                            ntn.If(
                                ntn.Call(
                                    ntn.Literal(ffuncs.lt), (i, ntn.Literal(half))
                                ),
                                ntn.Block(
                                    (
                                        ntn.Assign(
                                            a_i,
                                            ntn.Unwrap(
                                                ntn.Access(A_, ntn.Read(), (i,))
                                            ),
                                        ),
                                        ntn.Increment(
                                            ntn.Access(
                                                OUT_,
                                                ntn.Update(
                                                    ntn.Literal(ffuncs.overwrite)
                                                ),
                                                (i,),
                                            ),
                                            a_i,
                                        ),
                                    )
                                ),
                            ),
                        ),
                        ntn.Freeze(OUT_, ntn.Literal(ffuncs.overwrite)),
                        ntn.Repack(OUT_, OUT),
                        ntn.Return(OUT),
                    )
                ),
            ),
        )
    )

    # NOTATION interpreter is the reference semantics.
    ntn_mod = ntn.NotationInterpreter()(prgm)
    out_buf = BufferizedNDArray.from_numpy(np.zeros_like(a))
    ntn_result = ntn_mod.masked_copy(out_buf, a_buf).to_numpy()
    finch_assert_equal(ntn_result, expected)

    # ASSEMBLY path must agree (this is what regressed).
    asm_program = NotationCompiler(Reflector())(prgm)
    asm_mod = AssemblyInterpreter()(asm_program)
    out_buf = BufferizedNDArray.from_numpy(np.zeros_like(a))
    asm_result = asm_mod.masked_copy(out_buf, a_buf).to_numpy()
    finch_assert_equal(asm_result, expected)

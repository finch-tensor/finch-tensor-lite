import operator
from textwrap import dedent

import numpy as np

import finch.finch_notation as ntn
from finch.compile import dimension


def test_printer():
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
                        ntn.Repack(
                            val=C_,
                            obj=C,
                        ),
                        ntn.Return(C),
                    )
                ),
            ),
        )
    )

    actual = str(prgm)

    expected = dedent("""\
    def matmul(C: ndarray, A: ndarray, B: ndarray) -> ndarray:
        m: int64 = dimension(A, 0)
        n: int64 = dimension(B, 1)
        p: int64 = dimension(A, 1)
        A_: ndarray = unpack(A)
        B_: ndarray = unpack(B)
        C_: ndarray = unpack(C)
        declare(C_, 0.0, add, ['m', 'n'])
        loop(i, m):
            loop(j, n):
                loop(k, p):
                    a_ik: float64 = unwrap(read(A_, ['i', 'k']))
                    b_kj: float64 = unwrap(read(B_, ['k', 'j']))
                    c_ij: float64 = mul(a_ik, b_kj)
                    increment(update(C_, ['i', 'j'], add), c_ij)
        freeze(C_, add)
        repack(C_, C)
        return C
    """)

    assert expected == actual

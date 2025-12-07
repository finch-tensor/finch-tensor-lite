import _operator  # noqa: F401
import operator

import pytest

import numpy  # noqa: F401, ICN001
import numpy as np

import finchlite  # noqa: F401
import finchlite.finch_notation as ntn
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

from .util import finch_assert_equal


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

    mod = ntn.NotationInterpreter()(prgm)

    c = np.zeros(dtype=np.float64, shape=(a.shape[0], b.shape[1]))
    result = mod.matmul(c, a, b)

    expected = np.matmul(a, b)

    finch_assert_equal(result, expected)

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
        np.array([[1, 0, 1], [0, 0, 0], [1, 1, 0]], dtype=int),
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

    xi = ntn.Variable("xi", np.int64)
    yj = ntn.Variable("yj", np.int64)

    d_i = ntn.Variable("d_i", np.int64)
    d_i_j = ntn.Variable("d_i_j", np.int64)
    d_j = ntn.Variable("d_j", np.int64)
    d_j_i = ntn.Variable("d_j_i", np.int64)

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matrix_total_nnz", np.int64),
                (A,),
                ntn.Block(
                    (
                        ntn.Assign(
                            nj, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))
                        ),
                        ntn.Assign(
                            ni, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1)))
                        ),
                        ntn.Assign(dij, ntn.Literal(np.int64(0))),
                        ntn.Unpack(A_, A),
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
                                            ntn.Unwrap(
                                                ntn.Access(A_, ntn.Read(), (j, i))
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        ntn.Repack(A_, A),
                        ntn.Return(dij),
                    )
                ),
            ),
            ntn.Function(
                ntn.Variable("matrix_structure_to_dcs", tuple),
                (A,),
                ntn.Block(
                    (
                        ntn.Assign(
                            nj, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))
                        ),
                        ntn.Assign(
                            ni, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1)))
                        ),
                        ntn.Unpack(A_, A),
                        ntn.Assign(d_i, ntn.Literal(np.int64(0))),
                        ntn.Assign(d_i_j, ntn.Literal(np.int64(0))),
                        ntn.Loop(
                            i,
                            ni,
                            ntn.Block(
                                (
                                    ntn.Assign(xi, ntn.Literal(np.int64(0))),
                                    ntn.Loop(
                                        j,
                                        nj,
                                        ntn.Assign(
                                            xi,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (
                                                    xi,
                                                    ntn.Unwrap(
                                                        ntn.Access(
                                                            A_, ntn.Read(), (j, i)
                                                        )
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    ntn.If(
                                        ntn.Call(
                                            ntn.Literal(operator.ne),
                                            (xi, ntn.Literal(np.int64(0))),
                                        ),
                                        ntn.Assign(
                                            d_i,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (d_i, ntn.Literal(np.int64(1))),
                                            ),
                                        ),
                                    ),
                                    ntn.Assign(
                                        d_i_j, ntn.Call(ntn.Literal(max), (d_i_j, xi))
                                    ),
                                )
                            ),
                        ),
                        ntn.Assign(d_j, ntn.Literal(np.int64(0))),
                        ntn.Assign(d_j_i, ntn.Literal(np.int64(0))),
                        ntn.Loop(
                            j,
                            nj,
                            ntn.Block(
                                (
                                    ntn.Assign(yj, ntn.Literal(np.int64(0))),
                                    ntn.Loop(
                                        i,
                                        ni,
                                        ntn.Assign(
                                            yj,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (
                                                    yj,
                                                    ntn.Unwrap(
                                                        ntn.Access(
                                                            A_, ntn.Read(), (j, i)
                                                        )
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    ntn.If(
                                        ntn.Call(
                                            ntn.Literal(operator.ne),
                                            (yj, ntn.Literal(np.int64(0))),
                                        ),
                                        ntn.Assign(
                                            d_j,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (d_j, ntn.Literal(np.int64(1))),
                                            ),
                                        ),
                                    ),
                                    ntn.Assign(
                                        d_j_i, ntn.Call(ntn.Literal(max), (d_j_i, yj))
                                    ),
                                )
                            ),
                        ),
                        ntn.Repack(A_, A),
                        ntn.Return(
                            ntn.Call(
                                ntn.Literal(lambda a, b, c, d: (a, b, c, d)),
                                (d_i, d_i_j, d_j, d_j_i),
                            )
                        ),
                    )
                ),
            ),
        )
    )
    mod = ntn.NotationInterpreter()(prgm)

    d_ij = mod.matrix_total_nnz(a)
    d_i_, d_i_j_, d_j_, d_j_i_ = mod.matrix_structure_to_dcs(a)
    col_sums = a.sum(axis=0)
    row_sums = a.sum(axis=1)

    assert d_ij == int(np.count_nonzero(a))
    assert d_i_ == int((col_sums > 0).sum())
    assert d_i_j_ == int(col_sums.max(initial=0))
    assert d_j_ == int((row_sums > 0).sum())
    assert d_j_i_ == int(row_sums.max(initial=0))


@pytest.mark.parametrize(
    "a",
    [
        np.array(
            [
                [[1, 0, 0, 1], [0, 1, 0, 0]],
                [[0, 0, 0, 0], [1, 1, 0, 0]],
                [[0, 0, 1, 0], [0, 0, 0, 0]],
            ],
            dtype=int,
        ),
        np.zeros((2, 3, 4), dtype=int),
        np.ones((2, 3, 4), dtype=int),
    ],
)
def test_count_nonfill_3d(a):
    A = ntn.Variable("A", np.ndarray)
    A_ = ntn.Slot("A_", np.ndarray)

    i = ntn.Variable("i", np.int64)
    j = ntn.Variable("j", np.int64)
    k = ntn.Variable("k", np.int64)

    ni = ntn.Variable("ni", np.int64)
    nj = ntn.Variable("nj", np.int64)
    nk = ntn.Variable("nk", np.int64)

    dijk = ntn.Variable("dijk", np.int64)

    xi = ntn.Variable("xi", np.int64)
    yj = ntn.Variable("yj", np.int64)
    zk = ntn.Variable("zk", np.int64)

    d_i = ntn.Variable("d_i", np.int64)
    d_i_jk = ntn.Variable("d_i_jk", np.int64)
    d_j = ntn.Variable("d_j", np.int64)
    d_j_ik = ntn.Variable("d_j_ik", np.int64)
    d_k = ntn.Variable("d_k", np.int64)
    d_k_ij = ntn.Variable("d_k_ij", np.int64)

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("_3d_total_nnz", np.int64),
                (A,),
                ntn.Block(
                    (
                        ntn.Assign(
                            nk,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                        ),
                        ntn.Assign(
                            nj,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                        ),
                        ntn.Assign(
                            ni,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(2))),
                        ),
                        ntn.Assign(dijk, ntn.Literal(np.int64(0))),
                        ntn.Unpack(A_, A),
                        ntn.Loop(
                            i,
                            ni,
                            ntn.Loop(
                                j,
                                nj,
                                ntn.Loop(
                                    k,
                                    nk,
                                    ntn.Assign(
                                        dijk,
                                        ntn.Call(
                                            ntn.Literal(operator.add),
                                            (
                                                dijk,
                                                ntn.Unwrap(
                                                    ntn.Access(
                                                        A_, ntn.Read(), (k, j, i)
                                                    )
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        ntn.Repack(A_, A),
                        ntn.Return(dijk),
                    )
                ),
            ),
            ntn.Function(
                ntn.Variable("_3d_structure_to_dcs", tuple),
                (A,),
                ntn.Block(
                    (
                        ntn.Assign(
                            nk,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                        ),
                        ntn.Assign(
                            nj,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                        ),
                        ntn.Assign(
                            ni,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(2))),
                        ),
                        ntn.Unpack(A_, A),
                        ntn.Assign(d_i, ntn.Literal(np.int64(0))),
                        ntn.Assign(d_i_jk, ntn.Literal(np.int64(0))),
                        ntn.Loop(
                            i,
                            ni,
                            ntn.Block(
                                (
                                    ntn.Assign(xi, ntn.Literal(np.int64(0))),
                                    ntn.Loop(
                                        j,
                                        nj,
                                        ntn.Loop(
                                            k,
                                            nk,
                                            ntn.Assign(
                                                xi,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (
                                                        xi,
                                                        ntn.Unwrap(
                                                            ntn.Access(
                                                                A_,
                                                                ntn.Read(),
                                                                (k, j, i),
                                                            )
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    ntn.If(
                                        ntn.Call(
                                            ntn.Literal(operator.ne),
                                            (xi, ntn.Literal(np.int64(0))),
                                        ),
                                        ntn.Assign(
                                            d_i,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (d_i, ntn.Literal(np.int64(1))),
                                            ),
                                        ),
                                    ),
                                    ntn.Assign(
                                        d_i_jk,
                                        ntn.Call(ntn.Literal(max), (d_i_jk, xi)),
                                    ),
                                )
                            ),
                        ),
                        ntn.Assign(d_j, ntn.Literal(np.int64(0))),
                        ntn.Assign(d_j_ik, ntn.Literal(np.int64(0))),
                        ntn.Loop(
                            j,
                            nj,
                            ntn.Block(
                                (
                                    ntn.Assign(yj, ntn.Literal(np.int64(0))),
                                    ntn.Loop(
                                        i,
                                        ni,
                                        ntn.Loop(
                                            k,
                                            nk,
                                            ntn.Assign(
                                                yj,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (
                                                        yj,
                                                        ntn.Unwrap(
                                                            ntn.Access(
                                                                A_,
                                                                ntn.Read(),
                                                                (k, j, i),
                                                            )
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    ntn.If(
                                        ntn.Call(
                                            ntn.Literal(operator.ne),
                                            (yj, ntn.Literal(np.int64(0))),
                                        ),
                                        ntn.Assign(
                                            d_j,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (d_j, ntn.Literal(np.int64(1))),
                                            ),
                                        ),
                                    ),
                                    ntn.Assign(
                                        d_j_ik,
                                        ntn.Call(ntn.Literal(max), (d_j_ik, yj)),
                                    ),
                                )
                            ),
                        ),
                        ntn.Assign(d_k, ntn.Literal(np.int64(0))),
                        ntn.Assign(d_k_ij, ntn.Literal(np.int64(0))),
                        ntn.Loop(
                            k,
                            nk,
                            ntn.Block(
                                (
                                    ntn.Assign(zk, ntn.Literal(np.int64(0))),
                                    ntn.Loop(
                                        i,
                                        ni,
                                        ntn.Loop(
                                            j,
                                            nj,
                                            ntn.Assign(
                                                zk,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (
                                                        zk,
                                                        ntn.Unwrap(
                                                            ntn.Access(
                                                                A_,
                                                                ntn.Read(),
                                                                (k, j, i),
                                                            )
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    ntn.If(
                                        ntn.Call(
                                            ntn.Literal(operator.ne),
                                            (zk, ntn.Literal(np.int64(0))),
                                        ),
                                        ntn.Assign(
                                            d_k,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (d_k, ntn.Literal(np.int64(1))),
                                            ),
                                        ),
                                    ),
                                    ntn.Assign(
                                        d_k_ij,
                                        ntn.Call(ntn.Literal(max), (d_k_ij, zk)),
                                    ),
                                )
                            ),
                        ),
                        ntn.Repack(A_, A),
                        ntn.Return(
                            ntn.Call(
                                ntn.Literal(
                                    lambda a, b, c, d, e, f: (a, b, c, d, e, f)
                                ),
                                (d_i, d_i_jk, d_j, d_j_ik, d_k, d_k_ij),
                            )
                        ),
                    )
                ),
            ),
        )
    )
    mod = ntn.NotationInterpreter()(prgm)

    d_ijk = mod._3d_total_nnz(a)
    d_i_, d_i_jk_, d_j_, d_j_ik_, d_k_, d_k_ij_ = mod._3d_structure_to_dcs(a)
    X = a.sum(axis=(0, 1))
    Y = a.sum(axis=(0, 2))
    Z = a.sum(axis=(1, 2))

    assert d_ijk == int(np.count_nonzero(a))

    assert d_i_ == int((X > 0).sum())
    assert d_i_jk_ == int(X.max(initial=0))
    assert d_j_ == int((Y > 0).sum())
    assert d_j_ik_ == int(Y.max(initial=0))
    assert d_k_ == int((Z > 0).sum())
    assert d_k_ij_ == int(Z.max(initial=0))


@pytest.mark.parametrize(
    "a",
    [
        np.array(
            [
                [
                    [[1, 0], [0, 1]],
                    [[0, 1], [1, 0]],
                ],
                [
                    [[0, 0], [1, 0]],
                    [[1, 0], [0, 1]],
                ],
            ],
            dtype=int,
        ),
        np.zeros((2, 3, 1, 4), dtype=int),
        np.ones((1, 2, 3, 2), dtype=int),
    ],
)
def test_count_nonfill_4d(a):
    A = ntn.Variable("A", np.ndarray)
    A_ = ntn.Slot("A_", np.ndarray)

    i = ntn.Variable("i", np.int64)
    j = ntn.Variable("j", np.int64)
    k = ntn.Variable("k", np.int64)
    w = ntn.Variable("w", np.int64)

    ni = ntn.Variable("ni", np.int64)
    nj = ntn.Variable("nj", np.int64)
    nk = ntn.Variable("nk", np.int64)
    nw = ntn.Variable("nw", np.int64)

    dijkw = ntn.Variable("dijkw", np.int64)

    xi = ntn.Variable("xi", np.int64)
    yj = ntn.Variable("yj", np.int64)
    zk = ntn.Variable("zk", np.int64)
    uw = ntn.Variable("uw", np.int64)

    d_i = ntn.Variable("d_i", np.int64)
    d_i_jkw = ntn.Variable("d_i_jkw", np.int64)
    d_j = ntn.Variable("d_j", np.int64)
    d_j_ikw = ntn.Variable("d_j_ikw", np.int64)
    d_k = ntn.Variable("d_k", np.int64)
    d_k_ijw = ntn.Variable("d_k_ijw", np.int64)
    d_w = ntn.Variable("d_w", np.int64)
    d_w_ijk = ntn.Variable("d_l_ijw", np.int64)

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("_4d_total_nnz", np.int64),
                (A,),
                ntn.Block(
                    (
                        ntn.Assign(
                            nw,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                        ),
                        ntn.Assign(
                            nk,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                        ),
                        ntn.Assign(
                            nj,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(2))),
                        ),
                        ntn.Assign(
                            ni,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(3))),
                        ),
                        ntn.Assign(dijkw, ntn.Literal(np.int64(0))),
                        ntn.Unpack(A_, A),
                        ntn.Loop(
                            i,
                            ni,
                            ntn.Loop(
                                j,
                                nj,
                                ntn.Loop(
                                    k,
                                    nk,
                                    ntn.Loop(
                                        w,
                                        nw,
                                        ntn.Assign(
                                            dijkw,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (
                                                    dijkw,
                                                    ntn.Unwrap(
                                                        ntn.Access(
                                                            A_, ntn.Read(), (w, k, j, i)
                                                        )
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        ntn.Repack(A_, A),
                        ntn.Return(dijkw),
                    )
                ),
            ),
            ntn.Function(
                ntn.Variable("_4d_structure_to_dcs", tuple),
                (A,),
                ntn.Block(
                    (
                        ntn.Assign(
                            nw,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                        ),
                        ntn.Assign(
                            nk,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                        ),
                        ntn.Assign(
                            nj,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(2))),
                        ),
                        ntn.Assign(
                            ni,
                            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(3))),
                        ),
                        ntn.Unpack(A_, A),
                        ntn.Assign(d_i, ntn.Literal(np.int64(0))),
                        ntn.Assign(d_i_jkw, ntn.Literal(np.int64(0))),
                        ntn.Loop(
                            i,
                            ni,
                            ntn.Block(
                                (
                                    ntn.Assign(xi, ntn.Literal(np.int64(0))),
                                    ntn.Loop(
                                        j,
                                        nj,
                                        ntn.Loop(
                                            k,
                                            nk,
                                            ntn.Loop(
                                                w,
                                                nw,
                                                ntn.Assign(
                                                    xi,
                                                    ntn.Call(
                                                        ntn.Literal(operator.add),
                                                        (
                                                            xi,
                                                            ntn.Unwrap(
                                                                ntn.Access(
                                                                    A_,
                                                                    ntn.Read(),
                                                                    (w, k, j, i),
                                                                )
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    ntn.If(
                                        ntn.Call(
                                            ntn.Literal(operator.ne),
                                            (xi, ntn.Literal(np.int64(0))),
                                        ),
                                        ntn.Assign(
                                            d_i,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (d_i, ntn.Literal(np.int64(1))),
                                            ),
                                        ),
                                    ),
                                    ntn.Assign(
                                        d_i_jkw,
                                        ntn.Call(ntn.Literal(max), (d_i_jkw, xi)),
                                    ),
                                )
                            ),
                        ),
                        ntn.Assign(d_j, ntn.Literal(np.int64(0))),
                        ntn.Assign(d_j_ikw, ntn.Literal(np.int64(0))),
                        ntn.Loop(
                            j,
                            nj,
                            ntn.Block(
                                (
                                    ntn.Assign(yj, ntn.Literal(np.int64(0))),
                                    ntn.Loop(
                                        i,
                                        ni,
                                        ntn.Loop(
                                            k,
                                            nk,
                                            ntn.Loop(
                                                w,
                                                nw,
                                                ntn.Assign(
                                                    yj,
                                                    ntn.Call(
                                                        ntn.Literal(operator.add),
                                                        (
                                                            yj,
                                                            ntn.Unwrap(
                                                                ntn.Access(
                                                                    A_,
                                                                    ntn.Read(),
                                                                    (w, k, j, i),
                                                                )
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    ntn.If(
                                        ntn.Call(
                                            ntn.Literal(operator.ne),
                                            (yj, ntn.Literal(np.int64(0))),
                                        ),
                                        ntn.Assign(
                                            d_j,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (d_j, ntn.Literal(np.int64(1))),
                                            ),
                                        ),
                                    ),
                                    ntn.Assign(
                                        d_j_ikw,
                                        ntn.Call(ntn.Literal(max), (d_j_ikw, yj)),
                                    ),
                                )
                            ),
                        ),
                        ntn.Assign(d_k, ntn.Literal(np.int64(0))),
                        ntn.Assign(d_k_ijw, ntn.Literal(np.int64(0))),
                        ntn.Loop(
                            k,
                            nk,
                            ntn.Block(
                                (
                                    ntn.Assign(zk, ntn.Literal(np.int64(0))),
                                    ntn.Loop(
                                        i,
                                        ni,
                                        ntn.Loop(
                                            j,
                                            nj,
                                            ntn.Loop(
                                                w,
                                                nw,
                                                ntn.Assign(
                                                    zk,
                                                    ntn.Call(
                                                        ntn.Literal(operator.add),
                                                        (
                                                            zk,
                                                            ntn.Unwrap(
                                                                ntn.Access(
                                                                    A_,
                                                                    ntn.Read(),
                                                                    (w, k, j, i),
                                                                )
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    ntn.If(
                                        ntn.Call(
                                            ntn.Literal(operator.ne),
                                            (zk, ntn.Literal(np.int64(0))),
                                        ),
                                        ntn.Assign(
                                            d_k,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (d_k, ntn.Literal(np.int64(1))),
                                            ),
                                        ),
                                    ),
                                    ntn.Assign(
                                        d_k_ijw,
                                        ntn.Call(ntn.Literal(max), (d_k_ijw, zk)),
                                    ),
                                )
                            ),
                        ),
                        ntn.Assign(d_w, ntn.Literal(np.int64(0))),
                        ntn.Assign(d_w_ijk, ntn.Literal(np.int64(0))),
                        ntn.Loop(
                            w,
                            nw,
                            ntn.Block(
                                (
                                    ntn.Assign(uw, ntn.Literal(np.int64(0))),
                                    ntn.Loop(
                                        i,
                                        ni,
                                        ntn.Loop(
                                            j,
                                            nj,
                                            ntn.Loop(
                                                k,
                                                nk,
                                                ntn.Assign(
                                                    uw,
                                                    ntn.Call(
                                                        ntn.Literal(operator.add),
                                                        (
                                                            uw,
                                                            ntn.Unwrap(
                                                                ntn.Access(
                                                                    A_,
                                                                    ntn.Read(),
                                                                    (w, k, j, i),
                                                                )
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    ntn.If(
                                        ntn.Call(
                                            ntn.Literal(operator.ne),
                                            (uw, ntn.Literal(np.int64(0))),
                                        ),
                                        ntn.Assign(
                                            d_w,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (d_w, ntn.Literal(np.int64(1))),
                                            ),
                                        ),
                                    ),
                                    ntn.Assign(
                                        d_w_ijk,
                                        ntn.Call(ntn.Literal(max), (d_w_ijk, uw)),
                                    ),
                                )
                            ),
                        ),
                        ntn.Repack(A_, A),
                        ntn.Return(
                            ntn.Call(
                                ntn.Literal(
                                    lambda a, b, c, d, e, f, g, h: (
                                        a,
                                        b,
                                        c,
                                        d,
                                        e,
                                        f,
                                        g,
                                        h,
                                    )
                                ),
                                (
                                    d_i,
                                    d_i_jkw,
                                    d_j,
                                    d_j_ikw,
                                    d_k,
                                    d_k_ijw,
                                    d_w,
                                    d_w_ijk,
                                ),
                            )
                        ),
                    )
                ),
            ),
        )
    )
    mod = ntn.NotationInterpreter()(prgm)

    d_ijkw = mod._4d_total_nnz(a)
    d_i_, d_i_jkw_, d_j_, d_j_ikw_, d_k_, d_k_ijw_, d_w_, d_w_ijk_ = (
        mod._4d_structure_to_dcs(a)
    )

    X = a.sum(axis=(0, 1, 2))
    Y = a.sum(axis=(0, 1, 3))
    Z = a.sum(axis=(0, 2, 3))
    W = a.sum(axis=(1, 2, 3))

    assert d_ijkw == int(np.count_nonzero(a))

    assert d_i_ == int((X > 0).sum())
    assert d_i_jkw_ == int(X.max(initial=0))
    assert d_j_ == int((Y > 0).sum())
    assert d_j_ikw_ == int(Y.max(initial=0))
    assert d_k_ == int((Z > 0).sum())
    assert d_k_ijw_ == int(Z.max(initial=0))
    assert d_w_ == int((W > 0).sum())
    assert d_w_ijk_ == int(W.max(initial=0))

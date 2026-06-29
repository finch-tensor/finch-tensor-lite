from collections import OrderedDict

import pytest

import numpy as np

import finchlite as fl
from finchlite import ffuncs
from finchlite.algebra.utils import is_subsequence
from finchlite.autoschedule.loop_ordering import loop_order_cost
from finchlite.autoschedule.tensor_stats.exact_stats import ExactStatsFactory
from finchlite.finch_logic import Alias, Field, Literal, MapJoin, Table

AXES = {"i": 0, "j": 1, "k": 2}
ORDERS = [
    (Field("i"), Field("j"), Field("k")),
    (Field("j"), Field("i"), Field("k")),
    (Field("j"), Field("k"), Field("i")),
    (Field("k"), Field("j"), Field("i")),
]


def _sparse(shape, nnz, rng):
    mat = np.zeros(shape)
    if nnz:
        mat.flat[rng.choice(shape[0] * shape[1], nnz, replace=False)] = rng.uniform(
            0.1, 10, nnz
        )
    return mat


def _ref_cost(A, B, order):
    join = A[:, :, None] * B[None, :, :]
    cost = sum(
        (join != 0)
        .any(axis=tuple(AXES[n] for n in AXES if n not in set(order[:k])))
        .sum()
        for k in range(1, len(order) + 1)
    )
    if not is_subsequence(("i", "j"), order):
        cost += np.count_nonzero(A)
    if not is_subsequence(("j", "k"), order):
        cost += np.count_nonzero(B)
    return float(cost)


def _cost(A, B, order, sf):
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b = Alias("A"), Alias("B")
    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    bindings = OrderedDict({a: sf(fl.asarray(A), (i, j)), b: sf(fl.asarray(B), (j, k))})
    return loop_order_cost(expr, order, sf, bindings)


def test_loop_order_cost():
    rng = np.random.default_rng(42)
    sf = ExactStatsFactory()
    cases = [
        (
            np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]], float),
            np.array([[1, 0], [0, 3], [0, 0]], float),
        )
    ]
    for _ in range(10):
        m, p, n = rng.integers(4, 10, 3)
        cases.append(
            (
                _sparse((m, p), rng.integers(1, m * p // 3 + 1), rng),
                _sparse((p, n), rng.integers(1, p * n // 3 + 1), rng),
            )
        )

    for A, B in cases:
        for order in ORDERS:
            names = tuple(f.name for f in order)
            assert _cost(A, B, order, sf) == pytest.approx(_ref_cost(A, B, names))

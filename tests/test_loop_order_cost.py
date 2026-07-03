import time
from collections import OrderedDict
from functools import reduce

import pytest

import numpy as np

import finchlite as fl
from finchlite import ffuncs
from finchlite.algebra.utils import is_subsequence
from finchlite.autoschedule.loop_order_cost import loop_order_cost
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
    masks = {
        frozenset(("i", "j")): A[:, :, None] != 0,
        frozenset(("j", "k")): B[None, :, :] != 0,
    }
    cost = 0.0
    for k in range(1, len(order) + 1):
        prefix = set(order[:k])
        rels = [mask for axes, mask in masks.items() if axes & prefix]
        subquery = reduce(np.logical_and, rels)
        reduce_axes = tuple(AXES[n] for n in AXES if n not in prefix)
        cost += subquery.any(axis=reduce_axes).sum()
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
    test_start = time.perf_counter()
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

    n_costs = 0
    cost_time = 0.0
    ref_time = 0.0

    for A, B in cases:
        for order in ORDERS:
            names = tuple(f.name for f in order)

            t0 = time.perf_counter()
            got = _cost(A, B, order, sf)
            cost_time += time.perf_counter() - t0
            n_costs += 1

            t0 = time.perf_counter()
            expected = _ref_cost(A, B, names)
            ref_time += time.perf_counter() - t0

            assert got == pytest.approx(expected)

    avg_cost_ms = cost_time / n_costs * 1000
    avg_ref_ms = ref_time / n_costs * 1000
    total_time = time.perf_counter() - test_start
    print(
        f"loop_order_cost: {cost_time:.4f}s total "
        f"({n_costs} calls, {avg_cost_ms:.2f}ms avg)"
    )
    print(
        f"_ref_cost:       {ref_time:.4f}s total "
        f"({n_costs} calls, {avg_ref_ms:.2f}ms avg)"
    )
    print(f"test total:      {total_time:.4f}s")


def test_empty_relation():
    test_start = time.perf_counter()
    sf = ExactStatsFactory()
    # l_ is l, precommit throws bad name error otehrwise
    i, j, k, l_, m = (Field(name) for name in "ijklm")
    a, b, c, d = (Alias(name) for name in "ABCD")
    expr = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(a, (i, j)),
            Table(b, (j, k)),
            Table(c, (k, l_)),
            Table(d, (l_, m)),
        ),
    )
    bindings = OrderedDict(
        {
            a: sf(fl.asarray(np.ones((3, 3))), (i, j)),
            b: sf(fl.asarray(np.ones((3, 3))), (j, k)),
            c: sf(fl.asarray(np.ones((3, 3))), (k, l_)),
            d: sf(fl.asarray(np.zeros((3, 3))), (l_, m)),
        }
    )

    t0 = time.perf_counter()
    forward = loop_order_cost(expr, (i, j, k, l_, m), sf, bindings)
    forward_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    reverse = loop_order_cost(expr, (m, l_, k, j, i), sf, bindings)
    reverse_time = time.perf_counter() - t0

    assert forward > reverse

    total_time = time.perf_counter() - test_start
    print(f"loop_order_cost forward: {forward_time:.4f}s")
    print(f"loop_order_cost reverse: {reverse_time:.4f}s")
    print(f"test total:              {total_time:.4f}s")

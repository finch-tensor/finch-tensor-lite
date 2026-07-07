from collections import OrderedDict

import pytest

import numpy as np

import finchlite as fl
from finchlite import ffuncs
from finchlite.autoschedule.loop_order_cost import (
    SEQ_READ_COST,
    SEQ_WRITE_COST,
    get_conjunctive_and_disjunctive_inputs,
    get_loop_lookups,
    loop_order_cost,
)
from finchlite.autoschedule.tensor_stats import DatabaseStatsFactory
from finchlite.finch_logic import Alias, Field, Literal, MapJoin, Table

ORDERS = [
    (Field("i"), Field("j"), Field("k")),
    (Field("j"), Field("i"), Field("k")),
    (Field("j"), Field("k"), Field("i")),
    (Field("k"), Field("j"), Field("i")),
]

_AXES_A = {"i": 0, "j": 1}
_AXES_B = {"j": 0, "k": 1}


def _sparse(shape, nnz, rng):
    mat = np.zeros(shape)
    if nnz:
        mat.flat[rng.choice(shape[0] * shape[1], nnz, replace=False)] = rng.uniform(
            0.1, 10, nnz
        )
    return mat


def _expr_and_bindings(A, B, sf):
    i, j, k = Field("i"), Field("j"), Field("k")
    a, b = Alias("A"), Alias("B")
    expr = MapJoin(Literal(ffuncs.mul), (Table(a, (i, j)), Table(b, (j, k))))
    bindings = OrderedDict({a: sf(fl.asarray(A), (i, j)), b: sf(fl.asarray(B), (j, k))})
    return expr, bindings


def _prefix_lookup_factor(A, B, prefix):
    new_var = prefix[-1].name
    prefix_names = {field.name for field in prefix}
    lookup_factor = 0.0

    for mat, index_order, axes in (
        (A, ("i", "j"), _AXES_A),
        (B, ("j", "k"), _AXES_B),
    ):
        if new_var not in index_order:
            continue

        rel_dims = [dim for dim in index_order if dim in prefix_names]
        space = 1
        for dim in rel_dims:
            space *= mat.shape[axes[dim]]
        density = np.count_nonzero(mat) / space if space else 0.0
        is_dense = density > 0.05
        lookup_factor += SEQ_READ_COST / 5 if is_dense else SEQ_READ_COST

    lookup_factor += SEQ_WRITE_COST
    return lookup_factor


def _ref_cost(A, B, order, sf, expr, bindings):
    stats_bindings = bindings.copy()
    cache: dict[object, object] = {}
    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, sf, stats_bindings, cache
    )
    cost = 0.0
    for j in range(1, len(order) + 1):
        prefix = order[:j]
        lookups = get_loop_lookups(prefix, conjunct_stats, disjunct_stats, sf)
        cost += lookups * _prefix_lookup_factor(A, B, prefix)
    return cost


def test_loop_order_cost():
    rng = np.random.default_rng(42)
    sf = DatabaseStatsFactory()
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
            expr, bindings = _expr_and_bindings(A, B, sf)
            got = loop_order_cost(expr, order, sf, bindings)
            expected = _ref_cost(A, B, order, sf, expr, bindings)
            assert got == pytest.approx(expected)


def test_empty_relation():
    sf = DatabaseStatsFactory()
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

    forward = loop_order_cost(expr, (i, j, k, l_, m), sf, bindings)
    reverse = loop_order_cost(expr, (m, l_, k, j, i), sf, bindings)
    assert forward > reverse

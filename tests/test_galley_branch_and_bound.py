"""Tests: branch-and-bound (exact) cost vs greedy."""

from collections import OrderedDict

import pytest

import numpy as np

import finchlite as fl
from finchlite.algebra import ffunc
from finchlite.autoschedule.galley.logical_optimizer import AnnotatedQuery
from finchlite.autoschedule.galley.logical_optimizer.branch_and_bound import (
    branch_and_bound,
    pruned_query_to_plan,
)
from finchlite.autoschedule.tensor_stats import DenseStatsFactory
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Query,
    Table,
)

_DENSE_STATS_FACTORY = DenseStatsFactory()


def _make_aq_four_index_chain():
    """
    AnnotatedQuery for sum_{i,j,k,l} A[i,j]*B[j,k]*C[k,l].
    Chain with 4 indices. Dims: A 3x10, B 10x5, C 5x2.
    Exact search can find a cheaper order than greedy.
    """
    A = fl.asarray(np.ones((3, 10)))
    B = fl.asarray(np.ones((10, 5)))
    C = fl.asarray(np.ones((5, 2)))
    q = Query(
        Alias("out"),
        Aggregate(
            Literal(ffunc.add),
            Literal(0),
            MapJoin(
                Literal(ffunc.mul),
                (
                    Table(Literal(A), (Field("i"), Field("j"))),
                    Table(Literal(B), (Field("j"), Field("k"))),
                    Table(Literal(C), (Field("k"), Field("l"))),
                ),
            ),
            (Field("i"), Field("j"), Field("k"), Field("l")),
        ),
    )
    return AnnotatedQuery(_DENSE_STATS_FACTORY, q, bindings=OrderedDict())


def _make_aq_three_index_chain():
    """sum_{i,j,k} A[i,j]*B[j,k] — smaller chain."""
    A = fl.asarray(np.ones((4, 8)))
    B = fl.asarray(np.ones((8, 6)))
    q = Query(
        Alias("out"),
        Aggregate(
            Literal(ffunc.add),
            Literal(0),
            MapJoin(
                Literal(ffunc.mul),
                (
                    Table(Literal(A), (Field("i"), Field("j"))),
                    Table(Literal(B), (Field("j"), Field("k"))),
                ),
            ),
            (Field("i"), Field("j"), Field("k")),
        ),
    )
    return AnnotatedQuery(_DENSE_STATS_FACTORY, q, bindings=OrderedDict())


_CHAIN_FACTORIES = (_make_aq_three_index_chain, _make_aq_four_index_chain)


def test_pruned_exact_strictly_cheaper_than_pruned_greedy_on_four_index_chain():
    """Four-index chain: exact (pruned) cost is strictly below greedy (pruned)."""
    aq = _make_aq_four_index_chain()
    _, greedy_cost = pruned_query_to_plan(aq, use_greedy=True)
    _, exact_cost = pruned_query_to_plan(aq, use_greedy=False)
    assert exact_cost <= greedy_cost
    assert exact_cost < greedy_cost, (
        f"Expected exact ({exact_cost}) < greedy ({greedy_cost})"
    )


@pytest.mark.parametrize("factory", _CHAIN_FACTORIES)
def test_pruned_exact_never_more_expensive_than_pruned_greedy(factory):
    """Exact pruned plan cost is never above greedy-only pruned plan."""
    aq = factory()
    _, greedy_cost = pruned_query_to_plan(aq, use_greedy=True)
    _, exact_cost = pruned_query_to_plan(aq, use_greedy=False)
    assert exact_cost <= greedy_cost


@pytest.mark.parametrize("factory", _CHAIN_FACTORIES)
def test_bnb_exact_k_inf_cost_no_worse_than_greedy_k1(factory):
    """On one component, exact (k=inf) cost <= greedy (k=1) B&B cost."""
    aq = factory()
    component = aq.connected_components[0]
    r_greedy = branch_and_bound(aq, component, 1, OrderedDict())
    r_exact = branch_and_bound(aq, component, float("inf"), OrderedDict())
    assert r_greedy is not None and r_exact is not None
    (_, _, _, cost_k1), _ = r_greedy
    (_, _, _, cost_kinf), _ = r_exact
    assert cost_kinf <= cost_k1

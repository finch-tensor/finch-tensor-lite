"""Tests: branch-and-bound (exact) cost vs greedy."""

from collections import OrderedDict

import pytest

import numpy as np

import finchlite as fl
from finchlite.algebra import ffunc
from finchlite.autoschedule.galley.logical_optimizer import AnnotatedQuery
from finchlite.autoschedule.galley.logical_optimizer.branch_and_bound import (
    branch_and_bound,
    branch_and_bound_dfs,
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
    Reorder,
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


def test_layered_bnb_exact_matches_dfs_bnb_exact_on_matmul_chain():
    """Layered exact BnB vs DFS exact BnB: same optimal cost on a matmul chain."""
    rng = np.random.default_rng(42)
    shapes = ((3, 10), (10, 5), (5, 2))
    mats = [
        fl.asarray(rng.standard_normal((r, c)).astype(np.float64)) for r, c in shapes
    ]
    q = Query(
        Alias("out"),
        Aggregate(
            Literal(ffunc.add),
            Literal(0),
            MapJoin(
                Literal(ffunc.mul),
                (
                    Table(Literal(mats[0]), (Field("i"), Field("j"))),
                    Table(Literal(mats[1]), (Field("j"), Field("k"))),
                    Table(Literal(mats[2]), (Field("k"), Field("l"))),
                ),
            ),
            (Field("i"), Field("j"), Field("k"), Field("l")),
        ),
    )
    aq = AnnotatedQuery(_DENSE_STATS_FACTORY, q, bindings=OrderedDict())
    component = aq.connected_components[0]
    (_, _, _, cost_layered), _ = branch_and_bound(
        aq.copy(), component, float("inf"), OrderedDict()
    )
    (_, _, _, cost_dfs), _ = branch_and_bound_dfs(aq.copy(), component)
    assert cost_layered == pytest.approx(cost_dfs)


def test_pruned_exact_strictly_cheaper_than_pruned_greedy_on_four_index_chain():
    """Four-index chain: exact (pruned) cost is strictly below greedy (pruned)."""
    aq = _make_aq_four_index_chain()
    _, greedy_cost = pruned_query_to_plan(aq, optimizer="greedy")
    _, exact_cost = pruned_query_to_plan(aq, optimizer="bfs")
    assert exact_cost <= greedy_cost
    assert exact_cost < greedy_cost, (
        f"Expected exact ({exact_cost}) < greedy ({greedy_cost})"
    )


@pytest.mark.parametrize("factory", _CHAIN_FACTORIES)
def test_pruned_exact_never_more_expensive_than_pruned_greedy(factory):
    """Exact pruned plan cost is never above greedy-only pruned plan."""
    aq = factory()
    _, greedy_cost = pruned_query_to_plan(aq, optimizer="greedy")
    _, exact_cost = pruned_query_to_plan(aq, optimizer="bfs")
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


@pytest.mark.parametrize("factory", _CHAIN_FACTORIES)
def test_pruned_query_to_plan_use_dfs_exact_no_worse_than_greedy(factory):
    """DFS exact pruned plan cost is never above greedy-only pruned plan."""
    aq = factory()
    _, greedy_cost = pruned_query_to_plan(aq, optimizer="greedy")
    _, exact_dfs_cost = pruned_query_to_plan(aq, optimizer="dfs")
    assert exact_dfs_cost <= greedy_cost
    assert np.isfinite(exact_dfs_cost) and exact_dfs_cost > 0
    if factory is _make_aq_four_index_chain:
        assert exact_dfs_cost < greedy_cost, (
            f"Expected exact DFS ({exact_dfs_cost}) < greedy ({greedy_cost})"
        )


def _make_aq_passthrough_alias():
    """
    AnnotatedQuery for a bare Reorder(Table(alias, ...), ...) — no
    aggregation, so there are no reducible indices.  This is the minimal
    case that previously caused ``pruned_query_to_plan`` to return an empty
    list because ``get_remaining_query`` short-circuited on ``Table(Alias, _)``.
    """
    A = fl.asarray(np.ones((3, 4)))
    a_alias = Alias("A_in")
    bindings = OrderedDict()
    bindings[a_alias] = _DENSE_STATS_FACTORY(A, (Field("a_in_i_0"), Field("a_in_i_1")))
    q = Query(
        Alias("out"),
        Reorder(Table(a_alias, (Field("i"), Field("j"))), (Field("i"), Field("j"))),
    )
    return AnnotatedQuery(_DENSE_STATS_FACTORY, q, bindings=bindings)


def test_pruned_query_to_plan_nonempty_for_passthrough_alias():
    """pruned_query_to_plan must return at least one query for a passthrough alias."""
    aq = _make_aq_passthrough_alias()
    queries, _ = pruned_query_to_plan(aq)
    assert len(queries) >= 1, (
        "pruned_query_to_plan returned no queries for a passthrough"
    )


def test_pruned_query_to_plan_passthrough_lhs_matches_output_name():
    """The single query returned for a passthrough binds the correct output alias."""
    aq = _make_aq_passthrough_alias()
    queries, _ = pruned_query_to_plan(aq)
    assert queries[-1].lhs == Alias("out")


def test_pruned_query_to_plan_passthrough_body_references_input_alias():
    """The returned query body references the original input alias Table."""
    aq = _make_aq_passthrough_alias()
    queries, _ = pruned_query_to_plan(aq)
    last_rhs = queries[-1].rhs
    # Body should be Reorder(Table(Alias("A_in"), ...), ...) — the input alias
    assert isinstance(last_rhs, Reorder)
    assert isinstance(last_rhs.arg, Table)
    assert last_rhs.arg.tns == Alias("A_in")

"""Tests comparing branch_and_bound with greedy: case where exact beats greedy."""

import operator as op
from collections import OrderedDict

import numpy as np

import finchlite as fl
from finchlite.algebra import as_finch_operator
from finchlite.autoschedule.galley.logical_optimizer import AnnotatedQuery
from finchlite.autoschedule.galley.logical_optimizer.branch_and_bound import (
    pruned_query_to_plan,
    _aq_with_stats,
)
from finchlite.autoschedule.galley.logical_optimizer.greedy_optimizer import (
    greedy_query,
)
from finchlite.autoschedule.tensor_stats import DenseStats
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Query,
    Table,
)


def _make_aq_four_index_chain():
    """
    AnnotatedQuery for sum_{i,j,k,l} A[i,j]*B[j,k]*C[k,l].
    Chain with 4 indices. Dims: A 3x10, B 10x5, C 5x2.
    Exact search finds a cheaper order than greedy.
    """
    A = fl.asarray(np.ones((3, 10)))
    B = fl.asarray(np.ones((10, 5)))
    C = fl.asarray(np.ones((5, 2)))
    q = Query(
        Alias("out"),
        Aggregate(
            Literal(as_finch_operator(op.add)),
            Literal(0),
            MapJoin(
                Literal(as_finch_operator(op.mul)),
                (
                    Table(Literal(A), (Field("i"), Field("j"))),
                    Table(Literal(B), (Field("j"), Field("k"))),
                    Table(Literal(C), (Field("k"), Field("l"))),
                ),
            ),
            (Field("i"), Field("j"), Field("k"), Field("l")),
        ),
    )
    return AnnotatedQuery(DenseStats, q, bindings=OrderedDict())


def test_branch_and_bound_beats_greedy_on_four_index_chain():
    """
    Four-index chain sum_{i,j,k,l} A[i,j]*B[j,k]*C[k,l] where exact search
    finds a cheaper order than greedy_query.
    """
    # Use greedy_query (from greedy_optimizer)
    greedy_queries = greedy_query(_aq_with_stats(_make_aq_four_index_chain()))
    _, greedy_cost = pruned_query_to_plan(_make_aq_four_index_chain(), use_greedy=True)

    # Exact search (pruned with use_greedy=False uses branch_and_bound exact for small components)
    _, exact_cost = pruned_query_to_plan(_make_aq_four_index_chain(), use_greedy=False)

    print(f"Greedy cost: {greedy_cost}")
    print(f"Exact cost:  {exact_cost}")

    assert len(greedy_queries) >= 2  #check if greedy produces a plan
    assert exact_cost <= greedy_cost
    assert exact_cost < greedy_cost, (
        f"Expected exact ({exact_cost}) < greedy ({greedy_cost})"
    )
    
print(test_branch_and_bound_beats_greedy_on_four_index_chain())
"""
SImple benchmarks for greedy vs branch-and-bound.
"""

from __future__ import annotations

import operator as op
import time
from collections import OrderedDict

import pytest

import numpy as np

import finchlite as fl
from finchlite.algebra import as_finch_operator
from finchlite.autoschedule.galley.logical_optimizer import AnnotatedQuery
from finchlite.autoschedule.galley.logical_optimizer.branch_and_bound import (
    _aq_with_stats,
    branch_and_bound,
    pruned_query_to_plan,
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

NUM_TIMING_RUNS = 7


def _time_average_seconds(
    function_to_time, *, num_runs: int = NUM_TIMING_RUNS
) -> float:
    total_seconds = 0.0
    for _ in range(num_runs):
        time_start = time.perf_counter()
        function_to_time()
        total_seconds += time.perf_counter() - time_start
    return total_seconds / num_runs


def _make_chain_matmul(num_tensors: int, base: int = 6) -> AnnotatedQuery:
    fields = [Field(f"i{k}") for k in range(num_tensors + 1)]
    tables = []
    for tensor_index in range(num_tensors):
        array = fl.asarray(np.ones((base + tensor_index, base + tensor_index + 1)))
        tables.append(
            Table(Literal(array), (fields[tensor_index], fields[tensor_index + 1]))
        )
    q = Query(
        Alias("out"),
        Aggregate(
            Literal(as_finch_operator(op.add)),
            Literal(0),
            MapJoin(Literal(as_finch_operator(op.mul)), tuple(tables)),
            tuple(fields),
        ),
    )
    return AnnotatedQuery(DenseStats, q, bindings=OrderedDict())


def _make_four_index_chain() -> AnnotatedQuery:
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


@pytest.mark.parametrize(
    ("num_tensors", "base"),
    [
        (2, 5),
        (3, 5),
    ],
    ids=["chain_3idx", "chain_4idx"],
)
def test_benchmark_pruned_greedy_vs_exact_cost_and_time(num_tensors: int, base: int):
    """
    Pruned exact cost <= pruned greedy; print average timings over ``NUM_TIMING_RUNS``
    runs.
    """
    annotated_query = _make_chain_matmul(num_tensors, base=base)

    time_seconds_greedy_query = _time_average_seconds(
        lambda: greedy_query(_aq_with_stats(_make_chain_matmul(num_tensors, base)))
    )
    time_seconds_pruned_greedy = _time_average_seconds(
        lambda: pruned_query_to_plan(
            _make_chain_matmul(num_tensors, base), use_greedy=True
        )
    )
    time_seconds_pruned_exact = _time_average_seconds(
        lambda: pruned_query_to_plan(
            _make_chain_matmul(num_tensors, base), use_greedy=False
        )
    )

    _, cost_pruned_greedy = pruned_query_to_plan(annotated_query, use_greedy=True)
    _, cost_pruned_exact = pruned_query_to_plan(annotated_query, use_greedy=False)
    assert cost_pruned_exact <= cost_pruned_greedy

    print(
        f"\n[galley branch-and-bound benchmark] tensors={num_tensors} base={base} | "
        f"greedy_query={time_seconds_greedy_query * 1e3:8.3f} ms | "
        f"pruned greedy={time_seconds_pruned_greedy * 1e3:8.3f} ms | "
        f"pruned exact={time_seconds_pruned_exact * 1e3:8.3f} ms | "
        f"costs greedy={cost_pruned_greedy:g} exact={cost_pruned_exact:g}"
    )


def test_benchmark_four_index_chain_and_core_branch_and_bound():
    """
    Four-index chain: pruned costs and greedy versus core branch-and-bound
    (one versus unlimited candidates) times.
    """
    annotated_query = _make_four_index_chain()
    first_connected_component = _aq_with_stats(
        _make_four_index_chain()
    ).connected_components[0]

    time_seconds_greedy_query = _time_average_seconds(
        lambda: greedy_query(_aq_with_stats(_make_four_index_chain()))
    )
    time_seconds_pruned_greedy = _time_average_seconds(
        lambda: pruned_query_to_plan(_make_four_index_chain(), use_greedy=True)
    )
    time_seconds_pruned_exact = _time_average_seconds(
        lambda: pruned_query_to_plan(_make_four_index_chain(), use_greedy=False)
    )
    time_seconds_branch_and_bound_one_candidate = _time_average_seconds(
        lambda: branch_and_bound(
            _aq_with_stats(_make_four_index_chain()),
            first_connected_component,
            1,
            OrderedDict(),
        )
    )
    time_seconds_branch_and_bound_unlimited_candidates = (
        _time_average_seconds(
            lambda: branch_and_bound(
                _aq_with_stats(_make_four_index_chain()),
                first_connected_component,
                float("inf"),
                OrderedDict(),
            )
        )
    )

    _, cost_pruned_greedy = pruned_query_to_plan(annotated_query, use_greedy=True)
    _, cost_pruned_exact = pruned_query_to_plan(annotated_query, use_greedy=False)
    assert cost_pruned_exact <= cost_pruned_greedy

    branch_and_bound_one_candidate_milliseconds = (
        time_seconds_branch_and_bound_one_candidate * 1e3
    )
    branch_and_bound_unlimited_candidates_milliseconds = (
        time_seconds_branch_and_bound_unlimited_candidates * 1e3
    )
    print(
        f"\n[galley branch-and-bound benchmark] four-index chain | "
        f"greedy_query={time_seconds_greedy_query * 1e3:8.3f} ms | "
        f"pruned_greedy={time_seconds_pruned_greedy * 1e3:8.3f} ms | "
        f"pruned_exact={time_seconds_pruned_exact * 1e3:8.3f} ms | "
        f"branch-and-bound one candidate="
        f"{branch_and_bound_one_candidate_milliseconds:8.3f} ms | "
        f"branch-and-bound unlimited candidates="
        f"{branch_and_bound_unlimited_candidates_milliseconds:8.3f} ms | "
        f"costs greedy={cost_pruned_greedy:g} exact={cost_pruned_exact:g}"
    )

print(test_benchmark_four_index_chain_and_core_branch_and_bound())
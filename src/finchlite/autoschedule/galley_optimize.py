"""
Galley logical optimizer: applies greedy query rewriting to logical plans, with
an optional exact branch-and-bound path for query bodies.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from ..algebra.tensor import TensorFType
from ..finch_assembly import AssemblyLibrary
from ..finch_logic import (
    Alias,
    Field,
    LogicLoader,
    LogicStatement,
    Plan,
    Query,
    StatsFactory,
    TensorStats,
)
from ..util.logging import LOG_GALLEY
from .galley.logical_optimizer.annotated_query import AnnotatedQuery
from .galley.logical_optimizer.branch_and_bound import pruned_query_to_plan
from .galley.logical_optimizer.branch_and_bound_dfs import pruned_query_to_plan_dfs
from .galley.logical_optimizer.logic_to_stats import insert_statistics
from .galley.logical_optimizer.query_normalization import (
    postprocess_plan_after_galley,
    preprocess_plan_for_galley,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_GALLEY)

GalleyOptimizer = Literal["greedy", "bfs", "dfs"]


def optimize_query(
    query,
    stats_factory,
    stats_bindings,
    use_components: bool = True,
    *,
    optimizer: GalleyOptimizer = "greedy",
):
    """Rewrite a single logical Query using ``optimizer``:
    greedy,BFS, or DFS."""
    annotated_query = AnnotatedQuery(stats_factory, query, stats_bindings)
    if optimizer == "dfs":
        new_queries, _ = pruned_query_to_plan_dfs(
            annotated_query,
            use_components=use_components,
            use_greedy=False,
        )
    elif optimizer == "bfs":
        new_queries, _ = pruned_query_to_plan(
            annotated_query,
            use_components=use_components,
            use_greedy=False,
        )
    else:
        new_queries, _ = pruned_query_to_plan(
            annotated_query,
            use_components=use_components,
            use_greedy=True,
        )
    return new_queries


def optimize_plan(
    plan,
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
    use_components: bool = True,
    *,
    optimizer: GalleyOptimizer = "greedy",
):
    """
    Optimize a full Plan: run the Galley optimizer on each Query body,
    pass through non-Query bodies (Produces), and update stats bindings.
    """
    plan = preprocess_plan_for_galley(plan)
    optimized_queries = []
    cache_dict: dict[object, TensorStats] = {}
    for body in plan.bodies:
        if isinstance(body, Query):
            new_queries = optimize_query(
                body,
                stats_factory,
                stats_bindings,
                use_components=use_components,
                optimizer=optimizer,
            )
            for new_query in new_queries:
                insert_statistics(
                    stats_factory,
                    new_query,
                    stats_bindings,
                    replace=True,
                    cache=cache_dict,
                )
            optimized_queries.extend(new_queries)
        else:
            optimized_queries.append(body)

    return postprocess_plan_after_galley(Plan(tuple(optimized_queries)))


class GalleyLogicalOptimizer(LogicLoader):
    """
    LogicLoader stage that optimizes logical Plans with the Galley rewriter
    (greedy or exact layered/DFS branch-and-bound), then forwards to a
    downstream LogicLoader (ctx).
    """

    def __init__(
        self,
        ctx: LogicLoader,
        use_components: bool = True,
        *,
        optimizer: GalleyOptimizer = "bfs",
    ):
        self.ctx = ctx
        self.use_components = use_components
        self.optimizer = optimizer
        self.last_optimize_plan_s: float | None = None

    def __call__(
        self,
        term: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[Alias, TensorFType],
        dict[Alias, tuple[Field | None, ...]],
    ]:
        if not isinstance(term, Plan):
            raise ValueError(f"Unsupported program type: {type(term)}")
        logger.debug("Optimizing plan: %s", term)
        t0 = time.perf_counter()
        term = optimize_plan(
            term,
            stats_factory,
            stats,
            use_components=self.use_components,
            optimizer=self.optimizer,
        )
        self.last_optimize_plan_s = time.perf_counter() - t0
        return self.ctx(term, bindings, stats, stats_factory)

"""
Galley logical optimizer: applies greedy query rewriting to logical plans, with
an optional exact branch-and-bound path for query bodies.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import TypedDict

from ..finch_logic import (
    Alias,
    Field,
    LogicEvaluator,
    Plan,
    Query,
    StatsFactory,
    TensorStats,
)
from ..util.logging import LOG_GALLEY
from .galley.logical_optimizer.annotated_query import AnnotatedQuery
from .galley.logical_optimizer.branch_and_bound import pruned_query_to_plan
from .galley.logical_optimizer.logic_to_stats import insert_statistics
from .galley.logical_optimizer.query_normalization import (
    postprocess_plan_after_galley,
    preprocess_plan_for_galley,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_GALLEY)


def optimize_query(
    query,
    stats_factory,
    stats_bindings,
    use_components: bool = True,
    *,
    use_exact_branch_and_bound: bool = False,
):
    """Rewrite a single logical Query via greedy or exact branch-and-bound reduction."""
    annotated_query = AnnotatedQuery(stats_factory, query, stats_bindings)
    use_greedy = not use_exact_branch_and_bound
    new_queries, _ = pruned_query_to_plan(
        annotated_query, use_components=use_components, use_greedy=use_greedy
    )
    return new_queries


def optimize_plan(
    plan,
    stats_factory: StatsFactory,
    bindings,
    use_components: bool = True,
    *,
    use_exact_branch_and_bound: bool = False,
):
    """
    Optimize a full Plan: run the Galley optimizer on each Query body,
    pass through non-Query bodies (Produces), and update stats bindings.
    """
    plan = preprocess_plan_for_galley(plan)
    optimized_queries = []
    stats_bindings: OrderedDict[Alias, TensorStats] = OrderedDict(
        (
            var,
            stats_factory(T, tuple(Field(f"{var.name}_i_{i}") for i in range(T.ndim))),
        )
        for var, T in bindings.items()
    )
    cache_dict: dict[object, TensorStats] = {}
    for body in plan.bodies:
        if isinstance(body, Query):
            new_queries = optimize_query(
                body,
                stats_factory,
                stats_bindings,
                use_components=use_components,
                use_exact_branch_and_bound=use_exact_branch_and_bound,
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


class GalleyProfileTimes(TypedDict):
    """Time measurement for the compiler."""

    optimize_plan_s: float
    downstream_s: float


class GalleyLogicalOptimizer(LogicEvaluator):
    """
    Pipeline stage that optimizes logical Plans with the Galley greedy rewriter
    (or exact branch-and-bound when enabled), then forwards to an optional
    downstream LogicEvaluator (ctx).
    """

    def __init__(
        self,
        stats_factory: StatsFactory,
        ctx: LogicEvaluator | None = None,
        use_components: bool = True,
        profile: bool = False,
        *,
        use_exact_branch_and_bound: bool = True,
    ):
        self.stats_factory = stats_factory
        self.ctx = ctx
        self.use_components = use_components
        self.profile = profile
        self.use_exact_branch_and_bound = use_exact_branch_and_bound

    def __call__(self, prgm, bindings=None):
        if bindings is None:
            bindings = {}

        if isinstance(prgm, Plan):
            logger.debug("Optimizing plan: %s", prgm)
            t0 = time.perf_counter()
            prgm = optimize_plan(
                prgm,
                self.stats_factory,
                bindings,
                use_components=self.use_components,
                use_exact_branch_and_bound=self.use_exact_branch_and_bound,
            )
            t_opt = time.perf_counter() - t0
            if self.ctx is not None:
                t1 = time.perf_counter()
                out = self.ctx(prgm, bindings)
                t_down = time.perf_counter() - t1
            else:
                out = prgm
                t_down = 0.0
            times = GalleyProfileTimes(
                {
                    "optimize_plan_s": t_opt,
                    "downstream_s": t_down,
                }
            )
            #  End time
            if self.profile:
                return out, times
            return out
        raise ValueError(f"Unsupported program type: {type(prgm)}")

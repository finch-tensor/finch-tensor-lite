"""
Galley logical optimizer: applies greedy query rewriting to logical plans
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import TypedDict

from ..finch_logic import Alias, LogicEvaluator, Plan, Query
from .tensor_stats import TensorStats
from .galley.logical_optimizer.annotated_query import AnnotatedQuery
from .galley.logical_optimizer.greedy_optimizer import greedy_query
from .galley.logical_optimizer.logic_to_stats import insert_statistics
from .galley.logical_optimizer.query_normalization import (
    postprocess_plan_after_galley,
    preprocess_plan_for_galley,
)


def optimize_query(query, ST, stats_bindings, use_components: bool = True):
    """Rewrite a single logical Query via greedy reduction over reducible indices."""
    annotated_query = AnnotatedQuery(ST, query, stats_bindings)
    return greedy_query(annotated_query,  use_components=use_components)


def optimize_plan(plan, ST, bindings, use_components: bool = True):
    """
    Optimize a full Plan: run the Galley greedy optimizer on each Query body,
    pass through non-Query bodies (Produces), and update stats bindings.
    """
    # Preprocess the plan into the canonical form expected by AnnotatedQuery /
    # greedy_query.
    plan = preprocess_plan_for_galley(plan)
    # print("Preprocessed plan:")
    # print(plan)
    optimized_queries = []
    # Map alias -> tensor stats for cost/rewrite decisions
    stats_bindings: OrderedDict[Alias, TensorStats] = OrderedDict(
        (var, ST(T)) for var, T in bindings.items()
    )
    cache_dict: dict[object, TensorStats] = {}
    for body in plan.bodies:
        # Only put Queries through the greedy optimizer
        if isinstance(body, Query):
            new_queries = optimize_query(body, ST, stats_bindings, use_components=use_components)
            for new_query in new_queries:
                insert_statistics(
                    ST, new_query, stats_bindings, replace=True, cache=cache_dict
                )
            optimized_queries.extend(new_queries)
        else:
            # Produces(...)
            optimized_queries.append(body)

    return postprocess_plan_after_galley(Plan(tuple(optimized_queries)))


class GalleyProfileTimes(TypedDict):
    """Time measurement for the compiler."""

    optimize_plan_s: float
    downstream_s: float


class GalleyLogicalOptimizer(LogicEvaluator):
    """
    Pipeline stage that optimizes logical Plans with the Galley greedy rewriter,
    then forwards to an optional downstream LogicEvaluator (ctx)
    """

    def __init__(
        self,
        ST,
        ctx: LogicEvaluator | None = None,
        verbose: bool = False,
        use_components: bool = True,
        profile: bool = False,
    ):
        self.ST = ST
        self.ctx = ctx
        self.verbose = verbose
        self.use_components = use_components
        self.profile = profile

    def __call__(self, prgm, bindings=None):
        if bindings is None:
            bindings = {}

        if isinstance(prgm, Plan):
            if self.verbose:
                #print("Input plan:")
                #print(prgm)
                print("Filler")
            t0 = time.perf_counter()
            prgm = optimize_plan(
                prgm, self.ST, bindings, use_components=self.use_components
            )
            t_opt = time.perf_counter() - t0
            if self.ctx is not None:
                t1 = time.perf_counter()
                out = self.ctx(prgm, bindings)
                t_down = time.perf_counter() - t1
            else:
                out = prgm
                t_down = 0.0
            times: GalleyProfileTimes = {
                "optimize_plan_s": t_opt,
                "downstream_s": t_down,
            }
            #  End time
            if self.profile:
                return out, times
            return out
        # print("This probabiy should not happen")
        raise ValueError(f"Unsupported program type: {type(prgm)}")
"""
Galley logical optimizer: applies greedy query rewriting to logical plans
"""

from ..finch_logic import LogicEvaluator, Plan, Query
from ..galley.LogicalOptimizer.annotated_query import AnnotatedQuery
from ..galley.LogicalOptimizer.greedy_optimizer import greedy_query
from ..galley.LogicalOptimizer.logic_to_stats import insert_statistics
from ..galley.LogicalOptimizer.query_normalization import (
    postprocess_plan_after_galley,
    preprocess_plan_for_galley,
)


def optimize_query(query, ST, stats_bindings):
    """Rewrite a single logical Query via greedy reduction over reducible indices."""
    annotated_query = AnnotatedQuery(ST, query, stats_bindings)
    return greedy_query(annotated_query)


def optimize_plan(plan, ST, bindings):
    """
    Optimize a full Plan: run the Galley greedy optimizer on each Query body,
    pass through non-Query bodies (Produces), and update stats bindings.
    """
    # Preprocess the plan into the canonical form expected by AnnotatedQuery /
    # greedy_query.
    plan = preprocess_plan_for_galley(plan)
    optimized_queries = []
    # Map alias -> tensor stats for cost/rewrite decisions
    stats_bindings = {var: ST(T) for var, T in bindings.items()}
    cache_dict = {}
    for body in plan.bodies:
        # Only put Queries through the greedy optimizer
        if isinstance(body, Query):
            new_queries = optimize_query(body, ST, stats_bindings)
            for new_query in new_queries:
                insert_statistics(
                    ST, new_query, stats_bindings, replace=True, cache=cache_dict
                )
            optimized_queries.extend(new_queries)
        else:
            # Produces(...)
            optimized_queries.append(body)

    return postprocess_plan_after_galley(Plan(tuple(optimized_queries)))


class GalleyLogicalOptimizer(LogicEvaluator):
    """
    Pipeline stage that optimizes logical Plans with the Galley greedy rewriter,
    then forwards to an optional downstream LogicEvaluator (ctx)
    """

    def __init__(self, ST, ctx: LogicEvaluator | None = None, verbose: bool = False):
        self.ST = ST
        self.ctx = ctx

    def __call__(self, prgm, bindings=None):
        if bindings is None:
            bindings = {}

        if isinstance(prgm, Plan):
            prgm = optimize_plan(prgm, self.ST, bindings)
            if self.ctx is not None:
                return self.ctx(prgm, bindings)
            return prgm
        print("This probabiy should not happen")
        raise ValueError(f"Unsupported program type: {type(prgm)}")

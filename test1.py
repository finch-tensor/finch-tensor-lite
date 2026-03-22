"""
Print the program/plan for expr A@B + D@C to see Galley components in action.

Shows input plan, preprocessed plan, connected components, and optimized plan.
Works!
"""

import numpy as np

import finchlite.interface as fl_interface
from finchlite.autoschedule.galley.logical_optimizer.annotated_query import AnnotatedQuery
from finchlite.autoschedule.galley.logical_optimizer.logic_to_stats import insert_statistics
from finchlite.autoschedule.galley.logical_optimizer.query_normalization import (
    preprocess_plan_for_galley,
)
from finchlite.autoschedule.galley_optimize import optimize_plan
from finchlite.autoschedule.normalize import normalize_names
from finchlite.autoschedule.tensor_stats import DenseStats
from finchlite.finch_logic import Alias, Field, Plan, Produces, Query, Table
from finchlite.symbolic import gensym


def build_plan_from_expr(expr):
    """Build a Plan from a lazy expression (same as compute() does)."""
    args = (expr,) if not isinstance(expr, tuple) else expr
    vars = tuple(Alias(gensym("A")) for _ in args)
    ctx = args[0].ctx.join(*[x.ctx for x in args[1:]])
    bodies = tuple(
        Query(
            var,
            Table(
                arg.data,
                tuple(Field(gensym("i")) for _ in range(len(arg.shape))),
            ),
        )
        for arg, var in zip(args, vars, strict=True)
    )
    return Plan(ctx.trace() + bodies + (Produces(vars),))


def main():
    # Expression: A@B + D@C
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    D = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    expr = (
        fl_interface.lazy(A) @ fl_interface.lazy(B)
        + fl_interface.lazy(D) @ fl_interface.lazy(C)
    )

    # Build plan (no extract_tensors - plan has Literals, bindings empty)
    prgm = build_plan_from_expr(expr)
    root, bindings = normalize_names(prgm, {})

    print("Expression: A@B + D@C")
    print("=" * 60)
    print("1. Input plan (after normalize):")
    print("=" * 60)
    print(root)
    print()

    # Preprocess
    preprocessed = preprocess_plan_for_galley(root)
    print("=" * 60)
    print("2. Preprocessed plan (for Galley):")
    print("=" * 60)
    print(preprocessed)
    print()

    # Build stats via insert_statistics, then show connected components
    from collections import OrderedDict

    stats_bindings = OrderedDict()
    cache = {}
    for body in preprocessed.bodies:
        if isinstance(body, Query):
            insert_statistics(DenseStats, body, stats_bindings, replace=True, cache=cache)

    for i, body in enumerate(preprocessed.bodies):
        if isinstance(body, Query):
            aq = AnnotatedQuery(DenseStats, body, stats_bindings)
            print("=" * 60)
            print(f"3. Connected components (Query {i + 1}):")
            print("=" * 60)
            for j, comp in enumerate(aq.connected_components):
                print(f"  Component {j + 1}: {[idx.name for idx in comp]}")
            print()

    # Optimize with components
    opt_plan = optimize_plan(root, DenseStats, bindings, use_components=True)

    print("=" * 60)
    print("4. Optimized plan (with components=True):")
    print("=" * 60)
    print(opt_plan)
    print()

    # Compute and verify
    out = fl_interface.compute(expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
    expected = np.asarray(A) @ np.asarray(B) + np.asarray(D) @ np.asarray(C)
    out_arr = np.asarray(out)
    print("=" * 60)
    print("5. Result:")
    print("=" * 60)
    print("Output:  ", out_arr)
    print("Expected:", expected)
    print("Match:", np.allclose(out_arr, expected))


if __name__ == "__main__":
    main()

"""
Galley pruned optimizer BnB vs greedy: ``optimize_plan_s``, ``downstream_s``, and model
``cost`` per mode (exact BnB vs greedy).

TODO: Collapse query into a single constructor function.

NOTE: Despite lower cost, greedy is still faster than exact BnB.
BnB seems to be fusing queris togther more than greedy.



"""

from __future__ import annotations

import operator as op
from collections import OrderedDict
from functools import reduce

import numpy as np

from galley_compile_benchmarks import (
    _recursion_limit_ctx,
    chain10_shapes_benchmark,
    make_five_chain10_expr,
    plan_from_expr,
)

import finchlite as fl
import finchlite.interface as fl_interface
from finchlite.algebra import as_finch_operator
from finchlite.autoschedule import (
    DefaultLogicFormatter,
    LogicExecutor,
    LogicNormalizer,
    LogicStandardizer,
)
from finchlite.autoschedule.compiler import LogicCompiler
from finchlite.autoschedule.galley.logical_optimizer import AnnotatedQuery
from finchlite.autoschedule.galley.logical_optimizer.branch_and_bound import (
    pruned_query_to_plan,
)
from finchlite.autoschedule.galley_optimize import (
    GalleyLogicalOptimizer,
    GalleyProfileTimes,
)
from finchlite.autoschedule.tensor_stats import DenseStats
from finchlite.compile.lower import NotationCompiler
from finchlite.finch_assembly.interpreter import AssemblyInterpreter
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Query,
    Table,
)

DEFAULT_N = 1

_DOWNSTREAM = LogicExecutor(
    LogicStandardizer(
        DefaultLogicFormatter(LogicCompiler(NotationCompiler(AssemblyInterpreter())))
    )
)

GALLEY_PIPELINE_GREEDY = LogicNormalizer(
    GalleyLogicalOptimizer(
        DenseStats,
        _DOWNSTREAM,
        profile=True,
        use_exact_branch_and_bound=False,
    )
)

GALLEY_PIPELINE_EXACT_BNB = LogicNormalizer(
    GalleyLogicalOptimizer(
        DenseStats,
        _DOWNSTREAM,
        profile=True,
        use_exact_branch_and_bound=True,
    )
)


# --- Frontend expressions ---


def _four_index_chain_expr():
    A = fl.asarray(np.ones((3, 10)))
    B = fl.asarray(np.ones((10, 5)))
    C = fl.asarray(np.ones((5, 2)))
    return fl_interface.sum(
        fl_interface.lazy(A) @ fl_interface.lazy(B) @ fl_interface.lazy(C)
    )


def _three_index_chain_expr():
    A = fl.asarray(np.ones((4, 8)))
    B = fl.asarray(np.ones((8, 6)))
    return fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B))


def _chain_expr_from_shapes(shapes: list[tuple[int, int]]):
    mats = [fl.asarray(np.ones((r, c))) for r, c in shapes]
    return reduce(
        lambda a, b: a @ b,
        [fl_interface.lazy(m) for m in mats],
    )


def _skewed_four_matrix_expr():
    return _chain_expr_from_shapes(
        [(100, 5), (5, 1000), (1000, 3), (3, 50)],
    )


def _tapered_four_matrix_expr():
    return _chain_expr_from_shapes(
        [(1, 1000), (1000, 100), (100, 10), (10, 1)],
    )


base_n = 3
_BNB_GOOD_MATRIX_SHAPES: list[tuple[int, int]] = [
    (base_n**2, base_n**5),
    (base_n**5, base_n**3),
    (base_n**3, 1),
    (1, base_n**3),
    (base_n**3, base_n**5),
    (base_n**5, base_n**2),
]


def bnb_good_example():
    return _chain_expr_from_shapes(_BNB_GOOD_MATRIX_SHAPES)


# Four-matrix chain tuned so greedy cost is far above exact in both ratio (~1.2×)
# and absolute model cost (large gap vs the smaller skewed 4-matrix case above).
# Found by grid search on the same (a,5),(5,b),(b,c),(c,d) family as skewed.
_HEAVY_SKEW_FOUR_MATRIX_SHAPES: list[tuple[int, int]] = [
    (10, 5),
    (5, 10000),
    (10000, 3),
    (3, 10),
]


def _heavy_skew_four_matrix_expr():
    return _chain_expr_from_shapes(_HEAVY_SKEW_FOUR_MATRIX_SHAPES)


# --- Manual queries for AnnotatedQuery (cost lines) ---


def _four_index_chain_query() -> Query:
    A = fl.asarray(np.ones((3, 10)))
    B = fl.asarray(np.ones((10, 5)))
    C = fl.asarray(np.ones((5, 2)))
    return Query(
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


def _three_index_chain_query() -> Query:
    A = fl.asarray(np.ones((4, 8)))
    B = fl.asarray(np.ones((8, 6)))
    return Query(
        Alias("out"),
        Aggregate(
            Literal(as_finch_operator(op.add)),
            Literal(0),
            MapJoin(
                Literal(as_finch_operator(op.mul)),
                (
                    Table(Literal(A), (Field("i"), Field("j"))),
                    Table(Literal(B), (Field("j"), Field("k"))),
                ),
            ),
            (Field("i"), Field("j"), Field("k")),
        ),
    )


def _query_from_matmul_chain_shapes(
    shapes: list[tuple[int, int]],
    *,
    index_names: str | None = None,
) -> Query:
    n = len(shapes)
    names = index_names or "ijklmnopqrstuvwxyz"[: n + 1]
    assert len(names) == n + 1
    fields = [Field(names[i]) for i in range(n + 1)]
    tables = []
    for t, (r, c) in enumerate(shapes):
        arr = fl.asarray(np.ones((r, c)))
        tables.append(Table(Literal(arr), (fields[t], fields[t + 1])))
    return Query(
        Alias("out"),
        Aggregate(
            Literal(as_finch_operator(op.add)),
            Literal(0),
            MapJoin(Literal(as_finch_operator(op.mul)), tuple(tables)),
            tuple(fields[1:-1]),
        ),
    )


def _four_index_chain_aq() -> AnnotatedQuery:
    return AnnotatedQuery(DenseStats, _four_index_chain_query(), bindings=OrderedDict())


def _three_index_chain_aq() -> AnnotatedQuery:
    return AnnotatedQuery(
        DenseStats, _three_index_chain_query(), bindings=OrderedDict()
    )


def _skewed_four_matrix_aq() -> AnnotatedQuery:
    return AnnotatedQuery(
        DenseStats,
        _query_from_matmul_chain_shapes(
            [(100, 5), (5, 1000), (1000, 3), (3, 50)],
            index_names="ijklm",
        ),
        bindings=OrderedDict(),
    )


def _tapered_four_matrix_aq() -> AnnotatedQuery:
    return AnnotatedQuery(
        DenseStats,
        _query_from_matmul_chain_shapes(
            [(1, 1000), (1000, 100), (100, 10), (10, 1)],
            index_names="ijklm",
        ),
        bindings=OrderedDict(),
    )


def _heavy_skew_four_matrix_aq() -> AnnotatedQuery:
    return AnnotatedQuery(
        DenseStats,
        _query_from_matmul_chain_shapes(
            _HEAVY_SKEW_FOUR_MATRIX_SHAPES,
            index_names="ijklm",
        ),
        bindings=OrderedDict(),
    )


def bnb_good_aq() -> AnnotatedQuery:
    return AnnotatedQuery(
        DenseStats,
        _query_from_matmul_chain_shapes(
            _BNB_GOOD_MATRIX_SHAPES,
            index_names="ijklmnq",
        ),
        bindings=OrderedDict(),
    )


def _five_chain10_expr():
    """Five summed 10-matrix chains (same as ``make_five_chain10_expr`` benchmark)."""
    rng = np.random.default_rng(42)
    return make_five_chain10_expr(chain10_shapes_benchmark, rng)


def _query_five_chain10_matmul() -> Query:
    """
    Logical query for five 10-way matmul chains summed; outer dims ``i``, ``j`` shared.
    Shapes match ``chain10_shapes_benchmark`` per chain.
    """
    rng = np.random.default_rng(42)
    chain_exprs = []
    for k in range(5):
        chain_shapes = [tuple(m.shape) for m in chain10_shapes_benchmark(k, rng)]
        fields = [Field("i")] + [Field(f"c{k}_t{t}") for t in range(9)] + [Field("j")]
        tables = []
        for t, (r, c) in enumerate(chain_shapes):
            arr = fl.asarray(np.ones((r, c)))
            tables.append(Table(Literal(arr), (fields[t], fields[t + 1])))
        chain_exprs.append(
            Aggregate(
                Literal(as_finch_operator(op.add)),
                Literal(0),
                MapJoin(Literal(as_finch_operator(op.mul)), tuple(tables)),
                tuple(fields[1:-1]),
            )
        )
    return Query(
        Alias("out"),
        MapJoin(Literal(as_finch_operator(op.add)), tuple(chain_exprs)),
    )


def _five_chain10_matmul_aq() -> AnnotatedQuery:
    return AnnotatedQuery(
        DenseStats,
        _query_five_chain10_matmul(),
        bindings=OrderedDict(),
    )


def time_galley_bnb_compile_profile(
    expr,
    *,
    n: int = DEFAULT_N,
    recursion_limit: int | None = None,
) -> tuple[GalleyProfileTimes, GalleyProfileTimes]:
    """
    Average ``optimize_plan_s`` and ``downstream_s`` per iteration for exact BnB vs
    greedy Galley pipelines (same structure as ``time_compile_profile`` in
    ``galley_compile_benchmarks.py``).
    """
    bindings: dict = {}
    with _recursion_limit_ctx(recursion_limit):
        for _ in range(2):
            _ = GALLEY_PIPELINE_EXACT_BNB(plan_from_expr(expr), bindings)
            _ = GALLEY_PIPELINE_GREEDY(plan_from_expr(expr), bindings)

        opt_e = down_e = 0.0
        for _ in range(n):
            _, t = GALLEY_PIPELINE_EXACT_BNB(plan_from_expr(expr), bindings)
            opt_e += t["optimize_plan_s"]
            down_e += t["downstream_s"]
        exact_times: GalleyProfileTimes = {
            "optimize_plan_s": opt_e / n,
            "downstream_s": down_e / n,
        }

        opt_g = down_g = 0.0
        for _ in range(n):
            _, t = GALLEY_PIPELINE_GREEDY(plan_from_expr(expr), bindings)
            opt_g += t["optimize_plan_s"]
            down_g += t["downstream_s"]
        greedy_times: GalleyProfileTimes = {
            "optimize_plan_s": opt_g / n,
            "downstream_s": down_g / n,
        }

    return exact_times, greedy_times


def _format_bnb_block(
    title: str,
    exact_t: GalleyProfileTimes,
    greedy_t: GalleyProfileTimes,
    cost_exact: float,
    cost_greedy: float,
    nq_exact: int,
    nq_greedy: int,
) -> str:
    """Layout aligned with ``_format_block`` in ``galley_compile_benchmarks.py``."""
    lines = [
        "",
        "=" * 60,
        title,
        "  Exact BnB:",
        (
            f"    optimize_plan_s={exact_t['optimize_plan_s']:.6f}s  "
            f"downstream_s={exact_t['downstream_s']:.6f}s  "
            f"cost={cost_exact:g}  subqueries={nq_exact}"
        ),
        "  Greedy:",
        (
            f"    optimize_plan_s={greedy_t['optimize_plan_s']:.6f}s  "
            f"downstream_s={greedy_t['downstream_s']:.6f}s  "
            f"cost={cost_greedy:g}  subqueries={nq_greedy}"
        ),
        "=" * 60,
    ]
    return "\n".join(lines)


def main() -> None:

    expr = bnb_good_example()
    exact_t, greedy_t = time_galley_bnb_compile_profile(expr, n=DEFAULT_N)
    aq_e = bnb_good_aq()
    aq_g = bnb_good_aq()
    queries_exact, cost_exact = pruned_query_to_plan(aq_e, use_greedy=False)
    queries_greedy, cost_greedy = pruned_query_to_plan(aq_g, use_greedy=True)
    print(
        _format_bnb_block(
            "bnb-good example",
            exact_t,
            greedy_t,
            cost_exact,
            cost_greedy,
            len(queries_exact),
            len(queries_greedy),
        ),
        flush=True,
    )

    expr = _four_index_chain_expr()
    exact_t, greedy_t = time_galley_bnb_compile_profile(expr, n=DEFAULT_N)
    aq_e = _four_index_chain_aq()
    aq_g = _four_index_chain_aq()
    queries_exact, cost_exact = pruned_query_to_plan(aq_e, use_greedy=False)
    queries_greedy, cost_greedy = pruned_query_to_plan(aq_g, use_greedy=True)
    print(
        _format_bnb_block(
            "four-index chain",
            exact_t,
            greedy_t,
            cost_exact,
            cost_greedy,
            len(queries_exact),
            len(queries_greedy),
        ),
        flush=True,
    )

    expr = _three_index_chain_expr()
    exact_t, greedy_t = time_galley_bnb_compile_profile(expr, n=DEFAULT_N)
    aq_e = _three_index_chain_aq()
    aq_g = _three_index_chain_aq()
    queries_exact, cost_exact = pruned_query_to_plan(aq_e, use_greedy=False)
    queries_greedy, cost_greedy = pruned_query_to_plan(aq_g, use_greedy=True)
    print(
        _format_bnb_block(
            "three-index chain",
            exact_t,
            greedy_t,
            cost_exact,
            cost_greedy,
            len(queries_exact),
            len(queries_greedy),
        ),
        flush=True,
    )

    expr = _skewed_four_matrix_expr()
    exact_t, greedy_t = time_galley_bnb_compile_profile(expr, n=DEFAULT_N)
    aq_e = _skewed_four_matrix_aq()
    aq_g = _skewed_four_matrix_aq()
    queries_exact, cost_exact = pruned_query_to_plan(aq_e, use_greedy=False)
    queries_greedy, cost_greedy = pruned_query_to_plan(aq_g, use_greedy=True)
    print(
        _format_bnb_block(
            "skewed 4-matrix chain",
            exact_t,
            greedy_t,
            cost_exact,
            cost_greedy,
            len(queries_exact),
            len(queries_greedy),
        ),
        flush=True,
    )

    expr = _tapered_four_matrix_expr()
    exact_t, greedy_t = time_galley_bnb_compile_profile(expr, n=DEFAULT_N)
    aq_e = _tapered_four_matrix_aq()
    aq_g = _tapered_four_matrix_aq()
    queries_exact, cost_exact = pruned_query_to_plan(aq_e, use_greedy=False)
    queries_greedy, cost_greedy = pruned_query_to_plan(aq_g, use_greedy=True)
    print(
        _format_bnb_block(
            "tapered 4-matrix chain",
            exact_t,
            greedy_t,
            cost_exact,
            cost_greedy,
            len(queries_exact),
            len(queries_greedy),
        ),
        flush=True,
    )

    expr = _heavy_skew_four_matrix_expr()
    exact_t, greedy_t = time_galley_bnb_compile_profile(expr, n=DEFAULT_N)
    aq_e = _heavy_skew_four_matrix_aq()
    aq_g = _heavy_skew_four_matrix_aq()
    queries_exact, cost_exact = pruned_query_to_plan(aq_e, use_greedy=False)
    queries_greedy, cost_greedy = pruned_query_to_plan(aq_g, use_greedy=True)
    print(
        _format_bnb_block(
            "heavy skew 4-matrix chain",
            exact_t,
            greedy_t,
            cost_exact,
            cost_greedy,
            len(queries_exact),
            len(queries_greedy),
        ),
        flush=True,
    )

    expr = _five_chain10_expr()
    exact_t, greedy_t = time_galley_bnb_compile_profile(expr, n=DEFAULT_N)
    aq_e = _five_chain10_matmul_aq()
    aq_g = _five_chain10_matmul_aq()
    queries_exact, cost_exact = pruned_query_to_plan(aq_e, use_greedy=False)
    queries_greedy, cost_greedy = pruned_query_to_plan(aq_g, use_greedy=True)
    print(
        _format_bnb_block(
            "five chain10 terms",
            exact_t,
            greedy_t,
            cost_exact,
            cost_greedy,
            len(queries_exact),
            len(queries_greedy),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

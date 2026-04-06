"""
Unified Galley benchmarks (merged from ``galley_compile_benchmarks`` and
``galley_bnb_compile_benchmarks``).

**Compile profile:** ``optimize_plan_s`` / ``downstream_s`` with vs without
Galley connected components.

**BnB profile:** exact branch-and-bound vs greedy (``optimize_plan_s``,
``downstream_s``, model cost, subquery counts).

Run: ``poetry run python benchmarks/benchmarks.py``

Legacy entry points remain: ``galley_compile_benchmarks.py``,
``galley_bnb_compile_benchmarks.py`` (not removed).
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np

import finchlite as fl
import finchlite.interface as fl_interface
from finchlite.algebra import ffunc
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
from finchlite.autoschedule.tensor_stats import DenseStatsFactory
from finchlite.compile.lower import NotationCompiler
from finchlite.finch_assembly.interpreter import AssemblyInterpreter
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)
from finchlite.symbolic import gensym

# --- Defaults (compile uses more iterations; BnB uses 1 for heavy cases) ---

CHAIN_RECURSION_LIMIT = 4000
DEFAULT_COMPILE_PROFILE_N = 5
DEFAULT_BNB_PROFILE_N = 1

# Backwards-compatible names (match old modules)
DEFAULT_N = DEFAULT_COMPILE_PROFILE_N

# main() output (optional consolidation)
_BANNER_COMPILE_MAIN = "### Galley compile benchmarks (with vs without components) ###"
_BANNER_BNB_MAIN = "### Galley BnB vs greedy benchmarks ###"
_MSG_DONE_COMPILE = "Done compile benchmarks."
_MSG_DONE = "Done."

_DENSE_STATS_FACTORY = DenseStatsFactory()

# --- Downstream pipeline (shared structure) ---

_DOWNSTREAM = LogicExecutor(
    LogicStandardizer(
        DefaultLogicFormatter(LogicCompiler(NotationCompiler(AssemblyInterpreter())))
    )
)

GALLEY_COMPILE_PROFILE_WITH = LogicNormalizer(
    GalleyLogicalOptimizer(
        _DENSE_STATS_FACTORY,
        LogicExecutor(
            LogicStandardizer(
                DefaultLogicFormatter(
                    LogicCompiler(NotationCompiler(AssemblyInterpreter()))
                )
            )
        ),
        profile=True,
    )
)

GALLEY_COMPILE_PROFILE_WITHOUT = LogicNormalizer(
    GalleyLogicalOptimizer(
        _DENSE_STATS_FACTORY,
        LogicExecutor(
            LogicStandardizer(
                DefaultLogicFormatter(
                    LogicCompiler(NotationCompiler(AssemblyInterpreter()))
                )
            )
        ),
        use_components=False,
        profile=True,
    )
)

GALLEY_PIPELINE_GREEDY = LogicNormalizer(
    GalleyLogicalOptimizer(
        _DENSE_STATS_FACTORY,
        _DOWNSTREAM,
        profile=True,
        use_exact_branch_and_bound=False,
    )
)

GALLEY_PIPELINE_EXACT_BNB = LogicNormalizer(
    GalleyLogicalOptimizer(
        _DENSE_STATS_FACTORY,
        _DOWNSTREAM,
        profile=True,
        use_exact_branch_and_bound=True,
    )
)


@contextmanager
def _recursion_limit_ctx(limit: int | None):
    if limit is None:
        yield
        return
    old = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield
    finally:
        sys.setrecursionlimit(old)


def plan_from_expr(arg):
    """Build the same `Plan` as `finchlite.interface.fuse.compute`."""
    args = arg if isinstance(arg, tuple) else (arg,)
    vars_ = tuple(Alias(gensym("A")) for _ in args)
    ctx_2 = args[0].ctx.join(*[x.ctx for x in args[1:]])
    bodies = tuple(
        map(
            lambda a, var: Query(
                var,
                Table(a.data, tuple(Field(gensym("i")) for _ in range(len(a.shape)))),
            ),
            args,
            vars_,
        )
    )
    return Plan(ctx_2.trace() + bodies + (Produces(vars_),))


# =============================================================================
# Expression builders (shared / compile benchmarks)
# =============================================================================


def _default_rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    """Benchmarks use a fixed seed unless a generator is passed in."""
    return rng if rng is not None else np.random.default_rng(42)


def _lazy_matmul_chain(mats: Iterable) -> object:
    """Reduce matmul over ``fl_interface.lazy`` tensors from an iterable of arrays."""
    return reduce(
        lambda a, b: a @ b,
        [fl_interface.lazy(m) for m in mats],
    )


def make_summed_matmul_chains_expr(
    n_terms: int,
    shape_fn: Callable[[int, np.random.Generator], tuple],
    rng: np.random.Generator | None = None,
) -> object:
    """
    Sum of ``n_terms`` independent matmul chains; chain ``k`` from ``shape_fn(k, rng)``.
    Used for multi-chain benchmarks (e.g. three/five/50 terms × chainN).
    """
    rng = _default_rng(rng)
    terms = [_lazy_matmul_chain(shape_fn(k, rng)) for k in range(n_terms)]
    return reduce(lambda a, b: a + b, terms)


def make_chain10_expr(
    shape_fn: Callable[[int, np.random.Generator], tuple],
    rng: np.random.Generator | None = None,
) -> object:
    """Two terms: each is a 10-way matmul chain (A@B@...@Z) + (A1@B1@...@Z1)."""
    rng = _default_rng(rng)
    chain0 = shape_fn(0, rng)
    chain1 = shape_fn(1, rng)
    return _lazy_matmul_chain(chain0) + _lazy_matmul_chain(chain1)


def make_three_matmul_pairs_expr():
    """Three summed matmul pairs: A@B + C@D + E@F (20×10 @ 10×8 → 20×8)."""
    A = fl_interface.asarray(np.arange(20 * 10).reshape(20, 10).astype(float))
    B = fl_interface.asarray(np.arange(10 * 8).reshape(10, 8).astype(float))
    C = fl_interface.asarray(np.arange(20 * 10).reshape(20, 10).astype(float))
    D = fl_interface.asarray(np.arange(10 * 8).reshape(10, 8).astype(float))
    E = fl_interface.asarray(np.arange(20 * 10).reshape(20, 10).astype(float))
    F = fl_interface.asarray(np.arange(10 * 8).reshape(10, 8).astype(float))
    return (
        fl_interface.lazy(A) @ fl_interface.lazy(B)
        + fl_interface.lazy(C) @ fl_interface.lazy(D)
        + fl_interface.lazy(E) @ fl_interface.lazy(F)
    )


def make_fifty_matmul_pairs_expr():
    """Fifty summed matmul pairs distinct data per pair."""
    terms: list = []
    for k in range(50):
        off = float(k * 1_000)
        A = fl_interface.asarray(np.arange(20 * 10).reshape(20, 10).astype(float) + off)
        B = fl_interface.asarray(np.arange(10 * 8).reshape(10, 8).astype(float) + off)
        terms.append(fl_interface.lazy(A) @ fl_interface.lazy(B))
    return reduce(lambda a, b: a + b, terms)


def chain2_shapes_small(i, rng):
    """Two matrices per term: (4,5)@(5,8) -> (4,8)."""
    r = np.random.Generator(np.random.PCG64(42 + i * 1000))
    return (
        fl_interface.asarray(r.standard_normal((4, 5)).astype(float)),
        fl_interface.asarray(r.standard_normal((5, 8)).astype(float)),
    )


def chain2_shapes_benchmark(i, rng):
    """Two matrices per term: (8,10)@(10,16) -> (8,16)."""
    r = np.random.Generator(np.random.PCG64(42 + i * 1000))
    return (
        fl_interface.asarray(r.standard_normal((8, 10)).astype(float)),
        fl_interface.asarray(r.standard_normal((10, 16)).astype(float)),
    )


def make_fifty_chain2_terms_expr(
    shape_fn: Callable[[int, np.random.Generator], tuple],
    rng: np.random.Generator | None = None,
) -> object:
    """Fifty summed terms; each term is a 2-matrix matmul chain."""
    return make_summed_matmul_chains_expr(50, shape_fn, rng)


def make_three_chain10_expr(
    shape_fn: Callable[[int, np.random.Generator], tuple],
    rng: np.random.Generator | None = None,
) -> object:
    """Three summed chain10 terms: chain10_0 + chain10_1 + chain10_2."""
    return make_summed_matmul_chains_expr(3, shape_fn, rng)


def make_three_chain25_expr(
    shape_fn: Callable[[int, np.random.Generator], tuple],
    rng: np.random.Generator | None = None,
) -> object:
    """Three summed chain25 terms: chain25_0 + chain25_1 + chain25_2."""
    return make_summed_matmul_chains_expr(3, shape_fn, rng)


def make_five_chain10_expr(
    shape_fn: Callable[[int, np.random.Generator], tuple],
    rng: np.random.Generator | None = None,
) -> object:
    """Five summed chain10 terms: chain10_0 through chain10_4."""
    return make_summed_matmul_chains_expr(5, shape_fn, rng)


def chain10_shapes_small(i, rng):
    """10 matrices: (4,5)@(5,6)@...@(13,8) -> (4,8)."""
    r = np.random.Generator(np.random.PCG64(42 + i * 1000))
    return (
        fl_interface.asarray(r.standard_normal((4, 5)).astype(float)),
        fl_interface.asarray(r.standard_normal((5, 6)).astype(float)),
        fl_interface.asarray(r.standard_normal((6, 7)).astype(float)),
        fl_interface.asarray(r.standard_normal((7, 8)).astype(float)),
        fl_interface.asarray(r.standard_normal((8, 9)).astype(float)),
        fl_interface.asarray(r.standard_normal((9, 10)).astype(float)),
        fl_interface.asarray(r.standard_normal((10, 11)).astype(float)),
        fl_interface.asarray(r.standard_normal((11, 12)).astype(float)),
        fl_interface.asarray(r.standard_normal((12, 13)).astype(float)),
        fl_interface.asarray(r.standard_normal((13, 8)).astype(float)),
    )


def chain10_shapes_benchmark(i, rng):
    """10 matrices for benchmarks: (8,10)@(10,12)@...@(26,16) -> (8,16)."""
    r = np.random.Generator(np.random.PCG64(42 + i * 1000))
    return (
        fl_interface.asarray(r.standard_normal((8, 10)).astype(float)),
        fl_interface.asarray(r.standard_normal((10, 12)).astype(float)),
        fl_interface.asarray(r.standard_normal((12, 14)).astype(float)),
        fl_interface.asarray(r.standard_normal((14, 16)).astype(float)),
        fl_interface.asarray(r.standard_normal((16, 18)).astype(float)),
        fl_interface.asarray(r.standard_normal((18, 20)).astype(float)),
        fl_interface.asarray(r.standard_normal((20, 22)).astype(float)),
        fl_interface.asarray(r.standard_normal((22, 24)).astype(float)),
        fl_interface.asarray(r.standard_normal((24, 26)).astype(float)),
        fl_interface.asarray(r.standard_normal((26, 16)).astype(float)),
    )


def make_chain25_expr(
    shape_fn: Callable[[int, np.random.Generator], tuple],
    rng: np.random.Generator | None = None,
) -> object:
    """Build expr = (A@B@...@Z) + (A1@B1@...): 2 terms, each a 25-way matmul chain."""
    rng = _default_rng(rng)
    chain0 = shape_fn(0, rng)
    chain1 = shape_fn(1, rng)
    return _lazy_matmul_chain(chain0) + _lazy_matmul_chain(chain1)


def chain25_shapes_small(i, rng):
    """25 matrices: (4,5) @ (5,5)*23 @ (5,8) -> (4,8)."""
    r = np.random.Generator(np.random.PCG64(42 + i * 1000))
    mats = [
        fl_interface.asarray(r.standard_normal((4, 5)).astype(float)),
    ]
    mats.extend(
        fl_interface.asarray(r.standard_normal((5, 5)).astype(float)) for _ in range(23)
    )
    mats.append(fl_interface.asarray(r.standard_normal((5, 8)).astype(float)))
    return tuple(mats)


def chain25_shapes_benchmark(i, rng):
    """25 matrices for benchmarks: (8,8) throughout -> (8,8)."""
    r = np.random.Generator(np.random.PCG64(42 + i * 1000))
    return tuple(
        fl_interface.asarray(r.standard_normal((8, 8)).astype(float)) for _ in range(25)
    )


def make_sum_sum_benchmark_expr():
    """Large sum(A@B)+sum(C@D) expression used for frontend timing."""
    A = fl_interface.asarray(np.arange(100 * 50).reshape(100, 50).astype(float))
    B = fl_interface.asarray(np.arange(50 * 20).reshape(50, 20).astype(float))
    C = fl_interface.asarray(np.arange(100 * 50).reshape(100, 50).astype(float))
    D = fl_interface.asarray(np.arange(50 * 20).reshape(50, 20).astype(float))
    return fl_interface.sum(
        fl_interface.lazy(A) @ fl_interface.lazy(B), axis=1
    ) + fl_interface.sum(fl_interface.lazy(C) @ fl_interface.lazy(D), axis=1)


# =============================================================================
# BnB-specific frontend expressions and manual queries
# =============================================================================


def make_chain_expr_from_shapes(shapes: list[tuple[int, int]]) -> object:
    """Matmul chain from explicit matrix shapes (no outer sum); uses ones data."""
    mats = [fl.asarray(np.ones((r, c))) for r, c in shapes]
    return _lazy_matmul_chain(mats)


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
            Literal(ffunc.add),
            Literal(0),
            MapJoin(Literal(ffunc.mul), tuple(tables)),
            tuple(fields[1:-1]),
        ),
    )


_FOUR_INDEX_CHAIN_SHAPES: list[tuple[int, int]] = [(3, 10), (10, 5), (5, 2)]
_THREE_INDEX_CHAIN_SHAPES: list[tuple[int, int]] = [(4, 8), (8, 6)]


def make_four_index_chain_expr() -> object:
    """Sum over a 3-matrix chain (frontend + sum)."""
    return fl_interface.sum(make_chain_expr_from_shapes(_FOUR_INDEX_CHAIN_SHAPES))


def make_three_index_chain_expr() -> object:
    """Sum over a 2-matrix chain (frontend + sum)."""
    return fl_interface.sum(make_chain_expr_from_shapes(_THREE_INDEX_CHAIN_SHAPES))


_SKEWED_FOUR_MATRIX_SHAPES: list[tuple[int, int]] = [
    (100, 5),
    (5, 1000),
    (1000, 3),
    (3, 50),
]
_TAPERED_FOUR_MATRIX_SHAPES: list[tuple[int, int]] = [
    (1, 1000),
    (1000, 100),
    (100, 10),
    (10, 1),
]


def make_skewed_four_matrix_expr() -> object:
    return make_chain_expr_from_shapes(_SKEWED_FOUR_MATRIX_SHAPES)


def make_tapered_four_matrix_expr() -> object:
    return make_chain_expr_from_shapes(_TAPERED_FOUR_MATRIX_SHAPES)


_BNB_GOOD_BASE_N = 3
_BNB_GOOD_MATRIX_SHAPES: list[tuple[int, int]] = [
    (_BNB_GOOD_BASE_N**2, _BNB_GOOD_BASE_N**5),
    (_BNB_GOOD_BASE_N**5, _BNB_GOOD_BASE_N**3),
    (_BNB_GOOD_BASE_N**3, 1),
    (1, _BNB_GOOD_BASE_N**3),
    (_BNB_GOOD_BASE_N**3, _BNB_GOOD_BASE_N**5),
    (_BNB_GOOD_BASE_N**5, _BNB_GOOD_BASE_N**2),
]


def make_bnb_good_example() -> object:
    return make_chain_expr_from_shapes(_BNB_GOOD_MATRIX_SHAPES)


_HEAVY_SKEW_FOUR_MATRIX_SHAPES: list[tuple[int, int]] = [
    (10, 5),
    (5, 10000),
    (10000, 3),
    (3, 10),
]


def make_heavy_skew_four_matrix_expr() -> object:
    return make_chain_expr_from_shapes(_HEAVY_SKEW_FOUR_MATRIX_SHAPES)


def _four_index_chain_query() -> Query:
    return _query_from_matmul_chain_shapes(_FOUR_INDEX_CHAIN_SHAPES, index_names="ijkl")


def _three_index_chain_query() -> Query:
    return _query_from_matmul_chain_shapes(_THREE_INDEX_CHAIN_SHAPES, index_names="ijk")


def _annotated_query(q: Query) -> AnnotatedQuery:
    return AnnotatedQuery(_DENSE_STATS_FACTORY, q, bindings=OrderedDict())


def _four_index_chain_aq() -> AnnotatedQuery:
    return _annotated_query(_four_index_chain_query())


def _three_index_chain_aq() -> AnnotatedQuery:
    return _annotated_query(_three_index_chain_query())


def _skewed_four_matrix_aq() -> AnnotatedQuery:
    return _annotated_query(
        _query_from_matmul_chain_shapes(_SKEWED_FOUR_MATRIX_SHAPES, index_names="ijklm")
    )


def _tapered_four_matrix_aq() -> AnnotatedQuery:
    return _annotated_query(
        _query_from_matmul_chain_shapes(
            _TAPERED_FOUR_MATRIX_SHAPES, index_names="ijklm"
        )
    )


def _heavy_skew_four_matrix_aq() -> AnnotatedQuery:
    return _annotated_query(
        _query_from_matmul_chain_shapes(
            _HEAVY_SKEW_FOUR_MATRIX_SHAPES,
            index_names="ijklm",
        )
    )


def bnb_good_aq() -> AnnotatedQuery:
    return _annotated_query(
        _query_from_matmul_chain_shapes(
            _BNB_GOOD_MATRIX_SHAPES,
            index_names="ijklmnq",
        )
    )


def _query_five_chain10_matmul() -> Query:
    """
    Logical query for five 10-way matmul chains summed; outer dims ``i``, ``j`` shared.
    Shapes match ``chain10_shapes_benchmark`` per chain.
    """
    rng = _default_rng()
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
                Literal(ffunc.add),
                Literal(0),
                MapJoin(Literal(ffunc.mul), tuple(tables)),
                tuple(fields[1:-1]),
            )
        )
    return Query(
        Alias("out"),
        MapJoin(Literal(ffunc.add), tuple(chain_exprs)),
    )


def _five_chain10_matmul_aq() -> AnnotatedQuery:
    return _annotated_query(_query_five_chain10_matmul())


# =============================================================================
# Timing helpers
# =============================================================================


def _time_profile_pair(
    run_first: Callable[[], tuple[Any, GalleyProfileTimes]],
    run_second: Callable[[], tuple[Any, GalleyProfileTimes]],
    *,
    n: int,
    recursion_limit: int | None,
) -> tuple[GalleyProfileTimes, GalleyProfileTimes]:
    with _recursion_limit_ctx(recursion_limit):
        for _ in range(2):
            run_first()
            run_second()
        opt_a = down_a = 0.0
        for _ in range(n):
            _, t = run_first()
            opt_a += t["optimize_plan_s"]
            down_a += t["downstream_s"]
        first: GalleyProfileTimes = {
            "optimize_plan_s": opt_a / n,
            "downstream_s": down_a / n,
        }
        opt_b = down_b = 0.0
        for _ in range(n):
            _, t = run_second()
            opt_b += t["optimize_plan_s"]
            down_b += t["downstream_s"]
        second: GalleyProfileTimes = {
            "optimize_plan_s": opt_b / n,
            "downstream_s": down_b / n,
        }
        return first, second


def time_compile_profile(
    expr,
    *,
    n: int = DEFAULT_COMPILE_PROFILE_N,
    recursion_limit: int | None = None,
) -> tuple[GalleyProfileTimes, GalleyProfileTimes]:
    """
    Average ``optimize_plan_s`` and ``downstream_s`` per iteration for pipelines
    with and without Galley components.
    """

    def run_with() -> tuple[Any, GalleyProfileTimes]:
        return GALLEY_COMPILE_PROFILE_WITH(plan_from_expr(expr))

    def run_without() -> tuple[Any, GalleyProfileTimes]:
        return GALLEY_COMPILE_PROFILE_WITHOUT(plan_from_expr(expr))

    return _time_profile_pair(
        run_with, run_without, n=n, recursion_limit=recursion_limit
    )


def time_galley_bnb_compile_profile(
    expr,
    *,
    n: int = DEFAULT_BNB_PROFILE_N,
    recursion_limit: int | None = None,
) -> tuple[GalleyProfileTimes, GalleyProfileTimes]:
    """
    Average ``optimize_plan_s`` and ``downstream_s`` per iteration for exact BnB vs
    greedy Galley pipelines.
    """
    bindings: dict = {}

    def run_exact() -> tuple[Any, GalleyProfileTimes]:
        return GALLEY_PIPELINE_EXACT_BNB(plan_from_expr(expr), bindings)

    def run_greedy() -> tuple[Any, GalleyProfileTimes]:
        return GALLEY_PIPELINE_GREEDY(plan_from_expr(expr), bindings)

    return _time_profile_pair(
        run_exact, run_greedy, n=n, recursion_limit=recursion_limit
    )


def _print_profile_comparison(
    title: str,
    rows: list[tuple[str, GalleyProfileTimes, tuple[float, int] | None]],
) -> None:
    print("", flush=True)
    print("=" * 60, flush=True)
    print(title, flush=True)
    for label, times, extras in rows:
        print(f"  {label}:", flush=True)
        line = (
            f"    optimize_plan_s={times['optimize_plan_s']:.6f}s  "
            f"downstream_s={times['downstream_s']:.6f}s"
        )
        if extras is not None:
            cost, nq = extras
            line += f"  cost={cost:g}  subqueries={nq}"
        print(line, flush=True)
    print("=" * 60, flush=True)


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    """Shared benchmark case.

    Compile rows set ``aq_factory=None``; BnB sets ``recursion_limit=None``.
    """

    slug: str
    title: str
    build_expr: Callable[[], object]
    recursion_limit: int | None = None
    aq_factory: Callable[[], AnnotatedQuery] | None = None


def _exact_greedy_plan_stats(
    aq_factory: Callable[[], AnnotatedQuery],
) -> tuple[float, float, int, int]:
    aq_e = aq_factory()
    aq_g = aq_factory()
    queries_exact, cost_exact = pruned_query_to_plan(aq_e, use_greedy=False)
    queries_greedy, cost_greedy = pruned_query_to_plan(aq_g, use_greedy=True)
    return cost_exact, cost_greedy, len(queries_exact), len(queries_greedy)


def _run_compile_case(case: BenchmarkCaseSpec) -> None:
    if case.aq_factory is not None:
        raise ValueError("compile case must not set aq_factory")
    print(f"Compile benchmark: {case.slug}...", flush=True)
    expr = case.build_expr()
    w, wo = time_compile_profile(expr, recursion_limit=case.recursion_limit)
    _print_profile_comparison(
        case.title,
        [
            ("With components", w, None),
            ("Without components", wo, None),
        ],
    )


def _run_bnb_case(case: BenchmarkCaseSpec) -> None:
    if case.aq_factory is None:
        raise ValueError("BnB case must set aq_factory")
    if case.recursion_limit is not None:
        raise ValueError("BnB case must use recursion_limit=None")
    print(f"BnB benchmark: {case.slug}...", flush=True)
    expr = case.build_expr()
    exact_t, greedy_t = time_galley_bnb_compile_profile(expr, n=DEFAULT_BNB_PROFILE_N)
    cost_e, cost_g, nq_e, nq_g = _exact_greedy_plan_stats(case.aq_factory)
    _print_profile_comparison(
        case.title,
        [
            ("Exact BnB", exact_t, (cost_e, nq_e)),
            ("Greedy", greedy_t, (cost_g, nq_g)),
        ],
    )


def _compile_benchmark_cases(rng: np.random.Generator) -> tuple[BenchmarkCaseSpec, ...]:
    return (
        BenchmarkCaseSpec(
            slug="sum+sum matmul",
            title="Galley compile profile (sum+sum matmul)",
            build_expr=make_sum_sum_benchmark_expr,
        ),
        BenchmarkCaseSpec(
            slug="chain10",
            title="Galley compile profile (chain10)",
            build_expr=lambda: make_chain10_expr(chain10_shapes_benchmark, rng),
        ),
        BenchmarkCaseSpec(
            slug="three summed matmul pairs",
            title="Galley compile profile (three matmul pairs)",
            build_expr=make_three_matmul_pairs_expr,
        ),
        BenchmarkCaseSpec(
            slug="fifty terms × chain2",
            title="Galley compile profile (fifty terms × chain2)",
            build_expr=lambda: make_fifty_chain2_terms_expr(
                chain2_shapes_benchmark, rng
            ),
        ),
        BenchmarkCaseSpec(
            slug="three terms × chain10",
            title="Galley compile profile (three terms × chain10)",
            build_expr=lambda: make_three_chain10_expr(chain10_shapes_benchmark, rng),
        ),
        BenchmarkCaseSpec(
            slug="three terms × chain25",
            title="Galley compile profile (three terms × chain25)",
            build_expr=lambda: make_three_chain25_expr(chain25_shapes_benchmark, rng),
            recursion_limit=CHAIN_RECURSION_LIMIT,
        ),
        BenchmarkCaseSpec(
            slug="five terms × chain10",
            title="Galley compile profile (five terms × chain10)",
            build_expr=lambda: make_five_chain10_expr(chain10_shapes_benchmark, rng),
        ),
        BenchmarkCaseSpec(
            slug="chain25",
            title="Galley compile profile (chain25)",
            build_expr=lambda: make_chain25_expr(chain25_shapes_benchmark, rng),
            recursion_limit=CHAIN_RECURSION_LIMIT,
        ),
    )


BNB_CASES: tuple[BenchmarkCaseSpec, ...] = (
    BenchmarkCaseSpec(
        slug="bnb-good example",
        title="bnb-good example",
        build_expr=make_bnb_good_example,
        aq_factory=bnb_good_aq,
    ),
    BenchmarkCaseSpec(
        slug="four-index chain",
        title="four-index chain",
        build_expr=make_four_index_chain_expr,
        aq_factory=_four_index_chain_aq,
    ),
    BenchmarkCaseSpec(
        slug="three-index chain",
        title="three-index chain",
        build_expr=make_three_index_chain_expr,
        aq_factory=_three_index_chain_aq,
    ),
    BenchmarkCaseSpec(
        slug="skewed 4-matrix chain",
        title="skewed 4-matrix chain",
        build_expr=make_skewed_four_matrix_expr,
        aq_factory=_skewed_four_matrix_aq,
    ),
    BenchmarkCaseSpec(
        slug="tapered 4-matrix chain",
        title="tapered 4-matrix chain",
        build_expr=make_tapered_four_matrix_expr,
        aq_factory=_tapered_four_matrix_aq,
    ),
    BenchmarkCaseSpec(
        slug="heavy skew 4-matrix chain",
        title="heavy skew 4-matrix chain",
        build_expr=make_heavy_skew_four_matrix_expr,
        aq_factory=_heavy_skew_four_matrix_aq,
    ),
    BenchmarkCaseSpec(
        slug="five chain10 terms",
        title="five chain10 terms",
        build_expr=lambda: make_five_chain10_expr(
            chain10_shapes_benchmark, _default_rng()
        ),
        aq_factory=_five_chain10_matmul_aq,
    ),
)


# =============================================================================
# main sections (all tests from both original files)
# =============================================================================


def main_compile_benchmarks() -> None:
    """Galley compile profile: with vs without components."""
    rng = np.random.default_rng(42)
    for case in _compile_benchmark_cases(rng):
        _run_compile_case(case)


def main_bnb_benchmarks() -> None:
    """Galley exact BnB vs greedy (cost + subqueries)."""
    for case in BNB_CASES:
        _run_bnb_case(case)


def main() -> None:
    print("", flush=True)
    print(_BANNER_COMPILE_MAIN, flush=True)
    main_compile_benchmarks()
    print("", flush=True)
    print(_MSG_DONE_COMPILE, flush=True)
    print("", flush=True)
    print(_BANNER_BNB_MAIN, flush=True)
    main_bnb_benchmarks()
    print("", flush=True)
    print(_MSG_DONE, flush=True)


if __name__ == "__main__":
    main()

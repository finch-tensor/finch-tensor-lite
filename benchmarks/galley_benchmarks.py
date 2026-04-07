"""
Galley benchmarks: compile profile (with vs without connected components)
and exact branch-and-bound vs greedy.

Run: ``poetry run python benchmarks/galley_benchmarks.py``
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np

import finchlite as fl
import finchlite.interface as fl_interface
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
from finchlite.autoschedule.galley.logical_optimizer.query_normalization import (
    preprocess_plan_for_galley,
)
from finchlite.autoschedule.galley_optimize import (
    GalleyLogicalOptimizer,
    GalleyProfileTimes,
)
from finchlite.autoschedule.tensor_stats import DenseStatsFactory
from finchlite.compile.lower import NotationCompiler
from finchlite.finch_assembly.interpreter import AssemblyInterpreter
from finchlite.finch_logic import Alias, Field, Plan, Produces, Query, Table
from finchlite.interface.lazy import LazyTensor
from finchlite.symbolic import gensym

CHAIN_RECURSION_LIMIT = 4000
DEFAULT_COMPILE_PROFILE_N = 5
DEFAULT_BNB_PROFILE_N = 1

_DOWNSTREAM = LogicExecutor(
    LogicStandardizer(
        DefaultLogicFormatter(LogicCompiler(NotationCompiler(AssemblyInterpreter())))
    )
)

GALLEY_COMPILE_PROFILE_WITH = LogicNormalizer(
    GalleyLogicalOptimizer(
        DenseStatsFactory(),
        _DOWNSTREAM,
        profile=True,
    )
)

GALLEY_COMPILE_PROFILE_WITHOUT = LogicNormalizer(
    GalleyLogicalOptimizer(
        DenseStatsFactory(),
        _DOWNSTREAM,
        use_components=False,
        profile=True,
    )
)

GALLEY_PIPELINE_GREEDY = LogicNormalizer(
    GalleyLogicalOptimizer(
        DenseStatsFactory(),
        _DOWNSTREAM,
        profile=True,
        use_exact_branch_and_bound=False,
    )
)

GALLEY_PIPELINE_EXACT_BNB = LogicNormalizer(
    GalleyLogicalOptimizer(
        DenseStatsFactory(),
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


def _lazy_matmul_chain(mats: Iterable) -> LazyTensor:
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


def chain5_shapes_benchmark(i, rng):
    """5 matrices for benchmarks: (8,8) throughout -> (8,8)."""
    r = np.random.Generator(np.random.PCG64(42 + i * 1000))
    return tuple(
        fl_interface.asarray(r.standard_normal((8, 8)).astype(float)) for _ in range(5)
    )


# =============================================================================
# BnB-specific frontend expressions
# =============================================================================


def make_chain_expr_from_shapes(shapes: list[tuple[int, int]]) -> LazyTensor:
    """Matmul chain from explicit matrix shapes (no outer sum); uses ones data."""
    mats = [fl.asarray(np.ones((r, c))) for r, c in shapes]
    return _lazy_matmul_chain(mats)


def _query_from_lazy_expr(expr) -> Query:
    """One merged logical ``Query`` after ``preprocess_plan_for_galley``."""
    plan = preprocess_plan_for_galley(plan_from_expr(expr))
    queries = [b for b in plan.bodies if isinstance(b, Query)]
    if len(queries) != 1:
        raise ValueError(
            "expected exactly one merged Query in preprocessed plan for BnB; "
            f"got {len(queries)}"
        )
    return queries[0]


def _annotated_query_from_lazy_expr(expr) -> AnnotatedQuery:
    return AnnotatedQuery(
        DenseStatsFactory(),
        _query_from_lazy_expr(expr),
        bindings={},
    )


_BNB_GOOD_BASE_N = 3
_BNB_GOOD_MATRIX_SHAPES: list[tuple[int, int]] = [
    (_BNB_GOOD_BASE_N**2, _BNB_GOOD_BASE_N**5),
    (_BNB_GOOD_BASE_N**5, _BNB_GOOD_BASE_N**3),
    (_BNB_GOOD_BASE_N**3, 1),
    (1, _BNB_GOOD_BASE_N**3),
    (_BNB_GOOD_BASE_N**3, _BNB_GOOD_BASE_N**5),
    (_BNB_GOOD_BASE_N**5, _BNB_GOOD_BASE_N**2),
]


def make_bnb_benefits_example() -> LazyTensor:
    return make_chain_expr_from_shapes(_BNB_GOOD_MATRIX_SHAPES)


def make_bnb_slow_example() -> LazyTensor:
    return make_chain_expr_from_shapes([(5, 5) for _ in range(12)])


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

    Compile rows may set ``recursion_limit``; BnB derives ``AnnotatedQuery`` from
    ``build_expr()`` via ``preprocess_plan_for_galley(plan_from_expr(...))``.
    """

    title: str
    build_expr: Callable[[], object]
    recursion_limit: int | None = None


def _exact_greedy_plan_stats(
    aq_factory: Callable[[], AnnotatedQuery],
) -> tuple[float, float, int, int]:
    aq_e = aq_factory()
    aq_g = aq_factory()
    queries_exact, cost_exact = pruned_query_to_plan(aq_e, use_greedy=False)
    queries_greedy, cost_greedy = pruned_query_to_plan(aq_g, use_greedy=True)
    return cost_exact, cost_greedy, len(queries_exact), len(queries_greedy)


def _run_compile_case(case: BenchmarkCaseSpec) -> None:
    print(f"Compile benchmark: {case.title}...", flush=True)
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
    if case.recursion_limit is not None:
        raise ValueError("BnB case must use recursion_limit=None")
    print(f"BnB benchmark: {case.title}...", flush=True)
    expr = case.build_expr()
    exact_t, greedy_t = time_galley_bnb_compile_profile(expr, n=DEFAULT_BNB_PROFILE_N)
    cost_e, cost_g, nq_e, nq_g = _exact_greedy_plan_stats(
        lambda: _annotated_query_from_lazy_expr(expr)
    )
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
            title="Galley compile profile (12 x chain2)",
            build_expr=lambda: make_summed_matmul_chains_expr(
                12, chain2_shapes_benchmark, rng
            ),
        ),
        BenchmarkCaseSpec(
            title="Galley compile profile (3 x chain5)",
            build_expr=lambda: make_summed_matmul_chains_expr(
                3, chain5_shapes_benchmark, rng
            ),
            recursion_limit=CHAIN_RECURSION_LIMIT,
        ),
    )


BNB_CASES: tuple[BenchmarkCaseSpec, ...] = (
    BenchmarkCaseSpec(
        title="bnb-benefits example",
        build_expr=make_bnb_benefits_example,
    ),
    BenchmarkCaseSpec(
        title="bnb-slow example",
        build_expr=make_bnb_slow_example,
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
    print("### Galley compile benchmarks (with vs without components) ###", flush=True)
    main_compile_benchmarks()
    print("", flush=True)
    print("Done compile benchmarks.", flush=True)
    print("", flush=True)
    print("### Galley BnB vs greedy benchmarks ###", flush=True)
    main_bnb_benchmarks()
    print("", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

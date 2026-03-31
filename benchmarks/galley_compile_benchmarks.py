"""
Compile-path benchmark: Galley `optimize_plan` vs downstream pipeline timing
with and without components (`GalleyLogicalOptimizer` profile: optimize_plan_s,
downstream_s).

optimize_plan_s: time to optimize the plan in Galley
downstream_s: time in the downstream pipeline after optimize (e.g. rest of compile)
Does not take into account time to make a plan before optimize is called.

Uses the same expressions and ordering as galley_component_benchmarks.main().

Maybe remove downstream timing and change file to compile only

"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import reduce

import numpy as np

import finchlite.interface as fl_interface
from finchlite.autoschedule import (
    DefaultLogicFormatter,
    LogicExecutor,
    LogicNormalizer,
    LogicStandardizer,
)
from finchlite.autoschedule.compiler import LogicCompiler
from finchlite.autoschedule.galley_optimize import (
    GalleyLogicalOptimizer,
    GalleyProfileTimes,
)
from finchlite.autoschedule.tensor_stats import DenseStats
from finchlite.compile.lower import NotationCompiler
from finchlite.finch_assembly.interpreter import AssemblyInterpreter
from finchlite.finch_logic import Alias, Field, Plan, Produces, Query, Table
from finchlite.symbolic import gensym

CHAIN_RECURSION_LIMIT = 4000
DEFAULT_N = 5

GALLEY_COMPILE_PROFILE_WITH = LogicNormalizer(
    GalleyLogicalOptimizer(
        DenseStats,
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
        DenseStats,
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


# --- Expression builders ---


def make_chain10_expr(shape_fn, rng=None):
    """Two terms: each is a 10-way matmul chain (A@B@...@Z) + (A1@B1@...@Z1)."""
    if rng is None:
        rng = np.random.default_rng(42)
    chain0 = shape_fn(0, rng)
    chain1 = shape_fn(1, rng)
    term0 = reduce(
        lambda a, b: a @ b,
        [fl_interface.lazy(m) for m in chain0],
    )
    term1 = reduce(
        lambda a, b: a @ b,
        [fl_interface.lazy(m) for m in chain1],
    )
    return term0 + term1


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


# 2 Chain, 50 terms


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


def make_fifty_chain2_terms_expr(shape_fn, rng=None):
    """Fifty summed terms; each term is a 2-matrix matmul chain."""
    if rng is None:
        rng = np.random.default_rng(42)
    terms = []
    for k in range(50):
        chain = shape_fn(k, rng)
        t = reduce(
            lambda a, b: a @ b,
            [fl_interface.lazy(m) for m in chain],
        )
        terms.append(t)
    return reduce(lambda a, b: a + b, terms)


def make_three_chain10_expr(shape_fn, rng=None):
    """Three summed chain10 terms: chain10_0 + chain10_1 + chain10_2."""
    if rng is None:
        rng = np.random.default_rng(42)

    terms = []
    for k in range(3):
        chain = shape_fn(k, rng)
        term = reduce(
            lambda a, b: a @ b,
            [fl_interface.lazy(m) for m in chain],
        )
        terms.append(term)

    return terms[0] + terms[1] + terms[2]


def make_three_chain25_expr(shape_fn, rng=None):
    """Three summed chain25 terms: chain25_0 + chain25_1 + chain25_2."""
    if rng is None:
        rng = np.random.default_rng(42)
    terms = []
    for k in range(3):
        chain = shape_fn(k, rng)
        term = reduce(
            lambda a, b: a @ b,
            [fl_interface.lazy(m) for m in chain],
        )
        terms.append(term)
    return terms[0] + terms[1] + terms[2]


def make_five_chain10_expr(shape_fn, rng=None):
    """Five summed chain10 terms: chain10_0 through chain10_4."""
    if rng is None:
        rng = np.random.default_rng(42)

    terms = []
    for k in range(5):
        chain = shape_fn(k, rng)
        term = reduce(
            lambda a, b: a @ b,
            [fl_interface.lazy(m) for m in chain],
        )
        terms.append(term)

    # Force a strictly binary + tree at the Python level.
    # (May still get flattened by later lowering, but this minimizes n-ary + creation.)
    return terms[0] + terms[1] + terms[2] + terms[3] + terms[4]


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


def make_chain25_expr(shape_fn, rng=None):
    """Build expr = (A@B@...@Z) + (A1@B1@...): 2 terms, each a 25-way matmul chain."""
    if rng is None:
        rng = np.random.default_rng(42)
    chain0 = shape_fn(0, rng)
    chain1 = shape_fn(1, rng)
    term0 = reduce(
        lambda a, b: a @ b,
        [fl_interface.lazy(m) for m in chain0],
    )
    term1 = reduce(
        lambda a, b: a @ b,
        [fl_interface.lazy(m) for m in chain1],
    )
    return term0 + term1


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


def time_compile_profile(
    expr,
    *,
    n: int = DEFAULT_N,
    recursion_limit: int | None = None,
) -> tuple[GalleyProfileTimes, GalleyProfileTimes]:
    """
    Average `optimize_plan_s` and `downstream_s` per iteration for pipelines
    with and without Galley components.

    t is the structure that holds optimize_plan_s and downstream_s
    """
    with _recursion_limit_ctx(recursion_limit):
        for _ in range(2):
            _, _ = GALLEY_COMPILE_PROFILE_WITH(plan_from_expr(expr))
            _, _ = GALLEY_COMPILE_PROFILE_WITHOUT(plan_from_expr(expr))

        opt_w = down_w = 0.0
        for _ in range(n):
            _, t = GALLEY_COMPILE_PROFILE_WITH(plan_from_expr(expr))
            opt_w += t["optimize_plan_s"]
            down_w += t["downstream_s"]
        with_times: GalleyProfileTimes = {
            "optimize_plan_s": opt_w / n,
            "downstream_s": down_w / n,
        }

        opt_wo = down_wo = 0.0
        for _ in range(n):
            _, t = GALLEY_COMPILE_PROFILE_WITHOUT(plan_from_expr(expr))
            opt_wo += t["optimize_plan_s"]
            down_wo += t["downstream_s"]
        without_times: GalleyProfileTimes = {
            "optimize_plan_s": opt_wo / n,
            "downstream_s": down_wo / n,
        }

    return with_times, without_times


def _format_block(
    title: str, with_t: GalleyProfileTimes, without_t: GalleyProfileTimes
) -> str:
    lines = [
        "",
        "=" * 60,
        title,
        "  With components:",
        (
            f"    optimize_plan_s={with_t['optimize_plan_s']:.6f}s  "
            f"downstream_s={with_t['downstream_s']:.6f}s"
        ),
        "  Without components:",
        (
            f"    optimize_plan_s={without_t['optimize_plan_s']:.6f}s  "
            f"downstream_s={without_t['downstream_s']:.6f}s"
        ),
        "=" * 60,
    ]
    return "\n".join(lines)


def main() -> None:
    rng = np.random.default_rng(42)

    print("Compile benchmark: sum+sum matmul...", flush=True)
    expr_sum = make_sum_sum_benchmark_expr()
    w, wo = time_compile_profile(expr_sum)
    print(_format_block("Galley compile profile (sum+sum matmul)", w, wo), flush=True)

    print("Compile benchmark: chain10...", flush=True)
    expr_c10 = make_chain10_expr(chain10_shapes_benchmark, rng)
    w, wo = time_compile_profile(expr_c10)
    print(_format_block("Galley compile profile (chain10)", w, wo), flush=True)

    print("Compile benchmark: three summed matmul pairs...", flush=True)
    expr_3p = make_three_matmul_pairs_expr()
    w, wo = time_compile_profile(expr_3p)
    print(
        _format_block("Galley compile profile (three matmul pairs)", w, wo),
        flush=True,
    )

    print("Compile benchmark: fifty terms × chain2...", flush=True)
    expr_50c2 = make_fifty_chain2_terms_expr(chain2_shapes_benchmark, rng)
    w, wo = time_compile_profile(expr_50c2)
    print(
        _format_block("Galley compile profile (fifty terms × chain2)", w, wo),
        flush=True,
    )

    print("Compile benchmark: three terms × chain10...", flush=True)
    expr_3c10 = make_three_chain10_expr(chain10_shapes_benchmark, rng)
    w, wo = time_compile_profile(expr_3c10)
    print(
        _format_block("Galley compile profile (three terms × chain10)", w, wo),
        flush=True,
    )

    print("Compile benchmark: three terms × chain25...", flush=True)
    expr_3c25 = make_three_chain25_expr(chain25_shapes_benchmark, rng)
    w, wo = time_compile_profile(expr_3c25, recursion_limit=CHAIN_RECURSION_LIMIT)
    print(
        _format_block("Galley compile profile (three terms × chain25)", w, wo),
        flush=True,
    )

    print("Compile benchmark: five terms × chain10...", flush=True)
    expr_5c10 = make_five_chain10_expr(chain10_shapes_benchmark, rng)
    w, wo = time_compile_profile(expr_5c10)
    print(
        _format_block("Galley compile profile (five terms × chain10)", w, wo),
        flush=True,
    )

    print("Compile benchmark: chain25...", flush=True)
    expr_c25 = make_chain25_expr(chain25_shapes_benchmark, rng)
    w, wo = time_compile_profile(expr_c25, recursion_limit=CHAIN_RECURSION_LIMIT)
    print(_format_block("Galley compile profile (chain25)", w, wo), flush=True)

    print("", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

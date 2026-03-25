"""
Benchmark script comparing Galley with and without components.

Run: python tests/test_galley_benchmarks.py

NOTE:
Setting recursion limit to more than default for deep chains.
Iterations default to n=5 for averaging.


NOTE: where parameters were added: delete when done testing.
greedy_optimizer.py	greedy_query
galley_optimize.py	optimize_query
galley_optimize.py	optimize_plan
galley_optimize.py	GalleyLogicalOptimizer
test_galley_benchmarks.py	INTERPRET_NOTATION_GALLEY_NO_COMPONENTS

"""

from __future__ import annotations

import sys
import time
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
from finchlite.autoschedule.galley_optimize import GalleyLogicalOptimizer
from finchlite.autoschedule.normalize import normalize_names
from finchlite.autoschedule.tensor_stats import DenseStats
from finchlite.finch_logic import Alias, Field, Plan, Produces, Query, Table
from finchlite.finch_notation.interpreter import NotationInterpreter
from finchlite.symbolic import gensym

DEFAULT_N = 5
CHAIN_RECURSION_LIMIT = 4000

# TESTING PIPELINES 
INTERPRET_NOTATION_GALLEY_NO_COMPONENTS = LogicNormalizer(
    GalleyLogicalOptimizer(
        DenseStats,
        LogicExecutor(
            LogicStandardizer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        ),
        use_components=False,
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


def time_frontend_compute(
    expr, *, n: int = DEFAULT_N, recursion_limit: int | None = None
) -> tuple[float, float]:
    """
    Time full `compute()` with and without Galley components.

    Returns ``(components_with, components_without)`` — seconds per iteration.
    """
    with _recursion_limit_ctx(recursion_limit):
        t0 = time.perf_counter()
        for _ in range(n):
            fl_interface.compute(expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
        components_with = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for _ in range(n):
            fl_interface.compute(expr, ctx=INTERPRET_NOTATION_GALLEY_NO_COMPONENTS)
        components_without = (time.perf_counter() - t0) / n
    return components_with, components_without


def benchmark_compile_kernel(
    expr, *, n: int = DEFAULT_N, recursion_limit: int | None = None
) -> tuple[float, float, float, float]:
    """
    Time Galley optimize_plan vs downstream (notation codegen), with/without components.

    Uses ``GalleyLogicalOptimizer(..., profile=True)`` so timings match the real pipeline.

    Returns (compile_with_components, compile_without_components,
             kernel_with_components, kernel_without_components) seconds per iter.
    """
    with _recursion_limit_ctx(recursion_limit):
        prgm = build_plan_from_expr(expr)
        root, bindings = normalize_names(prgm, {})

        downstream = LogicExecutor(
            LogicStandardizer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        )

        def avg_profiled(use_components: bool) -> tuple[float, float]:
            opt = GalleyLogicalOptimizer(
                DenseStats,
                downstream,
                use_components=use_components,
                profile=True,
            )
            tot_optimize = 0.0
            tot_downstream = 0.0
            for _ in range(n):
                _out, times = opt(root, bindings)
                tot_optimize += times["optimize_plan_s"]
                tot_downstream += times["downstream_s"]
            return tot_optimize / n, tot_downstream / n

        compile_with_components, kernel_with_components = avg_profiled(True)
        compile_without_components, kernel_without_components = avg_profiled(False)

    return (
        compile_with_components,
        compile_without_components,
        kernel_with_components,
        kernel_without_components,
    )


def build_plan_from_expr(expr):
    """Replicate compute() plan building for benchmark use."""
    args = (expr,) if not isinstance(expr, tuple) else expr
    vars = tuple(Alias(gensym("A")) for _ in args)
    ctx_2 = args[0].ctx.join(*[x.ctx for x in args[1:]])
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
    return Plan(ctx_2.trace() + bodies + (Produces(vars),))


# --- Expression builders ---


def make_chain10_expr(shape_fn, rng=None):
    """Build expr = (A@B@...@Z) + (A1@B1@...@Z1): 2 terms, each a 10-way matmul chain."""
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
    for _ in range(23):
        mats.append(fl_interface.asarray(r.standard_normal((5, 5)).astype(float)))
    mats.append(fl_interface.asarray(r.standard_normal((5, 8)).astype(float)))
    return tuple(mats)


def chain25_shapes_benchmark(i, rng):
    """25 matrices for benchmarks: (8,8) throughout -> (8,8)."""
    r = np.random.Generator(np.random.PCG64(42 + i * 1000))
    return tuple(
        fl_interface.asarray(r.standard_normal((8, 8)).astype(float))
        for _ in range(25)
    )


def make_sum_sum_benchmark_expr():
    """Large sum(A@B)+sum(C@D) expression used for frontend timing."""
    A = fl_interface.asarray(np.arange(100 * 50).reshape(100, 50).astype(float))
    B = fl_interface.asarray(np.arange(50 * 20).reshape(50, 20).astype(float))
    C = fl_interface.asarray(np.arange(100 * 50).reshape(100, 50).astype(float))
    D = fl_interface.asarray(np.arange(50 * 20).reshape(50, 20).astype(float))
    return (
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=1)
        + fl_interface.sum(fl_interface.lazy(C) @ fl_interface.lazy(D), axis=1)
    )


def make_matmul_three_terms_compile_expr():
    """Smaller A@B + C@D + E@F for compile/kernel benchmark."""
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


def run_smoke_computes() -> None:
    """Quick compute smoke paths (no timing)."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    D = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    E = fl_interface.asarray(np.array([[2.0, 0.0], [0.0, 2.0]]))
    F = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    three = (
        fl_interface.lazy(A) @ fl_interface.lazy(B)
        + fl_interface.lazy(C) @ fl_interface.lazy(D)
        + fl_interface.lazy(E) @ fl_interface.lazy(F)
    )
    _ = fl_interface.compute(three, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
    _ = fl_interface.compute(three, ctx=INTERPRET_NOTATION_GALLEY_NO_COMPONENTS)

    rng = np.random.default_rng(42)
    e10 = make_chain10_expr(chain10_shapes_small, rng)
    _ = fl_interface.compute(e10, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
    _ = fl_interface.compute(e10, ctx=INTERPRET_NOTATION_GALLEY_NO_COMPONENTS)

    with _recursion_limit_ctx(CHAIN_RECURSION_LIMIT):
        e25 = make_chain25_expr(chain25_shapes_small, rng)
        _ = fl_interface.compute(e25, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
        _ = fl_interface.compute(e25, ctx=INTERPRET_NOTATION_GALLEY_NO_COMPONENTS)


def main() -> None:
    lines: list[str] = []

    lines.append("Smoke compute paths...")
    run_smoke_computes()

    lines.append("Frontend benchmark: sum+sum matmul...")
    expr_sum = make_sum_sum_benchmark_expr()
    components_with, components_without = time_frontend_compute(expr_sum)
    lines.append("")
    lines.append("=" * 60)
    lines.append("Galley benchmark results (sum+sum matmul)")
    lines.append(
        f"  frontend components vs no-components: "
        f"With components={components_with:.4f}s, Without={components_without:.4f}s"
    )
    lines.append("=" * 60)

    lines.append("Frontend benchmark: chain10...")
    rng = np.random.default_rng(42)
    expr_c10 = make_chain10_expr(chain10_shapes_benchmark, rng)
    components_with, components_without = time_frontend_compute(expr_c10)
    lines.append("")
    lines.append("=" * 60)
    lines.append("Galley chain10 frontend benchmark:")
    lines.append(f"  With components:   {components_with:.4f}s")
    lines.append(f"  Without components: {components_without:.4f}s")
    lines.append("=" * 60)

    lines.append("Frontend benchmark: chain25...")
    expr_c25 = make_chain25_expr(chain25_shapes_benchmark, rng)
    components_with, components_without = time_frontend_compute(
        expr_c25, recursion_limit=CHAIN_RECURSION_LIMIT
    )
    lines.append("")
    lines.append("=" * 60)
    lines.append("Galley chain25 frontend benchmark:")
    lines.append(f"  With components:   {components_with:.4f}s")
    lines.append(f"  Without components: {components_without:.4f}s")
    lines.append("=" * 60)

    lines.append("Compile vs kernel: matmul three terms...")
    expr_mk = make_matmul_three_terms_compile_expr()
    (
        compile_with_components,
        compile_without_components,
        kernel_with_components,
        kernel_without_components,
    ) = benchmark_compile_kernel(expr_mk)
    lines.append("")
    lines.append("=" * 60)
    lines.append("Galley compile vs kernel benchmark (matmul three terms):")
    lines.append(f"  Compile (with components):    {compile_with_components:.4f}s")
    lines.append(f"  Compile (without components):  {compile_without_components:.4f}s")
    lines.append(f"  Kernel (with components):     {kernel_with_components:.4f}s")
    lines.append(f"  Kernel (without components):   {kernel_without_components:.4f}s")
    lines.append("=" * 60)

    lines.append("Compile vs kernel: chain10...")
    expr_c10_small = make_chain10_expr(chain10_shapes_small, rng)
    (
        compile_with_components,
        compile_without_components,
        kernel_with_components,
        kernel_without_components,
    ) = benchmark_compile_kernel(expr_c10_small)
    lines.append("")
    lines.append("=" * 60)
    lines.append("Galley chain10 compile vs kernel benchmark:")
    lines.append(
        f"  Compile (with):    {compile_with_components:.4f}s  "
        f"(without): {compile_without_components:.4f}s"
    )
    lines.append(
        f"  Kernel (with):     {kernel_with_components:.4f}s  "
        f"(without): {kernel_without_components:.4f}s"
    )
    lines.append("=" * 60)

    lines.append("Compile vs kernel: chain25...")
    expr_c25_small = make_chain25_expr(chain25_shapes_small, rng)
    (
        compile_with_components,
        compile_without_components,
        kernel_with_components,
        kernel_without_components,
    ) = benchmark_compile_kernel(
        expr_c25_small, recursion_limit=CHAIN_RECURSION_LIMIT
    )
    lines.append("")
    lines.append("=" * 60)
    lines.append("Galley chain25 compile vs kernel benchmark:")
    lines.append(
        f"  Compile (with):    {compile_with_components:.4f}s  "
        f"(without): {compile_without_components:.4f}s"
    )
    lines.append(
        f"  Kernel (with):     {kernel_with_components:.4f}s  "
        f"(without): {kernel_without_components:.4f}s"
    )
    lines.append("=" * 60)

    lines.append("")
    lines.append("Done.")

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
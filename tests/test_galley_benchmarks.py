"""
Benchmark tests comparing greedy_query runtime with and without components.
Output is written to test_galley_benchmark_log.txt in the tests directory.

NOTE:
Setting recursion limit to more than default. (careful)
Only currently ruuning 1 time. Change n to get average.
Changed 100 chain test to 25 chain test.

============================================================
Galley chain10 frontend benchmark:
  With components:   31.0386s
  Without components: 44.5000s
============================================================

============================================================
Galley chain25 frontend benchmark:
  With components:   27.7452s
  Without components: 14.8611s
============================================================

Maybe non-frontend test is wrong somehow.
Run more than 1 time to get average


NOTE: where paramteres were added: delete when done testing.
greedy_optimizer.py	greedy_query	
galley_optimize.py	optimize_query	
galley_optimize.py	optimize_plan	
galley_optimize.py	GalleyLogicalOptimizer	
fuse.py	INTERPRET_NOTATION_GALLEY_NO_COMPONENTS

TODO: add A@B@C@D@E + ... (10 terms) benchmark after n-ary.



"""

import operator as op
import sys
import time
from collections import OrderedDict
from functools import reduce
from pathlib import Path

BENCHMARK_LOG_PATH = Path(__file__).parent / "test_galley_benchmark_log.txt"
_benchmark_log: list[str] = []


def write_log():
    """Write _benchmark_log to the log file."""
    with open(BENCHMARK_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_benchmark_log))


import pytest
import finchlite.interface as fl_interface

import numpy as np

import finchlite as fl
from finchlite.algebra import as_finch_operator

# Saved benchmark times, printed at end of test_frontend_benchmark_components_vs_no_components
_benchmark_times = []
from finchlite.autoschedule.galley.logical_optimizer import (
    AnnotatedQuery,
    greedy_query,
    insert_statistics,
)
from finchlite.autoschedule.galley_optimize import optimize_plan
from finchlite.autoschedule.normalize import normalize_names
from finchlite.autoschedule.tensor_stats import DenseStats
from finchlite.autoschedule import DefaultLogicFormatter, LogicExecutor, LogicStandardizer
from finchlite.autoschedule.compiler import LogicCompiler
from finchlite.finch_logic import (
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)
from finchlite.finch_notation.interpreter import NotationInterpreter
from finchlite.symbolic import gensym

def test_frontend_with_components():
    """Run sum(A@B, axis=0) + sum(C@D, axis=1) with components."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    D = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=0)
        + fl_interface.sum(fl_interface.lazy(C) @ fl_interface.lazy(D), axis=1),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )
    expected = np.sum(np.array(A) @ np.array(B), axis=0) + np.sum(np.array(C) @ np.array(D), axis=1)
    assert np.allclose(np.array(out), np.array(expected))

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
    """10 matrices: (4,5)@(5,6)@(6,7)@(7,8)@(8,9)@(9,10)@(10,11)@(11,12)@(12,13)@(13,8) -> (4,8)."""
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


def chain10_expected(shape_fn, rng=None):
    """NumPy reference for chain10: (A@B@...@Z) + (A1@B1@...@Z1)."""
    if rng is None:
        rng = np.random.default_rng(42)
    mats0 = [np.array(m) for m in shape_fn(0, rng)]
    mats1 = [np.array(m) for m in shape_fn(1, rng)]
    term0 = mats0[0]
    for m in mats0[1:]:
        term0 = term0 @ m
    term1 = mats1[0]
    for m in mats1[1:]:
        term1 = term1 @ m
    return term0 + term1


# --- Chain25: (25-way matmul) + (25-way matmul) ---


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


def chain25_expected(shape_fn, rng=None):
    """NumPy reference for chain25: (A@B@...@Z) + (A1@B1@...)."""
    if rng is None:
        rng = np.random.default_rng(42)
    mats0 = [np.array(m) for m in shape_fn(0, rng)]
    mats1 = [np.array(m) for m in shape_fn(1, rng)]
    term0 = mats0[0]
    for m in mats0[1:]:
        term0 = term0 @ m
    term1 = mats1[0]
    for m in mats1[1:]:
        term1 = term1 @ m
    return term0 + term1


def test_frontend_chain25_with_components():
    """Run (25-way chain) + (25-way chain) with components."""
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(4000)  # Deep 25-matmul tree needs more stack
        rng = np.random.default_rng(42)
        expr = make_chain25_expr(chain25_shapes_small, rng)
        out = fl_interface.compute(expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
        expected = chain25_expected(chain25_shapes_small, rng)
        assert np.allclose(np.array(out), np.array(expected))
    finally:
        sys.setrecursionlimit(old_limit)


def test_frontend_chain25_without_components():
    """Run (25-way chain) + (25-way chain) without components."""
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(4000)
        rng = np.random.default_rng(42)
        expr = make_chain25_expr(chain25_shapes_small, rng)
        out = fl_interface.compute(
            expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY_NO_COMPONENTS
        )
        expected = chain25_expected(chain25_shapes_small, rng)
        assert np.allclose(np.array(out), np.array(expected))
    finally:
        sys.setrecursionlimit(old_limit)


def test_frontend_chain10_with_components():
    """Run A@B@C@D@E + ... (10 terms) with components."""
    rng = np.random.default_rng(42)
    expr = make_chain10_expr(chain10_shapes_small, rng)
    out = fl_interface.compute(expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
    expected = chain10_expected(chain10_shapes_small, rng)
    assert np.allclose(np.array(out), np.array(expected))


def test_frontend_chain10_without_components():
    """Run (A@B@...@Z) + (A1@B1@...@Z1) without components."""
    rng = np.random.default_rng(42)
    expr = make_chain10_expr(chain10_shapes_small, rng)
    out = fl_interface.compute(
        expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY_NO_COMPONENTS
    )
    expected = chain10_expected(chain10_shapes_small, rng)
    assert np.allclose(np.array(out), np.array(expected))


def test_frontend_without_components():
    """Same computation with use_components=False; result must match."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    D = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=0)
        + fl_interface.sum(fl_interface.lazy(C) @ fl_interface.lazy(D), axis=1),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY_NO_COMPONENTS,
    )
    expected = np.sum(np.array(A) @ np.array(B), axis=0) + np.sum(np.array(C) @ np.array(D), axis=1)
    assert np.allclose(np.array(out), np.array(expected))
    
def test_frontend_benchmark_components_vs_no_components():
    """Benchmark frontend with components vs without."""
    A = fl_interface.asarray(np.arange(100 * 50).reshape(100, 50).astype(float))
    B = fl_interface.asarray(np.arange(50 * 20).reshape(50, 20).astype(float))
    C = fl_interface.asarray(np.arange(100 * 50).reshape(100, 50).astype(float))
    D = fl_interface.asarray(np.arange(50 * 20).reshape(50, 20).astype(float))
    expr = (
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=1)
        + fl_interface.sum(fl_interface.lazy(C) @ fl_interface.lazy(D), axis=1)
    )

    n = 5
    # Benchmark WITH components
    t0 = time.perf_counter()
    for _ in range(n):
        fl_interface.compute(expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
    t_with = (time.perf_counter() - t0) / n

    # Benchmark WITHOUT components
    t0 = time.perf_counter()
    for _ in range(n):
        fl_interface.compute(expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY_NO_COMPONENTS)
    t_without = (time.perf_counter() - t0) / n

    _benchmark_times.append(("frontend components vs no-components", t_with, t_without))
    _benchmark_log.append("")
    _benchmark_log.append("=" * 60)
    _benchmark_log.append("Galley benchmark results:")
    for name, tw, two in _benchmark_times:
        _benchmark_log.append(f"  {name}: With components={tw:.4f}s, Without={two:.4f}s")
    _benchmark_log.append("=" * 60)
    write_log()


def test_frontend_benchmark_chain10():
    """Benchmark frontend: A@B@C@D@E + ... (10 terms) with/without components."""
    rng = np.random.default_rng(42)
    expr = make_chain10_expr(chain10_shapes_benchmark, rng)

    n = 2
    t0 = time.perf_counter()
    for _ in range(n):
        fl_interface.compute(expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
    t_with = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for _ in range(n):
        fl_interface.compute(
            expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY_NO_COMPONENTS
        )
    t_without = (time.perf_counter() - t0) / n

    _benchmark_times.append(
        ("frontend chain10 (2× 10-way matmul)", t_with, t_without)
    )
    _benchmark_log.append("")
    _benchmark_log.append("=" * 60)
    _benchmark_log.append("Galley chain10 frontend benchmark:")
    _benchmark_log.append(f"  With components:   {t_with:.4f}s")
    _benchmark_log.append(f"  Without components: {t_without:.4f}s")
    _benchmark_log.append("=" * 60)
    write_log()


def test_frontend_benchmark_chain25():
    """Benchmark frontend: (25-way chain) + (25-way chain) with/without components."""
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(4000)
        rng = np.random.default_rng(42)
        expr = make_chain25_expr(chain25_shapes_benchmark, rng)

        n = 1  # Long chain - single run
        t0 = time.perf_counter()
        for _ in range(n):
            fl_interface.compute(expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY)
        t_with = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for _ in range(n):
            fl_interface.compute(
                expr, ctx=fl_interface.INTERPRET_NOTATION_GALLEY_NO_COMPONENTS
            )
        t_without = (time.perf_counter() - t0) / n

        _benchmark_times.append(
            ("frontend chain25 (2× 25-way matmul)", t_with, t_without)
        )
        _benchmark_log.append("")
        _benchmark_log.append("=" * 60)
        _benchmark_log.append("Galley chain25 frontend benchmark:")
        _benchmark_log.append(f"  With components:   {t_with:.4f}s")
        _benchmark_log.append(f"  Without components: {t_without:.4f}s")
        _benchmark_log.append("=" * 60)
        write_log()
    finally:
        sys.setrecursionlimit(old_limit)


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


def test_compile_vs_kernel_benchmark():
    """Measure compile time (Galley optimization) and kernel time separately."""
    # Use smaller matrices for faster benchmark
    A = fl_interface.asarray(np.arange(20 * 10).reshape(20, 10).astype(float))
    B = fl_interface.asarray(np.arange(10 * 8).reshape(10, 8).astype(float))
    C = fl_interface.asarray(np.arange(20 * 10).reshape(20, 10).astype(float))
    D = fl_interface.asarray(np.arange(10 * 8).reshape(10, 8).astype(float))
    expr = (
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=1)
        + fl_interface.sum(fl_interface.lazy(C) @ fl_interface.lazy(D), axis=1)
    )

    prgm = build_plan_from_expr(expr)
    root, bindings = normalize_names(prgm, {})

    downstream = LogicExecutor(
        LogicStandardizer(
            DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
        )
    )

    n = 2

    # Compile time: optimize_plan with components
    t0 = time.perf_counter()
    for _ in range(n):
        opt_plan = optimize_plan(root, DenseStats, bindings, use_components=True)
    t_compile_with = (time.perf_counter() - t0) / n

    # Compile time: optimize_plan without components
    t0 = time.perf_counter()
    for _ in range(n):
        opt_plan = optimize_plan(root, DenseStats, bindings, use_components=False)
    t_compile_without = (time.perf_counter() - t0) / n

    # Kernel time: run downstream on optimized plan (use components version)
    opt_plan_with = optimize_plan(root, DenseStats, bindings, use_components=True)
    downstream(opt_plan_with, bindings)  # warmup (JIT compile)
    t0 = time.perf_counter()
    for _ in range(n):
        downstream(opt_plan_with, bindings)
    t_kernel_with = (time.perf_counter() - t0) / n

    # Kernel time: run downstream on optimized plan (no components version)
    opt_plan_without = optimize_plan(root, DenseStats, bindings, use_components=False)
    downstream(opt_plan_without, bindings)  # warmup (JIT compile)
    t0 = time.perf_counter()
    for _ in range(n):
        downstream(opt_plan_without, bindings)
    t_kernel_without = (time.perf_counter() - t0) / n

    _benchmark_times.append(
        ("compile (optimize_plan)", t_compile_with, t_compile_without)
    )
    _benchmark_times.append(
        ("kernel (downstream execution)", t_kernel_with, t_kernel_without)
    )
    _benchmark_log.append("")
    _benchmark_log.append("=" * 60)
    _benchmark_log.append("Galley compile vs kernel benchmark:")
    _benchmark_log.append(f"  Compile (with components):    {t_compile_with:.4f}s")
    _benchmark_log.append(f"  Compile (without components):  {t_compile_without:.4f}s")
    _benchmark_log.append(f"  Kernel (with components):     {t_kernel_with:.4f}s")
    _benchmark_log.append(f"  Kernel (without components):   {t_kernel_without:.4f}s")
    _benchmark_log.append("=" * 60)
    write_log()


def test_compile_vs_kernel_benchmark_chain10():
    """Compile and kernel benchmarks for A@B@C@D@E + ... (10 terms)."""
    rng = np.random.default_rng(42)
    # Use small shapes - 10-term plan is large, full benchmark shapes would be slow
    expr = make_chain10_expr(chain10_shapes_small, rng)

    prgm = build_plan_from_expr(expr)
    root, bindings = normalize_names(prgm, {})

    downstream = LogicExecutor(
        LogicStandardizer(
            DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
        )
    )

    n = 1  # Single run - plan is large, keep benchmark tractable

    # Compile time
    t0 = time.perf_counter()
    for _ in range(n):
        optimize_plan(root, DenseStats, bindings, use_components=True)
    t_compile_with = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for _ in range(n):
        optimize_plan(root, DenseStats, bindings, use_components=False)
    t_compile_without = (time.perf_counter() - t0) / n

    # Kernel time
    opt_plan_with = optimize_plan(root, DenseStats, bindings, use_components=True)
    downstream(opt_plan_with, bindings)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        downstream(opt_plan_with, bindings)
    t_kernel_with = (time.perf_counter() - t0) / n

    opt_plan_without = optimize_plan(root, DenseStats, bindings, use_components=False)
    downstream(opt_plan_without, bindings)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        downstream(opt_plan_without, bindings)
    t_kernel_without = (time.perf_counter() - t0) / n

    _benchmark_times.append(
        ("compile chain10 (2× 10-way matmul)", t_compile_with, t_compile_without)
    )
    _benchmark_times.append(
        ("kernel chain10 (2× 10-way matmul)", t_kernel_with, t_kernel_without)
    )
    _benchmark_log.append("")
    _benchmark_log.append("=" * 60)
    _benchmark_log.append("Galley chain10 compile vs kernel benchmark:")
    _benchmark_log.append(f"  Compile (with):    {t_compile_with:.4f}s  (without): {t_compile_without:.4f}s")
    _benchmark_log.append(f"  Kernel (with):     {t_kernel_with:.4f}s  (without): {t_kernel_without:.4f}s")
    _benchmark_log.append("=" * 60)
    write_log()


def test_compile_vs_kernel_benchmark_chain25():
    """Compile and kernel benchmarks for (25-way chain) + (25-way chain)."""
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(4000)
        rng = np.random.default_rng(42)
        expr = make_chain25_expr(chain25_shapes_small, rng)

        prgm = build_plan_from_expr(expr)
        root, bindings = normalize_names(prgm, {})

        downstream = LogicExecutor(
            LogicStandardizer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        )

        n = 1  # Long chain - single run

        # Compile time
        t0 = time.perf_counter()
        for _ in range(n):
            optimize_plan(root, DenseStats, bindings, use_components=True)
        t_compile_with = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for _ in range(n):
            optimize_plan(root, DenseStats, bindings, use_components=False)
        t_compile_without = (time.perf_counter() - t0) / n

        # Kernel time
        opt_plan_with = optimize_plan(root, DenseStats, bindings, use_components=True)
        downstream(opt_plan_with, bindings)  # warmup
        t0 = time.perf_counter()
        for _ in range(n):
            downstream(opt_plan_with, bindings)
        t_kernel_with = (time.perf_counter() - t0) / n

        opt_plan_without = optimize_plan(root, DenseStats, bindings, use_components=False)
        downstream(opt_plan_without, bindings)  # warmup
        t0 = time.perf_counter()
        for _ in range(n):
            downstream(opt_plan_without, bindings)
        t_kernel_without = (time.perf_counter() - t0) / n

        _benchmark_times.append(
            ("compile chain25 (2× 25-way matmul)", t_compile_with, t_compile_without)
        )
        _benchmark_times.append(
            ("kernel chain25 (2× 25-way matmul)", t_kernel_with, t_kernel_without)
        )
        _benchmark_log.append("")
        _benchmark_log.append("=" * 60)
        _benchmark_log.append("Galley chain25 compile vs kernel benchmark:")
        _benchmark_log.append(f"  Compile (with):    {t_compile_with:.4f}s  (without): {t_compile_without:.4f}s")
        _benchmark_log.append(f"  Kernel (with):     {t_kernel_with:.4f}s  (without): {t_kernel_without:.4f}s")
        _benchmark_log.append("=" * 60)
        write_log()
    finally:
        sys.setrecursionlimit(old_limit)
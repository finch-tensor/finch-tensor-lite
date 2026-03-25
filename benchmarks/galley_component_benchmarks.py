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

NOTE: readd compile time benchmarks.
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
from finchlite.autoschedule.tensor_stats import DenseStats
from finchlite.finch_notation.interpreter import NotationInterpreter

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
        A = fl_interface.asarray(
            (np.arange(20 * 10).reshape(20, 10).astype(float) + off)
        )
        B = fl_interface.asarray(
            (np.arange(10 * 8).reshape(10, 8).astype(float) + off)
        )
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
    """Build expr = chain10_0 + chain10_1 + chain10_2 (3 terms, each a 10-way matmul chain)."""
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
    """Build expr = chain25_0 + chain25_1 + chain25_2 (3 terms, each a 25-way matmul chain)."""
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
    """Build expr = chain10_0 + chain10_1 + ... + chain10_4 (5 terms, each a 10-way matmul chain)."""
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


def main() -> None:
    lines: list[str] = []

    print("Frontend benchmark: sum+sum matmul...")
    expr_sum = make_sum_sum_benchmark_expr()
    _, _ = time_frontend_compute(expr_sum)
    components_with, components_without = time_frontend_compute(expr_sum)
    print("")
    print("=" * 60)
    print("Galley benchmark results (sum+sum matmul)")
    print(
        f"  frontend components vs no-components: "
        f"With components={components_with:.4f}s, Without={components_without:.4f}s"
    )
    print("=" * 60)

    print("Frontend benchmark: chain10...")
    rng = np.random.default_rng(42)
    expr_c10 = make_chain10_expr(chain10_shapes_benchmark, rng)
    _, _ = time_frontend_compute(expr_c10)
    components_with, components_without = time_frontend_compute(expr_c10)
    print("")
    print("=" * 60)
    print("Galley chain10 frontend benchmark:")
    print(f"  With components:   {components_with:.4f}s")
    print(f"  Without components: {components_without:.4f}s")
    print("=" * 60)
    
    print("Frontend benchmark: three summed matmul pairs...")
    expr_3p = make_three_matmul_pairs_expr()
    _, _ = time_frontend_compute(expr_3p)
    components_with, components_without = time_frontend_compute(expr_3p)
    print("")
    print("=" * 60)
    print("Galley three matmul pairs frontend benchmark:")
    print(f"  With components:   {components_with:.4f}s")
    print(f"  Without components: {components_without:.4f}s")
    print("=" * 60)
    
    # ERROR HERE:
    # 2 chain, 50 terms
    print("Frontend benchmark: fifty terms × chain2...")
    expr_50c2 = make_fifty_chain2_terms_expr(chain2_shapes_benchmark, rng)
    _, _ = time_frontend_compute(expr_50c2)
    components_with, components_without = time_frontend_compute(expr_50c2)
    print("")
    print("=" * 60)
    print("Galley fifty terms × chain2 frontend benchmark:")
    print(f"  With components:   {components_with:.4f}s")
    print(f"  Without components: {components_without:.4f}s")
    print("=" * 60)
    
    # 10 chain, 3 terms
    print("Frontend benchmark: three terms × chain10...")
    expr_3c10 = make_three_chain10_expr(chain10_shapes_benchmark, rng)
    _, _ = time_frontend_compute(expr_3c10)
    components_with, components_without = time_frontend_compute(expr_3c10)
    print("")
    print("=" * 60)
    print("Galley three terms × chain10 frontend benchmark:")
    print(f"  With components:   {components_with:.4f}s")
    print(f"  Without components: {components_without:.4f}s")
    print("=" * 60)
    
    print("Frontend benchmark: three terms × chain25...")
    expr_3c25 = make_three_chain25_expr(chain25_shapes_benchmark, rng)
    _, _ = time_frontend_compute(
        expr_3c25, recursion_limit=CHAIN_RECURSION_LIMIT)
    components_with, components_without = time_frontend_compute(
        expr_3c25, recursion_limit=CHAIN_RECURSION_LIMIT
    )
    print("")
    print("=" * 60)
    print("Galley three terms × chain25 frontend benchmark:")
    print(f"  With components:   {components_with:.4f}s")
    print(f"  Without components: {components_without:.4f}s")
    print("=" * 60)
    
    print("Frontend benchmark: five terms × chain10...")
    expr_5c10 = make_five_chain10_expr(chain10_shapes_benchmark, rng)
    _, _ = time_frontend_compute(expr_5c10)
    components_with, components_without = time_frontend_compute(expr_5c10)
    print("")
    print("=" * 60)
    print("Galley five terms × chain10 frontend benchmark:")
    print(f"  With components:   {components_with:.4f}s")
    print(f"  Without components: {components_without:.4f}s")
    print("=" * 60)

    print("Frontend benchmark: chain25...")
    expr_c25 = make_chain25_expr(chain25_shapes_benchmark, rng)
    _,_ = time_frontend_compute(
        expr_c25, recursion_limit=CHAIN_RECURSION_LIMIT
    )
    components_with, components_without = time_frontend_compute(
        expr_c25, recursion_limit=CHAIN_RECURSION_LIMIT
    )
    print("")
    print("=" * 60)
    print("Galley chain25 frontend benchmark:")
    print(f"  With components:   {components_with:.4f}s")
    print(f"  Without components: {components_without:.4f}s")
    print("=" * 60)

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
"""
Galley ASV benchmark: report Galley ``optimize_time`` and ``downstream_time`` for a
10-matrix matmul chain. One case uses dense matrices while the other
uses a sparse input for the last matrix.

Run: ``poetry run asv run --bench galley_benchmarks``
"""

from functools import reduce
from typing import Literal

import pytest

import numpy as np

import finchlite.interface as fl_interface
from finchlite.autoschedule import (
    DefaultLogicFormatter,
    DefaultLoopOrderer,
    LogicExecutor,
    LogicNormalizer,
)
from finchlite.autoschedule.compiler import LogicCompiler
from finchlite.autoschedule.galley_optimize import GalleyLogicalOptimizer
from finchlite.autoschedule.standardize import LogicStandardizer
from finchlite.autoschedule.tensor_stats import UniformStatsFactory
from finchlite.codegen.numba_codegen.numba import NumbaCompiler
from finchlite.compile.lower import NotationCompiler
from finchlite.finch_assembly.simplification import AssemblySimplify
from finchlite.finch_logic import Alias, Field, Plan, Produces, Query, Table
from finchlite.symbolic import gensym

from .utils import patch_benchmark

CHAIN_LEN = 10
MAT_DIM = 8

CASES = {
    "matmul10_dense": False,
    "matmul10_empty_last": True,
}


def _plan_from_lazy(expr):
    """
    Build the same `Plan` as `finchlite.interface.fuse.compute`.
    """
    args = (expr,)
    vars_ = tuple(Alias(gensym("A")) for _ in args)
    ctx = args[0].ctx.join()
    bodies = tuple(
        Query(
            var,
            Table(a.data, tuple(Field(gensym("i")) for _ in range(len(a.shape)))),
        )
        for a, var in zip(args, vars_, strict=True)
    )
    return Plan(ctx.trace() + bodies + (Produces(vars_),))


def _build_expr(empty_last):
    """
    Build a chain of 10 matmuls with the last matrix either dense or empty
    """
    rng = np.random.default_rng(42)
    mats = [rng.standard_normal((MAT_DIM, MAT_DIM)) for _ in range(CHAIN_LEN)]
    if empty_last:
        mats[-1].fill(0)
    lazies = [fl_interface.lazy(fl_interface.asarray(m)) for m in mats]
    return reduce(lambda a, b: a @ b, lazies)


def _make_pipeline():
    optimizer = GalleyLogicalOptimizer(
        DefaultLoopOrderer(
            LogicStandardizer(
                DefaultLogicFormatter(
                    LogicCompiler(
                        NotationCompiler(
                            NumbaCompiler(), ctx_transforms=(AssemblySimplify(),)
                        )
                    )
                )
            )
        )
    )
    executor = LogicExecutor(optimizer, stats_factory=UniformStatsFactory())
    return LogicNormalizer(executor)


@pytest.mark.parametrize("metric", ["optimize", "downstream"])
@pytest.mark.parametrize(
    "empty_last",
    [
        pytest.param(True, id="empty_last"),
        pytest.param(False, id="dense_last"),
    ],
)
def test_galley_matmul_chain(
    benchmark, monkeypatch, empty_last: bool, metric: Literal["optimize", "downstream"]
) -> None:
    import finchlite.autoschedule.galley_optimize as galley

    # Warmup
    pipeline = _make_pipeline()
    plan = _plan_from_lazy(_build_expr(empty_last))
    pipeline(plan)

    # Benchmark
    if metric == "optimize":
        patch_benchmark(benchmark, monkeypatch, galley, "optimize_plan")
    else:
        patch_benchmark(benchmark, monkeypatch, LogicStandardizer, "lower")

    pipeline(plan)

"""
Galley ASV benchmark: report Galley ``optimize_time`` and ``downstream_time`` for a
10-matrix matmul chain. One case uses dense matrices while the other
uses a sparse input for the last matrix.

Run: ``poetry run asv run --bench galley_benchmarks``
"""

import time
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
from finchlite.autoschedule.tensor_stats import UniformStatsFactory
from finchlite.finch_logic import Alias, Field, Plan, Produces, Query, Table
from finchlite.finch_notation.interpreter import NotationInterpreter
from finchlite.symbolic import gensym

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
        mats[-1] = np.zeros((MAT_DIM, MAT_DIM))
    lazies = [fl_interface.lazy(fl_interface.asarray(m)) for m in mats]
    return reduce(lambda a, b: a @ b, lazies)


def _make_pipeline():
    optimizer = GalleyLogicalOptimizer(
        LogicStandardizer(DefaultLogicFormatter(LogicCompiler(NotationInterpreter())))
    )
    executor = LogicExecutor(optimizer, stats_factory=UniformStatsFactory())
    return LogicNormalizer(executor), optimizer


class GalleyMatmulChain:
    params = list(CASES.keys())
    param_names = ["case"]

    def track_optimize_time(self, case):
        pipeline, optimizer = _make_pipeline()
        plan = _plan_from_lazy(_build_expr(CASES[case]))
        pipeline(plan)
        return optimizer.last_optimize_plan_s or 0.0

    track_optimize_time.unit = "seconds"

    def track_downstream_time(self, case):
        pipeline, optimizer = _make_pipeline()
        plan = _plan_from_lazy(_build_expr(CASES[case]))
        t0 = time.perf_counter()
        pipeline(plan)
        return time.perf_counter() - t0 - (optimizer.last_optimize_plan_s or 0.0)

    track_downstream_time.unit = "seconds"

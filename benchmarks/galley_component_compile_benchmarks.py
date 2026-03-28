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

import numpy as np

from galley_component_benchmarks import (
    CHAIN_RECURSION_LIMIT,
    DEFAULT_N,
    _recursion_limit_ctx,
    chain2_shapes_benchmark,
    chain10_shapes_benchmark,
    chain25_shapes_benchmark,
    make_chain10_expr,
    make_chain25_expr,
    make_fifty_chain2_terms_expr,
    make_five_chain10_expr,
    make_sum_sum_benchmark_expr,
    make_three_chain10_expr,
    make_three_chain25_expr,
    make_three_matmul_pairs_expr,
)

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
from finchlite.finch_logic import Alias, Field, Plan, Produces, Query, Table
from finchlite.finch_notation.interpreter import NotationInterpreter
from finchlite.symbolic import gensym

GALLEY_COMPILE_PROFILE_WITH = LogicNormalizer(
    GalleyLogicalOptimizer(
        DenseStats,
        LogicExecutor(
            LogicStandardizer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
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
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        ),
        use_components=False,
        profile=True,
    )
)


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
    print(_format_block("Galley compile profile (fifty terms × chain2)", w, wo), flush=True)

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
    print(_format_block("Galley compile profile (three terms × chain25)", w, wo), flush=True)

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

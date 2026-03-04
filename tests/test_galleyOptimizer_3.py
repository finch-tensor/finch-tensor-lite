"""
TEST 3

Same as test2 but run through the full Galley pipeline that is made here.
Fuze.py has circular import error, so we make the pipeline here.
(LogicNormalizer -> GalleyLogicalOptimizer -> LogicExecutor -> ... -> NotationInterpreter).
"""

# Same tests as test2 but run through the full Galley pipeline
# (LogicNormalizer -> GalleyLogicalOptimizer -> LogicExecutor -> ... -> NotationInterpreter).
import operator
import numpy as np
import finchlite as fl
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


def get_full_galley_pipeline():
    """Build the same pipeline as INTERPRET_NOTATION_GALLEY (no lazy proxy)."""
    from finchlite.autoschedule.GalleyLogicalOptimizer import GalleyLogicalOptimizer
    from finchlite.autoschedule import LogicExecutor, LogicNormalizer
    from finchlite.autoschedule.formatter import LogicFormatter
    from finchlite.autoschedule.optimize import DefaultLogicOptimizer
    from finchlite.autoschedule.standardize import LogicStandardizer
    from finchlite.autoschedule.compiler import LogicCompiler
    from finchlite.finch_notation.interpreter import NotationInterpreter
    from finchlite.galley.TensorStats import DenseStats

    return LogicNormalizer(
        GalleyLogicalOptimizer(
            DenseStats,
            LogicExecutor(
                DefaultLogicOptimizer(
                    LogicStandardizer(
                        LogicFormatter(
                            LogicCompiler(NotationInterpreter())
                        )
                    )
                )
            )
        )
    )


def test_full_pipeline_two_arrays():
    """Same as test2 test_galley_logical_optimizer: out = a * b, but through full pipeline."""
    a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = Alias("out")
    plan = Plan(
        (
            Query(
                out,
                MapJoin(
                    Literal(operator.mul),
                    (
                        Table(Literal(a), (Field("i"), Field("j"))),
                        Table(Literal(b), (Field("i"), Field("j"))),
                    ),
                ),
            ),
            Produces((out,)),
        )
    )

    ctx = get_full_galley_pipeline()
    try:
        result = ctx(plan, bindings={})
        print("test_full_pipeline_two_arrays: OK")
        print("  result:", result)
    except Exception as e:
        print("test_full_pipeline_two_arrays: FAILED")
        print("  error:", type(e).__name__, e)


def test_full_pipeline_greedy_example():
    """Same as test2 test_greedy_example: out = sum_i sum_j (A[i,j] * B[j,k]), but through full pipeline."""
    A = fl.asarray(np.ones((2, 3)))
    B = fl.asarray(np.ones((3, 4)))
    out = Alias("out")
    plan = Plan(
        (
            Query(
                out,
                Aggregate(
                    Literal(operator.add),
                    Literal(0.0),
                    Aggregate(
                        Literal(operator.add),
                        Literal(0.0),
                        MapJoin(
                            Literal(operator.mul),
                            (
                                Table(Literal(A), (Field("i"), Field("j"))),
                                Table(Literal(B), (Field("j"), Field("k"))),
                            ),
                        ),
                        (Field("i"),),
                    ),
                    (Field("j"),),
                ),
            ),
            Produces((out,)),
        )
    )

    ctx = get_full_galley_pipeline()
    try:
        result = ctx(plan, bindings={})
        print("test_full_pipeline_greedy_example: OK")
        print("  result:", result)
    except Exception as e:
        print("test_full_pipeline_greedy_example: FAILED")
        print("  error:", type(e).__name__, e)


def test_full_pipeline_lazy_matmul_sum_axis0():
    """Logical equivalent of test5: out = sum_i sum_j (A[i,j] * B[j,k]) with A,B from test5, via full pipeline."""
    A = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = Alias("out")

    # out[k] = sum_i sum_j A[i,j] * B[j,k]
    plan = Plan(
        (
            Query(
                out,
                Aggregate(
                    Literal(operator.add),
                    Literal(0.0),
                    Aggregate(
                        Literal(operator.add),
                        Literal(0.0),
                        MapJoin(
                            Literal(operator.mul),
                            (
                                Table(Literal(A), (Field("i"), Field("j"))),
                                Table(Literal(B), (Field("j"), Field("k"))),
                            ),
                        ),
                        (Field("i"),),
                    ),
                    (Field("j"),),
                ),
            ),
            Produces((out,)),
        )
    )

    ctx = get_full_galley_pipeline()
    try:
        result = ctx(plan, bindings={})
        print("test_full_pipeline_lazy_matmul_sum_axis0: OK")
        print("  result:", result)
    except Exception as e:
        print("test_full_pipeline_lazy_matmul_sum_axis0: FAILED")
        print("  error:", type(e).__name__, e)


if __name__ == "__main__":
    print("=== Full pipeline: out = a * b (2 arrays) ===")
    test_full_pipeline_two_arrays()
    print()
    #print("=== Full pipeline: out = sum_i sum_j (A[i,j] * B[j,k]) ===")
    #test_full_pipeline_greedy_example()
    #print()
    #print("=== Full pipeline: out = sum_i sum_j (A[i,j] * B[j,k]) (from lazy matmul + sum axis=0 shapes) ===")
    #test_full_pipeline_lazy_matmul_sum_axis0()
    
    
"""
Output: Runs correctly for simple a*b, different error in summation. (not KeyError)
Maybe pipeline in Fuze is wrong
"""

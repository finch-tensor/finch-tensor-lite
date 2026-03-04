"""
Combined Galley optimizer tests (originally test_galleyOptimizer_1 through test_galleyOptimizer_9).

- TEST 1: out = a * b via frontend (INTERPRET_NOTATION_GALLEY).
- TEST 2: GalleyLogicalOptimizer alone with a*b and triangle summation.
- TEST 3: Same as test2 but through full Galley pipeline (LogicNormalizer -> GalleyLogicalOptimizer -> ...).
- TEST 4: out = a * b + c * d via frontend.
- TEST 5: out = sum_i sum_j (A[i,j] * B[j,k]) via frontend (sum over axis=0).
- TEST 6: sum(A@B, axis=0) + sum(C@D, axis=1).
- TEST 7: Nested aggregates sum_i sum_j (A[i,j] * B[j,k]) (sum over all).
- TEST 8: sum((A @ B) @ C) — deeper nesting.
- TEST 9: expand_dims + sum over singleton axis (extra axis padding in logic_to_stats).
"""
import operator

import numpy as np
import pytest

import finchlite as fl
import finchlite.interface as fl_interface

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


# --- TEST 1: out = a * b via frontend ---
def test_galley_optimizer_1_elementwise_mul():
    """Running out = a * b with Finch/Galley pipeline using the frontend."""
    a = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.lazy(a) * fl_interface.lazy(b),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )
    expected = np.array([[1.0, 2.0], [3.0, 4.0]]) * np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 2: GalleyLogicalOptimizer alone ---
def test_galley_optimizer_2_logical_optimizer_two_arrays():
    """Test GalleyLogicalOptimizer with 2 arrays: out = a * b (elementwise)."""
    from finchlite.autoschedule.GalleyLogicalOptimizer import (
        GalleyLogicalOptimizer,
        optimize_plan,
    )
    from finchlite.galley.TensorStats import DenseStats

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

    optimizer = GalleyLogicalOptimizer(DenseStats, ctx=None)
    result = optimizer(plan, bindings={})

    assert isinstance(result, Plan)
    assert len(result.bodies) >= 1
    assert isinstance(result.bodies[-1], Produces)
    for body in result.bodies[:-1]:
        assert isinstance(body, Query)

    result2 = optimize_plan(plan, DenseStats, {})
    assert isinstance(result2, Plan)
    assert isinstance(result2.bodies[-1], Produces)


def test_galley_optimizer_2_greedy_example():
    """Example: out = sum_i sum_j (A[i,j] * B[j,k]) via GalleyLogicalOptimizer."""
    from finchlite.autoschedule.GalleyLogicalOptimizer import GalleyLogicalOptimizer
    from finchlite.galley.TensorStats import DenseStats

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

    optimizer = GalleyLogicalOptimizer(DenseStats, ctx=None)
    result = optimizer(plan, bindings={})
    assert isinstance(result, Plan)
    assert len(result.bodies) >= 1


# --- TEST 3: Full Galley pipeline ---
def get_full_galley_pipeline():
    """Build the same pipeline as INTERPRET_NOTATION_GALLEY (no lazy proxy)."""
    from finchlite.autoschedule import LogicExecutor, LogicNormalizer
    from finchlite.autoschedule.GalleyLogicalOptimizer import GalleyLogicalOptimizer
    from finchlite.autoschedule.compiler import LogicCompiler
    from finchlite.autoschedule.formatter import DefaultLogicFormatter
    from finchlite.autoschedule.optimize import DefaultLogicOptimizer
    from finchlite.autoschedule.standardize import LogicStandardizer
    from finchlite.finch_notation.interpreter import NotationInterpreter
    from finchlite.galley.TensorStats import DenseStats

    return LogicNormalizer(
        GalleyLogicalOptimizer(
            DenseStats,
            LogicExecutor(
                DefaultLogicOptimizer(
                    LogicStandardizer(
                        DefaultLogicFormatter(
                            LogicCompiler(NotationInterpreter())
                        )
                    )
                )
            ),
        )
    )


def test_galley_optimizer_3_full_pipeline_two_arrays():
    """Same as test2: out = a * b, but through full pipeline."""
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
    result = ctx(plan, bindings={})
    assert result is not None


def test_galley_optimizer_3_full_pipeline_greedy_example():
    """Same as test2 test_greedy_example: out = sum_i sum_j (A[i,j] * B[j,k]), through full pipeline."""
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
    result = ctx(plan, bindings={})
    assert result is not None


def test_galley_optimizer_3_full_pipeline_lazy_matmul_sum_axis0():
    """Logical equivalent of test5: out = sum_i sum_j (A[i,j] * B[j,k]) via full pipeline."""
    A = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
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
    result = ctx(plan, bindings={})
    assert result is not None


# --- TEST 4: out = a * b + c * d via frontend ---
def test_galley_optimizer_4_add_of_elementwise():
    """Running out = a * b + c * d with Finch/Galley pipeline using the frontend."""
    a = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    c = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    d = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.lazy(a) * fl_interface.lazy(b)
        + fl_interface.lazy(c) * fl_interface.lazy(d),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )
    expected = (
        np.array(a) * np.array(b) + np.array(c) * np.array(d)
    )
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 5: sum(A @ B, axis=0) via frontend ---
def test_galley_optimizer_5_matmul_sum_axis0():
    """Running out = sum_i sum_j (A[i,j] * B[j,k]) with Finch/Galley pipeline using the frontend."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=0),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )
    expected = np.sum(np.array(A) @ np.array(B), axis=0)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 6: sum(A@B, axis=0) + sum(C@D, axis=1) ---
def test_galley_optimizer_6_sum_axis0_plus_sum_axis1():
    """More complex: out = sum(A @ B, axis=0) + sum(C @ D, axis=1). Correctness vs NumPy."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    D = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=0)
        + fl_interface.sum(fl_interface.lazy(C) @ fl_interface.lazy(D), axis=1),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    left = np.sum(np.array(A) @ np.array(B), axis=0)
    right = np.sum(np.array(C) @ np.array(D), axis=1)
    expected = left + right
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 7: Nested aggregates sum(A @ B) ---
def test_galley_optimizer_7_nested_aggregates_full_sum():
    """Nested aggregates: out = sum_i sum_j (A[i,j] * B[j,k]). Verified against NumPy."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B)),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum(np.array(A) @ np.array(B))
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 8: sum((A @ B) @ C) ---
def test_galley_optimizer_8_deeper_nesting():
    """Deeper nesting: out = sum((A @ B) @ C). Verified against NumPy."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum((fl_interface.lazy(A) @ fl_interface.lazy(B)) @ fl_interface.lazy(C)),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum((np.array(A) @ np.array(B)) @ np.array(C))
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 9: expand_dims + sum over singleton (extra axis padding) ---
def test_galley_optimizer_9_expand_dims_sum_singleton():
    """Exercises extra axis padding in logic_to_stats: sum(expand_dims(A, 2), axis=2)."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))

    expanded = fl_interface.expand_dims(fl_interface.lazy(A), axis=2)
    out = fl_interface.compute(
        fl_interface.sum(expanded, axis=2),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum(np.expand_dims(np.array(A), axis=2), axis=2)
    assert np.allclose(np.array(out), np.array(expected))

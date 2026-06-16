import pytest

import numpy as np

import finchlite.interface as fl
from finchlite.algebra import ffuncs, ftype
from finchlite.autoschedule import (
    DefaultLogicOptimizer,
    DefaultLoopOrderer,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
    # validate,  # commented out: validate() grammar checker is no longer used
)
from finchlite.autoschedule.formatter import DefaultLogicFormatter
from finchlite.autoschedule.galley_optimize import GalleyLogicalOptimizer
from finchlite.autoschedule.standardize import LogicStandardizer
from finchlite.autoschedule.tensor_stats import DenseStatsFactory
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)
from finchlite.finch_notation.interpreter import NotationInterpreter

from .conftest import finch_assert_allclose


def capture_prgm(prgm, bindings, stats=None, stats_factory=None):
    return prgm, bindings


_DUMMY_STATS: dict = {}
_DUMMY_STATS_FACTORY = DenseStatsFactory()


def _dense_binding(shape: tuple[int, ...]):
    return ftype(fl.asarray(np.zeros(shape)))


"""
Output tests
"""


def test_default_loop_orderer_matmul_plan_output():
    """
    After ``DefaultLoopOrderer`` (reorder, concordize, drop_reorders, flatten_plans,
    normalize_names): matmul-shaped IR remains valid; loop nest is length-3 on the
    ``MapJoin``; reduction is one axis; operands are two ``Table`` nodes (swizzled
    left operand). Exact alias/field strings are not asserted because
    ``normalize_names`` rewrites them.
    """
    i, j, k = Field("i"), Field("j"), Field("k")
    plan = Plan(
        (
            Query(
                Alias("C"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Reorder(
                        MapJoin(
                            Literal(ffuncs.mul),
                            (
                                Table(Alias("A"), (i, k)),
                                Table(Alias("B"), (k, j)),
                            ),
                        ),
                        (i, j, k),
                    ),
                    (k,),
                ),
            ),
            Produces((Alias("C"),)),
        )
    )

    result, _ = DefaultLoopOrderer(capture_prgm)(
        plan,
        {
            Alias("A"): _dense_binding((2, 2)),
            Alias("B"): _dense_binding((2, 2)),
        },
        _DUMMY_STATS,
        _DUMMY_STATS_FACTORY,
    )
    # validate(result, kind="output")  # commented out: validate() removed

    assert isinstance(result, Plan)
    assert len(result.bodies) >= 3
    *swizzle_queries, q, prod = result.bodies
    assert len(swizzle_queries) >= 1
    for swizzle_q in swizzle_queries:
        assert isinstance(swizzle_q, Query)
        match swizzle_q.rhs:
            case Reorder(Table(tns, _), _):
                assert tns.name in ("A", "B") or tns.name.startswith(("A", "B"))
            case _:
                pytest.fail("expected swizzle Query(Reorder(Table(...), ...))")
    assert isinstance(q, Query)
    assert isinstance(prod, Produces)
    assert prod.args == (q.lhs,)

    match q.rhs:
        case Aggregate(op, init, Reorder(mj, outer), red):
            assert op.val is ffuncs.add
            assert init.val == 0
            assert isinstance(mj, MapJoin)
            assert mj.op.val is ffuncs.mul
            assert len(mj.args) == 2
            assert all(isinstance(a, Table) for a in mj.args)
            assert len(outer) == 3
            assert len(red) == 1
            assert red[0].name == "k"
            t0, t1 = mj.args
            assert set(t0.idxs) <= set(outer) and set(t1.idxs) <= set(outer)
        case _:
            pytest.fail("expected Aggregate with Reorder(MapJoin(...))")


def test_valid_plan_with_produces_passes_loop_ordering():
    i = Field("i")
    plan = Plan(
        (
            Query(Alias("B"), Reorder(Table(Alias("A"), (i,)), (i,))),
            Produces((Alias("B"),)),
        )
    )

    result, _ = DefaultLoopOrderer(capture_prgm)(
        plan, {Alias("A"): _dense_binding((2,))}, _DUMMY_STATS, _DUMMY_STATS_FACTORY
    )
    # validate(result, kind="output")  # commented out: validate() removed

    assert isinstance(result, Plan)
    assert len(result.bodies) == 2
    q, prod = result.bodies
    assert isinstance(q, Query)
    assert isinstance(prod, Produces)
    assert prod.args == (q.lhs,)
    match q.rhs:
        case Reorder(Table(_, idxs), reorder_idxs):
            assert idxs == reorder_idxs
        case _:
            pytest.fail("expected Reorder(Table(...), ...)")


"""
Input tests
"""
# Full pipeline test
INTERPRET_NOTATION_GALLEY_LOOP_ORDER = LogicNormalizer(
    LogicExecutor(
        GalleyLogicalOptimizer(
            LogicStandardizer(
                DefaultLoopOrderer(
                    DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
                )
            )
        )
    )
)


INTERPRET_NOTATION_TEST = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            LogicStandardizer(
                DefaultLoopOrderer(
                    DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
                )
            )
        )
    )
)


def test_galley_loop_order_frontend_pipeline():
    a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl.asarray(np.array([[5.0, 6.0], [7.0, 8.0]]))

    out = fl.compute(fl.lazy(a) @ fl.lazy(b), ctx=INTERPRET_NOTATION_GALLEY_LOOP_ORDER)

    expected = np.array(a) @ np.array(b)
    finch_assert_allclose(np.array(out), expected)


def test_galley_loop_order_frontend_elementwise_mul():
    """Galley + loop orderer pipeline: element-wise multiply (no contraction)."""
    a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl.asarray(np.array([[2.0, 0.5], [1.0, 3.0]]))

    out = fl.compute(fl.lazy(a) * fl.lazy(b), ctx=INTERPRET_NOTATION_GALLEY_LOOP_ORDER)

    expected = np.array(a) * np.array(b)
    finch_assert_allclose(np.array(out), expected)


def test_galley_loop_order_frontend_matmul_sum_axis0():
    """Galley + loop orderer pipeline: matmul then reduce along axis 0."""
    a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl.asarray(np.array([[5.0, 6.0], [7.0, 8.0]]))

    out = fl.compute(
        fl.sum(fl.lazy(a) @ fl.lazy(b), axis=0),
        ctx=INTERPRET_NOTATION_GALLEY_LOOP_ORDER,
    )

    expected = np.sum(np.array(a) @ np.array(b), axis=0)
    finch_assert_allclose(np.array(out), expected)


def test_optimizer_loop_order_frontend_pipeline():
    a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl.asarray(np.array([[5.0, 6.0], [7.0, 8.0]]))

    out = fl.compute(fl.lazy(a) @ fl.lazy(b), ctx=INTERPRET_NOTATION_TEST)

    expected = np.array(a) @ np.array(b)
    finch_assert_allclose(np.array(out), expected)


def test_optimizer_loop_order_frontend_elementwise_mul():
    """Default optimizer + loop orderer pipeline: element-wise multiply."""
    a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl.asarray(np.array([[2.0, 0.5], [1.0, 3.0]]))

    out = fl.compute(fl.lazy(a) * fl.lazy(b), ctx=INTERPRET_NOTATION_TEST)

    expected = np.array(a) * np.array(b)
    finch_assert_allclose(np.array(out), expected)


def test_optimizer_loop_order_frontend_matmul_sum_axis0():
    """Default optimizer + loop orderer pipeline: matmul then reduce along axis 0."""
    a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl.asarray(np.array([[5.0, 6.0], [7.0, 8.0]]))

    out = fl.compute(
        fl.sum(fl.lazy(a) @ fl.lazy(b), axis=0),
        ctx=INTERPRET_NOTATION_TEST,
    )

    expected = np.sum(np.array(a) @ np.array(b), axis=0)
    finch_assert_allclose(np.array(out), expected)

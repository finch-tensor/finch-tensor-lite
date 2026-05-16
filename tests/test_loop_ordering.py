import pytest

from finchlite.algebra import ffuncs
from finchlite.autoschedule import (
    DefaultLoopOrderer,
    validate,
)
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


def capture_prgm(prgm, bindings, stats=None, stats_factory=None):
    return prgm, bindings


_DUMMY_STATS: dict = {}
_DUMMY_STATS_FACTORY = DenseStatsFactory()


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
        plan, {}, _DUMMY_STATS, _DUMMY_STATS_FACTORY
    )
    validate(result, kind="output")

    assert isinstance(result, Plan)
    assert len(result.bodies) == 2
    q, prod = result.bodies
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
            assert red[0] in outer
            t0, t1 = mj.args
            assert len(t0.idxs) == 2 and len(t1.idxs) == 2
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
        plan, {}, _DUMMY_STATS, _DUMMY_STATS_FACTORY
    )
    validate(result, kind="output")

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


def test_input_rejects_invalid_root_node():
    with pytest.raises(
        ValueError, match="Invalid loop ordering input: expected Plan or Query"
    ):
        validate(Produces((Alias("A"),)), kind="input")


def test_input_rejects_bare_table_rhs():
    i = Field("i")

    with pytest.raises(
        ValueError, match="Invalid loop ordering input: Query rhs must be Reorder"
    ):
        validate(Query(Alias("B"), Table(Alias("A"), (i,))), kind="input")


def test_input_rejects_two_aggregates_on_rhs():
    """At most one Aggregate per Query rhs (single reduction around map-join)."""
    i = Field("i")
    inner = Aggregate(
        Literal(ffuncs.add),
        Literal(0),
        Reorder(Table(Alias("A"), (i,)), (i,)),
        (),
    )
    rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0),
        Reorder(inner, (i,)),
        (),
    )
    with pytest.raises(
        ValueError,
        match="Invalid loop ordering: at most one Aggregate per Query rhs",
    ):
        validate(Query(Alias("B"), rhs), kind="input")


def test_input_rejects_mapjoin_outside_aggregate():
    """MapJoin must sit under Aggregate(..., arg, ...), not a bare Reorder rhs."""
    i, j = Field("i"), Field("j")
    rhs = Reorder(
        MapJoin(
            Literal(ffuncs.mul),
            (
                Table(Alias("A"), (i, j)),
                Table(Alias("B"), (i, j)),
            ),
        ),
        (i, j),
    )
    with pytest.raises(
        ValueError,
        match="Invalid loop ordering: MapJoin is only allowed "
        "inside an Aggregate argument",
    ):
        validate(Query(Alias("C"), rhs), kind="input")


def test_input_accepts_mapjoin_inside_aggregate():
    i, j, k = Field("i"), Field("j"), Field("k")
    query = Query(
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
    )
    validate(query, kind="input")


def test_input_rejects_invalid_plan_body():
    i = Field("i")

    with pytest.raises(
        ValueError, match="Invalid loop ordering input: expected Query or Produces"
    ):
        validate(Plan((Table(Alias("A"), (i,)),)), kind="input")


def test_output_rejects_invalid():
    with pytest.raises(
        ValueError, match="Invalid loop ordering output: expected Plan or Query"
    ):
        validate(Produces((Alias("A"),)), kind="output")


def test_output_rejects_bare_rhs_on_query():
    i = Field("i")
    with pytest.raises(
        ValueError,
        match="Query rhs must be Reorder\\(\\.\\.\\.\\) or Aggregate",
    ):
        validate(Query(Alias("B"), Table(Alias("A"), (i,))), kind="output")


def test_output_rejects_nonstandard_inner_rhs():
    """Loop ordering output must not wrap Aggregate in an outer Reorder."""
    i = Field("i")
    bad = Query(
        Alias("B"),
        Reorder(
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                Reorder(Table(Alias("A"), (i,)), (i,)),
                (),
            ),
            (i,),
        ),
    )
    with pytest.raises(
        ValueError,
        match="Invalid loop ordering output: Query rhs must be Reorder",
    ):
        validate(bad, kind="output")


def test_output_rejects_two_aggregates_inside_reorder():
    """Query rhs must still have at most one Aggregate (nested aggregate invalid)."""
    i = Field("i")
    inner = Aggregate(
        Literal(ffuncs.add),
        Literal(0),
        Reorder(Table(Alias("A"), (i,)), (i,)),
        (),
    )
    rhs = Aggregate(
        Literal(ffuncs.add),
        Literal(0),
        Reorder(inner, (i,)),
        (),
    )
    bad = Query(Alias("B"), rhs)
    with pytest.raises(
        ValueError,
        match="Invalid loop ordering: at most one Aggregate per Query rhs",
    ):
        validate(bad, kind="output")


def test_output_rejects_mapjoin_outside_aggregate_inside_reorder():
    i, j = Field("i"), Field("j")
    inner = Reorder(
        MapJoin(
            Literal(ffuncs.mul),
            (
                Table(Alias("A"), (i, j)),
                Table(Alias("B"), (i, j)),
            ),
        ),
        (i, j),
    )
    bad = Query(Alias("C"), Reorder(inner, (i, j)))
    with pytest.raises(
        ValueError,
        match="Invalid loop ordering: MapJoin is only allowed "
        "inside an Aggregate argument",
    ):
        validate(bad, kind="output")


def test_output_rejects_produces_followed_by_query():
    i = Field("i")
    plan = Plan(
        (
            Produces((Alias("A"),)),
            Query(Alias("B"), Reorder(Table(Alias("A"), (i,)), (i,))),
        )
    )
    with pytest.raises(
        ValueError, match="Invalid loop ordering output: Produces must be final body"
    ):
        validate(plan, kind="output")

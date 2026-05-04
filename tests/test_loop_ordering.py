import pytest

from finchlite.algebra import ffuncs
from finchlite.autoschedule import (
    DefaultLoopOrderer,
    validate_input,
    validate_output,
)
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


def capture_prgm(prgm, bindings):
    return prgm, bindings


"""
Output tests
"""


def test_valid_single_query_is_wrapped_in_loop_reorder():
    i, j = Field("i"), Field("j")
    query = Query(
        Alias("B"),
        Reorder(Table(Alias("A"), (i, j)), (i, j)),
    )

    result, bindings = DefaultLoopOrderer(capture_prgm)(query, {})

    assert bindings == {}
    assert result == Query(Alias("B"), Reorder(query.rhs, (i, j)))


def test_valid_aggregate_query_is_wrapped_in_loop_reorder():
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

    result, _ = DefaultLoopOrderer(capture_prgm)(query, {})

    # k appears on both operands; nest k outermost — order merged into inner Reorder
    assert result == Query(
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
                (k, i, j),
            ),
            (k,),
        ),
    )


def test_valid_plan_with_produces_passes_loop_ordering():
    i = Field("i")
    plan = Plan(
        (
            Query(Alias("B"), Reorder(Table(Alias("A"), (i,)), (i,))),
            Produces((Alias("B"),)),
        )
    )

    result, _ = DefaultLoopOrderer(capture_prgm)(plan, {})

    assert result == Plan(
        (
            Query(Alias("B"), Reorder(plan.bodies[0].rhs, (i,))),
            Produces((Alias("B"),)),
        )
    )


"""
Input tests
"""


def test_input_rejects_invalid_root_node():
    with pytest.raises(
        ValueError, match="Invalid loop ordering input: expected Plan or Query"
    ):
        validate_input(Produces((Alias("A"),)))


def test_input_rejects_bare_table_rhs():
    i = Field("i")

    with pytest.raises(
        ValueError, match="Invalid loop ordering input: Query rhs must be Reorder"
    ):
        validate_input(Query(Alias("B"), Table(Alias("A"), (i,))))


def test_input_rejects_aggregate_without_inner_reorder():
    i = Field("i")

    with pytest.raises(
        ValueError, match="Invalid loop ordering input: Query rhs must be Reorder"
    ):
        validate_input(
            Query(
                Alias("B"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(Alias("A"), (i,)),
                    (i,),
                ),
            )
        )


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
        validate_input(Query(Alias("B"), rhs))


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
        validate_input(Query(Alias("C"), rhs))


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
    validate_input(query)


def test_input_rejects_invalid_plan_body():
    i = Field("i")

    with pytest.raises(
        ValueError, match="Invalid loop ordering input: expected Query or Produces"
    ):
        validate_input(Plan((Table(Alias("A"), (i,)),)))


def test_output_rejects_invalid():
    with pytest.raises(
        ValueError, match="Invalid loop ordering output: expected Plan or Query"
    ):
        validate_output(Produces((Alias("A"),)))


def test_output_rejects_bare_rhs_on_query():
    i = Field("i")
    with pytest.raises(
        ValueError,
        match="Query rhs must be Reorder\\(\\.\\.\\.\\) or Aggregate",
    ):
        validate_output(Query(Alias("B"), Table(Alias("A"), (i,))))


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
        validate_output(bad)


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
        validate_output(bad)


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
        validate_output(bad)


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
        validate_output(plan)


def test_default_loop_orderer_passes_validate_output():
    i, j = Field("i"), Field("j")
    query = Query(
        Alias("B"),
        Reorder(Table(Alias("A"), (i, j)), (i, j)),
    )
    reorder_query, _ = DefaultLoopOrderer(capture_prgm)(query, {})
    validate_output(reorder_query)

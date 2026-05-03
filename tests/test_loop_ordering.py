import pytest

from finchlite.algebra import ffuncs
from finchlite.autoschedule import (
    DefaultLoopOrderer,
    validate_input,
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

    assert result == Query(Alias("C"), Reorder(query.rhs, (k, i, j)))


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


def test_input_rejects_invalid_root_node():
    with pytest.raises(
        ValueError, match="Invalid loop ordering input: expected Plan or Query"
    ):
        validate_input(Produces((Alias("A"),)))


def test_input_rejects_bare_table_rhs():
    i = Field("i")

    with pytest.raises(
        ValueError, match="Invalid loop ordering input: expected standardized Query rhs"
    ):
        validate_input(Query(Alias("B"), Table(Alias("A"), (i,))))


def test_input_rejects_aggregate_without_inner_reorder():
    i = Field("i")

    with pytest.raises(
        ValueError, match="Invalid loop ordering input: expected standardized Query rhs"
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


def test_input_rejects_invalid_plan_body():
    i = Field("i")

    with pytest.raises(
        ValueError, match="Invalid loop ordering input: expected Query or Produces"
    ):
        validate_input(Plan((Table(Alias("A"), (i,)),)))



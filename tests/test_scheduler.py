from finch.autoschedule import propagate_map_queries, lift_subqueries, push_fields
from finch.finch_logic import (
    Plan,
    Query,
    Alias,
    Field,
    Aggregate,
    Immediate,
    MapJoin,
    Produces,
    Relabel,
    Reorder,
    Subquery,
    Table,
)


def test_propagate_map_queries_simple():
    plan = Plan(
        (
            Query(
                Alias("A10"),
                Aggregate(Immediate("+"), Immediate(0), Immediate("[1,2,3]"), ()),
            ),
            Query(Alias("A11"), Alias("A10")),
            Produces((Alias("A11"),)),
        )
    )
    expected = Plan(
        (
            Query(
                Alias("A11"),
                MapJoin(Immediate("+"), (Immediate(0), Immediate("[1,2,3]"))),
            ),
            Produces((Alias("A11"),)),
        )
    )

    result = propagate_map_queries(plan)
    assert result == expected


def test_lift_subqueries():
    plan = Plan(
        (
            Query(
                Alias("A10"),
                Plan(
                    (
                        Subquery(Alias("C10"), Immediate(0)),
                        Subquery(
                            Alias("B10"),
                            MapJoin(
                                Immediate("+"),
                                (
                                    Subquery(Alias("C10"), Immediate(0)),
                                    Immediate("[1,2,3]"),
                                ),
                            ),
                        ),
                        Subquery(Alias("B10"), Immediate(0)),
                        Produces((Alias("B10"),)),
                    )
                ),
            ),
            Produces((Alias("A10"),)),
        )
    )

    expected = Plan(
        (
            Plan(
                (
                    Query(Alias("C10"), Immediate(0)),
                    Query(
                        Alias("B10"),
                        MapJoin(Immediate("+"), (Alias("C10"), Immediate("[1,2,3]"))),
                    ),
                    Query(
                        Alias("A10"),
                        Plan(
                            (
                                Alias("C10"),
                                Alias("B10"),
                                Alias("B10"),
                                Produces((Alias("B10"),)),
                            )
                        ),
                    ),
                ),
            ),
            Produces((Alias("A10"),)),
        )
    )

    result = lift_subqueries(plan)
    assert result == expected


def test_push_fields():
    plan = Plan(
        (
            Relabel(
                MapJoin(
                    Immediate("+"),
                    (
                        Table(Immediate("tbl1"), (Field("A1"), Field("A2"))),
                        Table(Immediate("tbl2"), (Field("A2"), Field("A1"))),
                    ),
                ),
                (Field("B1"), Field("B2")),
            ),
            Relabel(
                Aggregate(
                    Immediate("+"),
                    Immediate(0),
                    Table(Immediate(""), (Field("A1"), Field("A2"), Field("A3"))),
                    (Field("A2"),),
                ),
                (Field("B1"), Field("B3")),
            ),
        )
    )

    expected = Plan(
        (
            MapJoin(
                Immediate("+"),
                (
                    Reorder(
                        Table(Immediate("tbl1"), (Field("B1"), Field("B2"))),
                        (Field("B1"), Field("B2")),
                    ),
                    Reorder(
                        Table(Immediate("tbl2"), (Field("B2"), Field("B1"))),
                        (Field("B1"), Field("B2")),
                    ),
                ),
            ),
            Aggregate(
                Immediate("+"),
                Immediate(0),
                Table(Immediate(""), (Field("B1"), Field("A2"), Field("B3"))),
                (Field("A2"),),
            ),
        ),
    )

    result = push_fields(plan)
    assert result == expected

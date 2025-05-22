import pytest

from finch.autoschedule import (
    isolate_aggregates,
    isolate_reformats,
    isolate_tables,
    lift_fields,
    lift_subqueries,
    pretty_labels,
    propagate_copy_queries,
    propagate_fields,
    propagate_into_reformats,
    propagate_map_queries,
    propagate_transpose_queries,
    push_fields,
)
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Immediate,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reformat,
    Relabel,
    Reorder,
    Subquery,
    Table,
)
from finch.symbolic.gensym import _sg


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


def test_propagate_fields():
    plan = Plan(
        (
            Query(
                Alias("A10"),
                MapJoin(
                    Immediate("op"),
                    (
                        Table(Immediate("tbl1"), (Field("A1"), Field("A2"))),
                        Table(Immediate("tbl2"), (Field("A2"), Field("A3"))),
                    ),
                ),
            ),
            Alias("A10"),
        )
    )

    expected = Plan(
        (
            Query(
                Alias("A10"),
                MapJoin(
                    Immediate("op"),
                    (
                        Table(Immediate("tbl1"), (Field("A1"), Field("A2"))),
                        Table(Immediate("tbl2"), (Field("A2"), Field("A3"))),
                    ),
                ),
            ),
            Relabel(Alias("A10"), (Field("A1"), Field("A2"), Field("A3"))),
        )
    )

    result = propagate_fields(plan)
    assert result == expected


@pytest.mark.parametrize(
    "node,pass_fn",
    [
        (
            Aggregate(Immediate(""), Immediate(""), Immediate(""), ()),
            isolate_aggregates,
        ),
        (Reformat(Immediate(""), Immediate("")), isolate_reformats),
        (Table(Immediate(""), ()), isolate_tables),
    ],
)
def test_isolate_passes(node, pass_fn):
    plan = Plan((node, node, node))
    expected = Plan(
        (
            Subquery(Alias(f"#A#{_sg.counter}"), node),
            Subquery(Alias(f"#A#{_sg.counter + 1}"), node),
            Subquery(Alias(f"#A#{_sg.counter + 2}"), node),
        )
    )

    result = pass_fn(plan)
    assert result == expected


def test_pretty_labels():
    plan = Plan(
        (
            Field("AA"),
            Alias("BB"),
            Alias("CC"),
            Subquery(Alias("BB"), Field("AA")),
            Subquery(Alias("CC"), Field("AA")),
        )
    )
    expected = Plan(
        (
            Field(":i0"),
            Alias(":A0"),
            Alias(":A1"),
            Subquery(Alias(":A0"), Field(":i0")),
            Subquery(Alias(":A1"), Field(":i0")),
        )
    )

    result = pretty_labels(plan)
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
            Reorder(
                Aggregate(
                    Immediate("+"),
                    Immediate(0),
                    Table(Immediate(""), (Field("A1"), Field("A2"), Field("A3"))),
                    (Field("A2"),),
                ),
                (Field("A3"), Field("A1")),
            ),
        )
    )

    expected = Plan(
        (
            MapJoin(
                op=Immediate(val="+"),
                args=(
                    Table(
                        tns=Immediate(val="tbl1"),
                        idxs=(Field(name="B1"), Field(name="B2")),
                    ),
                    Table(
                        tns=Immediate(val="tbl2"),
                        idxs=(Field(name="B1"), Field(name="B2")),
                    ),
                ),
            ),
            Aggregate(
                op=Immediate(val="+"),
                init=Immediate(val=0),
                arg=Table(
                    tns=Immediate(val=""),
                    idxs=(Field(name="B1"), Field(name="A2"), Field(name="B3")),
                ),
                idxs=(Field(name="A2"),),
            ),
            Reorder(
                Aggregate(
                    Immediate("+"),
                    Immediate(0),
                    Reorder(
                        Table(Immediate(""), (Field("A1"), Field("A2"), Field("A3"))),
                        (Field("A3"), Field("A2"), Field("A1")),
                    ),
                    (Field("A2"),),
                ),
                (Field("A3"), Field("A1")),
            ),
        )
    )

    result = push_fields(plan)
    assert result == expected


def test_propagate_copy_queries():
    plan = Plan(
        (
            Query(Alias("A0"), Alias("A0")),
            Query(Alias("A1"), Alias("A2")),
            Query(Alias("A1"), Immediate(0)),
        )
    )

    expected = Plan(
        (
            Plan(),
            Plan(),
            Query(Alias("A2"), Immediate(0)),
        )
    )

    result = propagate_copy_queries(plan)
    assert result == expected


def test_propagate_into_reformats():
    plan = Plan(
        (
            Query(Alias("A1"), Alias("A0")),
            Query(
                Alias("D0"),
                Aggregate(Immediate("*"), Immediate(1), Alias("A1"), (Field("i2"),)),
            ),
            Query(
                Alias("B0"),
                Aggregate(Immediate("+"), Immediate(0), Alias("A1"), (Field("i1"),)),
            ),
            Immediate(1),
            Query(Alias("C0"), Reformat(Immediate(3), Alias("B0"))),
            Query(Alias("E0"), Reformat(Immediate(4), Alias("D0"))),
            Immediate(2),
        )
    )

    expected = Plan(
        (
            Query(Alias("A1"), Alias("A0")),
            Query(
                Alias("E0"),
                Reformat(
                    Immediate(4),
                    Aggregate(
                        Immediate("*"), Immediate(1), Alias("A1"), (Field("i2"),)
                    ),
                ),
            ),
            Query(
                Alias("C0"),
                Reformat(
                    Immediate(3),
                    Aggregate(
                        Immediate("+"), Immediate(0), Alias("A1"), (Field("i1"),)
                    ),
                ),
            ),
            Immediate(1),
            Immediate(2),
        )
    )

    result = propagate_into_reformats(plan)
    assert result == expected


def test_propagate_transpose_queries():
    plan = Plan(
        (
            Query(
                Alias("A1"),
                Relabel(
                    Relabel(
                        Alias("XD"),
                        (Field("i1"), Field("i2")),
                    ),
                    (Field("j1"), Field("j2")),
                ),
            ),
            Query(Alias("A2"), Reorder(Alias("A1"), (Field("j2"), Field("j1")))),
            Produces((Alias("A2"),)),
        )
    )

    expected = Plan(
        (
            Plan(),
            Plan(),
            Produces(
                (
                    Reorder(
                        Relabel(Alias("XD"), (Field("j1"), Field("j2"))),
                        (Field("j2"), Field("j1")),
                    ),
                )
            ),
        )
    )

    result = propagate_transpose_queries(plan)
    assert result == expected


def test_lift_fields():
    plan = Plan(
        (
            Aggregate(
                Immediate("*"),
                Immediate(1),
                Table(Immediate(2), (Field("i1"), Field("i2"))),
                (Field("i2"),),
            ),
            Query(
                Alias("A0"),
                MapJoin(
                    Immediate("*"),
                    (
                        Table(Immediate(2), (Field("i1"), Field("i2"))),
                        Table(Immediate(4), (Field("i1"), Field("i2"))),
                    ),
                ),
            ),
            Query(
                Alias("A0"),
                Reformat(
                    Immediate(0),
                    MapJoin(
                        Immediate("*"),
                        (
                            Table(Immediate(2), (Field("i1"), Field("i2"))),
                            Table(Immediate(4), (Field("i1"), Field("i2"))),
                        ),
                    ),
                ),
            ),
        )
    )

    expected = Plan(
        (
            Aggregate(
                Immediate("*"),
                Immediate(1),
                Reorder(
                    Table(Immediate(2), (Field("i1"), Field("i2"))),
                    (Field("i1"), Field("i2")),
                ),
                (Field("i2"),),
            ),
            Query(
                Alias("A0"),
                Reorder(
                    MapJoin(
                        Immediate("*"),
                        (
                            Table(Immediate(2), (Field("i1"), Field("i2"))),
                            Table(Immediate(4), (Field("i1"), Field("i2"))),
                        ),
                    ),
                    (Field("i1"), Field("i2")),
                ),
            ),
            Query(
                Alias("A0"),
                Reformat(
                    Immediate(0),
                    Reorder(
                        MapJoin(
                            Immediate("*"),
                            (
                                Table(Immediate(2), (Field("i1"), Field("i2"))),
                                Table(Immediate(4), (Field("i1"), Field("i2"))),
                            ),
                        ),
                        (Field("i1"), Field("i2")),
                    ),
                ),
            ),
        )
    )

    result = lift_fields(plan)
    assert result == expected

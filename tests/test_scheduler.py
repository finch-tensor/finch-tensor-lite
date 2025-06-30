from operator import add, mul

import pytest

import numpy as np

from finch.autoschedule import (
    concordize,
    flatten_plans,
    isolate_aggregates,
    isolate_reformats,
    isolate_tables,
    lift_fields,
    lift_subqueries,
    materialize_squeeze_expand_productions,
    normalize_names,
    optimize,
    pretty_labels,
    propagate_copy_queries,
    propagate_fields,
    propagate_into_reformats,
    propagate_map_queries,
    propagate_map_queries_backward,
    propagate_transpose_queries,
    push_fields,
)
from finch.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
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
from finch.finch_logic.interpreter import FinchLogicInterpreter
from finch.symbolic.gensym import _sg


def test_propagate_map_queries():
    plan = Plan(
        (
            Query(
                Alias("A10"),
                Aggregate(Literal("+"), Literal(0), Literal("[1,2,3]"), ()),
            ),
            Query(Alias("A11"), Alias("A10")),
            Produces((Alias("A11"),)),
        )
    )
    expected = Plan(
        (
            Query(
                Alias("A11"),
                MapJoin(Literal("+"), (Literal(0), Literal("[1,2,3]"))),
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
                        Subquery(Alias("C10"), Literal(0)),
                        Subquery(
                            Alias("B10"),
                            MapJoin(
                                Literal("+"),
                                (
                                    Subquery(Alias("C10"), Literal(0)),
                                    Literal("[1,2,3]"),
                                ),
                            ),
                        ),
                        Subquery(Alias("B10"), Literal(0)),
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
                    Query(Alias("C10"), Literal(0)),
                    Query(
                        Alias("B10"),
                        MapJoin(Literal("+"), (Alias("C10"), Literal("[1,2,3]"))),
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
                    Literal("op"),
                    (
                        Table(Literal("tbl1"), (Field("A1"), Field("A2"))),
                        Table(Literal("tbl2"), (Field("A2"), Field("A3"))),
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
                    Literal("op"),
                    (
                        Table(Literal("tbl1"), (Field("A1"), Field("A2"))),
                        Table(Literal("tbl2"), (Field("A2"), Field("A3"))),
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
            Aggregate(Literal(""), Literal(""), Reorder(Literal(""), ()), ()),
            isolate_aggregates,
        ),
        (Reformat(Literal(""), Reorder(Literal(""), ())), isolate_reformats),
        (Table(Literal(""), ()), isolate_tables),
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
                    Literal("+"),
                    (
                        Table(Literal("tbl1"), (Field("A1"), Field("A2"))),
                        Table(Literal("tbl2"), (Field("A2"), Field("A1"))),
                    ),
                ),
                (Field("B1"), Field("B2")),
            ),
            Relabel(
                Aggregate(
                    Literal("+"),
                    Literal(0),
                    Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
                    (Field("A2"),),
                ),
                (Field("B1"), Field("B3")),
            ),
            Reorder(
                Aggregate(
                    Literal("+"),
                    Literal(0),
                    Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
                    (Field("A2"),),
                ),
                (Field("A3"), Field("A1")),
            ),
        )
    )

    expected = Plan(
        (
            MapJoin(
                op=Literal(val="+"),
                args=(
                    Table(
                        tns=Literal(val="tbl1"),
                        idxs=(Field(name="B1"), Field(name="B2")),
                    ),
                    Table(
                        tns=Literal(val="tbl2"),
                        idxs=(Field(name="B2"), Field(name="B1")),
                    ),
                ),
            ),
            Aggregate(
                op=Literal(val="+"),
                init=Literal(val=0),
                arg=Table(
                    tns=Literal(val=""),
                    idxs=(Field(name="B1"), Field(name="A2"), Field(name="B3")),
                ),
                idxs=(Field(name="A2"),),
            ),
            Reorder(
                Aggregate(
                    Literal("+"),
                    Literal(0),
                    Reorder(
                        Table(Literal(""), (Field("A1"), Field("A2"), Field("A3"))),
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
            Query(Alias("A1"), Literal(0)),
        )
    )

    expected = Plan(
        (
            Plan(),
            Plan(),
            Query(Alias("A2"), Literal(0)),
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
                Aggregate(Literal("*"), Literal(1), Alias("A1"), (Field("i2"),)),
            ),
            Query(
                Alias("B0"),
                Aggregate(Literal("+"), Literal(0), Alias("A1"), (Field("i1"),)),
            ),
            Literal(1),
            Query(Alias("C0"), Reformat(Literal(3), Alias("B0"))),
            Query(Alias("E0"), Reformat(Literal(4), Alias("D0"))),
            Literal(2),
        )
    )

    expected = Plan(
        (
            Query(Alias("A1"), Alias("A0")),
            Query(
                Alias("E0"),
                Reformat(
                    Literal(4),
                    Aggregate(Literal("*"), Literal(1), Alias("A1"), (Field("i2"),)),
                ),
            ),
            Query(
                Alias("C0"),
                Reformat(
                    Literal(3),
                    Aggregate(Literal("+"), Literal(0), Alias("A1"), (Field("i1"),)),
                ),
            ),
            Literal(1),
            Literal(2),
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
                Literal("*"),
                Literal(1),
                Table(Literal(2), (Field("i1"), Field("i2"))),
                (Field("i2"),),
            ),
            Query(
                Alias("A0"),
                MapJoin(
                    Literal("*"),
                    (
                        Table(Literal(2), (Field("i1"), Field("i2"))),
                        Table(Literal(4), (Field("i1"), Field("i2"))),
                    ),
                ),
            ),
            Query(
                Alias("A0"),
                Reformat(
                    Literal(0),
                    MapJoin(
                        Literal("*"),
                        (
                            Table(Literal(2), (Field("i1"), Field("i2"))),
                            Table(Literal(4), (Field("i1"), Field("i2"))),
                        ),
                    ),
                ),
            ),
        )
    )

    expected = Plan(
        (
            Aggregate(
                Literal("*"),
                Literal(1),
                Reorder(
                    Table(Literal(2), (Field("i1"), Field("i2"))),
                    (Field("i1"), Field("i2")),
                ),
                (Field("i2"),),
            ),
            Query(
                Alias("A0"),
                Reorder(
                    MapJoin(
                        Literal("*"),
                        (
                            Table(Literal(2), (Field("i1"), Field("i2"))),
                            Table(Literal(4), (Field("i1"), Field("i2"))),
                        ),
                    ),
                    (Field("i1"), Field("i2")),
                ),
            ),
            Query(
                Alias("A0"),
                Reformat(
                    Literal(0),
                    Reorder(
                        MapJoin(
                            Literal("*"),
                            (
                                Table(Literal(2), (Field("i1"), Field("i2"))),
                                Table(Literal(4), (Field("i1"), Field("i2"))),
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


def test_normalize_names():
    plan = Plan(
        (
            Field("##foo#8"),
            Field("##foo#1"),
            Field("#2#foo"),
            Alias("##foo#9"),
            Field("#10#A"),
            Alias("bar"),
            Field("j"),
            Alias("##test#0"),
        )
    )

    expected = Plan(
        (
            Field("foo_2"),
            Field("foo_3"),
            Field("foo_4"),
            Alias("foo"),
            Field("A"),
            Alias("bar"),
            Field("j"),
            Alias("test"),
        )
    )

    result = normalize_names(plan)
    assert result == expected


def test_concordize():
    plan = Plan(
        (
            Query(Alias("A0"), Table(Literal(0), (Field("i0"), Field("i1")))),
            Query(
                Alias("A1"),
                Reorder(
                    Relabel(Alias("A0"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i0")),
                ),
            ),
            Query(
                Alias("A2"),
                Reorder(
                    Relabel(Alias("A0"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i1")),
                ),
            ),
            Produces((Alias("A1"), Alias("A2"))),
        )
    )

    expected = Plan(
        (
            Query(Alias("A0"), Table(Literal(0), (Field("i0"), Field("i1")))),
            Query(
                Alias("A0_2"),
                Reorder(
                    Relabel(Alias("A0"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i0")),
                ),
            ),
            Query(
                Alias("A0_3"),
                Reorder(
                    Relabel(Alias("A0"), (Field("i0"), Field("i1"))),
                    (Field("i0"), Field("i1")),
                ),
            ),
            Query(
                Alias("A1"),
                Reorder(
                    Relabel(Alias("A0_2"), (Field("i1"), Field("i0"))),
                    (Field("i1"), Field("i0")),
                ),
            ),
            Query(
                Alias("A2"),
                Reorder(
                    Relabel(Alias("A0_3"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i1")),
                ),
            ),
            Produces((Alias("A1"), Alias("A2"))),
        )
    )

    result = concordize(plan)
    assert result == expected


def test_flatten_plans():
    plan = Plan(
        (
            Plan(
                (
                    Field("i0"),
                    Field("i1"),
                )
            ),
            Alias("A0"),
            Plan(
                (
                    Plan(
                        (
                            Field("i3"),
                            Produces((Alias("A1"),)),
                        )
                    ),
                )
            ),
            Field("i4"),
            Alias("A2"),
        )
    )

    expected = Plan(
        (
            Field("i0"),
            Field("i1"),
            Alias("A0"),
            Field("i3"),
            Produces((Alias("A1"),)),
        )
    )

    result = flatten_plans(plan)
    assert result == expected


def test_propagate_map_queries_backward():
    plan = Plan(
        (
            Query(Alias("A0"), Alias("A1")),
            Alias("A0"),
            MapJoin(
                Literal(mul),
                (
                    Table(Literal(10), (Field("i1"),)),
                    Aggregate(
                        Literal(add),
                        Literal(0),
                        Table(Literal(10), (Field("i1"), Field("i2"), Field("i3"))),
                        (Field("i2"),),
                    ),
                    Table(Literal(10), (Field("i3"),)),
                ),
            ),
            Aggregate(
                Literal(add),
                Literal(10),
                Aggregate(Literal(add), Literal(0), Alias("A2"), (Field("i4"),)),
                (Field("i5"),),
            ),
            Reorder(
                Aggregate(
                    Literal(mul),
                    Literal(1),
                    Table(Literal(10), (Field("i6"), Field("i7"), Field("i8"))),
                    (Field("i7"),),
                ),
                (Field("i6"), Field("i8")),
            ),
        )
    )

    expected = Plan(
        (
            Plan(bodies=()),
            Alias("A1"),
            Aggregate(
                Literal(add),
                Literal(0),
                MapJoin(
                    Literal(mul),
                    (
                        Table(Literal(10), (Field("i1"),)),
                        Table(Literal(10), (Field("i1"), Field("i2"), Field("i3"))),
                        Table(Literal(10), (Field("i3"),)),
                    ),
                ),
                (Field("i2"),),
            ),
            Aggregate(
                Literal(add), Literal(10), Alias("A2"), (Field("i4"), Field("i5"))
            ),
            Aggregate(
                Literal(mul),
                Literal(1),
                Reorder(
                    Table(
                        Literal(10),
                        (
                            Field("i6"),
                            Field("i7"),
                            Field("i8"),
                        ),
                    ),
                    (Field("i6"), Field("i7"), Field("i8")),
                ),
                (Field("i7"),),
            ),
        )
    )

    result = propagate_map_queries_backward(plan)
    assert result == expected


def test_materialize_squeeze_expand_productions():
    plan = Plan(
        (
            Produces(
                (
                    Reorder(
                        Relabel(Alias("A0"), (Field("i2"), Field("i1"))),
                        (Field("i1"), Field("i2"), Field("i3")),
                    ),
                    Reorder(
                        Relabel(Alias("A0"), (Field("i1"), Field("i2"))),
                        (Field("i1"), Field("i2")),
                    ),
                )
            ),
        )
    )

    expected = Plan(
        (
            Plan(
                (
                    Query(
                        Alias(f"#A#{_sg.counter}"),
                        Reorder(
                            Relabel(Alias("A0"), (Field("i2"), Field("i1"))),
                            (Field("i2"), Field("i1"), Field("i3")),
                        ),
                    ),
                    Produces(
                        (
                            Reorder(
                                Relabel(
                                    Alias(f"#A#{_sg.counter}"),
                                    (Field("i2"), Field("i1"), Field("i3")),
                                ),
                                (Field("i1"), Field("i2"), Field("i3")),
                            ),
                            Alias("A0"),
                        )
                    ),
                )
            ),
        )
    )

    result = materialize_squeeze_expand_productions(plan)
    assert result == expected


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        (np.array([[2, 0], [1, 3]]), np.array([[4, 1], [2, 2]])),
    ],
)
def test_scheduler_e2e_matmul(a, b):
    i, j, k = Field("i"), Field("j"), Field("k")

    plan = Plan(
        (
            Query(Alias("A"), Table(Literal(a), (i, k))),
            Query(Alias("B"), Table(Literal(b), (k, j))),
            Query(Alias("AB"), MapJoin(Literal(mul), (Alias("A"), Alias("B")))),
            Query(
                Alias("C"),
                Aggregate(Literal(add), Literal(0), Alias("AB"), (k,)),
            ),
            Produces((Alias("C"),)),
        )
    )

    plan_opt = optimize(plan)

    result = FinchLogicInterpreter()(plan_opt)[0]

    expected = np.matmul(a, b)

    np.testing.assert_equal(result, expected)


def test_scheduler_e2e_sddmm():
    s = np.array([[2, 4], [6, 0]])
    a = np.array([[1, 2], [3, 2]])
    b = np.array([[9, 8], [6, 5]])
    i, j, k = Field("i"), Field("j"), Field("k")

    plan = Plan(
        (
            Query(Alias("S"), Table(Literal(s), (i, j))),
            Query(Alias("A"), Table(Literal(a), (i, k))),
            Query(Alias("B"), Table(Literal(b), (k, j))),
            Query(Alias("AB"), MapJoin(Literal(mul), (Alias("A"), Alias("B")))),
            # matmul
            Query(Alias("C"), Aggregate(Literal(add), Literal(0), Alias("AB"), (k,))),
            # elemwise
            Query(Alias("RES"), MapJoin(Literal(mul), (Alias("C"), Alias("S")))),
            Produces((Alias("RES"),)),
        )
    )

    expected_plan = Plan(
        (
            Query(Alias(":A0"), Table(Literal(a), (Field(":i0"), Field(":i1")))),
            Query(Alias(":A1"), Table(Literal(b), (Field(":i1"), Field(":i2")))),
            Query(Alias(":A2"), Table(Literal(s), (Field(":i0"), Field(":i2")))),
            Query(
                Alias(":A3"),
                Aggregate(
                    Literal(add),
                    Literal(0),
                    Reorder(
                        MapJoin(
                            Literal(mul),
                            (
                                Reorder(
                                    MapJoin(
                                        Literal(mul),
                                        (
                                            Reorder(
                                                Relabel(
                                                    Alias(":A0"),
                                                    (Field(":i0"), Field(":i1")),
                                                ),
                                                (Field(":i0"), Field(":i1")),
                                            ),
                                            Reorder(
                                                Relabel(
                                                    Alias(":A1"),
                                                    (Field(":i1"), Field(":i2")),
                                                ),
                                                (Field(":i1"), Field(":i2")),
                                            ),
                                        ),
                                    ),
                                    (Field(":i0"), Field(":i1"), Field(":i2")),
                                ),
                                Reorder(
                                    Relabel(Alias(":A2"), (Field(":i0"), Field(":i2"))),
                                    (Field(":i0"), Field(":i2")),
                                ),
                            ),
                        ),
                        (Field(":i0"), Field(":i1"), Field(":i2")),
                    ),
                    (Field(":i1"),),
                ),
            ),
            Plan((Produces((Relabel(Alias(":A3"), (Field(":i0"), Field(":i2"))),)),)),
        )
    )

    plan_opt = optimize(plan)

    assert plan_opt == expected_plan

    result = FinchLogicInterpreter()(plan_opt)[0]

    expected = s * np.matmul(a, b)

    np.testing.assert_equal(result, expected)

from operator import add, mul

import numpy as np

import finchlite
from finchlite.autoschedule import (
    normalize_names,
)
from finchlite.autoschedule.optimize import (
    lift_fields,
    optimize,
    propagate_copy_queries,
    propagate_map_queries,
    propagate_single_use_queries_and_combine_aggregates,
    propagate_transpose_queries,
    set_loop_order,
)
from finchlite.autoschedule.standardize import (
    concordize,
    flatten_plans,
    isolate_aggregates,
    push_fields,
    standardize,
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
    Relabel,
    Reorder,
    Table,
)
from finchlite.symbolic.ftype import ftype
from finchlite.symbolic.gensym import _sg


def test_propagate_map_queries():
    plan = Plan(
        (
            Query(
                Alias("A10"),
                Aggregate(Literal("+"), Literal(0), Literal("[1,2,3]"), ()),
            ),
            Query(Alias("A11"), Table(Alias("A10"), ())),
            Produces((Alias("A11"),)),
        )
    )
    expected = Plan(
        (
            Query(
                Alias("A11"),
                Relabel(MapJoin(Literal("+"), (Literal(0), Literal("[1,2,3]"))), ()),
            ),
            Produces((Alias("A11"),)),
        )
    )

    result = propagate_map_queries(plan)
    assert result == expected


def test_propagate_single_use_queries_and_combine_aggregates():
    plan = Plan(
        (
            Query(Alias("A0"), Alias("A1")),
            Table(Alias("A0"), (Field("i0"), Field("i1"))),
            MapJoin(
                Literal(mul),
                (
                    Table(Literal(10), (Field("i2"),)),
                    Aggregate(
                        Literal(add),
                        Literal(0),
                        Table(Literal(10), (Field("i2"), Field("i3"), Field("i4"))),
                        (Field("i3"),),
                    ),
                    Table(Literal(10), (Field("i4"),)),
                ),
            ),
            Aggregate(
                Literal(add),
                Literal(10),
                Aggregate(Literal(add), Literal(0), Alias("A2"), (Field("i5"),)),
                (Field("i6"),),
            ),
            Aggregate(
                Literal(add),
                Literal(0),
                Reorder(
                    Aggregate(
                        Literal(add),
                        Literal(0),
                        Table(
                            Alias("A3"),
                            (Field("i10"), Field("i7"), Field("i9"), Field("i8")),
                        ),
                        (Field("i7"), Field("i8")),
                    ),
                    (Field("i9"), Field("i10")),
                ),
                (Field("i9"),),
            ),
            Produces(()),
        )
    )

    expected = Plan(
        (
            Plan(()),
            Relabel(Alias("A1"), (Field("i0"), Field("i1"))),
            Aggregate(
                Literal(add),
                Literal(0),
                MapJoin(
                    Literal(mul),
                    (
                        Table(Literal(10), (Field("i2"),)),
                        Table(Literal(10), (Field("i2"), Field("i3"), Field("i4"))),
                        Table(Literal(10), (Field("i4"),)),
                    ),
                ),
                (Field("i3"),),
            ),
            Aggregate(
                Literal(add), Literal(10), Alias("A2"), (Field("i5"), Field("i6"))
            ),
            Reorder(
                Aggregate(
                    Literal(add),
                    Literal(0),
                    Reorder(
                        Table(
                            Alias("A3"),
                            (Field("i10"), Field("i7"), Field("i9"), Field("i8")),
                        ),
                        (
                            Field("i9"),
                            Field("i7"),
                            Field("i10"),
                            Field("i8"),
                        ),
                    ),
                    (Field("i7"), Field("i8"), Field("i9")),
                ),
                (Field("i10"),),
            ),
            Produces(()),
        )
    )

    result = propagate_single_use_queries_and_combine_aggregates(plan)
    assert result == expected


def test_isolate_aggregates():
    plan = Plan(
        (
            Query(
                Alias("A0"),
                Aggregate(
                    Literal("+"),
                    Literal(0),
                    Aggregate(
                        Literal("*"),
                        Literal(1),
                        Table(Literal(10), (Field("i1"), Field("i2"), Field("i3"))),
                        (Field("i2"),),
                    ),
                    (Field("i1"),),
                ),
            ),
        )
    )

    expected = Plan(
        (
            Plan(
                (
                    Query(
                        Alias(f"#A#{_sg.counter}"),
                        Aggregate(
                            Literal("*"),
                            Literal(1),
                            Table(Literal(10), (Field("i1"), Field("i2"), Field("i3"))),
                            (Field("i2"),),
                        ),
                    ),
                    Query(
                        Alias("A0"),
                        Aggregate(
                            Literal("+"),
                            Literal(0),
                            Table(
                                Alias(f"#A#{_sg.counter}"), (Field("i1"), Field("i3"))
                            ),
                            (Field("i1"),),
                        ),
                    ),
                )
            ),
        )
    )

    result = isolate_aggregates(plan)
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
            Query(Alias("A0"), Table(Alias("A0"), (Field("i0"),))),
            Query(Alias("A1"), Table(Alias("A2"), (Field("i1"),))),
            Query(Alias("A1"), Table(Literal(0), (Field("i1"),))),
            Produces((Alias("A1"),)),
        )
    )

    expected = Plan(
        (
            Plan(),
            Plan(),
            Query(Alias("A2"), Table(Literal(0), (Field("i1"),))),
            Produces((Alias("A2"),)),
        )
    )

    result = propagate_copy_queries(plan)
    assert result == expected


def test_propagate_transpose_queries():
    plan = Plan(
        (
            Query(
                Alias("A1"),
                Relabel(
                    Table(
                        Alias("XD"),
                        (Field("i1"), Field("i2")),
                    ),
                    (Field("j1"), Field("j2")),
                ),
            ),
            Query(
                Alias("A2"),
                Reorder(
                    Table(Alias("A1"), (Field("j1"), Field("j2"))),
                    (Field("j2"), Field("j1")),
                ),
            ),
            Produces((Alias("A2"),)),
        )
    )

    expected = Plan(
        (
            Query(
                Alias("A2"),
                Reorder(
                    Table(Alias("XD"), (Field("j1"), Field("j2"))),
                    (Field("j2"), Field("j1")),
                ),
            ),
            Produces((Alias("A2"),)),
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
                MapJoin(
                    Literal("*"),
                    (
                        Table(Literal(2), (Field("i1"), Field("i2"))),
                        Table(Literal(4), (Field("i1"), Field("i2"))),
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
            Field("i"),
            Field("i_2"),
            Field("i_3"),
            Alias("A"),
            Field("i_4"),
            Alias("A_2"),
            Field("i_5"),
            Alias("A_3"),
        )
    )

    result, bindings = normalize_names(plan, {})
    assert result == expected


def test_concordize():
    plan = Plan(
        (
            Query(Alias("A0"), Table(Literal(0), (Field("i0"), Field("i1")))),
            Query(
                Alias("A1"),
                Reorder(
                    Table(Alias("A0"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i0")),
                ),
            ),
            Query(
                Alias("A2"),
                Reorder(
                    Table(Alias("A0"), (Field("i0"), Field("i1"))),
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
                Alias("A0_4"),
                Reorder(
                    Table(Alias("A0"), (Field("i_0"), Field("i_1"))),
                    (Field("i_1"), Field("i_0")),
                ),
            ),
            Query(
                Alias("A0_5"),
                Reorder(
                    Table(Alias("A0"), (Field("i_0"), Field("i_1"))),
                    (Field("i_0"), Field("i_1")),
                ),
            ),
            Query(
                Alias("A1"),
                Reorder(
                    Table(Alias("A0_4"), (Field("i1"), Field("i0"))),
                    (Field("i1"), Field("i0")),
                ),
            ),
            Query(
                Alias("A2"),
                Reorder(
                    Table(Alias("A0_5"), (Field("i0"), Field("i1"))),
                    (Field("i1"), Field("i1")),
                ),
            ),
            Produces((Alias("A1"), Alias("A2"))),
        )
    )

    result = concordize(plan, bindings={})
    assert result == expected


def test_set_loop_order():
    plan = Query(
        Alias("C"),
        Aggregate(
            Literal(add),
            Literal(0),
            Reorder(
                MapJoin(
                    Literal(mul),
                    (
                        Reorder(
                            Table(Alias("A"), (Field("i0"), Field("i1"))),
                            (Field("i0"), Field("i1")),
                        ),
                        Reorder(
                            Table(Alias("B"), (Field("i1"), Field("i2"))),
                            (Field("i1"), Field("i2")),
                        ),
                    ),
                ),
                (Field("i0"), Field("i2"), Field("i1")),
            ),
            (Field("i1"),),
        ),
    )

    expected = Query(
        Alias("C"),
        Aggregate(
            Literal(add),
            Literal(0),
            Reorder(
                Reorder(
                    MapJoin(
                        Literal(mul),
                        (
                            Reorder(
                                Table(Alias("A"), (Field("i0"), Field("i1"))),
                                (Field("i0"), Field("i1")),
                            ),
                            Reorder(
                                Table(Alias("B"), (Field("i1"), Field("i2"))),
                                (Field("i1"), Field("i2")),
                            ),
                        ),
                    ),
                    (Field("i0"), Field("i2"), Field("i1")),
                ),
                (Field("i0"), Field("i1"), Field("i2")),
            ),
            (Field("i1"),),
        ),
    )

    result = set_loop_order(plan)
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


def test_scheduler_e2e_matmul(file_regression):
    a = np.array([[1, 2], [3, 4]])
    b = (np.array([[5, 6], [7, 8]]),)
    i, j, k = Field("i"), Field("j"), Field("k")

    plan = Plan(
        (
            Query(
                Alias("AB"),
                MapJoin(
                    Literal(mul), (Table(Alias("A"), (i, k)), Table(Alias("B"), (k, j)))
                ),
            ),
            Query(
                Alias("C"),
                Aggregate(
                    Literal(add), Literal(0), Table(Alias("AB"), (i, k, j)), (k,)
                ),
            ),
            Produces((Alias("C"),)),
        )
    )

    plan_opt, bindings = optimize(
        plan,
        {
            Alias("A"): ftype(finchlite.asarray(a)),
            Alias("B"): ftype(finchlite.asarray(b)),
        },
    )
    plan_opt, bindings = standardize(plan_opt, bindings)

    file_regression.check(
        str(plan_opt), extension=".txt", basename="test_scheduler_e2e_matmul_plan"
    )


def test_scheduler_e2e_sddmm(file_regression):
    s = np.array([[2, 4], [6, 0]])
    a = np.array([[1, 2], [3, 2]])
    b = np.array([[9, 8], [6, 5]])
    i, j, k = Field("i"), Field("j"), Field("k")

    plan = Plan(
        (
            Query(
                Alias("AB"),
                MapJoin(
                    Literal(mul), (Table(Alias("A"), (i, j)), Table(Alias("B"), (k, j)))
                ),
            ),
            # matmul
            Query(
                Alias("C"),
                Aggregate(
                    Literal(add), Literal(0), Table(Alias("AB"), (i, k, j)), (k,)
                ),
            ),
            # elemwise
            Query(
                Alias("RES"),
                MapJoin(
                    Literal(mul), (Table(Alias("C"), (i, j)), Table(Alias("S"), (j, i)))
                ),
            ),
            Produces((Alias("RES"),)),
        )
    )

    plan_opt, bindings = optimize(
        plan,
        {
            Alias("S"): ftype(finchlite.asarray(s)),
            Alias("A"): ftype(finchlite.asarray(a)),
            Alias("B"): ftype(finchlite.asarray(b)),
        },
    )
    plan_opt, bindings = standardize(plan_opt, bindings)

    file_regression.check(
        str(plan_opt), extension=".txt", basename="test_scheduler_e2e_sddmm_plan"
    )

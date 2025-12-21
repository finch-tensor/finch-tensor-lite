import operator

import numpy as np

import finchlite.finch_logic as logic
from finchlite import ftype
from finchlite.autoschedule import NotationGenerator
from finchlite.compile.bufferized_ndarray import (
    BufferizedNDArray,
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
    TableValue,
)
from finchlite.interface import INTERPRET_NOTATION, OPTIMIZE_LOGIC

from .conftest import finch_assert_equal


def test_logic_compiler(file_regression):
    plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name=":A2"),
                rhs=Aggregate(
                    op=logic.Literal(val=operator.add),
                    init=logic.Literal(val=0),
                    arg=Reorder(
                        arg=MapJoin(
                            op=logic.Literal(val=operator.mul),
                            args=(
                                Relabel(
                                    arg=Alias(name=":A0"),
                                    idxs=(Field(name=":i0"), Field(name=":i1")),
                                ),
                                Relabel(
                                    arg=Alias(name=":A1"),
                                    idxs=(Field(name=":i1"), Field(name=":i2")),
                                ),
                            ),
                        ),
                        idxs=(Field(name=":i0"), Field(name=":i1"), Field(name=":i2")),
                    ),
                    idxs=(Field(name=":i1"),),
                ),
            ),
            Plan(
                bodies=(Produces(args=(Alias(name=":A2"),)),),
            ),
        ),
    )

    bindings = {
        Alias(name=":A0"): TableValue(
            BufferizedNDArray(np.array([[1, 2], [3, 4]])),
            (Field(name=":i0"), Field(name=":i1")),
        ),
        Alias(name=":A1"): TableValue(
            BufferizedNDArray(np.array([[5, 6], [7, 8]])),
            (Field(name=":i1"), Field(name=":i2")),
        ),
        Alias(name=":A2"): TableValue(
            BufferizedNDArray(np.array([[5, 6], [7, 8]])),
            (Field(name=":i0"), Field(name=":i2")),
        ),
    }

    program = NotationGenerator()(
        plan, {var: ftype(val) for var, val in bindings.items()}
    )

    file_regression.check(
        str(program), extension=".txt", basename="test_logic_compiler_program"
    )

    result = INTERPRET_NOTATION(plan, bindings)

    expected = np.matmul(
        bindings[Alias(name=":A0")].tns.to_numpy(),
        bindings[Alias(name=":A1")].tns.to_numpy(),
        dtype=float,
    )

    finch_assert_equal(result[0].tns.to_numpy(), expected)


def test_logic_compiler_pipeline():
    plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name="#A#0"),
                rhs=Table(
                    tns=Literal(
                        val=BufferizedNDArray(val=np.array([[1, 2, 3], [0, 2, 1]]))
                    ),
                    idxs=(Field(name="#i#1"), Field(name="#i#2")),
                ),
            ),
            Query(
                lhs=Alias(name="#A#3"),
                rhs=Table(
                    tns=Literal(
                        val=BufferizedNDArray(val=np.array([[1, 2], [0, 3], [4, 0]]))
                    ),
                    idxs=(Field(name="#i#4"), Field(name="#i#5")),
                ),
            ),
            Query(
                lhs=Alias(name="#A#17"),
                rhs=Reorder(
                    arg=MapJoin(
                        op=Literal(val=operator.mul),
                        args=(
                            Reorder(
                                arg=Relabel(
                                    arg=Reorder(
                                        arg=Relabel(
                                            arg=Alias(name="#A#0"),
                                            idxs=(
                                                Field(name="#i#6"),
                                                Field(name="#i#7"),
                                            ),
                                        ),
                                        idxs=(
                                            Field(name="#i#6"),
                                            Field(name="#i#7"),
                                            Field(name="#i#8"),
                                        ),
                                    ),
                                    idxs=(
                                        Field(name="#i#12"),
                                        Field(name="#i#13"),
                                        Field(name="#j#15"),
                                    ),
                                ),
                                idxs=(Field(name="#i#12"), Field(name="#i#13")),
                            ),
                            Reorder(
                                arg=Relabel(
                                    arg=Reorder(
                                        arg=Relabel(
                                            arg=Alias(name="#A#3"),
                                            idxs=(
                                                Field(name="#i#9"),
                                                Field(name="#i#10"),
                                            ),
                                        ),
                                        idxs=(
                                            Field(name="#i#11"),
                                            Field(name="#i#9"),
                                            Field(name="#i#10"),
                                        ),
                                    ),
                                    idxs=(
                                        Field(name="#j#16"),
                                        Field(name="#i#13"),
                                        Field(name="#i#14"),
                                    ),
                                ),
                                idxs=(Field(name="#i#13"), Field(name="#i#14")),
                            ),
                        ),
                    ),
                    idxs=(
                        Field(name="#i#12"),
                        Field(name="#i#13"),
                        Field(name="#i#14"),
                    ),
                ),
            ),
            Query(
                lhs=Alias(name="#A#21"),
                rhs=Aggregate(
                    op=Literal(val=operator.add),
                    init=Literal(val=np.float64(0.0)),
                    arg=Relabel(
                        arg=Alias(name="#A#17"),
                        idxs=(
                            Field(name="#i#18"),
                            Field(name="#i#19"),
                            Field(name="#i#20"),
                        ),
                    ),
                    idxs=(Field(name="#i#19"),),
                ),
            ),
            Plan(bodies=()),
            Query(lhs=Alias(name="#A#22"), rhs=Alias(name="#A#21")),
            Produces(args=(Alias(name="#A#22"),)),
        )
    )

    expected_plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name="A"),
                rhs=Reorder(
                    arg=Alias(name="A_11"), idxs=(Field(name="i"), Field(name="i_2"))
                ),
            ),
            Query(
                lhs=Alias(name="A_2"),
                rhs=Reorder(
                    arg=Alias(name="A_12"), idxs=(Field(name="i_3"), Field(name="i_4"))
                ),
            ),
            Query(
                lhs=Alias(name="#A#23"),
                rhs=Aggregate(
                    op=Literal(val=operator.add),
                    init=Literal(val=np.float64(0.0)),
                    arg=Relabel(
                        arg=Reorder(
                            arg=MapJoin(
                                op=Literal(val=operator.mul),
                                args=(
                                    Relabel(
                                        arg=Alias(name="A"),
                                        idxs=(Field(name="i_8"), Field(name="i_9")),
                                    ),
                                    Relabel(
                                        arg=Alias(name="A_2"),
                                        idxs=(Field(name="i_9"), Field(name="i_15")),
                                    ),
                                ),
                            ),
                            idxs=(
                                Field(name="i_8"),
                                Field(name="i_9"),
                                Field(name="i_15"),
                            ),
                        ),
                        idxs=(
                            Field(name="i_16"),
                            Field(name="i_17"),
                            Field(name="i_18"),
                        ),
                    ),
                    idxs=(Field(name="i_17"),),
                ),
            ),
            Query(
                lhs=Alias(name="A_4"),
                rhs=Reorder(
                    arg=Alias(name="#A#23"),
                    idxs=(Field(name="i_16"), Field(name="i_18")),
                ),
            ),
            Produces(args=(Alias(name="A_4"),)),
        )
    )

    ctx = OPTIMIZE_LOGIC
    debug_ctx = {}
    ctx(plan, debug_ctx=debug_ctx)

    assert debug_ctx["post_logic_standardizer"] == expected_plan

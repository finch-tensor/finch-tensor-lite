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
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    TableValue,
)
from finchlite.interface import INTERPRET_NOTATION

from .conftest import finch_assert_equal, reset_name_counts


def test_logic_compiler(file_regression):
    plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name="A2"),
                rhs=Aggregate(
                    op=logic.Literal(val=operator.add),
                    init=logic.Literal(val=0),
                    arg=Reorder(
                        arg=MapJoin(
                            op=logic.Literal(val=operator.mul),
                            args=(
                                Relabel(
                                    arg=Alias(name="A0"),
                                    idxs=(Field(name="i0"), Field(name="i1")),
                                ),
                                Relabel(
                                    arg=Alias(name="A1"),
                                    idxs=(Field(name="i1"), Field(name="i2")),
                                ),
                            ),
                        ),
                        idxs=(Field(name="i0"), Field(name="i1"), Field(name="i2")),
                    ),
                    idxs=(Field(name="i1"),),
                ),
            ),
            Plan(
                bodies=(Produces(args=(Alias(name="A2"),)),),
            ),
        ),
    )

    bindings = {
        Alias(name="A0"): TableValue(
            BufferizedNDArray(np.array([[1, 2], [3, 4]])),
            (Field(name="i0"), Field(name="i1")),
        ),
        Alias(name="A1"): TableValue(
            BufferizedNDArray(np.array([[5, 6], [7, 8]])),
            (Field(name="i1"), Field(name="i2")),
        ),
        Alias(name="A2"): TableValue(
            BufferizedNDArray(np.array([[5, 6], [7, 8]])),
            (Field(name="i0"), Field(name="i2")),
        ),
    }

    program = NotationGenerator()(
        plan, {var: ftype(val) for var, val in bindings.items()}
    )

    file_regression.check(
        reset_name_counts(str(program)),
        extension=".txt",
        basename="test_logic_compiler_program",
    )

    result = INTERPRET_NOTATION(plan, bindings)

    expected = np.matmul(
        bindings[Alias(name="A0")].tns.to_numpy(),
        bindings[Alias(name="A1")].tns.to_numpy(),
        dtype=float,
    )

    finch_assert_equal(result[0].tns.to_numpy(), expected)

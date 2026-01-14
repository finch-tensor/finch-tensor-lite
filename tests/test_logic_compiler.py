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
    Reorder,
    Table,
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
                                Table(
                                    Alias(name="A0"),
                                    (Field(name="i0"), Field(name="i1")),
                                ),
                                Table(
                                    Alias(name="A1"),
                                    (Field(name="i1"), Field(name="i2")),
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
        Alias(name="A0"): BufferizedNDArray(np.array([[1, 2], [3, 4]])),
        Alias(name="A1"): BufferizedNDArray(np.array([[5, 6], [7, 8]])),
        Alias(name="A2"): BufferizedNDArray(np.array([[5, 6], [7, 8]])),
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
        bindings[Alias(name="A0")].to_numpy(),
        bindings[Alias(name="A1")].to_numpy(),
        dtype=float,
    )

    finch_assert_equal(result[0].to_numpy(), expected)

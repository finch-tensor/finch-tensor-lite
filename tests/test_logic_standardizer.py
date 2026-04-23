import numpy as np

import finchlite.finch_logic as logic
from finchlite import ffunc, ftype
from finchlite.autoschedule import LogicStandardizer
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
    Reorder,
    Table,
)

from .conftest import reset_name_counts


def test_logic_standardizer_inplace(file_regression):
    plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name="A2"),
                rhs=Reorder(
                    arg=MapJoin(
                        op=Literal(ffunc.add),
                        args=(
                            Reorder(
                                arg=Table(
                                    Alias("A2"), (Field(name="i0"), Field(name="i2"))
                                ),
                                idxs=(Field(name="i0"), Field(name="i2")),
                            ),
                            Aggregate(
                                op=logic.Literal(val=ffunc.add),
                                init=logic.Literal(val=0),
                                arg=Reorder(
                                    arg=MapJoin(
                                        op=logic.Literal(val=ffunc.mul),
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
                                    idxs=(
                                        Field(name="i0"),
                                        Field(name="i1"),
                                        Field(name="i2"),
                                    ),
                                ),
                                idxs=(Field(name="i1"),),
                            ),
                        ),
                    ),
                    idxs=(Field(name="i0"), Field(name="i2")),
                ),
            ),
            Plan(
                bodies=(Produces(args=(Alias(name="A2"),)),),
            ),
        ),
    )

    bindings = {
        Alias(name="A0"): BufferizedNDArray.from_numpy(np.array([[1, 2], [3, 4]])),
        Alias(name="A1"): BufferizedNDArray.from_numpy(np.array([[5, 6], [7, 8]])),
        Alias(name="A2"): BufferizedNDArray.from_numpy(np.array([[1, 1], [1, 1]])),
    }

    program = LogicStandardizer()(
        plan, {var: ftype(val) for var, val in bindings.items()}, None, None
    )

    file_regression.check(
        reset_name_counts(str(program)),
        extension=".txt",
        basename="test_logic_standardizer_inplace_program",
    )

import operator

import numpy as np

import finchlite.finch_logic as logic
from finchlite.finch_logic import TableValue
from finchlite.autoschedule import (
    LogicCompiler,
    NotationGenerator
)
from finchlite.codegen.numpy_buffer import NumpyBufferFType
from finchlite.compile import dimension
from finchlite.compile.bufferized_ndarray import (
    BufferizedNDArray,
    BufferizedNDArrayFType,
)
from finchlite.interface import INTERPRET_NOTATION
from finchlite import ftype
from finchlite.compile.lower import ExtentFType
from finchlite.finch_assembly.struct import TupleFType
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
    Table,
)
from finchlite.finch_notation import (
    Access,
    Assign,
    Block,
    Call,
    Declare,
    Freeze,
    Function,
    Increment,
    Literal,
    Loop,
    Module,
    NotationInterpreter,
    Read,
    Return,
    Slot,
    Unpack,
    Unwrap,
    Update,
    Variable,
)
from finchlite.finch_notation.nodes import Repack
from finchlite.interface.fuse import provision_tensors

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
                bodies=(
                    Produces(
                        args=(Alias(name=":A2"),)
                    ),
                ),
            )
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

    program = NotationGenerator()(plan, {var:ftype(val) for var, val in bindings.items()})

    file_regression.check(str(program), extension=".txt", basename="test_logic_compiler_program")

    result = INTERPRET_NOTATION(plan, bindings)

    expected = np.matmul(bindings[Alias(name=":A0")].tns.to_numpy(), bindings[Alias(name=":A1")].tns.to_numpy(), dtype=float)

    finch_assert_equal(result[0].tns.to_numpy(), expected)



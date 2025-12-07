import operator

import numpy as np

import finchlite.finch_logic as logic
from finchlite.autoschedule import (
    LogicCompiler,
)
from finchlite.codegen.numpy_buffer import NumpyBufferFType
from finchlite.compile import dimension
from finchlite.compile.bufferized_ndarray import (
    BufferizedNDArray,
    BufferizedNDArrayFType,
)
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

from .util import finch_assert_equal


def test_logic_compiler():
    plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name=":A0"),
                rhs=Table(
                    tns=logic.Literal(
                        val=BufferizedNDArray(np.array([[1, 2], [3, 4]]))
                    ),
                    idxs=(Field(name=":i0"), Field(name=":i1")),
                ),
            ),
            Query(
                lhs=Alias(name=":A1"),
                rhs=Table(
                    tns=logic.Literal(
                        val=BufferizedNDArray(np.array([[5, 6], [7, 8]]))
                    ),
                    idxs=(Field(name=":i1"), Field(name=":i2")),
                ),
            ),
            Query(
                lhs=Alias(name=":A2"),
                rhs=Aggregate(
                    op=logic.Literal(val=operator.add),
                    init=logic.Literal(val=0),
                    arg=Reorder(
                        arg=MapJoin(
                            op=logic.Literal(val=operator.mul),
                            args=(
                                Reorder(
                                    arg=Relabel(
                                        arg=Alias(name=":A0"),
                                        idxs=(Field(name=":i0"), Field(name=":i1")),
                                    ),
                                    idxs=(Field(name=":i0"), Field(name=":i1")),
                                ),
                                Reorder(
                                    arg=Relabel(
                                        arg=Alias(name=":A1"),
                                        idxs=(Field(name=":i1"), Field(name=":i2")),
                                    ),
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
                        args=(
                            Relabel(
                                arg=Alias(name=":A2"),
                                idxs=(Field(name=":i0"), Field(name=":i2")),
                            ),
                        )
                    ),
                )
            ),
        )
    )

    bufferized_ndarray_ftype = BufferizedNDArrayFType(
        buffer_type=NumpyBufferFType(np.dtype(int)),
        ndim=np.intp(2),
        dimension_type=TupleFType.from_tuple((np.intp, np.intp)),
    )

    expected_program = Module(
        funcs=(
            Function(
                name=Variable(name="func", type_=bufferized_ndarray_ftype),
                args=(
                    Variable(name=":A0", type_=bufferized_ndarray_ftype),
                    Variable(name=":A1", type_=bufferized_ndarray_ftype),
                    Variable(name=":A2", type_=bufferized_ndarray_ftype),
                ),
                body=Block(
                    bodies=(
                        Assign(
                            lhs=Variable(
                                name=":i0_size", type_=ExtentFType(np.intp, np.intp)
                            ),
                            rhs=Call(
                                op=Literal(val=dimension),
                                args=(
                                    Variable(
                                        name=":A0",
                                        type_=bufferized_ndarray_ftype,
                                    ),
                                    Literal(val=0),
                                ),
                            ),
                        ),
                        Assign(
                            lhs=Variable(
                                name=":i1_size", type_=ExtentFType(np.intp, np.intp)
                            ),
                            rhs=Call(
                                op=Literal(val=dimension),
                                args=(
                                    Variable(
                                        name=":A0",
                                        type_=bufferized_ndarray_ftype,
                                    ),
                                    Literal(val=1),
                                ),
                            ),
                        ),
                        Assign(
                            lhs=Variable(
                                name=":i2_size", type_=ExtentFType(np.intp, np.intp)
                            ),
                            rhs=Call(
                                op=Literal(val=dimension),
                                args=(
                                    Variable(
                                        name=":A1",
                                        type_=bufferized_ndarray_ftype,
                                    ),
                                    Literal(val=1),
                                ),
                            ),
                        ),
                        Unpack(
                            Slot(name=":A0_slot", type=bufferized_ndarray_ftype),
                            Variable(name=":A0", type_=bufferized_ndarray_ftype),
                        ),
                        Unpack(
                            Slot(name=":A1_slot", type=bufferized_ndarray_ftype),
                            Variable(name=":A1", type_=bufferized_ndarray_ftype),
                        ),
                        Unpack(
                            Slot(name=":A2_slot", type=bufferized_ndarray_ftype),
                            Variable(name=":A2", type_=bufferized_ndarray_ftype),
                        ),
                        Declare(
                            tns=Slot(name=":A2_slot", type=bufferized_ndarray_ftype),
                            init=Literal(val=0),
                            op=Literal(val=operator.add),
                            shape=(
                                Variable(
                                    name=":i0_size", type_=ExtentFType(np.intp, np.intp)
                                ),
                                Variable(
                                    name=":i2_size", type_=ExtentFType(np.intp, np.intp)
                                ),
                            ),
                        ),
                        Loop(
                            idx=Variable(name=":i0", type_=np.intp),
                            ext=Variable(
                                name=":i0_size", type_=ExtentFType(np.intp, np.intp)
                            ),
                            body=Loop(
                                idx=Variable(name=":i1", type_=np.intp),
                                ext=Variable(
                                    name=":i1_size", type_=ExtentFType(np.intp, np.intp)
                                ),
                                body=Loop(
                                    idx=Variable(name=":i2", type_=np.intp),
                                    ext=Variable(
                                        name=":i2_size",
                                        type_=ExtentFType(np.intp, np.intp),
                                    ),
                                    body=Block(
                                        bodies=(
                                            Increment(
                                                lhs=Access(
                                                    tns=Slot(
                                                        name=":A2_slot",
                                                        type=bufferized_ndarray_ftype,
                                                    ),
                                                    mode=Update(
                                                        op=Literal(val=operator.add)
                                                    ),
                                                    idxs=(
                                                        Variable(
                                                            name=":i0", type_=np.intp
                                                        ),
                                                        Variable(
                                                            name=":i2", type_=np.intp
                                                        ),
                                                    ),
                                                ),
                                                rhs=Call(
                                                    op=Literal(val=operator.mul),
                                                    args=(
                                                        Unwrap(
                                                            arg=Access(
                                                                tns=Slot(
                                                                    name=":A0_slot",
                                                                    type=bufferized_ndarray_ftype,
                                                                ),
                                                                mode=Read(),
                                                                idxs=(
                                                                    Variable(
                                                                        name=":i0",
                                                                        type_=np.intp,
                                                                    ),
                                                                    Variable(
                                                                        name=":i1",
                                                                        type_=np.intp,
                                                                    ),
                                                                ),
                                                            )
                                                        ),
                                                        Unwrap(
                                                            arg=Access(
                                                                tns=Slot(
                                                                    name=":A1_slot",
                                                                    type=bufferized_ndarray_ftype,
                                                                ),
                                                                mode=Read(),
                                                                idxs=(
                                                                    Variable(
                                                                        name=":i1",
                                                                        type_=np.intp,
                                                                    ),
                                                                    Variable(
                                                                        name=":i2",
                                                                        type_=np.intp,
                                                                    ),
                                                                ),
                                                            )
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        )
                                    ),
                                ),
                            ),
                        ),
                        Freeze(
                            tns=Slot(name=":A2_slot", type=bufferized_ndarray_ftype),
                            op=Literal(val=operator.add),
                        ),
                        Repack(
                            val=Slot(name=":A0_slot", type=bufferized_ndarray_ftype),
                            obj=Variable(name=":A0", type_=bufferized_ndarray_ftype),
                        ),
                        Repack(
                            val=Slot(name=":A1_slot", type=bufferized_ndarray_ftype),
                            obj=Variable(name=":A1", type_=bufferized_ndarray_ftype),
                        ),
                        Repack(
                            val=Slot(name=":A2_slot", type=bufferized_ndarray_ftype),
                            obj=Variable(name=":A2", type_=bufferized_ndarray_ftype),
                        ),
                        Return(
                            val=Variable(name=":A2", type_=bufferized_ndarray_ftype)
                        ),
                    )
                ),
            ),
        )
    )

    program, table_vars, tables = LogicCompiler()(plan)

    assert program == expected_program

    mod = NotationInterpreter()(program)

    args = provision_tensors(program, table_vars, tables)
    result = mod.func(*args)

    expected = np.matmul(args[0].to_numpy(), args[1].to_numpy(), dtype=float)

    finch_assert_equal(result.to_numpy(), expected)

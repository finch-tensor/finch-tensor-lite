import operator

import numpy as np

import finch.finch_logic as logic
from finch.algebra.tensor import NDArrayFType
from finch.autoschedule import (
    LogicCompiler,
)
from finch.compile import dimension
from finch.finch_logic import (
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
from finch.finch_notation import (
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
from finch.finch_notation.nodes import Repack


def test_logic_compiler():
    plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name=":A0"),
                rhs=Table(
                    tns=logic.Literal(val=np.array([[1, 2], [3, 4]])),
                    idxs=(Field(name=":i0"), Field(name=":i1")),
                ),
            ),
            Query(
                lhs=Alias(name=":A1"),
                rhs=Table(
                    tns=logic.Literal(val=np.array([[5, 6], [7, 8]])),
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

    expected_program = Module(
        funcs=(
            Function(
                name=Variable(name="func", type_=np.ndarray),
                args=(
                    Variable(name=":A0", type_=NDArrayFType(np.dtype(int), 2)),
                    Variable(name=":A1", type_=NDArrayFType(np.dtype(int), 2)),
                    Variable(name=":A2", type_=NDArrayFType(np.dtype(int), 2)),
                ),
                body=Block(
                    bodies=(
                        Assign(
                            lhs=Variable(name=":i0_size", type_=int),
                            rhs=Call(
                                op=Literal(val=dimension),
                                args=(
                                    Variable(
                                        name=":A0",
                                        type_=NDArrayFType(np.dtype(int), 2),
                                    ),
                                    Literal(val=0),
                                ),
                            ),
                        ),
                        Assign(
                            lhs=Variable(name=":i1_size", type_=int),
                            rhs=Call(
                                op=Literal(val=dimension),
                                args=(
                                    Variable(
                                        name=":A0",
                                        type_=NDArrayFType(np.dtype(int), 2),
                                    ),
                                    Literal(val=1),
                                ),
                            ),
                        ),
                        Assign(
                            lhs=Variable(name=":i2_size", type_=int),
                            rhs=Call(
                                op=Literal(val=dimension),
                                args=(
                                    Variable(
                                        name=":A1",
                                        type_=NDArrayFType(np.dtype(int), 2),
                                    ),
                                    Literal(val=1),
                                ),
                            ),
                        ),
                        Unpack(
                            Slot(name=":A0_slot", type=NDArrayFType(np.dtype(int), 2)),
                            Variable(name=":A0", type_=NDArrayFType(np.dtype(int), 2)),
                        ),
                        Unpack(
                            Slot(name=":A1_slot", type=NDArrayFType(np.dtype(int), 2)),
                            Variable(name=":A1", type_=NDArrayFType(np.dtype(int), 2)),
                        ),
                        Unpack(
                            Slot(name=":A2_slot", type=NDArrayFType(np.dtype(int), 2)),
                            Variable(name=":A2", type_=NDArrayFType(np.dtype(int), 2)),
                        ),
                        Declare(
                            tns=Slot(
                                name=":A2_slot", type=NDArrayFType(np.dtype(int), 2)
                            ),
                            init=Literal(val=0),
                            op=Literal(val=operator.add),
                            shape=(
                                Variable(name=":i0_size", type_=int),
                                Variable(name=":i2_size", type_=int),
                            ),
                        ),
                        Loop(
                            idx=Variable(name=":i2", type_=int),
                            ext=Variable(name=":i2_size", type_=int),
                            body=Loop(
                                idx=Variable(name=":i1", type_=int),
                                ext=Variable(name=":i1_size", type_=int),
                                body=Loop(
                                    idx=Variable(name=":i0", type_=int),
                                    ext=Variable(name=":i0_size", type_=int),
                                    body=Block(
                                        bodies=(
                                            Increment(
                                                lhs=Access(
                                                    tns=Slot(
                                                        name=":A2_slot",
                                                        type=NDArrayFType(
                                                            np.dtype(int), 2
                                                        ),
                                                    ),
                                                    mode=Update(
                                                        op=Literal(val=operator.add)
                                                    ),
                                                    idxs=(
                                                        Variable(name=":i0", type_=int),
                                                        Variable(name=":i2", type_=int),
                                                    ),
                                                ),
                                                rhs=Call(
                                                    op=Literal(val=operator.mul),
                                                    args=(
                                                        Unwrap(
                                                            arg=Access(
                                                                tns=Slot(
                                                                    name=":A0_slot",
                                                                    type=NDArrayFType(
                                                                        np.dtype(int), 2
                                                                    ),
                                                                ),
                                                                mode=Read(),
                                                                idxs=(
                                                                    Variable(
                                                                        name=":i0",
                                                                        type_=int,
                                                                    ),
                                                                    Variable(
                                                                        name=":i1",
                                                                        type_=int,
                                                                    ),
                                                                ),
                                                            )
                                                        ),
                                                        Unwrap(
                                                            arg=Access(
                                                                tns=Slot(
                                                                    name=":A1_slot",
                                                                    type=NDArrayFType(
                                                                        np.dtype(int), 2
                                                                    ),
                                                                ),
                                                                mode=Read(),
                                                                idxs=(
                                                                    Variable(
                                                                        name=":i1",
                                                                        type_=int,
                                                                    ),
                                                                    Variable(
                                                                        name=":i2",
                                                                        type_=int,
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
                            tns=Slot(
                                name=":A2_slot", type=NDArrayFType(np.dtype(int), 2)
                            ),
                            op=Literal(val=operator.add),
                        ),
                        Repack(
                            val=Slot(
                                name=":A0_slot", type=NDArrayFType(np.dtype(int), 2)
                            ),
                            obj=Variable(
                                name=":A0", type_=NDArrayFType(np.dtype(int), 2)
                            ),
                        ),
                        Repack(
                            val=Slot(
                                name=":A1_slot", type=NDArrayFType(np.dtype(int), 2)
                            ),
                            obj=Variable(
                                name=":A1", type_=NDArrayFType(np.dtype(int), 2)
                            ),
                        ),
                        Repack(
                            val=Slot(
                                name=":A2_slot", type=NDArrayFType(np.dtype(int), 2)
                            ),
                            obj=Variable(
                                name=":A2", type_=NDArrayFType(np.dtype(int), 2)
                            ),
                        ),
                        Return(
                            val=Variable(
                                name=":A2", type_=NDArrayFType(np.dtype(int), 2)
                            )
                        ),
                    )
                ),
            ),
        )
    )

    program, tables = LogicCompiler()(plan)

    assert program == expected_program

    mod = NotationInterpreter()(program)
    args = [tables[logic.Alias(arg.name)].tns.val for arg in program.funcs[0].args]

    result = mod.func(*args)

    expected = np.matmul(args[0], args[1], dtype=float)

    np.testing.assert_equal(result, expected)

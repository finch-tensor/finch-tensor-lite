import operator

import numpy as np

import finch.finch_logic as logic
import finch.finch_notation as ntn
from finch.autoschedule import (
    LogicCompiler,
)
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
    Loop,
    Module,
    Read,
    Return,
    Unwrap,
    Update,
    Variable,
)


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
                    Variable(name=":A0", type_=np.ndarray),
                    Variable(name=":A1", type_=np.ndarray),
                    Variable(name=":A2", type_=np.ndarray),
                ),
                body=Block(
                    bodies=(
                        Assign(
                            lhs=Variable(name=":i0_size", type_=int),
                            rhs=Call(
                                op=ntn.Literal(val=ntn.dimension),
                                args=(
                                    Variable(name=":A0", type_=np.ndarray),
                                    ntn.Literal(val=0),
                                ),
                            ),
                        ),
                        Assign(
                            lhs=Variable(name=":i1_size", type_=int),
                            rhs=Call(
                                op=ntn.Literal(val=ntn.dimension),
                                args=(
                                    Variable(name=":A0", type_=np.ndarray),
                                    ntn.Literal(val=1),
                                ),
                            ),
                        ),
                        Assign(
                            lhs=Variable(name=":i2_size", type_=int),
                            rhs=Call(
                                op=ntn.Literal(val=ntn.dimension),
                                args=(
                                    Variable(name=":A1", type_=np.ndarray),
                                    ntn.Literal(val=1),
                                ),
                            ),
                        ),
                        Assign(
                            lhs=Variable(name=":A2", type_=np.ndarray),
                            rhs=Declare(
                                tns=Variable(name=":A2", type_=np.ndarray),
                                init=ntn.Literal(val=0),
                                op=ntn.Literal(val=operator.add),
                                shape=(
                                    Variable(name=":i0_size", type_=int),
                                    Variable(name=":i2_size", type_=int),
                                ),
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
                                                    tns=Variable(
                                                        name=":A2", type_=np.ndarray
                                                    ),
                                                    mode=Update(
                                                        op=ntn.Literal(val=operator.add)
                                                    ),
                                                    idxs=(
                                                        Variable(name=":i0", type_=int),
                                                        Variable(name=":i2", type_=int),
                                                    ),
                                                ),
                                                rhs=Call(
                                                    op=ntn.Literal(val=operator.mul),
                                                    args=(
                                                        Unwrap(
                                                            arg=Access(
                                                                tns=Variable(
                                                                    name=":A0",
                                                                    type_=np.ndarray,
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
                                                                tns=Variable(
                                                                    name=":A1",
                                                                    type_=np.ndarray,
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
                        Assign(
                            lhs=Variable(name=":A2", type_=np.ndarray),
                            rhs=Freeze(
                                tns=Variable(name=":A2", type_=np.ndarray),
                                op=ntn.Literal(val=operator.add),
                            ),
                        ),
                        Return(val=Variable(name=":A2", type_=np.ndarray)),
                    )
                ),
            ),
        )
    )

    program, tables = LogicCompiler()(plan)

    assert program == expected_program

    mod = ntn.NotationInterpreter()(program)
    args = [tables[logic.Alias(arg.name)] for arg in program.funcs[0].args]

    result = mod.func(*args)

    expected = np.matmul(args[0], args[1], dtype=float)

    np.testing.assert_equal(result, expected)

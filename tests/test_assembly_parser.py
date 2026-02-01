import operator

import numpy as np

from finchlite.finch_assembly import (
    Assign,
    Block,
    Call,
    ForLoop,
    Literal,
    Load,
    Store,
    Variable,
    parse_assembly,
)


def test_parser():
    lvl_ptr = Variable("lvl_ptr", np.ndarray)
    pos_stop = Variable("pos_stop", int)
    qos_stop = Variable("qos_stop", int)
    lvl_idx = Variable("lvl_idx", np.ndarray)
    p = Variable("p", int)

    expr = """finch
    resize(lvl_ptr, pos_stop + 1)
    for (p in 0:pos_stop)
        lvl_ptr[p + 1] += lvl_ptr[p]
    end
    qos_stop = lvl_ptr[pos_stop] - 1
    // some comment
    resize(lvl_idx, qos_stop)
    """

    result = parse_assembly(expr, locals())

    expected = Block(
        (
            Call(
                op=Literal(val=np.resize),
                args=(
                    Variable(name="lvl_ptr", type=np.ndarray),
                    Call(
                        op=Literal(val=operator.add),
                        args=(Variable(name="pos_stop", type=int), Literal(val=1)),
                    ),
                ),
            ),
            ForLoop(
                var=Variable(name="p", type=int),
                start=Literal(val=0),
                end=Variable(name="pos_stop", type=int),
                body=Block(
                    bodies=(
                        Store(
                            buffer=Variable(name="lvl_ptr", type=np.ndarray),
                            index=Call(
                                op=Literal(val=operator.add),
                                args=(Variable(name="p", type=int), Literal(val=1)),
                            ),
                            value=Call(
                                op=Literal(val=operator.add),
                                args=(
                                    Load(
                                        buffer=Variable(
                                            name="lvl_ptr", type=np.ndarray
                                        ),
                                        index=Call(
                                            op=Literal(val=operator.add),
                                            args=(
                                                Variable(name="p", type=int),
                                                Literal(val=1),
                                            ),
                                        ),
                                    ),
                                    Load(
                                        buffer=Variable(
                                            name="lvl_ptr", type=np.ndarray
                                        ),
                                        index=Variable(name="p", type=int),
                                    ),
                                ),
                            ),
                        ),
                    )
                ),
            ),
            Assign(
                lhs=Variable(name="qos_stop", type=int),
                rhs=Call(
                    op=Literal(val=operator.sub),
                    args=(
                        Load(
                            buffer=Variable(name="lvl_ptr", type=np.ndarray),
                            index=Variable(name="pos_stop", type=int),
                        ),
                        Literal(val=1),
                    ),
                ),
            ),
            Call(
                op=Literal(val=np.resize),
                args=(
                    Variable(name="lvl_idx", type=np.ndarray),
                    Variable(name="qos_stop", type=int),
                ),
            ),
        )
    )

    assert result == expected

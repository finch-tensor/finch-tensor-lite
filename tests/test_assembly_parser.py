import operator

import numpy as np

from finchlite.finch_assembly import (
    Assign,
    Block,
    Call,
    ForLoop,
    IfElse,
    Literal,
    Load,
    Store,
    Variable,
    parse_assembly,
)


def test_for_loop():
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


def test_if_statement():
    lvl_ptr = Variable("lvl_ptr", np.ndarray)
    lvl_idx = Variable("lvl_idx", np.ndarray)
    pos = Variable("pos", int)
    q = Variable("q", int)
    q_stop = Variable("q_stop", int)
    i = Variable("i", int)
    i1 = Variable("i1", int)

    expr = """finch
    q = lvl_ptr[pos]
    q_stop = lvl_ptr[pos + 1]
    if (q < q_stop)
        i = lvl_idx[q]
        i1 = lvl_idx[q_stop - 1]
    else
        i = 1
        i1 = 0
    end
    """

    result = parse_assembly(expr, locals())

    expected = Block(
        (
            Assign(q, Load(lvl_ptr, pos)),
            Assign(
                q_stop,
                Load(lvl_ptr, Call(Literal(operator.add), (pos, Literal(np.intp(1))))),
            ),
            IfElse(
                Call(Literal(operator.lt), (q, q_stop)),
                Block(
                    (
                        Assign(i, Load(lvl_idx, q)),
                        Assign(
                            i1,
                            Load(
                                lvl_idx,
                                Call(
                                    Literal(operator.sub), (q_stop, Literal(np.intp(1)))
                                ),
                            ),
                        ),
                    )
                ),
                Block(
                    (
                        Assign(i, Literal(np.intp(1))),
                        Assign(i1, Literal(np.intp(0))),
                    )
                ),
            ),
        )
    )

    assert result == expected

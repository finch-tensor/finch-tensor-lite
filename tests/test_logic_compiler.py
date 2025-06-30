import operator

import numpy as np

from finch.autoschedule import (
    LogicCompiler,
)
from finch.finch_logic import (
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
)


def test_logic_compiler():
    plan = Plan(
        bodies=(
            Query(
                lhs=Alias(name=":A0"),
                rhs=Table(
                    tns=Literal(val=np.array([[1, 2], [3, 4]])),
                    idxs=(Field(name=":i0"), Field(name=":i1")),
                ),
            ),
            Query(
                lhs=Alias(name=":A1"),
                rhs=Table(
                    tns=Literal(val=np.array([[5, 6], [7, 8]])),
                    idxs=(Field(name=":i1"), Field(name=":i2")),
                ),
            ),
            Query(
                lhs=Alias(name=":A1_2"),
                rhs=Reorder(
                    arg=Relabel(
                        arg=Alias(name=":A1"),
                        idxs=(Field(name=":i1"), Field(name=":i2")),
                    ),
                    idxs=(Field(name=":i2"), Field(name=":i1")),
                ),
            ),
            Query(
                lhs=Alias(name=":A2"),
                rhs=Aggregate(
                    op=Literal(val=operator.add),
                    init=Literal(val=0),
                    arg=Reorder(
                        arg=MapJoin(
                            op=Literal(val=operator.mul),
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
                                        arg=Alias(name=":A1_2"),
                                        idxs=(Field(name=":i2"), Field(name=":i1")),
                                    ),
                                    idxs=(Field(name=":i2"), Field(name=":i1")),
                                ),
                            ),
                        ),
                        idxs=(Field(name=":i0"), Field(name=":i2"), Field(name=":i1")),
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

    prgm = LogicCompiler()(plan)

    from pprint import pprint

    pprint(prgm)
    assert False

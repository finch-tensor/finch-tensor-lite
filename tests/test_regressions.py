"""
Tests to check if regression fixtures work as expected.
"""

import operator

import numpy as np

import finch.finch_logic as logic
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


def test_c_program(file_regression):
    """
    Example test for a C program using the program_regression fixture.
    This test will generate a program and compare it against the stored regression.
    """
    # Your C program logic here
    program = 'int main() { int a= 5; printf("Hello, World!"); return 0; }'
    # Compare the generated program against the stored regression
    file_regression.check(program, extension=".c")


def test_file_regression(file_regression):
    content = "This is a test file content."
    file_regression.check(content)


def test_tree_regression(file_regression):
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
    program, tables = LogicCompiler()(plan)
    file_regression.check(str(program), extension=".txt")

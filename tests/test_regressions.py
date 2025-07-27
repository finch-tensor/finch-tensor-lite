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


def test_c_program(program_regression):
    """
    Example test for a C program using the program_regression fixture.
    This test will generate a program and compare it against the stored regression.
    """
    # Your C program logic here
    program = 'int main() { int a= 5; printf("Hello, World!"); return 0; }'
    # Compare the generated program against the stored regression
    program_regression(program, extension=".c")


def test_file_regression(file_regression):
    content = "This is a test file content."
    file_regression.check(content)


def test_tree_regression(program_regression):
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
    program_regression(program)


def test_regex_substitution_patterns(program_regression):
    """
    Test comprehensive regex substitution functionality including:
    - Global substitution from conftest.py
    - Custom substitutions
    - Multiple pattern applications
    """
    program = (
        "ProcessResult(\n"
        "  objects=[\n"
        "    Object(name='obj1', id=12345, func=<function process at 0x7f8c2c0d1e50>),"
        "\n"
        "    Object(name='obj2', id=67890, func=<function validate at 0x7f8c2c0d2a80>)"
        "\n"
        "  ],\n"
        "  status='complete'\n"
        ")"
    )

    substitutions = {
        r"Object\(name='[^']+', id=\d+,": r"Object(name='...', id=...,",
        r"status='[^']+'": r"status='...'",
    }

    program_regression(program, substitutions=substitutions, extension=".txt")

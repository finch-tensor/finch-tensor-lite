import _operator  # noqa: F401
import operator
from collections import namedtuple

import pytest

import numpy  # noqa: F401, ICN001
import numpy as np

from finch import finch_assembly as asm
from finch.codegen import NumpyBuffer
from finch.finch_assembly import (  # noqa: F401
    AssemblyInterpreter,
    Assign,
    Block,
    Call,
    Function,
    If,
    IfElse,
    Literal,
    Module,
    Return,
    Variable,
)
from finch.symbolic import ftype


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([1, 2, 3], dtype=np.float64), np.array([4, 5, 6], dtype=np.float64)),
        (np.array([0], dtype=np.float64), np.array([7], dtype=np.float64)),
        (
            np.array([1.5, 2.5], dtype=np.float64),
            np.array([3.5, 4.5], dtype=np.float64),
        ),
    ],
)
def test_dot_product(a, b):
    # Simple dot product using numpy for expected result
    c = asm.Variable("c", np.float64)
    i = asm.Variable("i", np.int64)
    ab = NumpyBuffer(a)
    bb = NumpyBuffer(b)
    ab_v = asm.Variable("a", ab.ftype)
    ab_slt = asm.Slot("a_", ab.ftype)
    bb_v = asm.Variable("b", bb.ftype)
    bb_slt = asm.Slot("b_", bb.ftype)

    mod = AssemblyInterpreter()(
        asm.Module(
            (
                asm.Function(
                    asm.Variable("dot_product", np.float64),
                    (
                        ab_v,
                        bb_v,
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, asm.Literal(np.float64(0.0))),
                            asm.Unpack(ab_slt, ab_v),
                            asm.Unpack(bb_slt, bb_v),
                            asm.ForLoop(
                                i,
                                asm.Literal(np.int64(0)),
                                asm.Length(ab_slt),
                                asm.Block(
                                    (
                                        asm.Assign(
                                            c,
                                            asm.Call(
                                                asm.Literal(operator.add),
                                                (
                                                    c,
                                                    asm.Call(
                                                        asm.Literal(operator.mul),
                                                        (
                                                            asm.Load(ab_slt, i),
                                                            asm.Load(bb_slt, i),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                            asm.Repack(ab_slt),
                            asm.Repack(bb_slt),
                            asm.Return(c),
                        )
                    ),
                ),
            )
        )
    )

    result = mod.dot_product(ab, bb)
    expected = np.dot(a, b)
    assert np.allclose(result, expected)


def test_if_statement():
    var = asm.Variable("a", np.int64)
    root = asm.Module(
        (
            asm.Function(
                asm.Variable("if_else", np.int64),
                (),
                asm.Block(
                    (
                        asm.Assign(var, asm.Literal(np.int64(5))),
                        asm.If(
                            asm.Call(
                                asm.Literal(operator.eq),
                                (var, asm.Literal(np.int64(5))),
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(operator.add),
                                            (var, asm.Literal(np.int64(10))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.IfElse(
                            asm.Call(
                                asm.Literal(operator.lt),
                                (var, asm.Literal(np.int64(15))),
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(operator.sub),
                                            (var, asm.Literal(np.int64(3))),
                                        ),
                                    ),
                                )
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(operator.mul),
                                            (var, asm.Literal(np.int64(2))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.Return(var),
                    )
                ),
            ),
        )
    )

    mod = AssemblyInterpreter()(root)

    result = mod.if_else()
    assert result == 30

    assert root == eval(repr(root))


def test_simple_struct():
    Point = namedtuple("Point", ["x", "y"])
    p = Point(np.float64(1.0), np.float64(2.0))
    x = (1, 4)

    p_var = asm.Variable("p", ftype(p))
    x_var = asm.Variable("x", ftype(x))
    res_var = asm.Variable("res", np.float64)
    mod = AssemblyInterpreter()(
        asm.Module(
            (
                asm.Function(
                    asm.Variable("simple_struct", np.float64),
                    (p_var, x_var),
                    asm.Block(
                        (
                            asm.Assign(
                                res_var,
                                asm.Call(
                                    asm.Literal(operator.mul),
                                    (
                                        asm.GetAttr(p_var, asm.Literal("x")),
                                        asm.GetAttr(x_var, asm.Literal("element_0")),
                                    ),
                                ),
                            ),
                            asm.Assign(
                                res_var,
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (
                                        res_var,
                                        asm.Call(
                                            asm.Literal(operator.mul),
                                            (
                                                asm.GetAttr(p_var, asm.Literal("y")),
                                                asm.GetAttr(
                                                    x_var, asm.Literal("element_1")
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            asm.Return(res_var),
                        )
                    ),
                ),
            ),
        )
    )

    result = mod.simple_struct(p, x)
    assert result == 9.0

import operator
from textwrap import dedent

import numpy as np

import finch.finch_assembly as asm
from finch.codegen.numpy_buffer import NumpyBuffer
from finch.finch_assembly.printer import PrinterCompiler


def test_printer_if():
    pc = PrinterCompiler()

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

    actual = pc(root)

    expected = dedent("""\
    def if_else() -> int64:
        a: int64 = 5
        if eq(a, 5):
            a: int64 = add(a, 10)
        if lt(a, 15):
            a: int64 = sub(a, 3)
        else:
            a: int64 = mul(a, 2)
        return a
    """)

    assert expected == actual


def test_printer_dot():
    pc = PrinterCompiler()

    c = asm.Variable("c", np.float64)
    i = asm.Variable("i", np.int64)
    ab = NumpyBuffer(np.array([1, 2, 3], dtype=np.float64))
    bb = NumpyBuffer(np.array([4, 5, 6], dtype=np.float64))
    ab_v = asm.Variable("a", ab.format)
    ab_slt = asm.Slot("a_", ab.format)
    bb_v = asm.Variable("b", bb.format)
    bb_slt = asm.Slot("b_", bb.format)

    prgm = asm.Module(
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

    actual = pc(prgm)

    expected = dedent("""\
    def dot_product(a: format(float64), b: format(float64)) -> float64:
        c: float64 = 0.0
        a_: format(float64) = unpack(a)
        b_: format(float64) = unpack(b)
        for i in range(0, length(slot(a_, format(float64)))):
            c: float64 = add(c, mul(load(slot(a_, format(float64)), i), load(slot(b_, format(float64)), i)))
        repack(a_)
        repack(b_)
        return c
    """)  # noqa: E501

    assert expected == actual

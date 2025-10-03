import operator

import numpy as np

import finchlite.finch_assembly as asm
from finchlite.codegen.numpy_buffer import NumpyBuffer
from finchlite.finch_assembly.assembly_dataflow import build_finch_assembly_cfg


def test_asm_cfg_printer_if(file_regression):
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

    root = asm.nodes.number_assembly_ast(root)
    cfg = build_finch_assembly_cfg(root)
    file_regression.check(str(cfg), extension=".txt")


def test_asm_cfg_printer_dot(file_regression):
    c = asm.Variable("c", np.float64)
    i = asm.Variable("i", np.int64)
    ab = NumpyBuffer(np.array([1, 2, 3], dtype=np.float64))
    bb = NumpyBuffer(np.array([4, 5, 6], dtype=np.float64))
    ab_v = asm.Variable("a", ab.ftype)
    ab_slt = asm.Slot("a_", ab.ftype)
    bb_v = asm.Variable("b", bb.ftype)
    bb_slt = asm.Slot("b_", bb.ftype)

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

    prgm = asm.nodes.number_assembly_ast(prgm)
    cfg = build_finch_assembly_cfg(prgm)
    file_regression.check(str(cfg), extension=".txt")

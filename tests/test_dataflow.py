import operator

import numpy as np

import finchlite.finch_assembly as asm
from finchlite.codegen.numpy_buffer import NumpyBuffer
from finchlite.finch_assembly.cfg_builder import assembly_build_cfg
from finchlite.finch_assembly.dataflow import assembly_copy_propagation_debug


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

    cfg = assembly_build_cfg(root)
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

    cfg = assembly_build_cfg(prgm)
    file_regression.check(str(cfg), extension=".txt")


def test_asm_copy_propagation_if(file_regression):
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

    copy_propagation = assembly_copy_propagation_debug(root)
    file_regression.check(str(copy_propagation), extension=".txt")


def test_asm_copy_propagation_dot(file_regression):
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

    copy_propagation = assembly_copy_propagation_debug(prgm)
    file_regression.check(str(copy_propagation), extension=".txt")


def test_asm_cfg_printer_comprehensive(file_regression):
    a = asm.Variable("a", np.int64)
    b = asm.Variable("b", np.int64)
    c = asm.Variable("c", np.int64)
    d = asm.Variable("d", np.int64)
    result = asm.Variable("result", np.int64)
    i = asm.Variable("i", np.int64)
    j = asm.Variable("j", np.int64)
    temp = asm.Variable("temp", np.int64)

    helper_func = asm.Function(
        asm.Variable("compute", np.int64),
        (asm.Variable("x", np.int64), asm.Variable("y", np.int64)),
        asm.Block(
            (
                asm.Assign(temp, asm.Variable("x", np.int64)),
                asm.Assign(
                    temp,
                    asm.Call(
                        asm.Literal(operator.add),
                        (temp, asm.Variable("y", np.int64)),
                    ),
                ),
                asm.Return(temp),
            )
        ),
    )

    main_func = asm.Function(
        asm.Variable("main", np.int64),
        (),
        asm.Block(
            (
                asm.Assign(a, asm.Literal(np.int64(10))),
                asm.Assign(b, a),
                asm.Assign(c, b),
                asm.Assign(d, asm.Literal(np.int64(5))),
                asm.If(
                    asm.Call(
                        asm.Literal(operator.gt),
                        (c, asm.Literal(np.int64(5))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(
                                a,
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (a, d),
                                ),
                            ),
                            asm.Assign(b, a),
                        )
                    ),
                ),
                asm.Assign(result, asm.Literal(np.int64(0))),
                asm.ForLoop(
                    i,
                    asm.Literal(np.int64(0)),
                    asm.Literal(np.int64(5)),
                    asm.Block(
                        (
                            asm.Assign(temp, i),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (result, temp),
                                ),
                            ),
                            asm.If(
                                asm.Call(
                                    asm.Literal(operator.eq),
                                    (
                                        asm.Call(
                                            asm.Literal(operator.mod),
                                            (i, asm.Literal(np.int64(2))),
                                        ),
                                        asm.Literal(np.int64(0)),
                                    ),
                                ),
                                asm.Block(
                                    (
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(operator.mul),
                                                (result, asm.Literal(np.int64(2))),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                asm.IfElse(
                    asm.Call(
                        asm.Literal(operator.gt),
                        (result, asm.Literal(np.int64(20))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, result),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (c, b),
                                ),
                            ),
                        )
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, b),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(operator.mul),
                                    (c, asm.Literal(np.int64(3))),
                                ),
                            ),
                        )
                    ),
                ),
                asm.ForLoop(
                    i,
                    asm.Literal(np.int64(0)),
                    asm.Literal(np.int64(3)),
                    asm.Block(
                        (
                            asm.Assign(a, i),
                            asm.ForLoop(
                                j,
                                asm.Literal(np.int64(0)),
                                asm.Literal(np.int64(2)),
                                asm.Block(
                                    (
                                        asm.Assign(b, j),
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(operator.add),
                                                (
                                                    result,
                                                    asm.Call(
                                                        asm.Literal(operator.add),
                                                        (a, b),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                asm.Assign(d, result),
                asm.IfElse(
                    asm.Call(
                        asm.Literal(operator.lt),
                        (d, asm.Literal(np.int64(100))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, d),
                            asm.IfElse(
                                asm.Call(
                                    asm.Literal(operator.gt),
                                    (c, asm.Literal(np.int64(50))),
                                ),
                                asm.Block(
                                    (
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(operator.mul),
                                                (c, asm.Literal(np.int64(2))),
                                            ),
                                        ),
                                    )
                                ),
                                asm.Block((asm.Assign(result, c),)),
                            ),
                        )
                    ),
                    asm.Block((asm.Assign(result, d),)),
                ),
                asm.Assign(a, result),
                asm.Assign(b, a),
                asm.Return(b),
            )
        ),
    )

    root = asm.Module((helper_func, main_func))

    cfg = assembly_build_cfg(root)
    file_regression.check(str(cfg), extension=".txt")


def test_asm_copy_propagation_comprehensive(file_regression):
    a = asm.Variable("a", np.int64)
    b = asm.Variable("b", np.int64)
    c = asm.Variable("c", np.int64)
    d = asm.Variable("d", np.int64)
    result = asm.Variable("result", np.int64)
    i = asm.Variable("i", np.int64)
    j = asm.Variable("j", np.int64)
    temp = asm.Variable("temp", np.int64)

    helper_func = asm.Function(
        asm.Variable("compute", np.int64),
        (asm.Variable("x", np.int64), asm.Variable("y", np.int64)),
        asm.Block(
            (
                asm.Assign(temp, asm.Variable("x", np.int64)),
                asm.Assign(
                    temp,
                    asm.Call(
                        asm.Literal(operator.add),
                        (temp, asm.Variable("y", np.int64)),
                    ),
                ),
                asm.Return(temp),
            )
        ),
    )

    main_func = asm.Function(
        asm.Variable("main", np.int64),
        (),
        asm.Block(
            (
                asm.Assign(a, asm.Literal(np.int64(10))),
                asm.Assign(b, a),
                asm.Assign(c, b),
                asm.Assign(d, asm.Literal(np.int64(5))),
                asm.If(
                    asm.Call(
                        asm.Literal(operator.gt),
                        (c, asm.Literal(np.int64(5))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(
                                a,
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (a, d),
                                ),
                            ),
                            asm.Assign(b, a),
                        )
                    ),
                ),
                asm.Assign(result, asm.Literal(np.int64(0))),
                asm.ForLoop(
                    i,
                    asm.Literal(np.int64(0)),
                    asm.Literal(np.int64(5)),
                    asm.Block(
                        (
                            asm.Assign(temp, i),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (result, temp),
                                ),
                            ),
                            asm.If(
                                asm.Call(
                                    asm.Literal(operator.eq),
                                    (
                                        asm.Call(
                                            asm.Literal(operator.mod),
                                            (i, asm.Literal(np.int64(2))),
                                        ),
                                        asm.Literal(np.int64(0)),
                                    ),
                                ),
                                asm.Block(
                                    (
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(operator.mul),
                                                (result, asm.Literal(np.int64(2))),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                asm.IfElse(
                    asm.Call(
                        asm.Literal(operator.gt),
                        (result, asm.Literal(np.int64(20))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, result),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (c, b),
                                ),
                            ),
                        )
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, b),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(operator.mul),
                                    (c, asm.Literal(np.int64(3))),
                                ),
                            ),
                        )
                    ),
                ),
                asm.ForLoop(
                    i,
                    asm.Literal(np.int64(0)),
                    asm.Literal(np.int64(3)),
                    asm.Block(
                        (
                            asm.Assign(a, i),
                            asm.ForLoop(
                                j,
                                asm.Literal(np.int64(0)),
                                asm.Literal(np.int64(2)),
                                asm.Block(
                                    (
                                        asm.Assign(b, j),
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(operator.add),
                                                (
                                                    result,
                                                    asm.Call(
                                                        asm.Literal(operator.add),
                                                        (a, b),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                asm.Assign(d, result),
                asm.IfElse(
                    asm.Call(
                        asm.Literal(operator.lt),
                        (d, asm.Literal(np.int64(100))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, d),
                            asm.IfElse(
                                asm.Call(
                                    asm.Literal(operator.gt),
                                    (c, asm.Literal(np.int64(50))),
                                ),
                                asm.Block(
                                    (
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(operator.mul),
                                                (c, asm.Literal(np.int64(2))),
                                            ),
                                        ),
                                    )
                                ),
                                asm.Block((asm.Assign(result, c),)),
                            ),
                        )
                    ),
                    asm.Block((asm.Assign(result, d),)),
                ),
                asm.Assign(a, result),
                asm.Assign(b, a),
                asm.Return(b),
            )
        ),
    )

    root = asm.Module((helper_func, main_func))

    copy_propagation = assembly_copy_propagation_debug(root)
    file_regression.check(str(copy_propagation), extension=".txt")

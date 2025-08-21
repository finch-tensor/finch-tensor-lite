import operator
from collections import namedtuple

import pytest

import numpy as np

import finch.finch_assembly as asm
from finch.codegen import NumpyBuffer
from finch.symbolic import FType, ftype


def test_lit_basic():
    checker = asm.AssemblyTypeChecker()
    assert checker(asm.Literal(np.float64(1.0))) is np.float64
    assert checker(asm.Literal(True)) is bool


def test_var_basic():
    checker = asm.AssemblyTypeChecker()
    checker.ctxt["x"] = np.float64
    x_type = checker(asm.Variable("x", np.float64))
    assert x_type == np.float64
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Variable("y", np.float64))
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Variable("x", float))
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Variable("x", 42))


def test_slot_basic():
    checker = asm.AssemblyTypeChecker()
    b = NumpyBuffer(np.array([1, 2, 3]))
    checker.ctxt["b"] = b.ftype
    b_type = checker(asm.Slot("b", b.ftype))
    assert isinstance(b_type, FType)
    assert b_type == b.ftype
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Slot("b", float))
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Slot("b", 42))


def test_getattr_basic():
    checker = asm.AssemblyTypeChecker()
    p = (1, "one")
    p_var = asm.Variable("p", ftype(p))
    checker.ctxt["p"] = ftype(p)
    assert checker(asm.GetAttr(p_var, asm.Literal("element_0"))) is int
    assert checker(asm.GetAttr(p_var, asm.Literal("element_1"))) is str
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.GetAttr(p_var, asm.Literal("element_3")))
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.GetAttr(p_var, asm.Literal("x")))
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.GetAttr(asm.Literal("not a struct"), asm.Literal("element_0")))
    with pytest.raises(ValueError):
        checker(asm.GetAttr(p_var, "x"))


def test_call_basic():
    checker = asm.AssemblyTypeChecker()
    assert (
        checker(
            asm.Call(
                asm.Literal(operator.add),
                (
                    asm.Literal(np.float64(2.0)),
                    asm.Literal(np.float64(3.0)),
                ),
            )
        )
        == np.float64
    )
    assert (
        checker(asm.Call(asm.Literal(np.sin), (asm.Literal(np.float64(3.0)),)))
        == np.float64
    )
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Call(asm.Literal(np.sin), (asm.Literal("string"),)))


def test_load_basic():
    checker = asm.AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1.0]))
    checker.ctxt["a"] = a.ftype
    assert (
        checker(asm.Load(asm.Slot("a", a.ftype), asm.Literal(np.int64(0))))
        == np.float64
    )
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Load(asm.Slot("a", a.ftype), asm.Literal(0.0)))
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Load(asm.Literal(0.0), asm.Literal(0)))


def test_length_basic():
    checker = asm.AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1, 2, 3]))
    checker.ctxt["a"] = a.ftype
    assert checker(asm.Length(asm.Slot("a", a.ftype))) == np.int64
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Length(asm.Literal(0.0)))


def test_unpack_basic():
    checker = asm.AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1, 2, 3]))
    var_a = asm.Variable("a", a.ftype)
    slot_a = asm.Slot("a_", a.ftype)
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Unpack(slot_a, var_a))
    checker.ctxt["a"] = a.ftype
    assert checker(asm.Unpack(slot_a, var_a)) is None
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Unpack(slot_a, var_a))


def test_repack_basic():
    checker = asm.AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1, 2, 3]))
    slot_a = asm.Slot("a_", a.ftype)
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Repack(slot_a))
    checker.ctxt["a_"] = a.ftype
    assert checker(asm.Repack(slot_a)) is None
    checker.ctxt["a_"] = int
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Repack(slot_a))
    with pytest.raises(ValueError):
        checker(asm.Repack(asm.Literal(np.int64(42))))


def test_assign_basic():
    checker = asm.AssemblyTypeChecker()
    assert (
        checker(asm.Assign(asm.Variable("x", np.float64), asm.Literal(np.float64(2.0))))
        is None
    )
    assert checker(asm.Variable("x", np.float64)) is np.float64
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.Assign(
                asm.Variable("x", asm.Literal(np.float64(2.0))),
                asm.Literal(np.float64(2.0)),
            )
        )
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Assign(asm.Variable("x", np.float64), asm.Literal(True)))


def test_setattr_basic():
    checker = asm.AssemblyTypeChecker()
    Point = namedtuple("Point", ["x", "y"])
    p = Point(np.float64(1.0), True)
    p_var = asm.Variable("p", ftype(p))
    checker.ctxt["p"] = ftype(p)
    assert (
        checker(asm.SetAttr(p_var, asm.Literal("x"), asm.Literal(np.float64(2.0))))
        is None
    )
    assert checker(asm.SetAttr(p_var, asm.Literal("y"), asm.Literal(False))) is None
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.SetAttr(p_var, asm.Literal("x"), asm.Literal(1)))
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.SetAttr(p_var, asm.Literal("z"), asm.Literal(1)))
    with pytest.raises(ValueError):
        checker(asm.SetAttr(p_var, "x", asm.Literal(np.float64(3.0))))
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.SetAttr(
                asm.Literal("not a struct"),
                asm.Literal("x"),
                asm.Literal(np.float64(2.0)),
            )
        )


def test_store_basic():
    checker = asm.AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1, 2, 3]))
    checker.ctxt["a"] = a.ftype
    assert (
        checker(
            asm.Store(
                asm.Slot("a", a.ftype),
                asm.Literal(np.int64(0)),
                asm.Literal(np.int64(42)),
            )
        )
        is None
    )
    assert (
        checker(asm.Load(asm.Slot("a", a.ftype), asm.Literal(np.int64(0)))) is np.int64
    )
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.Store(asm.Slot("a", a.ftype), asm.Literal(0), asm.Literal(np.int64(42)))
        )
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.Store(asm.Slot("a", a.ftype), asm.Literal(np.int64(0)), asm.Literal(42))
        )
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.Store(
                asm.Literal(0.0), asm.Literal(np.int64(0)), asm.Literal(np.int64(42))
            )
        )


def test_resize_basic():
    checker = asm.AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1, 2, 3]))
    checker.ctxt["a"] = a.ftype
    assert (
        checker(asm.Resize(asm.Slot("a", a.ftype), asm.Literal(np.int64(20)))) is None
    )
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Resize(asm.Slot("a", a.ftype), asm.Literal(20)))
    with pytest.raises(asm.AssemblyTypeError):
        checker(asm.Resize(asm.Literal(0.0), asm.Literal(20)))


def test_forloop_basic():
    checker = asm.AssemblyTypeChecker()
    assert (
        checker(
            asm.ForLoop(
                asm.Variable("x", np.int64),
                asm.Literal(np.int64(0)),
                asm.Literal(np.int64(10)),
                asm.Assign(
                    asm.Variable("i", np.int64),
                    asm.Variable("x", np.int64),
                ),
            )
        )
        is None
    )
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.ForLoop(
                asm.Variable("x", np.float64),
                asm.Literal(np.float64(0)),
                asm.Literal(np.float64(10)),
                asm.Assign(
                    asm.Variable("i", np.float64),
                    asm.Variable("x", np.float64),
                ),
            )
        )
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.ForLoop(
                asm.Variable("x", int),
                asm.Literal(np.int64(0)),
                asm.Literal(np.int64(10)),
                asm.Assign(
                    asm.Variable("i", np.int64),
                    asm.Variable("x", np.int64),
                ),
            )
        )
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.ForLoop(
                asm.Variable("x", int),
                asm.Literal(0),
                asm.Literal(np.int64(10)),
                asm.Assign(
                    asm.Variable("i", np.int64),
                    asm.Variable("x", np.int64),
                ),
            )
        )


def test_bufferloop_basic():
    checker = asm.AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1.0, 2.0, 3.0]))
    checker.ctxt["a"] = a.ftype
    assert (
        checker(
            asm.BufferLoop(
                asm.Slot("a", a.ftype),
                asm.Variable("x", np.float64),
                asm.Assign(
                    asm.Variable("i", np.float64),
                    asm.Variable("x", np.float64),
                ),
            )
        )
        is None
    )
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.BufferLoop(
                asm.Slot("a", a.ftype),
                asm.Variable("x", np.int64),
                asm.Assign(
                    asm.Variable("i", np.float64),
                    asm.Variable("x", np.float64),
                ),
            )
        )


def test_whileloop_basic():
    checker = asm.AssemblyTypeChecker()
    assert (
        checker(
            asm.WhileLoop(
                asm.Call(
                    asm.Literal(operator.and_),
                    (
                        asm.Literal(True),
                        asm.Literal(0),
                    ),
                ),
                asm.Assign(asm.Variable("x", int), asm.Literal(0)),
            )
        )
        is None
    )
    a = NumpyBuffer(np.array([1.0, 2.0, 3.0]))
    checker.ctxt["a"] = a.ftype
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.WhileLoop(
                asm.Slot("a", a.ftype),
                asm.Assign(asm.Variable("x", int), asm.Literal(0)),
            )
        )


def test_if_basic():
    checker = asm.AssemblyTypeChecker()
    assert (
        checker(
            asm.If(
                asm.Call(
                    asm.Literal(operator.and_),
                    (
                        asm.Literal(True),
                        asm.Literal(0),
                    ),
                ),
                asm.Assign(asm.Variable("x", int), asm.Literal(0)),
            )
        )
        is None
    )
    a = NumpyBuffer(np.array([1.0, 2.0, 3.0]))
    checker.ctxt["a"] = a.ftype
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.If(
                asm.Slot("a", a.ftype),
                asm.Assign(asm.Variable("x", int), asm.Literal(0)),
            )
        )


def test_ifelse_basic():
    checker = asm.AssemblyTypeChecker()
    assert (
        checker(
            asm.IfElse(
                asm.Call(
                    asm.Literal(operator.and_),
                    (
                        asm.Literal(True),
                        asm.Literal(0),
                    ),
                ),
                asm.Assign(asm.Variable("x", int), asm.Literal(0)),
                asm.Assign(asm.Variable("x", int), asm.Literal(1)),
            )
        )
        is None
    )
    a = NumpyBuffer(np.array([1.0, 2.0, 3.0]))
    checker.ctxt["a"] = a.ftype
    with pytest.raises(asm.AssemblyTypeError):
        checker(
            asm.IfElse(
                asm.Slot("a", a.ftype),
                asm.Assign(asm.Variable("x", int), asm.Literal(0)),
                asm.Assign(asm.Variable("x", int), asm.Literal(1)),
            )
        )


def test_function_basic():
    checker = asm.AssemblyTypeChecker()
    fun = asm.Function(
        asm.Variable("add", np.int64),
        (
            asm.Variable("x", np.int64),
            asm.Variable("y", np.int64),
        ),
        asm.Return(
            asm.Call(
                asm.Literal(operator.add),
                (
                    asm.Variable("x", np.int64),
                    asm.Variable("y", np.int64),
                ),
            )
        ),
    )
    assert checker(fun) is None
    with pytest.raises(asm.AssemblyTypeError):
        fun = asm.Function(
            asm.Variable("add", np.float64),
            (
                asm.Variable("x", np.int64),
                asm.Variable("y", np.int64),
            ),
            asm.Return(
                asm.Call(
                    asm.Literal(operator.add),
                    (
                        asm.Variable("x", np.int64),
                        asm.Variable("y", np.int64),
                    ),
                )
            ),
        )
        checker(fun)
    with pytest.raises(asm.AssemblyTypeError):
        other_fun = asm.Function(
            asm.Variable("sub", np.float64),
            (
                asm.Variable("x", np.int64),
                asm.Variable("y", np.int64),
            ),
            asm.Block(
                (
                    fun,
                    asm.Return(
                        asm.Call(
                            asm.Literal(operator.sub),
                            (
                                asm.Variable("x", np.int64),
                                asm.Variable("y", np.int64),
                            ),
                        )
                    ),
                )
            ),
        )
        checker(other_fun)


def test_return_basic():
    checker = asm.AssemblyTypeChecker()
    fun = asm.Function(
        asm.Variable("foo", np.int64), (), asm.Return(asm.Literal(np.int64(0)))
    )
    assert checker(fun) is None
    with pytest.raises(asm.AssemblyTypeError):
        fun = asm.Function(
            asm.Variable("foo", np.int64),
            (),
            asm.If(asm.Literal(True), asm.Return(asm.Literal(np.int64(0)))),
        )
        checker(fun)
    fun = asm.Function(
        asm.Variable("foo", np.int64),
        (),
        asm.Block(
            (
                asm.If(asm.Literal(True), asm.Return(asm.Literal(np.int64(0)))),
                asm.Return(asm.Literal(np.int64(1))),
            )
        ),
    )
    assert checker(fun) is None
    with pytest.raises(asm.AssemblyTypeError):
        fun = asm.Function(
            asm.Variable("foo", np.int64),
            (),
            asm.Block(
                (
                    asm.If(asm.Literal(False), asm.Return(asm.Literal(np.float64(0)))),
                    asm.Return(asm.Literal(np.int64(1))),
                )
            ),
        )
        checker(fun)


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
    # Simple dot product
    # Borrowed from test_assembly_interpreter.py
    c = asm.Variable("c", np.float64)
    i = asm.Variable("i", np.int64)
    ab = NumpyBuffer(a)
    bb = NumpyBuffer(b)
    ab_v = asm.Variable("a", ab.ftype)
    ab_slt = asm.Slot("a_", ab.ftype)
    bb_v = asm.Variable("b", bb.ftype)
    bb_slt = asm.Slot("b_", bb.ftype)

    mod = asm.Module(
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

    checker = asm.AssemblyTypeChecker()
    assert checker(mod) is None


def test_if_statement():
    # borrowed from test_assembly_interpreter.py
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

    assert asm.AssemblyTypeChecker()(root) is None


def test_simple_struct():
    # borrowed from test_assembly_interpreter.py
    Point = namedtuple("Point", ["x", "y"])
    p = Point(np.float64(1.0), np.float64(2.0))
    x = (1, 4)

    p_var = asm.Variable("p", ftype(p))
    x_var = asm.Variable("x", ftype(x))
    res_var = asm.Variable("res", np.float64)
    mod = asm.Module(
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

    assert asm.AssemblyTypeChecker()(mod) is None

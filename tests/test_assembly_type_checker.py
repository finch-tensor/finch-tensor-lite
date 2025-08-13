import operator
from collections import namedtuple

import pytest

import numpy as np

from finch.codegen import NumpyBuffer
from finch.finch_assembly import (
    AssemblyTypeChecker,
    AssemblyTypeError,
    Assign,
    BufferLoop,
    Call,
    ForLoop,
    GetAttr,
    If,
    IfElse,
    Length,
    Literal,
    Load,
    Resize,
    SetAttr,
    Slot,
    Store,
    Variable,
    WhileLoop,
)
from finch.symbolic import FType, ftype


def test_lit_basic():
    checker = AssemblyTypeChecker()
    assert checker(Literal(np.float64(1.0))) is np.float64
    assert checker(Literal(True)) is bool


def test_var_basic():
    checker = AssemblyTypeChecker()
    checker.ctxt["x"] = np.float64
    x_type = checker(Variable("x", np.float64))
    assert x_type == np.float64
    with pytest.raises(AssemblyTypeError):
        checker(Variable("y", np.float64))
    with pytest.raises(AssemblyTypeError):
        checker(Variable("x", float))
    with pytest.raises(AssemblyTypeError):
        checker(Variable("x", 42))


def test_slot_basic():
    checker = AssemblyTypeChecker()
    b = NumpyBuffer(np.array([1, 2, 3]))
    checker.ctxt["b"] = b.ftype
    b_type = checker(Slot("b", b.ftype))
    assert isinstance(b_type, FType)
    assert b_type == b.ftype
    with pytest.raises(AssemblyTypeError):
        checker(Slot("b", float))
    with pytest.raises(AssemblyTypeError):
        checker(Slot("b", 42))


def test_assign_basic():
    checker = AssemblyTypeChecker()
    checker(Assign(Variable("x", np.float64), Literal(np.float64(2.0))))
    assert checker(Variable("x", np.float64)) is np.float64
    with pytest.raises(AssemblyTypeError):
        checker(
            Assign(Variable("x", Literal(np.float64(2.0))), Literal(np.float64(2.0)))
        )
    with pytest.raises(AssemblyTypeError):
        checker(Assign(Variable("x", np.float64), Literal(True)))


def test_getattr_basic():
    checker = AssemblyTypeChecker()
    p = (1, "one")
    p_var = Variable("p", ftype(p))
    checker.ctxt["p"] = ftype(p)
    assert checker(GetAttr(p_var, Literal("element_0"))) is int
    assert checker(GetAttr(p_var, Literal("element_1"))) is str
    with pytest.raises(AssemblyTypeError):
        checker(GetAttr(p_var, Literal("element_3")))
    with pytest.raises(AssemblyTypeError):
        checker(GetAttr(p_var, Literal("x")))
    with pytest.raises(AssemblyTypeError):
        checker(GetAttr(Literal("not a struct"), Literal("element_0")))
    with pytest.raises(AssemblyTypeError):
        checker(GetAttr(p_var, "x"))


def test_setattr_basic():
    checker = AssemblyTypeChecker()
    Point = namedtuple("Point", ["x", "y"])
    p = Point(np.float64(1.0), True)
    p_var = Variable("p", ftype(p))
    checker.ctxt["p"] = ftype(p)
    checker(SetAttr(p_var, Literal("x"), Literal(np.float64(2.0))))
    checker(SetAttr(p_var, Literal("y"), Literal(False)))
    with pytest.raises(AssemblyTypeError):
        checker(SetAttr(p_var, Literal("x"), Literal(1)))
    with pytest.raises(AssemblyTypeError):
        checker(SetAttr(p_var, Literal("z"), Literal(1)))
    with pytest.raises(AssemblyTypeError):
        checker(SetAttr(p_var, "x", Literal(np.float64(3.0))))
    with pytest.raises(AssemblyTypeError):
        checker(
            SetAttr(Literal("not a struct"), Literal("x"), Literal(np.float64(2.0)))
        )


def test_call_basic():
    checker = AssemblyTypeChecker()
    assert (
        checker(
            Call(
                Literal(operator.add),
                (
                    Literal(np.float64(2.0)),
                    Literal(np.float64(3.0)),
                ),
            )
        )
        == np.float64
    )
    assert checker(Call(Literal(np.sin), (Literal(np.float64(3.0)),))) == np.float64


def test_call_sin():
    checker = AssemblyTypeChecker()
    with pytest.raises(AssemblyTypeError):
        checker(Call(Literal(np.sin), (Literal("string"),)))


def test_load_basic():
    checker = AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1.0]))
    checker.ctxt["a"] = a.ftype
    assert checker(Load(Slot("a", a.ftype), Literal(np.int64(0)))) == np.float64
    with pytest.raises(AssemblyTypeError):
        checker(Load(Slot("a", a.ftype), Literal(0.0)))
    with pytest.raises(AssemblyTypeError):
        checker(Load(Literal(0.0), Literal(0)))


def test_store_basic():
    checker = AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1, 2, 3]))
    checker.ctxt["a"] = a.ftype
    checker(Store(Slot("a", a.ftype), Literal(np.int64(0)), Literal(np.int64(42))))
    assert checker(Load(Slot("a", a.ftype), Literal(np.int64(0)))) == np.int64
    with pytest.raises(AssemblyTypeError):
        checker(Store(Slot("a", a.ftype), Literal(0), Literal(np.int64(42))))
    with pytest.raises(AssemblyTypeError):
        checker(Store(Slot("a", a.ftype), Literal(np.int64(0)), Literal(42)))
    with pytest.raises(AssemblyTypeError):
        checker(Store(Literal(0.0), Literal(np.int64(0)), Literal(np.int64(42))))


def test_resize_basic():
    checker = AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1, 2, 3]))
    checker.ctxt["a"] = a.ftype
    checker(Resize(Slot("a", a.ftype), Literal(np.int64(20))))
    with pytest.raises(AssemblyTypeError):
        checker(Resize(Slot("a", a.ftype), Literal(20)))
    with pytest.raises(AssemblyTypeError):
        checker(Resize(Literal(0.0), Literal(20)))


def test_length_basic():
    checker = AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1, 2, 3]))
    checker.ctxt["a"] = a.ftype
    assert checker(Length(Slot("a", a.ftype))) == np.int64
    with pytest.raises(AssemblyTypeError):
        checker(Length(Literal(0.0)))


def test_forloop_basic():
    checker = AssemblyTypeChecker()
    checker(
        ForLoop(
            Variable("x", np.int64),
            Literal(np.int64(0)),
            Literal(np.int64(10)),
            Assign(
                Variable("i", np.int64),
                Variable("x", np.int64),
            ),
        )
    )
    with pytest.raises(AssemblyTypeError):
        checker(
            ForLoop(
                Variable("x", np.float64),
                Literal(np.float64(0)),
                Literal(np.float64(10)),
                Assign(
                    Variable("i", np.float64),
                    Variable("x", np.float64),
                ),
            )
        )
    with pytest.raises(AssemblyTypeError):
        checker(
            ForLoop(
                Variable("x", int),
                Literal(np.int64(0)),
                Literal(np.int64(10)),
                Assign(
                    Variable("i", np.int64),
                    Variable("x", np.int64),
                ),
            )
        )
    with pytest.raises(AssemblyTypeError):
        checker(
            ForLoop(
                Variable("x", int),
                Literal(0),
                Literal(np.int64(10)),
                Assign(
                    Variable("i", np.int64),
                    Variable("x", np.int64),
                ),
            )
        )


def test_bufferloop_basic():
    checker = AssemblyTypeChecker()
    a = NumpyBuffer(np.array([1.0, 2.0, 3.0]))
    checker.ctxt["a"] = a.ftype
    checker(
        BufferLoop(
            Slot("a", a.ftype),
            Variable("x", np.float64),
            Assign(
                Variable("i", np.float64),
                Variable("x", np.float64),
            ),
        )
    )
    with pytest.raises(AssemblyTypeError):
        checker(
            BufferLoop(
                Slot("a", a.ftype),
                Variable("x", np.int64),
                Assign(
                    Variable("i", np.float64),
                    Variable("x", np.float64),
                ),
            )
        )


def test_whileloop_basic():
    checker = AssemblyTypeChecker()
    checker(
        WhileLoop(
            Call(
                Literal(operator.and_),
                (
                    Literal(True),
                    Literal(0),
                ),
            ),
            Assign(Variable("x", int), Literal(0)),
        )
    )
    a = NumpyBuffer(np.array([1.0, 2.0, 3.0]))
    checker.ctxt["a"] = a.ftype
    with pytest.raises(AssemblyTypeError):
        checker(
            WhileLoop(
                Slot("a", a.ftype),
                Assign(Variable("x", int), Literal(0)),
            )
        )


def test_if_basic():
    checker = AssemblyTypeChecker()
    checker(
        If(
            Call(
                Literal(operator.and_),
                (
                    Literal(True),
                    Literal(0),
                ),
            ),
            Assign(Variable("x", int), Literal(0)),
        )
    )
    a = NumpyBuffer(np.array([1.0, 2.0, 3.0]))
    checker.ctxt["a"] = a.ftype
    with pytest.raises(AssemblyTypeError):
        checker(
            If(
                Slot("a", a.ftype),
                Assign(Variable("x", int), Literal(0)),
            )
        )


def test_ifelse_basic():
    checker = AssemblyTypeChecker()
    checker(
        IfElse(
            Call(
                Literal(operator.and_),
                (
                    Literal(True),
                    Literal(0),
                ),
            ),
            Assign(Variable("x", int), Literal(0)),
            Assign(Variable("x", int), Literal(1)),
        )
    )
    a = NumpyBuffer(np.array([1.0, 2.0, 3.0]))
    checker.ctxt["a"] = a.ftype
    with pytest.raises(AssemblyTypeError):
        checker(
            IfElse(
                Slot("a", a.ftype),
                Assign(Variable("x", int), Literal(0)),
                Assign(Variable("x", int), Literal(1)),
            )
        )

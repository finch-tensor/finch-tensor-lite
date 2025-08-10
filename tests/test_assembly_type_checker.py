from collections import namedtuple

import pytest

import numpy as np

from finch.codegen import NumpyBuffer
from finch.finch_assembly import (
    AssemblyTypeChecker,
    AssemblyTypeError,
    Assign,
    GetAttr,
    Literal,
    SetAttr,
    Slot,
    Variable,
)
from finch.symbolic import FType, ftype


def test_lit_basic():
    assert AssemblyTypeChecker()(Literal(np.float64(1.0))) == np.float64


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

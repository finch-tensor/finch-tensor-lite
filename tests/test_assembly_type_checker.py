import pytest

import numpy as np

from finch.codegen import NumpyBuffer
from finch.finch_assembly import (
    AssemblyTypeChecker,
    Assign,
    Literal,
    Slot,
    Variable,
)
from finch.symbolic import FType


def test_lit_basic():
    assert AssemblyTypeChecker()(Literal(np.float64(1.0))) == np.float64


def test_var_basic():
    checker = AssemblyTypeChecker()
    checker.ctxt["x"] = np.float64
    x_type = checker(Variable("x", np.float64))
    assert x_type == np.float64
    with pytest.raises(TypeError):
        checker(Variable("y", np.float64))
    with pytest.raises(TypeError):
        checker(Variable("x", float))
    with pytest.raises(TypeError):
        checker(Variable("x", 42))


def test_slot_basic():
    checker = AssemblyTypeChecker()
    b = NumpyBuffer(np.array([1, 2, 3]))
    checker.ctxt["b"] = b.ftype
    b_type = checker(Slot("b", b.ftype))
    assert isinstance(b_type, FType)
    assert b_type == b.ftype
    with pytest.raises(TypeError):
        checker(Slot("b", float))
    with pytest.raises(TypeError):
        checker(Slot("b", 42))


def test_assign_basic():
    checker = AssemblyTypeChecker()
    checker(Assign(Variable("x", np.float64), Literal(np.float64(2.0))))
    assert checker(Variable("x", np.float64)) == np.float64
    with pytest.raises(TypeError):
        checker(
            Assign(Variable("x", Literal(np.float64(2.0))), Literal(np.float64(2.0)))
        )
    with pytest.raises(TypeError):
        checker(Assign(Variable("x", np.float64), Literal(True)))

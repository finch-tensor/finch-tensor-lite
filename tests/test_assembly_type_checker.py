import pytest

import numpy as np

from finch.codegen import NumpyBuffer
from finch.finch_assembly import (
    AssemblyTypeChecker,
    Literal,
    Slot,
    Variable,
)
from finch.symbolic import FType


def test_lit():
    assert AssemblyTypeChecker()(Literal(np.float64(1.0))) == np.float64


def test_var():
    check = AssemblyTypeChecker()
    check.ctxt["x"] = np.float64
    x_type = check(Variable("x", np.float64))
    assert x_type == np.float64
    with pytest.raises(TypeError):
        check(Variable("y", np.float64))
    with pytest.raises(TypeError):
        check(Variable("x", float))
    with pytest.raises(TypeError):
        check(Variable("x", 42))


def test_slot():
    check = AssemblyTypeChecker()
    b = NumpyBuffer(np.array([1, 2, 3]))
    check.ctxt["b"] = b.ftype
    b_type = check(Slot("b", b.ftype))
    assert isinstance(b_type, FType)
    assert b_type == b.ftype
    with pytest.raises(TypeError):
        check(Slot("b", float))
    with pytest.raises(TypeError):
        check(Slot("b", 42))

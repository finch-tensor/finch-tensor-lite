from finch.finch_assembly import (
    AssemblyInterpreter,
    AssemblyInterpreterKernel,
)
from finch import finch_assembly as asm

from finch.codegen import NumpyBuffer

import pytest
import numpy as np
import operator


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([1, 2, 3], dtype=np.float64), np.array([4, 5, 6], dtype=np.float64)),
        (np.array([0], dtype=np.float64), np.array([7], dtype=np.float64)),
        (np.array([1.5, 2.5], dtype=np.float64), np.array([3.5, 4.5], dtype=np.float64)),
    ],
)
def test_dot_product(a, b):
    # Simple dot product using numpy for expected result
    c = asm.Variable("c", np.float64)
    i = asm.Variable("i", np.int64)
    ab = NumpyBuffer(a)
    bb = NumpyBuffer(b)
    f = asm.Variable("dot_product", np.float64)
    prgm = asm.Module((
        asm.Function(f, (
            asm.Variable("a", ab.format()),
            asm.Variable("b", bb.format())
        ), asm.Block((
            asm.Assign(c, asm.Immediate(np.float64(0.0))),
            asm.ForLoop(i, asm.Immediate(0), asm.Length(a), asm.Block((
                asm.Assign(
                    c,
                    asm.Call(
                        asm.Immediate(operator.add), (
                            c,
                            asm.Call(asm.Immediate(operator.mul), (
                                asm.Load(a, i), asm.Load(b, i)
                            ))
                        ),
                    )
                ),
            ))),
            asm.Return(c),
        ))),
    ))
    kernel = AssemblyInterpreterKernel(prgm, f)
    result = kernel(ab, bb)
    expected = np.dot(a, b)
    assert np.allclose(result, expected)
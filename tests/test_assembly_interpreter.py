import operator

import pytest

import numpy as np

from finch import finch_assembly as asm
from finch.codegen import NumpyBuffer
from finch.finch_assembly import (
    AssemblyInterpreter,
)


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
    ab_v = asm.Variable("a", ab.get_format())
    bb_v = asm.Variable("b", bb.get_format())
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
                            asm.Assign(c, asm.Immediate(np.float64(0.0))),
                            asm.ForLoop(
                                i,
                                asm.Immediate(np.int64(0)),
                                asm.Length(ab_v),
                                asm.Block(
                                    (
                                        asm.Assign(
                                            c,
                                            asm.Call(
                                                asm.Immediate(operator.add),
                                                (
                                                    c,
                                                    asm.Call(
                                                        asm.Immediate(operator.mul),
                                                        (
                                                            asm.Load(ab_v, i),
                                                            asm.Load(bb_v, i),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                ),
                            ),
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

import ctypes
from typing import NamedTuple

import numpy as np

from finch.codegen.numpy_buffer import (
    BufferFields,
    CNumpyBuffer,
    NumpyBufferFType,
    NumpyBuffer,
)

import finch.finch_assembly as asm
from finch.finch_assembly import Buffer
from finch.util import qual_str
from finch.codegen.c import CBufferFType, CCompiler, CContext, CStackFType, c_type
from finch.codegen.numba_backend import NumbaBufferFType


def SafeBuffer(
    buffer_type: type[Buffer], buffer_ftype: type[CBufferFType]
) -> "tuple[type, type]":
    """
    SafeBuffer is a factory which attaches index safety methods to a buffer.

    We keep it standard from the standard buffer codegen implementation since
    this can result in a fairly big performance tradeoff.
    """

    if not isinstance(buffer_type, type):
        raise TypeError(
            f"buffer_type {buffer_type} supplied to SafeBuffer must be a buffer type"
        )
    if not isinstance(buffer_ftype, type):
        raise TypeError(
            f"buffer ftype {buffer_type} supplied to SafeBuffer must be a buffer type"
        )

    class SafeBufferType(buffer_type):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, *kwargs)

        @property
        def ftype(self):
            """
            Returns the ftype of the buffer.
            """
            return SafeBufferFType(self.arr.dtype.type)

        def load(self, index: int):
            if index < 0 or index >= self.length():
                raise IndexError(f"{self} received an index out of bounds!")
            return super().load(index)

        def store(self, index: int, value):
            if index < 0 or index >= self.length():
                raise IndexError(f"{self} received an index out of bounds!")
            return super().store(index, value)

    class SafeBufferFType(buffer_ftype):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, *kwargs)

        def check(self, ctx, buf, idx):
            ctx.add_header("#include <stdio.h>")
            ctx.add_header("#include <stdlib.h>")
            idx_n = ctx.freshen(idx, "computed")
            ctx.exec(
                f"{ctx.feed}size_t {idx_n} = ({ctx(idx)});\n"
                f"{ctx.feed}if ({idx_n} < 0 || {idx_n} >= ({self.c_length(ctx, buf)}))"
                f'{{ fprintf(stderr, "Encountered an index out of bounds error!"); exit(1); }}'
            )
            return asm.Variable(idx_n, np.intp)

        def c_load(self, ctx, buf, idx):
            """
            A c_load function with preemptive index checking.

            self.check returns the value of the computed index so things don't
            get computed twice.
            """
            return super().c_load(ctx, buf, self.check(ctx, buf, idx))

        def c_store(self, ctx, buf, idx, value):
            """
            A c_store function with preemptive index checking.

            self.check returns the variable name of the computed index so
            things don't get computed twice.
            """
            super().c_store(ctx, buf, self.check(ctx, buf, idx), value)

    return SafeBufferType, SafeBufferFType


SafeNumpyBuffer, SafeNumpyBufferFType = SafeBuffer(NumpyBuffer, NumpyBufferFType)

__all__ = [
    "SafeNumpyBuffer",
    "SafeNumpyBufferFType",
]

if __name__ == "__main__":

    a = np.array([1, 2, 3], dtype=ctypes.c_double)
    ab = SafeNumpyBuffer(a)

    ab_v = asm.Variable("a", ab.ftype)
    ab_slt = asm.Slot("a_", ab.ftype)
    idx = asm.Variable("idx", ctypes.c_size_t)
    val = asm.Variable("val", ctypes.c_double)

    res_var = asm.Variable("val", ab.ftype.element_type)

    prgm = asm.Module(
        (
            asm.Function(
                asm.Variable("finch_access", ab.ftype.element_type),
                (ab_v, idx),
                asm.Block(
                    (
                        asm.Unpack(ab_slt, ab_v),
                        asm.Assign(
                            res_var,
                            asm.Load(ab_slt, idx),
                        ),
                        asm.Return(res_var),
                    )
                ),
            ),
            asm.Function(
                asm.Variable("finch_change", ab.ftype.element_type),
                (ab_v, idx, val),
                asm.Block(
                    (
                        asm.Unpack(ab_slt, ab_v),
                        asm.Store(
                            ab_slt,
                            idx,
                            val,
                        ),
                    )
                ),
            ),
        )
    )
    compiler = CCompiler()
    mod = compiler(prgm)
    access = mod.finch_access
    change = mod.finch_change
    print(ab)
    print(access(ab, ctypes.c_size_t(0)))
    change(ab, ctypes.c_size_t(0), ctypes.c_double(2))
    print(ab)
    # should fail here.
    print(access(ab, ctypes.c_size_t(3)))
    change(ab, ctypes.c_size_t(3), ctypes.c_double(2))
    print(ab)

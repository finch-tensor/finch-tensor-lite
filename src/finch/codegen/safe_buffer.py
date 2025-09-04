import ctypes
from typing import NamedTuple

import numpy as np

from finch.codegen.numpy_buffer import (
    BufferFields,
    CNumpyBuffer,
    NumpyBufferFType,
    NumpyBuffer,
)

from ..finch_assembly import Buffer
from ..util import qual_str
from .c import CBufferFType, CContext, CStackFType, c_type
from .numba_backend import NumbaBufferFType


def SafeBuffer(buffer_type: type[Buffer], buffer_ftype: type[CBufferFType]) -> "tuple[type, type]":
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

        def check(self, ctx, buf: "BufferFields", idx) -> str:
            idx_n = ctx.freshen(idx, "computed")
            ctx.exec(
                f"{ctx.feed}size_t {idx_n} = ({ctx(idx)});\n"
                f"{ctx.feed}if ({idx_n} < 0 || {idx_n} >= {buf.length})"
                f' {{ printf("Encountered invalid indexing for {buf.data}!"); exit(1); }}'
            )
            return idx_n

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

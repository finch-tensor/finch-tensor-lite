import ctypes

import finchlite.finch_assembly as asm
from finchlite.codegen.c import CBufferFType
from finchlite.codegen.numba_backend import NumbaBufferFType
from finchlite.finch_assembly import Buffer


class SafeBuffer(Buffer):
    def __init__(self, buffer: Buffer):
        self._underlying = buffer

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer.
        """
        return SafeBufferFType(self._underlying.ftype)

    def load(self, index: int):
        if index < 0 or index >= self.length():
            raise IndexError(f"{self} received an index out of bounds!")
        return self._underlying.load(index)

    def store(self, index: int, value):
        if index < 0 or index >= self.length():
            raise IndexError(f"{self} received an index out of bounds!")
        return self._underlying.store(index, value)

    def length(self):
        return self._underlying.length()

    def resize(self, len: int):
        return self._underlying.resize(len)

    def __str__(self) -> str:
        return f"safe({self._underlying})"

    def __getattr__(self, name):
        return getattr(self._underlying, name)


class SafeBufferFType(CBufferFType, NumbaBufferFType):
    def __init__(self, underlying_format):
        self._underlying_format = underlying_format

    def __eq__(self, other):
        if not isinstance(other, SafeBufferFType):
            return False
        return self._underlying_format == other._underlying_format

    def __hash__(self):
        return hash(("SafeBufferFType", self._underlying_format))

    def c_type(self, *args, **kwargs):
        return self._underlying_format.c_type(*args, **kwargs)

    def c_length(self, *args, **kwargs):
        return self._underlying_format.c_length(*args, **kwargs)

    def c_data(self, *args, **kwargs):
        return self._underlying_format.c_data(*args, **kwargs)

    def _c_check(self, ctx, buf, idx):
        ctx.add_header("#include <stdio.h>")
        ctx.add_header("#include <stdlib.h>")
        idx_n = ctx.freshen("computed")
        ctx.exec(
            f"{ctx.feed}size_t {idx_n} = ({ctx(idx)});\n"
            f"{ctx.feed}if ({idx_n} < 0 || {idx_n} >= ({self.c_length(ctx, buf)}))"
            "{"
            f'fprintf(stderr, "Encountered an index out of bounds error!");\n'
            f"exit(1);\n"
            "}"
        )
        return asm.Variable(idx_n, ctypes.c_size_t)

    def c_load(self, ctx, buf, idx):
        """
        A c_load function with preemptive index checking.

        self.check returns the value of the computed index so things don't
        get computed twice.
        """
        return self._underlying_format.c_load(ctx, buf, self._c_check(ctx, buf, idx))

    def c_store(self, ctx, buf, idx, value):
        """
        A c_store function with preemptive index checking.

        self.check returns the variable name of the computed index so
        things don't get computed twice.
        """
        self._underlying_format.c_store(ctx, buf, self._c_check(ctx, buf, idx), value)

    def c_resize(self, *args, **kwargs):
        return self._underlying_format.c_resize(*args, **kwargs)

    def c_unpack(self, *args, **kwargs):
        return self._underlying_format.c_unpack(*args, **kwargs)

    def c_repack(self, *args, **kwargs):
        return self._underlying_format.c_repack(*args, **kwargs)

    def serialize_to_c(self, *args, **kwargs):
        return self._underlying_format.serialize_to_c(*args, **kwargs)

    def deserialize_from_c(self, *args, **kwargs):
        return self._underlying_format.deserialize_from_c(*args, **kwargs)

    def construct_from_c(self, *args, **kwargs):
        return self._underlying_format.construct_from_c(*args, **kwargs)

    # numba definitions

    def numba_type(self, *args, **kwargs):
        return self._underlying_format.numba_type(*args, **kwargs)

    def numba_length(self, *args, **kwargs):
        return self._underlying_format.numba_length(*args, **kwargs)

    def numba_check(self, ctx, buf, idx):
        idx_n = ctx.freshen("computed")
        ctx.exec(
            f"{ctx.feed}{idx_n} = ({ctx(idx)})\n"
            f"{ctx.feed}if {idx_n} < 0 or {idx_n} >= ({self.numba_length(ctx, buf)}):\n"
            f"{ctx.feed}    raise IndexError()"
        )
        return asm.Variable(idx_n, int)

    def numba_load(self, ctx, buf, idx):
        """
        A numba_load function with preemptive index checking.

        self.numba_check returns the value of the computed index so things don't
        get computed twice.
        """
        return self._underlying_format.numba_load(
            ctx, buf, self.numba_check(ctx, buf, idx)
        )

    def numba_store(self, ctx, buf, idx, value):
        """
        A numba_store function with preemptive index checking.

        self.numba_check returns the variable name of the computed index so
        things don't get computed twice.
        """
        self._underlying_format.numba_store(
            ctx, buf, self.numba_check(ctx, buf, idx), value
        )

    def numba_resize(self, *args, **kwargs):
        return self._underlying_format.numba_resize(*args, **kwargs)

    def numba_unpack(self, *args, **kwargs):
        return self._underlying_format.numba_unpack(*args, **kwargs)

    def numba_repack(self, *args, **kwargs):
        return self._underlying_format.numba_repack(*args, **kwargs)

    def serialize_to_numba(self, *args, **kwargs):
        return self._underlying_format.serialize_to_numba(*args, **kwargs)

    def deserialize_from_numba(self, *args, **kwargs):
        return self._underlying_format.deserialize_from_numba(*args, **kwargs)

    def construct_from_numba(self, *args, **kwargs):
        return self._underlying_format.construct_from_numba(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Create a SafeBuffer wrapping a buffer created by the underlying format.
        Args:
            *args: Arguments to pass to the underlying format
            **kwargs: Keyword arguments to pass to the underlying format
        Returns:
            A SafeBuffer instance wrapping the created buffer
        """
        underlying_buffer = self._underlying_format(*args, **kwargs)
        return SafeBuffer(underlying_buffer)

    @property
    def element_type(self):
        """
        Return the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self._underlying_format.element_type

    @property
    def underlying_format(self):
        """
        Provide access to the underlying format.
        """
        return self._underlying_format

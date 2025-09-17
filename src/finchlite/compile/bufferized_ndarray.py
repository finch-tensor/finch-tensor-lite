import operator
from textwrap import dedent
from typing import Any, NamedTuple

import numpy as np

import numba

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import Tensor
from ..codegen import NumpyBuffer, NumpyBufferFType
from ..codegen.numba_backend import (
    numba_globals,
    numba_structs,
    numba_type,
)
from ..finch_assembly import AssemblyStructFType, TupleFType
from ..symbolic import ftype
from . import looplets as lplt
from .lower import FinchTensorFType


class BufferizedNDArray(Tensor):
    def __init__(self, arr: np.ndarray):
        itemsize = arr.dtype.itemsize
        for stride in arr.strides:
            if stride % itemsize != 0:
                raise ValueError("Array must be aligned to multiple of itemsize")
        self.strides = tuple(np.intp(stride // itemsize) for stride in arr.strides)
        self._shape = tuple(np.asarray(s)[()] for s in arr.shape)
        self.buf = NumpyBuffer(arr.reshape(-1, copy=False))

    def to_numpy(self):
        """
        Convert the bufferized NDArray to a NumPy array.
        This is used to get the underlying NumPy array from the bufferized NDArray.
        """
        return self.buf.arr.reshape(self._shape, copy=False)

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a BufferizedNDArrayFType.
        """
        return BufferizedNDArrayFType(ftype(self.buf), self.ndim, ftype(self.strides))

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return np.intp(len(self._shape))

    def declare(self, init, op, shape):
        """
        Declare a bufferized NDArray with the given initialization value,
        operation, and shape.
        """
        for dim, size in zip(shape, self._shape, strict=False):
            if dim.start != 0:
                raise ValueError(
                    f"Invalid dimension start value {dim.start} for ndarray"
                    f" declaration."
                )
            if dim.end != size:
                raise ValueError(
                    f"Invalid dimension end value {dim.end} for ndarray declaration."
                )
        for i in range(self.buf.length()):
            self.buf.store(i, init)
        return self

    def freeze(self, op):
        return self

    def thaw(self, op):
        return self

    def access(self, indices, op):
        return BufferizedNDArrayAccessor(self).access(indices, op)

    def __getitem__(self, index):
        """
        Get an item from the bufferized NDArray.
        This allows for indexing into the bufferized array.
        """
        if isinstance(index, tuple):
            index = 0 if index == () else np.dot(index, self.strides)
        return self.buf.load(index)

    def __setitem__(self, index, value):
        """
        Set an item in the bufferized NDArray.
        This allows for indexing into the bufferized array.
        """
        if isinstance(index, tuple):
            index = np.ravel_multi_index(index, self._shape)
        self.buf.store(index, value)

    def __str__(self):
        return f"BufferizedNDArray(shape={self.shape})"

    def __repr__(self):
        return f"BufferizedNDArray(shape={self.shape})"


class BufferizedNDArrayFields(NamedTuple):
    stride: tuple[asm.Variable, ...]
    buf: asm.Variable
    buf_s: asm.Slot


class BufferizedNDArrayFType(FinchTensorFType, AssemblyStructFType):
    """
    A ftype for bufferized NumPy arrays that provides metadata about the array.
    This includes the fill value, element type, and shape type.
    """

    @property
    def struct_name(self):
        return "BufferizedNDArray"

    @property
    def struct_fields(self):
        return [
            ("buf", self.buf_t),
            ("shape", self.shape_t),
            ("strides", self.strides_t),
        ]

    def __init__(self, buf_t: NumpyBufferFType, ndim: np.intp, strides_t: TupleFType):
        self.buf_t = buf_t
        self._ndim = ndim
        self.shape_t = strides_t  # assuming shape is the same type as strides
        self.strides_t = strides_t

    def __eq__(self, other):
        if not isinstance(other, BufferizedNDArrayFType):
            return False
        return self.buf_t == other.buf_t and self._ndim == other._ndim

    def __hash__(self):
        return hash((self.buf_t, self._ndim))

    def __str__(self):
        return f"{self.struct_name}(ndim={self.ndim})"

    def __repr__(self):
        return f"{self.struct_name}({repr(self.buf_t)})"

    @property
    def ndim(self) -> np.intp:
        return self._ndim

    @property
    def fill_value(self) -> Any:
        return np.zeros((), dtype=self.buf_t.element_type)[()]

    @property
    def element_type(self):
        return self.buf_t.element_type

    @property
    def shape_type(self) -> tuple:
        return tuple(np.intp for _ in range(self._ndim))

    def lower_declare(self, ctx, tns, init, op, shape):
        i_var = asm.Variable("i", self.buf_t.length_type)
        body = asm.Store(
            tns.obj.buf_s,
            i_var,
            asm.Literal(init.val),
        )
        ctx.exec(
            asm.ForLoop(i_var, asm.Literal(np.intp(0)), asm.Length(tns.obj.buf_s), body)
        )
        return

    def lower_freeze(self, ctx, tns, op):
        return tns

    def lower_thaw(self, ctx, tns, op):
        return tns

    def unfurl(self, ctx, tns, ext, mode, proto):
        op = None
        if isinstance(mode, ntn.Update):
            op = mode.op
        tns = ctx.resolve(tns).obj
        acc_t = BufferizedNDArrayAccessorFType(self, 0, self.buf_t.length_type, op)
        obj = BufferizedNDArrayAccessorFields(
            tns, 0, asm.Literal(self.buf_t.length_type(0)), op
        )
        return acc_t.unfurl(ctx, ntn.Stack(obj, acc_t), ext, mode, proto)

    def lower_unwrap(self, ctx, obj): ...

    def lower_increment(self, ctx, obj, val): ...

    def asm_unpack(self, ctx, var_n, val):
        """
        Unpack the into asm context.
        """
        stride = []
        for i in range(self._ndim):
            stride_i = asm.Variable(f"{var_n}_stride_{i}", self.buf_t.length_type)
            stride.append(stride_i)
            stride_e = asm.GetAttr(val, asm.Literal("strides"))
            stride_i_e = asm.GetAttr(stride_e, asm.Literal(f"element_{i}"))
            ctx.exec(asm.Assign(stride_i, stride_i_e))
        buf = asm.Variable(f"{var_n}_buf", self.buf_t)
        buf_e = asm.GetAttr(val, asm.Literal("buf"))
        ctx.exec(asm.Assign(buf, buf_e))
        buf_s = asm.Slot(f"{var_n}_buf_slot", self.buf_t)
        ctx.exec(asm.Unpack(buf_s, buf))

        return BufferizedNDArrayFields(tuple(stride), buf, buf_s)

    def asm_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from C context.
        """
        ctx.exec(asm.Repack(obj.buf_s))
        return

    def __call__(self, numba_buffer):
        return BufferizedNDArray(
            numba_buffer.buf[0].reshape(
                tuple(
                    getattr(numba_buffer.shape, f)
                    for (f, _) in self.shape_t.struct_fields
                )
            )
        )

    def numba_type(self) -> type:
        if self in numba_structs:
            return numba_structs[self]

        spec = [
            (
                "buf",
                numba.types.ListType(
                    numba.types.Array(numba.from_dtype(self.buf_t.element_type), 1, "C")
                ),
            ),
            ("shape", numba_type(self.shape_t).class_type.instance_type),
            ("strides", numba_type(self.strides_t).class_type.instance_type),
        ]

        def _str_format(types: list[type]) -> str:
            return "_".join(np.dtype(t).char for t in types)

        class_name = (
            f"Numba_BufferizedNDArray_{np.dtype(self.buf_t.element_type).char}_"
            f"sh_{_str_format(self.shape_t.struct_fieldformats)}_"
            f"str_{_str_format(self.strides_t.struct_fieldformats)}"
        )
        class_src = dedent(
            f"""\
            class {class_name}:
                def __init__(self, buf, shape, strides):
                    self.buf = buf
                    self.shape = shape
                    self.strides = strides
                @staticmethod
                def numba_name():
                    return '{class_name}'
            """
        )
        ns: dict[str, object] = {}
        exec(class_src, ns)
        new_struct = numba.experimental.jitclass(ns[class_name], spec)
        numba_structs[self] = new_struct
        numba_globals[new_struct.__name__] = new_struct
        return new_struct


class BufferizedNDArrayAccessor(Tensor):
    """
    A class representing a tensor view that is bufferized.
    This is used to create a view of a tensor with a specific extent.
    """

    def __init__(self, tns: BufferizedNDArray, nind=None, pos=None, op=None):
        self.tns = tns
        if pos is None:
            pos = ftype(self.tns).buf_t.length_type(0)
        self.pos = pos
        self.op = op
        if nind is None:
            nind = 0
        self.nind = nind

    @property
    def ftype(self):
        return BufferizedNDArrayAccessorFType(
            ftype(self.tns), self.nind, ftype(self.pos), self.op
        )

    @property
    def shape(self):
        return self.tns.shape[self.nind :]

    def access(self, indices, op):
        pos = self.pos + np.dot(
            indices, self.tns.strides[self.nind : self.nind + len(indices)]
        )
        return BufferizedNDArrayAccessor(self.tns, self.nind + len(indices), pos, op)

    def unwrap(self):
        """
        Unwrap the tensor view to get the underlying tensor.
        This is used to get the original tensor from a tensor view.
        """
        assert self.ndim == 0, "Cannot unwrap a tensor view with non-zero dimension."
        return self.tns.buf.load(self.pos)

    def increment(self, val):
        """
        Increment the tensor view with a value.
        This updates the tensor at the specified index with the operation and value.
        """
        if self.op is None:
            raise ValueError("No operation defined for increment.")
        assert self.ndim == 0, "Cannot unwrap a tensor view with non-zero dimension."
        self.tns.buf.store(self.pos, self.op(self.tns.buf.load(self.pos), val))
        return self


class BufferizedNDArrayAccessorFields(NamedTuple):
    tns: BufferizedNDArrayFields
    nind: int
    pos: asm.AssemblyNode
    op: Any


class BufferizedNDArrayAccessorFType(FinchTensorFType):
    def __init__(self, tns, nind, pos, op):
        self.tns = tns
        self.nind = nind
        self.pos = pos
        self.op = op

    def __eq__(self, other):
        return (
            isinstance(other, BufferizedNDArrayAccessorFType)
            and self.tns == other.tns
            and self.nind == other.nind
            and self.pos == other.pos
            and self.op == other.op
        )

    def __hash__(self):
        return hash((self.tns, self.nind, self.pos, self.op))

    @property
    def ndim(self) -> np.intp:
        return self.tns.ndim - self.nind

    @property
    def shape_type(self) -> tuple:
        return self.tns.shape_type[self.nind :]

    @property
    def fill_value(self) -> Any:
        return self.tns.fill_value

    @property
    def element_type(self):
        return self.tns.element_type

    def lower_declare(self, ctx, tns, init, op, shape):
        raise NotImplementedError(
            "BufferizedNDArrayAccessorFType does not support lower_declare."
        )

    def lower_freeze(self, ctx, tns, op):
        raise NotImplementedError(
            "BufferizedNDArrayAccessorFType does not support lower_freeze."
        )

    def lower_thaw(self, ctx, tns, op):
        raise NotImplementedError(
            "BufferizedNDArrayAccessorFType does not support lower_thaw."
        )

    # TODO: We should unpack arrays before passing them to freeze/thaw
    # def asm_unpack(self, ctx, var_n, val):
    #     """
    #     Unpack the into asm context.
    #     """
    #     tns = self.tns.asm_unpack(ctx, f"{var_n}_tns", asm.GetAttr(val, "tns"))
    #     nind = asm.Variable(f"{var_n}_nind", self.nind)
    #     pos = asm.Variable(f"{var_n}_pos", self.pos)
    #     op = asm.Variable(f"{var_n}_op", self.op)
    #     ctx.exec(asm.Assign(pos, asm.GetAttr(val, "pos")))
    #     ctx.exec(asm.Assign(nind, asm.GetAttr(val, "nind")))
    #     ctx.exec(asm.Assign(op, asm.GetAttr(val, "op")))
    #     return BufferizedNDArrayFields(tns, pos, nind, op)

    def asm_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from C context.
        """
        (self.tns.asm_repack(ctx, lhs.tns, obj.tns),)
        ctx.exec(
            asm.Block(
                asm.SetAttr(lhs, "tns", obj.tns),
                asm.SetAttr(lhs, "pos", obj.pos),
                asm.SetAttr(lhs, "nind", obj.nind),
                asm.SetAttr(lhs, "op", obj.op),
            )
        )

    def lower_unwrap(self, ctx, obj):
        return asm.Load(obj.tns.buf_s, obj.pos)

    def lower_increment(self, ctx, obj, val):
        lowered_pos = asm.Variable(obj.pos.name, obj.pos.type)
        ctx.exec(
            asm.Store(
                obj.tns.buf_s,
                lowered_pos,
                asm.Call(
                    asm.Literal(self.op.val),
                    [asm.Load(obj.tns.buf_s, lowered_pos), val],
                ),
            )
        )

    def unfurl(self, ctx, tns, ext, mode, proto):
        def child_accessor(ctx, idx):
            pos_2 = asm.Variable(
                ctx.freshen(ctx.idx, f"_pos_{self.ndim - 1}"), self.pos
            )
            ctx.exec(
                asm.Assign(
                    pos_2,
                    asm.Call(
                        asm.Literal(operator.add),
                        [
                            tns.obj.pos,
                            asm.Call(
                                asm.Literal(operator.mul),
                                [
                                    tns.obj.tns.stride[self.nind],
                                    asm.Variable(ctx.idx.name, ctx.idx.type_),
                                ],
                            ),
                        ],
                    ),
                )
            )
            return ntn.Stack(
                BufferizedNDArrayAccessorFields(
                    tns=tns.obj.tns,
                    nind=self.nind - 1,
                    pos=pos_2,
                    op=self.op,
                ),
                BufferizedNDArrayAccessorFType(
                    self.tns, self.nind + 1, self.pos, self.op
                ),
            )

        return lplt.Lookup(
            body=lambda ctx, idx: lplt.Leaf(
                body=lambda ctx: child_accessor(ctx, idx),
            )
        )

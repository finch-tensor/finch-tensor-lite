import operator
from typing import Any, NamedTuple

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import Tensor
from ..codegen import NumpyBuffer
from ..finch_assembly import AssemblyStructFormat
from ..symbolic import format
from . import looplets as lplt
from .lower import FinchTensorFormat


class BufferizedNDArray(Tensor):
    def __init__(self, arr: np.ndarray):
        itemsize = arr.dtype.itemsize
        for stride in arr.strides:
            if stride % itemsize != 0:
                raise ValueError("Array must be aligned to multiple of itemsize")
        self.strides = tuple(stride // itemsize for stride in arr.strides)
        self._shape = arr.shape
        self.buf = NumpyBuffer(
            np.lib.stride_tricks.as_strided(
                arr,
                shape=(np.dot(arr.strides, arr.shape) // itemsize,),
                strides=(itemsize,),
            )
        )

    def to_numpy(self):
        """
        Convert the bufferized NDArray to a NumPy array.
        This is used to get the underlying NumPy array from the bufferized NDArray.
        """
        itemsize = self.buf.arr.dtype.itemsize
        return np.lib.stride_tricks.as_strided(
            self.buf.arr,
            shape=self._shape,
            strides=(stride * itemsize for stride in self.strides),
        )

    @property
    def format(self):
        """
        Returns the format of the buffer, which is a BufferizedNDArrayFormat.
        """
        return BufferizedNDArrayFormat(format(self.buf), len(self.strides))

    @property
    def shape(self):
        return self._shape

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
        shape = tuple(dim.end for dim in shape)
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
            index = np.dot(index, self.strides)
        return self.buf.load(index)

    def __setitem__(self, index, value):
        """
        Set an item in the bufferized NDArray.
        This allows for indexing into the bufferized array.
        """
        if isinstance(index, tuple):
            index = np.ravel_multi_index(index, self._shape)
        self.buf.store(index, value)


class BufferizedNDArrayFields(NamedTuple):
    stride: tuple[asm.Variable, ...]
    buf: asm.Variable
    buf_s: asm.Slot


class BufferizedNDArrayFormat(FinchTensorFormat, AssemblyStructFormat):
    """
    A format for bufferized NumPy arrays that provides metadata about the array.
    This includes the fill value, element type, and shape type.
    """

    @property
    def struct_name(self):
        return "BufferizedNDArray"

    @property
    def struct_fields(self):
        return [
            ("buf", self.buf),
            ("_ndim", self._ndim),
        ]

    def __init__(self, buf, ndim: int):
        self.buf = buf
        self._ndim = ndim

    def __eq__(self, other):
        if not isinstance(other, BufferizedNDArrayFormat):
            return False
        return self.buf == other.buf and self._ndim == other._ndim

    def __hash__(self):
        return hash((self.buf, self._ndim))

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def fill_value(self) -> Any:
        return np.zeros((), dtype=self.buf.dtype)[()]

    @property
    def element_type(self):
        return self.buf.element_type

    @property
    def shape_type(self) -> tuple:
        return tuple(np.int_ for _ in range(self._ndim))

    def lower_declare(self, ctx, tns, init, op, shape):
        i_var = asm.Variable("i", self.buf.length_type)
        buf = asm.Stack(tns.obj.buf, self.buf)
        body = asm.Store(
            buf,
            i_var,
            asm.Literal(init.val),
        )
        ctx.exec(asm.ForLoop(i_var, asm.Literal(0), asm.Length(buf), body))
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
        acc_t = BufferizedNDArrayAccessorFormat(self, 0, self.buf.length_type, op)
        obj = BufferizedNDArrayAccessorFields(
            tns, 0, asm.Literal(self.buf.length_type(0)), op
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
            stride_i = asm.Variable(f"{var_n}_stride_{i}", self.buf.length_type)
            stride.append(stride_i)
            stride_e = asm.GetAttr(val, "stride")
            stride_i_e = asm.GetAttr(stride_e, f"element_{i}")
            ctx.exec(asm.Assign(stride_i, stride_i_e))
        buf = asm.Variable(f"{var_n}_buf", self.buf)
        buf_e = asm.GetAttr(val, "buf")
        ctx.exec(asm.Assign(buf, buf_e))
        buf_s = asm.Slot(f"{var_n}_buf", self.buf)
        ctx.exec(asm.Unpack(buf_s, buf))

        return BufferizedNDArrayFields(tuple(stride), buf, buf_s)

    def asm_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from C context.
        """
        ctx.exec(asm.Repack(obj.buf))
        return


class BufferizedNDArrayAccessor(Tensor):
    """
    A class representing a tensor view that is bufferized.
    This is used to create a view of a tensor with a specific extent.
    """

    def __init__(self, tns: BufferizedNDArray, nind=None, pos=None, op=None):
        self.tns = tns
        if pos is None:
            pos = format(self.tns).buf.length_type(0)
        self.pos = pos
        self.op = op
        if nind is None:
            nind = 0
        self.nind = nind

    @property
    def format(self):
        return BufferizedNDArrayAccessorFormat(
            format(self.tns), self.nind, format(self.pos), self.op
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


class BufferizedNDArrayAccessorFormat(FinchTensorFormat):
    def __init__(self, tns, nind, pos, op):
        self.tns = tns
        self.nind = nind
        self.pos = pos
        self.op = op

    def __eq__(self, other):
        return (
            isinstance(other, BufferizedNDArrayAccessorFormat)
            and self.tns == other.tns
            and self.nind == other.nind
            and self.pos == other.pos
            and self.op == other.op
        )

    def __hash__(self):
        return hash((self.tns, self.nind, self.pos, self.op))

    @property
    def ndim(self) -> int:
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
            "BufferizedNDArrayAccessorFormat does not support lower_declare."
        )

    def lower_freeze(self, ctx, tns, op):
        raise NotImplementedError(
            "BufferizedNDArrayAccessorFormat does not support lower_freeze."
        )

    def lower_thaw(self, ctx, tns, op):
        raise NotImplementedError(
            "BufferizedNDArrayAccessorFormat does not support lower_thaw."
        )

    def asm_unpack(self, ctx, var_n, val):
        """
        Unpack the into asm context.
        """
        tns = self.tns.asm_unpack(ctx, f"{var_n}_tns", asm.GetAttr(val, "tns"))
        nind = asm.Variable(f"{var_n}_nind", self.nind)
        pos = asm.Variable(f"{var_n}_pos", self.pos)
        op = asm.Variable(f"{var_n}_op", self.op)
        ctx.exec(asm.Assign(pos, asm.GetAttr(val, "pos")))
        ctx.exec(asm.Assign(nind, asm.GetAttr(val, "nind")))
        ctx.exec(asm.Assign(op, asm.GetAttr(val, "op")))
        return BufferizedNDArrayFields(tns, pos, nind, op)

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
        return asm.Load(obj.tns.buf, obj.pos)

    def lower_increment(self, ctx, obj, val):
        ctx.exec(
            asm.Store(
                obj.tns.buf,
                obj.pos,
                asm.Call(asm.Literal(self.op), [asm.Load(obj.tns.buf, obj.pos), val]),
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
                            self.pos,
                            asm.Call(
                                asm.Literal(operator.mul),
                                [
                                    tns.obj.tns.stride[self.nind],
                                    ctx.freshen(ctx.idx, f"_pos_{self.ndim - 1}"),
                                ],
                            ),
                        ],
                    ),
                )
            )
            return ntn.Stack(
                BufferizedNDArrayAccessorFields(
                    tns=tns.obj.tns, nind=self.nind - 1, pos=pos_2, op=self.op
                ),
                BufferizedNDArrayAccessorFormat(
                    self.tns, self.nind + 1, pos_2, self.op
                ),
            )

        return lplt.Lookup(
            body=lambda ctx, idx: lplt.Leaf(
                body=lambda ctx: child_accessor(ctx, idx),
            )
        )

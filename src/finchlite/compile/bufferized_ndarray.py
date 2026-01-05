import operator
from typing import Any, NamedTuple

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import Tensor
from ..codegen import NumpyBuffer, NumpyBufferFType
from ..finch_assembly import AssemblyStructFType, TupleFType
from ..symbolic import ftype
from . import looplets as lplt
from .lower import FinchTensorFType


class BufferizedNDArray(Tensor):
    def __init__(
        self,
        val: np.ndarray | NumpyBuffer,
        shape: tuple[np.integer, ...] | None = None,
        strides: tuple[np.integer, ...] | None = None,
    ):
        self._shape: tuple[np.integer, ...]
        self.strides: tuple[np.integer, ...]
        if shape is None and strides is None and isinstance(val, np.ndarray):
            itemsize = val.dtype.itemsize
            for stride in val.strides:
                if stride % itemsize != 0:
                    raise ValueError("Array must be aligned to multiple of itemsize")
            self.strides = tuple(np.intp(stride // itemsize) for stride in val.strides)
            self._shape = tuple(np.intp(s) for s in val.shape)
            self.val = NumpyBuffer(val.reshape(-1, copy=False))
        elif shape is not None and strides is not None and isinstance(val, NumpyBuffer):
            self.strides = strides
            self._shape = shape
            self.val = val
        else:
            raise Exception("Invalid constructor arguments")

    def to_numpy(self):
        """
        Convert the bufferized NDArray to a NumPy array.
        This is used to get the underlying NumPy array from the bufferized NDArray.
        """
        return self.val.arr.reshape(self._shape, copy=False)

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a BufferizedNDArrayFType.
        """
        return BufferizedNDArrayFType(
            buffer_type=ftype(self.val),
            ndim=self.ndim,
            dimension_type=ftype(self.strides),
        )

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
        for i in range(self.val.length()):
            self.val.store(i, init)
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
        return self.val.load(index)

    def __setitem__(self, index, value):
        """
        Set an item in the bufferized NDArray.
        This allows for indexing into the bufferized array.
        """
        if isinstance(index, tuple):
            index = np.ravel_multi_index(index, self._shape)
        self.val.store(index, value)

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
        def str_format(types):
            return "_".join(
                f"{np.dtype(t).kind}{np.dtype(t).itemsize * 8}" for t in types
            )

        dt = np.dtype(self.buf_t.element_type)
        return (
            f"BufferizedNDArray_{dt.kind}{dt.itemsize * 8}_"
            f"shape_{str_format(self.shape_t.struct_fieldformats)}_"
            f"strides_{str_format(self.strides_t.struct_fieldformats)}"
        )

    @property
    def struct_fields(self):
        return [
            ("val", self.buf_t),
            ("shape", self.shape_t),
            ("strides", self.strides_t),
        ]

    def from_fields(self, buf, shape, strides):
        return BufferizedNDArray(
            buf,
            shape,
            strides,
        )

    def __init__(
        self,
        *,
        buffer_type: NumpyBufferFType,
        ndim: np.intp,
        dimension_type: TupleFType,
    ):
        self.buf_t = buffer_type
        self._ndim = ndim
        self.shape_t = dimension_type  # assuming shape is the same type as strides
        self.strides_t = dimension_type

    def __call__(
        self,
        shape: tuple[int, ...],
        val=None,
    ) -> BufferizedNDArray:
        if val is None:
            tns = np.full(shape, self.fill_value, dtype=self.buf_t.element_type)
        else:
            tns = val
        tns_2 = BufferizedNDArray(tns)
        return BufferizedNDArray(
            tns_2.val,
            shape=tuple(
                t(s)
                for s, t in zip(shape, self.shape_t.struct_fieldformats, strict=True)
            ),
            strides=tuple(
                t(s)
                for (s, t) in zip(
                    tns_2.strides, self.strides_t.struct_fieldformats, strict=True
                )
            ),
        )

    def __eq__(self, other):
        if not isinstance(other, BufferizedNDArrayFType):
            return False
        return self.buf_t == other.buf_t and self.ndim == other.ndim

    def __hash__(self):
        return hash((self.buf_t, self.ndim))

    def __str__(self):
        return str(self.struct_name)

    def __repr__(self):
        return f"{self.struct_name}({repr(self.buf_t)})"

    @property
    def ndim(self) -> np.intp:
        return self._ndim

    @ndim.setter
    def ndim(self, val):
        self._ndim = val

    def from_kwargs(self, **kwargs) -> "BufferizedNDArrayFType":
        b_t = kwargs.get("buffer_type", self.buf_t)
        ndim = kwargs.get("ndim", self.ndim)
        if "shape_type" in kwargs:
            s_t = kwargs["shape_type"]
            d_t = s_t if isinstance(s_t, TupleFType) else TupleFType.from_tuple(s_t)
        else:
            d_t = self.shape_t
        return BufferizedNDArrayFType(buffer_type=b_t, ndim=ndim, dimension_type=d_t)

    def to_kwargs(self):
        return {
            "buffer_type": self.buf_t,
            "ndim": self.ndim,
            "shape_type": self.shape_t,
        }

    # TODO: temporary approach for suitable rep and traits
    def add_levels(self, idxs: list[int]):
        return self

    # TODO: temporary approach for suitable rep and traits
    def remove_levels(self, idxs: list[int]):
        return self

    @property
    def fill_value(self) -> Any:
        return np.zeros((), dtype=self.buf_t.element_type)[()]

    @property
    def element_type(self):
        return self.buf_t.element_type

    @property
    def shape_type(self) -> tuple:
        return tuple(np.intp for _ in range(self.ndim))

    def lower_dim(self, ctx, obj, r):
        return asm.GetAttr(
            asm.GetAttr(obj, asm.Literal("shape")),
            asm.Literal(f"element_{r}"),
        )

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
        for i in range(self.ndim):
            stride_i = asm.Variable(f"{var_n}_stride_{i}", self.buf_t.length_type)
            stride.append(stride_i)
            stride_e = asm.GetAttr(val, asm.Literal("strides"))
            stride_i_e = asm.GetAttr(stride_e, asm.Literal(f"element_{i}"))
            ctx.exec(asm.Assign(stride_i, stride_i_e))
        buf = asm.Variable(f"{var_n}_buf", self.buf_t)
        buf_e = asm.GetAttr(val, asm.Literal("val"))
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
        if len(indices) + self.nind > self.tns.ndim:
            raise IndexError(
                f"Too many indices for tensor access: "
                f"got {len(indices)} indices for tensor with "
                f"{self.tns.ndim - self.nind} dimensions."
            )
        for i, idx in enumerate(indices):
            if not (0 <= idx < self.tns.shape[self.nind + i]):
                raise IndexError(
                    f"Index {idx} out of bounds for axis {self.nind + i} "
                    f"with size {self.tns.shape[self.nind + i]}"
                )
        pos = self.pos
        for i, idx in enumerate(indices):
            pos += idx * self.tns.strides[self.nind + i]
        return BufferizedNDArrayAccessor(self.tns, self.nind + len(indices), pos, op)

    def unwrap(self):
        """
        Unwrap the tensor view to get the underlying tensor.
        This is used to get the original tensor from a tensor view.
        """
        assert self.ndim == 0, "Cannot unwrap a tensor view with non-zero dimension."
        return self.tns.val.load(self.pos)

    def increment(self, val):
        """
        Increment the tensor view with a value.
        This updates the tensor at the specified index with the operation and value.
        """
        if self.op is None:
            raise ValueError("No operation defined for increment.")
        assert self.ndim == 0, "Cannot unwrap a tensor view with non-zero dimension."
        self.tns.val.store(self.pos, self.op(self.tns.val.load(self.pos), val))
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

    def __call__(self, shape: tuple) -> BufferizedNDArrayAccessor:
        raise NotImplementedError(
            "Cannot directly instantiate BufferizedNDArrayAccessor from ftype"
        )

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

    def lower_dim(self, ctx, obj, r):
        return self.tns.lower_dim(ctx, obj.tns, r)

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
        self.tns.asm_repack(ctx, lhs.tns, obj.tns)
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

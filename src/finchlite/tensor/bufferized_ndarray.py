from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from finchlite import finch_assembly as asm
from finchlite import finch_notation as ntn
from finchlite.algebra import (
    FType,
    ImmutableStructFType,
    Tensor,
    TupleFType,
    ffuncs,
    ftype,
)
from finchlite.codegen import NumpyBuffer, NumpyBufferFType
from finchlite.codegen.numba_codegen import to_numpy_type
from finchlite.compile import looplets as lplt
from finchlite.compile.lower import AssemblyContext, FinchTensorFType

from .override_tensor import OverrideTensor


def _get_default_strides(size: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(np.cumprod((1,) + size[::-1]).astype(int))[-2::-1]


class BufferizedNDArray(OverrideTensor):
    def __init__(
        self,
        val: NumpyBuffer,
        shape: tuple[np.integer, ...],
        strides: tuple[np.integer, ...],
    ):
        self.val = val
        self._shape = shape
        self.strides = strides

    def to_numpy(self):
        """
        Convert the bufferized NDArray to a NumPy array.
        This is used to get the underlying NumPy array from the bufferized NDArray.
        """
        return self.val.arr.reshape(self._shape, copy=False)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "BufferizedNDArray":
        itemsize = arr.dtype.itemsize
        strides = tuple(np.intp(stride // itemsize) for stride in arr.strides)
        shape = tuple(np.intp(s) for s in arr.shape)
        val = NumpyBuffer(arr.reshape(-1, copy=False))
        return BufferizedNDArray(val, shape, strides)

    def __array__(self):
        return self.to_numpy()

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

    @property
    def fill_value(self) -> Any:
        """Default value to fill the tensor."""
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        """Data type of the tensor elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple:
        """Shape type of the tensor."""
        return self.ftype.shape_type

    def declare(self, init, op, shape):
        """
        Declare a bufferized NDArray with the given initialization value,
        operation, and shape.
        """
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
        return f"{self.ftype}(shape={self.shape})"

    def __repr__(self):
        return f"{self.ftype}(shape={self.shape})"


@dataclass(unsafe_hash=True)
class BufferizedNDArrayFields:
    stride: tuple[asm.Variable, ...]
    buf: asm.Variable
    buf_s: asm.Slot
    dirty_bit: bool


class BufferizedNDArrayFType(FinchTensorFType, ImmutableStructFType):
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

    def from_numpy(self, arr):
        val = NumpyBuffer(arr.reshape(-1, copy=False))
        strides = _get_default_strides(arr.shape)
        return BufferizedNDArray(
            val=val,
            shape=tuple(
                t(s)
                for s, t in zip(arr.shape, self.shape_t.struct_fieldtypes, strict=True)
            ),
            strides=tuple(
                t(s)
                for (s, t) in zip(
                    strides, self.strides_t.struct_fieldtypes, strict=True
                )
            ),
        )

    def __init__(
        self,
        *,
        buffer_type: NumpyBufferFType,
        ndim: int,
        dimension_type: TupleFType | tuple[FType, ...],
    ):
        if not isinstance(dimension_type, TupleFType):
            dimension_type = TupleFType.from_tuple(dimension_type)
        assert isinstance(dimension_type, TupleFType)
        # Normalize dimension field types to Finch ftypes so generated
        # result_type values are consistent with strict asm type checks.
        dimension_type = TupleFType.from_tuple(
            tuple(ftype(t) for t in dimension_type.struct_fieldtypes)
        )
        self.buf_t = buffer_type
        self._ndim = ndim
        self.shape_t = dimension_type
        self.strides_t = dimension_type  # assuming strides is the same type as shape

    def construct(
        self,
        shape: tuple[int, ...],
    ) -> BufferizedNDArray:
        arr = np.zeros(shape, dtype=to_numpy_type(self.element_type))
        return self.from_numpy(arr)

    def __call__(
        self,
        val: Any,
    ) -> BufferizedNDArray:
        """
        Convert a tensor to this bufferized ndarray type.

        Args:
            val: A tensor to convert to this type.
        Returns:
            A BufferizedNDArray instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
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
        return (
            f"BufferizedNDArrayFType(buffer_type={repr(self.buf_t)},"
            f" ndim = {self.ndim}, dimension_type ={repr(self.shape_t)})"
        )

    @property
    def ndim(self) -> np.intp:
        return np.intp(self._ndim)

    @ndim.setter
    def ndim(self, val):
        self._ndim = val

    @property
    def fill_value(self) -> Any:
        return np.zeros((), dtype=to_numpy_type(self.buf_t.element_type))[()]

    @property
    def element_type(self):
        return self.buf_t.element_type

    @property
    def shape_type(self) -> tuple:
        return tuple(self.shape_t.struct_fieldtypes)

    def lower_dim(self, ctx, obj, r):
        return asm.GetAttr(
            asm.GetAttr(obj.buf, asm.Literal("shape")),
            asm.Literal(f"element_{r}"),
        )

    def lower_declare(self, ctx, tns: ntn.Stack, init, op, shape):
        i_var = asm.Variable("i", self.buf_t.length_type)
        body = asm.Store(
            tns.obj.buf_s,
            i_var,
            asm.Literal(init.val),
        )
        ctx.exec(
            asm.ForLoop(i_var, asm.Literal(np.intp(0)), asm.Length(tns.obj.buf_s), body)
        )
        tns.obj.dirty_bit = True
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

    def lower_increment(self, ctx, obj, op, val): ...

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

        return BufferizedNDArrayFields(tuple(stride), val, buf_s, dirty_bit=False)

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
            tns_ftype = cast(BufferizedNDArrayFType, ftype(self.tns))
            pos = tns_ftype.buf_t.length_type(0)
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

    @property
    def fill_value(self) -> Any:
        """Default value to fill the tensor."""
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        """Data type of the tensor elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple:
        """Shape type of the tensor."""
        return self.ftype.shape_type

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


@dataclass(eq=True, frozen=True)
class BufferizedNDArrayAccessorFields:
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

    def construct(self, shape: tuple) -> BufferizedNDArrayAccessor:
        raise NotImplementedError(
            "Cannot directly instantiate BufferizedNDArrayAccessor from ftype"
        )

    def __call__(self, val: Any) -> BufferizedNDArrayAccessor:
        """
        Convert a tensor to this bufferized ndarray accessor type.

        Args:
            val: A tensor to convert to this type.
        Returns:
            A BufferizedNDArrayAccessor instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )

    def from_numpy(self, arr):
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

    def lower_unwrap(self, ctx, tns):
        return asm.Load(tns.obj.tns.buf_s, tns.obj.pos)

    def lower_increment(
        self,
        ctx: AssemblyContext,
        tns: ntn.Stack,
        op: ntn.Literal,
        val: ntn.NotationExpression,
    ):
        obj = tns.obj
        op_e, pos_e, val_e = ctx(op), obj.pos, ctx(val)
        increment_call = asm.Call(
            op_e,
            (asm.Load(obj.tns.buf_s, pos_e), val_e),
        )
        if obj.tns.dirty_bit and op.val is ffuncs.overwrite:
            increment_call = asm.Call(
                asm.Literal(ffuncs.init_write(tns.type.fill_value)),
                (asm.Load(obj.tns.buf_s, pos_e), increment_call),
            )

        ctx.exec(asm.Store(obj.tns.buf_s, pos_e, increment_call))

    def unfurl(self, ctx: AssemblyContext, tns, ext, mode, proto):
        def child_accessor(ctx, idx):
            pos_2 = asm.Variable(ctx.freshen(idx, f"_pos_{self.ndim - 1}"), self.pos)
            ctx.exec(
                asm.Assign(
                    pos_2,
                    asm.Call(
                        asm.Literal(ffuncs.add),
                        (
                            tns.obj.pos,
                            asm.Call(
                                asm.Literal(ffuncs.mul),
                                (
                                    tns.obj.tns.stride[self.nind],
                                    asm.Variable(idx.name, idx.type_),
                                ),
                            ),
                        ),
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

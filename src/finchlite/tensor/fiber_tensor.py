from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, NamedTuple, TypeVar

import numpy as np

from finchlite.algebra import Tensor
from finchlite.algebra.algebra import register_property
from finchlite.codegen.numpy_buffer import NumpyBuffer
from finchlite.symbolic import FType, FTyped

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..compile.lower import FinchTensorFType


class LevelFType(FType, ABC):
    """
    An abstract base class representing the ftype of levels.
    """

    @property
    @abstractmethod
    def ndim(self):
        """
        Number of dimensions of the fibers in the structure.
        """
        ...

    @property
    @abstractmethod
    def fill_value(self):
        """
        Fill value of the fibers, or `None` if dynamic.
        """
        ...

    @property
    @abstractmethod
    def element_type(self):
        """
        Type of elements stored in the fibers.
        """
        ...

    @property
    @abstractmethod
    def shape_type(self):
        """
        Tuple of types of the dimensions in the shape
        """
        ...

    @property
    @abstractmethod
    def position_type(self):
        """
        Type of positions within the levels.
        """
        ...

    @property
    @abstractmethod
    def buffer_factory(self):
        """
        Function to create default buffers for the fibers.
        """
        ...


class Level(FTyped, ABC):
    """
    An abstract base class representing a fiber allocator that manages fibers in
    a tensor.
    """

    @property
    @abstractmethod
    def shape(self):
        """
        Shape of the fibers in the structure.
        """
        ...

    @property
    def ndim(self):
        return self.ftype.ndim

    @property
    def fill_value(self):
        return self.ftype.fill_value

    @property
    def element_type(self):
        return self.ftype.element_type

    @property
    def shape_type(self):
        return self.ftype.shape_type

    @property
    def position_type(self):
        return self.ftype.position_type

    @property
    def buffer_factory(self):
        return self.ftype.buffer_factory


Tp = TypeVar("Tp")


@dataclass
class FiberTensor(Generic[Tp], Tensor):
    """
    A class representing a tensor with fiber structure.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl: Level
    pos: Tp

    def __repr__(self):
        res = f"FiberTensor(lvl={self.lvl}"
        if self.pos is not None:
            res += f", pos={self.pos}"
        res += ")"
        return res

    @property
    def ftype(self):
        """
        Returns the ftype of the fiber tensor, which is a FiberTensorFType.
        """
        return FiberTensorFType(self.lvl.ftype, type(self.pos))

    @property
    def shape(self):
        return self.lvl.shape

    @property
    def ndim(self):
        return self.lvl.ndim

    @property
    def shape_type(self):
        return self.lvl.shape_type

    @property
    def element_type(self):
        return self.lvl.element_type

    @property
    def fill_value(self):
        return self.lvl.fill_value

    @property
    def position_type(self):
        return self.lvl.position_type

    @property
    def buffer_factory(self):
        """
        Returns the ftype of the buffer used for the fibers.
        This is typically a NumpyBufferFType or similar.
        """
        return self.lvl.buffer_factory


class FiberTensorFields(NamedTuple):
    stride: tuple[asm.Variable, ...]
    buf: asm.Variable
    buf_s: asm.Slot


@dataclass(unsafe_hash=True)
class FiberTensorFType(FinchTensorFType):
    """
    An abstract base class representing the ftype of a fiber tensor.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl: LevelFType
    _position_type: type | None = None

    def __post_init__(self):
        if self._position_type is None:
            self._position_type = self.lvl.position_type

    def __call__(self, *, shape=None, val=None):
        """
        Creates an instance of a FiberTensor with the given arguments.
        """
        if shape is None:
            shape = val.shape
            val = NumpyBuffer(val.reshape[-1])
        return FiberTensor(self.lvl(shape, val), self.lvl.position_type(1))

    @property
    def shape(self):
        return self.lvl.shape

    @property
    def ndim(self):
        return self.lvl.ndim

    @property
    def shape_type(self):
        return self.lvl.shape_type

    @property
    def element_type(self):
        return self.lvl.element_type

    @property
    def fill_value(self):
        return self.lvl.fill_value

    @property
    def position_type(self):
        return self._position_type

    @property
    def buffer_factory(self):
        return self.lvl.buffer_factory

    def unfurl(self, ctx, tns, ext, mode, proto):
        op = None
        if isinstance(mode, ntn.Update):
            op = mode.op
        tns = ctx.resolve(tns).obj
        acc_t = self.lvl.unfurl(
            ctx,
        )

        # acc_t = BufferizedNDArrayAccessorFType(self, 0, self.buf_t.length_type, op)
        # obj = BufferizedNDArrayAccessorFields(
        #     tns, 0, asm.Literal(self.buf_t.length_type(0)), op
        # )
        return acc_t.unfurl(ctx, ntn.Stack(obj, acc_t), ext, mode, proto)

    def lower_freeze(self, ctx, tns, op):
        raise NotImplementedError

    def lower_thaw(self, ctx, tns, op):
        raise NotImplementedError

    def lower_unwrap(self, ctx, obj):
        raise NotImplementedError

    def lower_increment(self, ctx, obj, val):
        raise NotImplementedError

    def lower_declare(self, ctx, tns, init, op, shape):
        i_var = asm.Variable("i", self.buffer_factory.length_type)
        body = asm.Store(
            tns.obj.buf_s,
            i_var,
            asm.Literal(init.val),
        )
        ctx.exec(
            asm.ForLoop(i_var, asm.Literal(np.intp(0)), asm.Length(tns.obj.buf_s), body)
        )
        return

    def asm_unpack(self, ctx, var_n, val):
        """
        Unpack the into asm context.
        """
        stride = []
        shape_type = self.shape_type
        for i in range(self.ndim):
            stride_i = asm.Variable(f"{var_n}_stride_{i}", shape_type[i])
            stride.append(stride_i)
            stride_e = asm.GetAttr(val, asm.Literal("strides"))
            stride_i_e = asm.GetAttr(stride_e, asm.Literal(f"element_{i}"))
            ctx.exec(asm.Assign(stride_i, stride_i_e))
        buf = asm.Variable(f"{var_n}_buf", self.buffer_factory)
        buf_e = asm.GetAttr(val, asm.Literal("buf"))
        ctx.exec(asm.Assign(buf, buf_e))
        buf_s = asm.Slot(f"{var_n}_buf_slot", self.buffer_factory)
        ctx.exec(asm.Unpack(buf_s, buf))

        return FiberTensorFields(tuple(stride), buf, buf_s)


def fiber_tensor(lvl: LevelFType, position_type: type | None = None):
    """
    Creates a FiberTensorFType with the given level ftype and position type.

    Args:
        lvl: The level ftype to be used for the tensor.
        val: The value to store in the tensor.
        pos_type: The type of positions within the tensor. Defaults to None.

    Returns:
        An instance of FiberTensorFType.
    """
    # mypy does not understand that dataclasses generate __hash__ and __eq__
    # https://github.com/python/mypy/issues/19799
    return FiberTensorFType(lvl, position_type)  # type: ignore[abstract]


register_property(FiberTensor, "asarray", "__attr__", lambda x: x)

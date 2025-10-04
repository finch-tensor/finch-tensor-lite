from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, TypeVar

import numpy as np

from finchlite.finch_assembly.struct import TupleFType

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import Tensor, register_property
from ..codegen.numpy_buffer import NumpyBuffer
from ..compile.lower import FinchTensorFType
from ..symbolic import FTyped


class LevelFType(FinchTensorFType, ABC):
    """
    An abstract base class representing the ftype of levels.
    """

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

    @property
    @abstractmethod
    def buffer_type(self): ...


class Level(FTyped, ABC):
    """
    An abstract base class representing a fiber allocator that manages fibers in
    a tensor.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """
        Shape of the fibers in the structure.
        """
        ...

    @property
    @abstractmethod
    def stride(self) -> np.integer: ...

    @property
    @abstractmethod
    def buf(self) -> Any: ...

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

    @property
    def buffer_type(self):
        return self.ftype.buffer_type


Tp = TypeVar("Tp")


@dataclass
class FiberTensor(Tensor, Generic[Tp]):
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
    def stride(self):
        return self.lvl.stride

    @property
    def buf(self):
        return self.lvl.buf

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

    def to_numpy(self) -> np.ndarray:
        # TODO: temporary for dense only. TBD in sparse_level PR
        return np.reshape(self.buf.arr, self.shape, copy=False)


class FiberTensorFields(NamedTuple):
    lvl: asm.Variable
    buf_s: asm.Slot


@dataclass(unsafe_hash=True)
class FiberTensorFType(FinchTensorFType, asm.AssemblyStructFType):
    """
    An abstract base class representing the ftype of a fiber tensor.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl_t: LevelFType
    position_type: type | None = None

    @property
    def struct_name(self):
        # TODO: include dt = np.dtype(self.buf_t.element_type)
        return "FiberTensorFType"

    @property
    def struct_fields(self):
        return [
            ("lvl", self.lvl_t),
            ("shape", TupleFType.from_tuple(self.shape_type)),
        ]

    def __post_init__(self):
        if self.position_type is None:
            self.position_type = self.lvl_t.position_type

    def __call__(self, *, lvl=None, shape=None, val=None):
        """
        Creates an instance of a FiberTensor with the given arguments.
        """
        if lvl is not None:
            return FiberTensor(lvl, self.lvl_t.position_type(1))
        if shape is None:
            shape = val.shape
            val = NumpyBuffer(val.reshape(-1))
        return FiberTensor(
            self.lvl_t(shape=shape, val=val), self.lvl_t.position_type(1)
        )

    def __str__(self):
        return f"FiberTensorFType({self.lvl_t})"

    @property
    def ndim(self):
        return self.lvl_t.ndim

    @property
    def shape_type(self):
        return self.lvl_t.shape_type

    @property
    def element_type(self):
        return self.lvl_t.element_type

    @property
    def fill_value(self):
        return self.lvl_t.fill_value

    @property
    def buffer_factory(self):
        return self.lvl_t.buffer_factory

    @property
    def buffer_type(self):
        return self.lvl_t.buffer_type

    def from_kwargs(self, **kwargs) -> "FiberTensorFType":
        pos_t = kwargs.get("position_type", self.position_type)
        return FiberTensorFType(self.lvl_t.from_kwargs(**kwargs), pos_t)

    def to_kwargs(self):
        return {
            "position_type": self.position_type,
            "shape_type": self.shape_type,
        } | self.lvl_t.to_kwargs()

    # TODO: temporary approach for suitable rep and traits
    def add_levels(self, idxs: list[int]):
        from .level.dense_level import dense

        copy = deepcopy(self)
        lvl = copy
        for idx in range(max(idxs) + 1):
            if idx in idxs:
                lvl.lvl_t = dense(lvl.lvl_t, dimension_type=np.intp)
            lvl = lvl.lvl_t
        return copy

    # TODO: temporary approach for suitable rep and traits
    def remove_levels(self, idxs: list[int]):
        copy = deepcopy(self)
        lvl = copy
        for i in range(self.ndim):
            if i in idxs:
                lvl.lvl_t = lvl.lvl_t.lvl_t
            lvl = lvl.lvl_t
        return copy

    def unfurl(self, ctx, tns, ext, mode, proto):
        op = None
        if isinstance(mode, ntn.Update):
            op = mode.op
        tns = ctx.resolve(tns).obj
        obj = self.lvl_t.get_fields_class(
            tns.lvl, tns.buf_s, 0, asm.Literal(self.position_type(0)), op
        )
        return self.lvl_t.unfurl(ctx, ntn.Stack(obj, self.lvl_t), ext, mode, proto)

    def lower_freeze(self, ctx, tns, op):
        return tns

    def lower_thaw(self, ctx, tns, op):
        raise NotImplementedError

    def lower_unwrap(self, ctx, obj):
        raise NotImplementedError

    def lower_increment(self, ctx, obj, val):
        raise NotImplementedError

    def lower_declare(self, ctx, tns, init, op, shape):
        return self.lvl_t.lower_declare(ctx, tns.obj.buf_s, init, op, shape)

    def asm_unpack(self, ctx, var_n, val):
        """
        Unpack the into asm context.
        """
        val_lvl = asm.GetAttr(val, asm.Literal("lvl"))
        buf_s = self.lvl_t.asm_unpack(ctx, var_n, val_lvl)
        return FiberTensorFields(val_lvl, buf_s)

    def asm_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from the context.
        """
        ctx.exec(asm.Repack(obj.buf_s))
        return


def fiber_tensor(lvl: LevelFType, position_type: type | None = None):
    """
    Creates a FiberTensorFType with the given level ftype and position type.

    Args:
        lvl: The level ftype to be used for the tensor.
        position_type: The type of positions within the tensor. Defaults to None.

    Returns:
        An instance of FiberTensorFType.
    """
    # mypy does not understand that dataclasses generate __hash__ and __eq__
    # https://github.com/python/mypy/issues/19799
    return FiberTensorFType(lvl, position_type)  # type: ignore[abstract]


register_property(FiberTensor, "asarray", "__attr__", lambda x: x)

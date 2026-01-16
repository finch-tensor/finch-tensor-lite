from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import Tensor, register_property
from ..compile.lower import FinchTensorFType
from ..symbolic import FType, FTyped


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
        Tuple of types of the dimensions in the shape.
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

    @property
    @abstractmethod
    def buffer_type(self): ...

    @property
    @abstractmethod
    def next_level(self):
        """
        Get the nested level.
        """
        ...

    @abstractmethod
    def get_fields_class(self, tns, buf_s, pos, op, dirty_bit): ...

    @abstractmethod
    def level_unfurl(self, ctx, tns, ext, mode, proto, pos):
        """
        Emit code to unfurl the fiber at position `pos` in the level.
        """
        ...

    @abstractmethod
    def level_lower_freeze(self, ctx, tns, op, pos):
        """
        Emit code to freeze `pos` previously assembled positions in the level.
        """
        ...

    @abstractmethod
    def level_lower_thaw(self, ctx, tns, op, pos):
        """
        Emit code to thaw `pos` previously assembled positions in the level.
        """
        ...

    @abstractmethod
    def level_lower_unwrap(self, ctx, obj, pos):
        """
        Emit code to return the unwrapped scalar at position `pos` in the level.
        """
        ...

    @abstractmethod
    def level_lower_increment(self, ctx, obj, val, pos):
        """
        Emit code to increment position `pos` in the level.
        """
        ...

    @abstractmethod
    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        """
        Emit code to lower a declare of `pos` previously assembled positions in
        the level.
        """
        ...

    @abstractmethod
    def level_lower_dim(self, ctx, obj, r):
        """
        Emit code to return the size of dimension `r` of the subtensors in the level.
        """
        ...

    @abstractmethod
    def level_asm_unpack(self, ctx, var_n, val) -> asm.Slot:
        """
        Emit code unpacking the level.
        """
        ...


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
    def val(self) -> Any: ...

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


@dataclass
class FiberTensor(Tensor):
    """
    A class representing a tensor with fiber structure.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl: Level

    def __repr__(self):
        return f"FiberTensor(lvl={self.lvl})"

    @property
    def ftype(self):
        """
        Returns the ftype of the fiber tensor, which is a FiberTensorFType.
        """
        return FiberTensorFType(self.lvl.ftype)

    @property
    def shape(self):
        return self.lvl.shape

    @property
    def stride(self):
        return self.lvl.stride

    @property
    def val(self):
        return self.lvl.val

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
        return self.lvl.buffer_factory

    def to_numpy(self) -> np.ndarray:
        # TODO: temporary for dense only. TBD in sparse_level PR
        return np.reshape(self.lvl.val.arr, self.shape, copy=False)


@dataclass(eq=True, frozen=True)
class FiberTensorFields:
    lvl: asm.AssemblyExpression
    buf_s: asm.Slot


@dataclass(unsafe_hash=True)
class FiberTensorFType(FinchTensorFType, asm.AssemblyStructFType):
    """
    An abstract base class representing the ftype of a fiber tensor.

    Attributes:
        lvl_t: The level ftype to be used for the tensor.
    """

    lvl_t: LevelFType

    @property
    def struct_name(self):
        # TODO: include dt = np.dtype(self.buf_t.element_type)
        return "FiberTensorFType"

    @property
    def struct_fields(self):
        return [
            ("lvl", self.lvl_t),
            ("shape", asm.TupleFType.from_tuple(self.shape_type)),
        ]

    def __call__(self, shape):
        """
        Creates an instance of a FiberTensor with the given arguments.
        """
        return FiberTensor(self.lvl_t(shape=shape))

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
    def position_type(self):
        return self.lvl_t.position_type

    @property
    def buffer_factory(self):
        return self.lvl_t.buffer_factory

    @property
    def buffer_type(self):
        return self.lvl_t.buffer_type

    def unfurl(self, ctx, tns, ext, mode, proto):
        tns = ctx.resolve(tns).obj
        pos = tns.pos if hasattr(tns, "pos") else asm.Literal(self.position_type(0))
        dirty_bit = tns.dirty_bit if hasattr(tns, "dirty_bit") else asm.Literal(False)
        op = mode.op if isinstance(mode, ntn.Update) else None
        obj = self.lvl_t.get_fields_class(
            tns.lvl,
            tns.buf_s,
            pos,
            op,
            dirty_bit,
        )
        return self.lvl_t.level_unfurl(
            ctx, ntn.Stack(obj, self), ext, mode, proto, obj.pos
        )

    def lower_freeze(self, ctx, tns, op):
        return self.lvl_t.level_lower_freeze(ctx, tns.obj.buf_s, op, tns.obj.pos)

    def lower_thaw(self, ctx, tns, op):
        return self.lvl_t.level_lower_thaw(ctx, tns.obj.buf_s, op, tns.obj.pos)

    def lower_unwrap(self, ctx, tns):
        return self.lvl_t.level_lower_unwrap(ctx, tns.obj, tns.obj.pos)

    def lower_increment(self, ctx, tns, val):
        return self.lvl_t.level_lower_increment(ctx, tns.obj, val, tns.obj.pos)

    def lower_declare(self, ctx, tns, init, op, shape):
        return self.lvl_t.level_lower_declare(
            ctx, tns.obj.buf_s, init, op, shape, tns.obj.pos
        )

    def lower_dim(self, ctx, obj, r):
        return self.lvl_t.level_lower_dim(ctx, obj.lvl, r)

    def asm_unpack(self, ctx, var_n, val) -> FiberTensorFields:
        """
        Unpack the into asm context.
        """
        val_lvl = asm.GetAttr(val, asm.Literal("lvl"))
        buf_s = self.lvl_t.level_asm_unpack(ctx, var_n, val_lvl)
        return FiberTensorFields(val_lvl, buf_s)

    def asm_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from the context.
        """
        ctx.exec(asm.Repack(obj.buf_s))
        return

    def from_fields(self, *args) -> "FiberTensor":
        return FiberTensor(*args)


def fiber_tensor(lvls: tuple):
    """
    Creates a FiberTensorFType with the given level ftype and position type.

    Args:
        lvl: The level ftype to be used for the tensor.
    Returns:
        An instance of a fiber tensor format.
    """
    return (FiberTensor, lvls)


register_property(FiberTensor, "asarray", "__attr__", lambda x: x)

import operator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ... import finch_assembly as asm
from ... import finch_notation as ntn
from ...compile import LoopletContext
from ...compile import looplets as lplt
from ..fiber_tensor import FiberTensorFields, FiberTensorFType, Level, LevelFType


class DenseLevelFields(NamedTuple):
    pass


@dataclass(unsafe_hash=True)
class DenseLevelFType(LevelFType, asm.AssemblyStructFType):
    _lvl_t: LevelFType
    dimension_type: Any = None

    @property
    def struct_name(self):
        return "DenseLevelFType"

    @property
    def struct_fields(self):
        return [
            ("lvl", self.lvl_t),
            ("dimension", self.dimension_type),
            ("stride", self.dimension_type),
        ]

    def __post_init__(self):
        if self.dimension_type is None:
            self.dimension_type = np.intp

    def __call__(self, *, shape):
        """
        Creates an instance of DenseLevel with the given ftype.

        Args:
            shape: The shape to be used for the level. (mandatory)
        Returns:
            An instance of DenseLevel.
        """
        lvl = self.lvl_t(shape=shape[1:])
        return DenseLevel(lvl, self.dimension_type(shape[0]))

    def from_numpy(self, shape, val):
        """
        Creates an instance of DenseLevel with the given shape.

        Args:
            shape: The shape to be used for the level.
            val: Value to pass to ElementLevel.
        Returns:
            An instance of DenseLevel.
        """
        lvl = self.lvl_t.from_numpy(shape[1:], val)
        return DenseLevel(lvl, self.dimension_type(shape[0]))

    def __str__(self):
        return f"DenseLevelFType({self.lvl_t})"

    @property
    def ndim(self):
        return 1 + self.lvl_t.ndim

    @property
    def fill_value(self):
        return self.lvl_t.fill_value

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        return self.lvl_t.element_type

    @property
    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        return (self.dimension_type, *self.lvl_t.shape_type)

    @property
    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        return self.lvl_t.position_type

    @property
    def buffer_type(self):
        return self.lvl_t.buffer_type

    @property
    def buffer_factory(self):
        """
        Returns the ftype of the buffer used for the fibers.
        """
        return self.lvl_t.buffer_factory

    @property
    def lvl_t(self):
        return self._lvl_t

    def level_asm_unpack(self, ctx, var_n, val) -> tuple[DenseLevelFields, ...]:
        val_lvl = asm.GetAttr(val, asm.Literal("lvl"))
        return (DenseLevelFields(),) + self.lvl_t.level_asm_unpack(ctx, var_n, val_lvl)

    def level_asm_repack(self, ctx, lvls_slots):
        self.lvl_t.level_asm_repack(ctx, lvls_slots[1:])

    def level_lower_dim(self, ctx, obj, r):
        if r == 0:
            return asm.GetAttr(obj, asm.Literal("dimension"))
        obj = asm.GetAttr(obj, asm.Literal("lvl"))
        return self.lvl_t.level_lower_dim(ctx, obj, r - 1)

    def level_lower_declare(self, ctx, lvls_slots, init, op, shape, pos):
        return self.lvl_t.level_lower_declare(ctx, lvls_slots[1:], init, op, shape, pos)

    def level_lower_freeze(self, ctx, lvls_slots, op, pos):
        return self.lvl_t.level_lower_freeze(ctx, lvls_slots[1:], op, pos)

    def level_lower_thaw(self, ctx, lvls_slots, op, pos):
        return self.lvl_t.level_lower_thaw(ctx, lvls_slots[1:], op, pos)

    def level_lower_increment(self, ctx, obj, op, val, pos):
        raise NotImplementedError(
            "DenseLevelFType does not support level_lower_increment."
        )

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError(
            "DenseLevelFType does not support level_lower_unwrap."
        )

    def level_unfurl(self, ctx, stack: asm.Stack, ext, mode, proto, pos):
        tns: FiberTensorFields = stack.obj
        ft_ftype: FiberTensorFType = stack.type

        def child_accessor(ctx: LoopletContext, idx):
            pos_2 = asm.Variable(
                ctx.freshen(ctx.idx, f"_pos_{self.ndim - 1}"), self.position_type
            )
            ctx.exec(
                asm.Assign(
                    pos_2,
                    asm.Call(
                        asm.Literal(operator.add),
                        (
                            pos,
                            asm.Call(
                                asm.Literal(operator.mul),
                                (
                                    asm.GetAttr(tns.lvl, asm.Literal("stride")),
                                    asm.Variable(ctx.idx.name, ctx.idx.type_),
                                ),
                            ),
                        ),
                    ),
                )
            )
            return ntn.Stack(
                FiberTensorFields(
                    asm.GetAttr(tns.lvl, asm.Literal("lvl")),
                    tns.lvls_slots[1:],
                    pos_2,
                    tns.dirty_bit,
                ),
                FiberTensorFType(ft_ftype.lvl_t.lvl_t),  # type: ignore[abstract]
            )

        return lplt.Lookup(
            body=lambda ctx, idx: lplt.Leaf(
                body=lambda ctx: child_accessor(ctx, idx),
            )
        )

    def from_fields(self, lvl, dimension, stride) -> "DenseLevel":
        return DenseLevel(lvl=lvl, dimension=dimension)


def dense(lvl, dimension_type=None):
    return DenseLevelFType(lvl, dimension_type=dimension_type)


@dataclass
class DenseLevel(Level):
    """
    A class representing dense level.
    """

    lvl: Level
    dimension: np.integer

    @property
    def shape(self) -> tuple:
        return (self.dimension, *self.lvl.shape)

    @property
    def stride(self) -> np.integer:
        stride = self.lvl.stride
        if self.lvl.ndim == 0:
            return stride
        return self.lvl.shape[0] * stride

    @property
    def ftype(self) -> DenseLevelFType:
        # mypy does not understand that dataclasses generate __hash__ and __eq__
        # https://github.com/python/mypy/issues/19799
        return DenseLevelFType(self.lvl.ftype, type(self.dimension))  # type: ignore[abstract]

    @property
    def val(self) -> Any:
        return self.lvl.val

    def __str__(self):
        return f"DenseLevel(lvl={self.lvl}, dim={self.dimension})"

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ... import finch_assembly as asm
from ... import finch_notation as ntn
from ...compile import AssemblyContext, LoopletContext
from ...compile import looplets as lplt
from ...compile.lower import SymbolicExtent
from ..fiber_tensor import FiberTensorFields, FiberTensorFType, Level, LevelFType
from ...algebra import ffunc

class DenseLevelFields(NamedTuple):
    lvl_asm: asm.AssemblyExpression  # assembly expression of the current level
    next_lvl: NamedTuple


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

    def level_asm_unpack(self, ctx, var_n, val) -> DenseLevelFields:
        return DenseLevelFields(
            val,
            self.lvl_t.level_asm_unpack(
                ctx, var_n, asm.GetAttr(val, asm.Literal("lvl"))
            ),
        )

    def level_asm_repack(self, ctx, lvl_fields: DenseLevelFields):
        self.lvl_t.level_asm_repack(ctx, lvl_fields.next_lvl)

    def level_lower_dim(self, ctx, lvl_fields: DenseLevelFields, r):
        if r == 0:
            return asm.GetAttr(lvl_fields.lvl_asm, asm.Literal("dimension"))
        return self.lvl_t.level_lower_dim(ctx, lvl_fields.next_lvl, r - 1)

    def level_lower_declare(
        self, ctx, lvl_fields: DenseLevelFields, init, op, shape, pos
    ):
        return self.lvl_t.level_lower_declare(
            ctx, lvl_fields.next_lvl, init, op, shape, pos
        )

    def level_lower_freeze(self, ctx, lvl_fields: DenseLevelFields, op, pos):
        return self.lvl_t.level_lower_freeze(ctx, lvl_fields.next_lvl, op, pos)

    def level_lower_thaw(self, ctx, lvl_fields: DenseLevelFields, op, pos):
        return self.lvl_t.level_lower_thaw(ctx, lvl_fields.next_lvl, op, pos)

    def level_lower_increment(self, ctx, obj, op, val, pos):
        raise NotImplementedError(
            "DenseLevelFType does not support level_lower_increment."
        )

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError(
            "DenseLevelFType does not support level_lower_unwrap."
        )

    def level_unfurl(
        self,
        ctx: AssemblyContext,
        stack: ntn.Stack,
        ext: SymbolicExtent,
        mode,
        proto,
        pos: asm.AssemblyExpression,
    ):
        tns: FiberTensorFields = stack.obj
        ft_ftype: FiberTensorFType = stack.type
        assert isinstance(tns.lvl_fields, DenseLevelFields)
        lvl = tns.lvl_fields.lvl_asm
        next_lvl = tns.lvl_fields.next_lvl

        def child_accessor(ctx: LoopletContext, idx: ntn.Variable):
            pos_2 = asm.Variable(
                ctx.freshen(idx, f"_pos_{self.ndim - 1}"), self.position_type
            )
            ctx.exec(
                asm.Assign(
                    pos_2,
                    asm.Call(
                        asm.Literal(ffunc.add),
                        (
                            pos,
                            asm.Call(
                                asm.Literal(ffunc.mul),
                                (
                                    asm.GetAttr(lvl, asm.Literal("stride")),
                                    asm.Variable(
                                        idx.name, idx.type_
                                    ),  # TODO: lower with ctx.ctx
                                ),
                            ),
                        ),
                    ),
                )
            )
            return ntn.Stack(
                FiberTensorFields(
                    next_lvl, pos_2, tns.dirty_bit, tns.visited_idxs + (idx,)
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
        if self.lvl.ndim == 0 or self.lvl.stride == 0:
            return np.intp(1)
        return self.lvl.shape[0] * self.lvl.stride

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

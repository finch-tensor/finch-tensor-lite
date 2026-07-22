from dataclasses import dataclass
from typing import Any

import numpy as np

from finchlite import finch_assembly as asm
from finchlite import finch_notation as ntn
from finchlite.algebra import FType, ImmutableStructFType, ffuncs, ftype, ftypes
from finchlite.compile import AssemblyContext, LoopletContext
from finchlite.compile import looplets as lplt
from finchlite.compile.lower import SymbolicExtent
from finchlite.tensor.fiber_tensor import (
    FiberTensorFType,
    Level,
    LevelFType,
)
from finchlite.tensor.traits import Dense


@dataclass(unsafe_hash=True)
class DenseLevelFType(LevelFType, ImmutableStructFType):
    _lvl_t: LevelFType
    dimension_type: FType = ftypes.intp

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
    
    def level_iter_cost(self, fields, stats, stats_factory, num_pos, l):
        """
        For all the parent num_pos passed we go through all the positions given dense
        """
        n = stats.get_dim_size(fields[l])
        return num_pos*n + self.lvl_t.level_iter_cost(fields,stats,stats_factory,num_pos*n,l+1)
    
    def level_cost(self,fields,stats,stats_factory,num_pos,l)->float:
        n = stats.get_dim_size(fields[l])
        return self.lvl_t.level_cost(fields,stats,stats_factory,num_pos*n,l+1)


    def __post_init__(self):
        self.dimension_type = ftype(self.dimension_type)

    def construct(self, shape: tuple[Any, ...], *, pos: int) -> "DenseLevel":
        """
        Creates an instance of DenseLevel with the given ftype.

        Args:
            shape: The shape to be used for the level. (mandatory)
        Returns:
            An instance of DenseLevel.
        """
        dimension = self.dimension_type(shape[0])
        lvl = self.lvl_t.construct(
            shape=shape[1:],
            pos=int(pos) * int(dimension),
        )
        return DenseLevel(lvl, dimension)

    def __call__(self, val: Any) -> "DenseLevel":
        """
        Convert a level to this dense level type.

        Args:
            val: A value to convert to this type.
        Returns:
            A DenseLevel instance of this type.
        """
        raise NotImplementedError(
            f"Level conversion not yet implemented for {type(self).__name__}"
        )

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

    def level_format_properties(self, n):
        return [Dense(tuple(range(n + 1)))] + self.lvl_t.level_format_properties(n + 1)

    def level_lower_dim(self, ctx, lvl, r):
        if r == 0:
            return asm.GetAttr(lvl, asm.Literal("dimension"))
        return self.lvl_t.level_lower_dim(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), r - 1
        )

    def level_lower_declare(self, ctx, lvl, init, op, shape, pos):
        return self.lvl_t.level_lower_declare(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), init, op, shape, pos
        )

    def level_lower_freeze(self, ctx, lvl, op, pos):
        return self.lvl_t.level_lower_freeze(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), op, pos
        )

    def level_lower_thaw(self, ctx, lvl, op, pos):
        return self.lvl_t.level_lower_thaw(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), op, pos
        )

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
        fiber: ntn.Fiber,
        ext: SymbolicExtent,
        mode,
        proto,
        pos: asm.AssemblyExpression,
    ):
        tns = fiber
        ft_ftype: FiberTensorFType = fiber.type
        lvl = ctx.fiber_level(tns)

        def child_accessor(ctx: LoopletContext, idx: ntn.Variable):
            if idx.type_ is None:
                raise TypeError(f"Expected loop variable type for {idx.name}")
            pos_2 = asm.Variable(
                ctx.freshen(idx, f"_pos_{self.ndim - 1}"), self.position_type
            )
            ctx.exec(
                asm.Assign(
                    pos_2,
                    asm.Call(
                        asm.Literal(ffuncs.add),
                        (
                            pos,
                            asm.Call(
                                asm.Literal(ffuncs.mul),
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
            child_type = FiberTensorFType(ft_ftype.lvl_t.lvl_t)  # type: ignore[abstract]
            return ntn.Fiber(
                tns.root,
                ntn.Child(tns.lvl),
                pos_2,
                child_type,
                (*tns.idxs, idx),
            )

        return lplt.Lookup(
            body=lambda ctx, idx: lplt.Leaf(
                body=lambda ctx: child_accessor(ctx, idx),
            )
        )

    def from_fields(self, lvl, dimension, stride) -> "DenseLevel":
        return DenseLevel(lvl=lvl, dimension=dimension)


def dense(lvl, dimension_type=ftypes.intp):
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
        return DenseLevelFType(self.lvl.ftype, ftype(self.dimension))  # type: ignore[abstract]

    @property
    def val(self) -> Any:
        return self.lvl.val

    def __str__(self):
        return f"DenseLevel(lvl={self.lvl}, dim={self.dimension})"

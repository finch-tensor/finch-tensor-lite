import operator
from abc import ABC
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ... import finch_assembly as asm
from ... import finch_notation as ntn
from ...compile import looplets as lplt
from ..fiber_tensor import FiberTensorFields, Level, LevelFType


class DenseLevelFields(NamedTuple):
    tns: FiberTensorFields
    nind: int
    pos: asm.AssemblyNode
    op: Any


@dataclass(unsafe_hash=True)
class DenseLevelFType(LevelFType, ABC):
    lvl: Any
    dimension_type: Any = None
    pos: asm.AssemblyNode | None = None
    op: Any = None

    def __post_init__(self):
        if self.dimension_type is None:
            self.dimension_type = np.intp

    def __call__(self, shape, val=None):
        """
        Creates an instance of DenseLevel with the given ftype.
        Args:
            shape: The shape to be used for the level. (mandatory)
        Returns:
            An instance of DenseLevel.
        """
        lvl = self.lvl(shape[1:], val)
        return DenseLevel(self, lvl, self.dimension_type(shape[0]))

    def __str__(self):
        return f"DenseLevelFType({self.lvl})"

    @property
    def ndim(self):
        return 1 + self.lvl.ndim

    @property
    def fill_value(self):
        return self.lvl.fill_value

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        return self.lvl.element_type

    @property
    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        return (self.dimension_type, *self.lvl.shape_type)

    @property
    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        return self.lvl.position_type

    @property
    def buffer_factory(self):
        """
        Returns the ftype of the buffer used for the fibers.
        """
        return self.lvl.buffer_factory

    def get_fields_class(self, tns, nind, pos, op):
        return DenseLevelFields(tns, nind, pos, op)

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
                                    tns.obj.tns.stride[tns.obj.nind],
                                    asm.Variable(ctx.idx.name, ctx.idx.type_),
                                ],
                            ),
                        ],
                    ),
                )
            )
            return ntn.Stack(
                self.lvl.get_fields_class(
                    tns.obj.tns, tns.obj.nind + 1, pos_2, tns.obj.op
                ),
                self.lvl,
            )

        return lplt.Lookup(
            body=lambda ctx, idx: lplt.Leaf(
                body=lambda ctx: child_accessor(ctx, idx),
            )
        )


def dense(lvl, dimension_type=None):
    return DenseLevelFType(lvl, dimension_type=dimension_type)


@dataclass
class DenseLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    _format: DenseLevelFType
    lvl: Any
    dimension: Any
    pos: asm.AssemblyNode | None = None

    @property
    def shape(self) -> tuple:
        return (self.dimension, *self.lvl.shape)

    @property
    def ftype(self):
        return self._format

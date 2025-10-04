import operator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ... import finch_assembly as asm
from ... import finch_notation as ntn
from ...codegen import NumpyBufferFType
from ...compile import looplets as lplt
from ..fiber_tensor import Level, LevelFType


class DenseLevelFields(NamedTuple):
    lvl: asm.Variable
    buf_s: NumpyBufferFType
    nind: int
    pos: asm.AssemblyNode
    op: Any


@dataclass(unsafe_hash=True)
class DenseLevelFType(LevelFType, asm.AssemblyStructFType):
    lvl_t: LevelFType
    dimension_type: Any = None
    op: Any = None

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

    def __call__(self, *, lvl=None, dimension=None, stride=None, shape=None, val=None):
        """
        Creates an instance of DenseLevel with the given ftype.
        Args:
            shape: The shape to be used for the level. (mandatory)
        Returns:
            An instance of DenseLevel.
        """
        if lvl is not None and dimension is not None:
            return DenseLevel(self, lvl, dimension)
        lvl = self.lvl_t(shape=shape[1:], val=val)
        return DenseLevel(self, lvl, self.dimension_type(shape[0]))

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

    def from_kwargs(self, **kwargs) -> "DenseLevelFType":
        dimension_type = kwargs.get("dimension_type", self.position_type)
        if "shape_type" in kwargs:
            shape_type = kwargs["shape_type"]
            dimension_type = shape_type[0]
            kwargs["shape_type"] = shape_type[1:]
        op = kwargs.get("op", self.op)
        return DenseLevelFType(self.lvl_t.from_kwargs(**kwargs), dimension_type, op)

    def to_kwargs(self):
        return {
            "dimension_type": self.position_type,
            "op": self.op,
        } | self.lvl_t.to_kwargs()

    def asm_unpack(self, ctx, var_n, val):
        val_lvl = asm.GetAttr(val, asm.Literal("lvl"))
        return self.lvl_t.asm_unpack(ctx, var_n, val_lvl)

    def get_fields_class(self, tns, buf_s, nind, pos, op):
        return DenseLevelFields(tns, buf_s, nind, pos, op)

    def lower_declare(self, ctx, tns, init, op, shape):
        return self.lvl_t.lower_declare(ctx, tns, init, op, shape)

    def lower_freeze(self, ctx, tns, op):
        raise NotImplementedError("DenseLevelFType does not support lower_freeze.")

    def lower_increment(self, ctx, obj, val):
        raise NotImplementedError("DenseLevelFType does not support lower_increment.")

    def lower_thaw(self, ctx, tns, op):
        raise NotImplementedError("DenseLevelFType does not support lower_thaw.")

    def lower_unwrap(self, ctx, obj):
        raise NotImplementedError("DenseLevelFType does not support lower_unwrap.")

    def unfurl(self, ctx, tns, ext, mode, proto):
        def child_accessor(ctx, idx):
            pos_2 = asm.Variable(
                ctx.freshen(ctx.idx, f"_pos_{self.ndim - 1}"), self.position_type
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
                                    asm.GetAttr(tns.obj.lvl, asm.Literal("stride")),
                                    asm.Variable(ctx.idx.name, ctx.idx.type_),
                                ],
                            ),
                        ],
                    ),
                )
            )
            return ntn.Stack(
                self.lvl_t.get_fields_class(
                    asm.GetAttr(tns.obj.lvl, asm.Literal("lvl")),
                    tns.obj.buf_s,
                    tns.obj.nind + 1,
                    pos_2,
                    tns.obj.op,
                ),
                self.lvl_t,
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
    A class representing dense level.
    """

    _format: DenseLevelFType
    lvl: Level
    dimension: np.intp
    pos: asm.AssemblyNode | None = None

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
        return self._format

    @property
    def buf(self):
        return self.lvl.buf

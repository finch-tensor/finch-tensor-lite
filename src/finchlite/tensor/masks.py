from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import Tensor, register_property
from ..compile import looplets as lplt
from ..compile.lower import FinchTensorFType
from ..finch_assembly.struct import TupleFType
from ..tensor import Level, LevelFType


@dataclass
class LoTriMask(Tensor):
    """
    Lower triangular mask wrapper tensor.
    """

    body: Tensor

    def __post_init__(self):
        # insert column level in appropriate place
        lvl = self.body.lvl
        for _ in range(self.body.ndim - 2):
            lvl = lvl.lvl
        low_tri_mask_column = LoTriMaskColumn(
            LoTriMaskColumnFType(lvl.lvl.ftype), lvl.lvl
        )
        lvl.lvl = low_tri_mask_column
        lvl._format._lvl_t = low_tri_mask_column.ftype

    @property
    def ftype(self):
        return LoTriMaskFType(self.body.ftype)

    @property
    def shape(self):
        return self.body.shape


@dataclass(unsafe_hash=True)
class LoTriMaskFType(FinchTensorFType, asm.AssemblyStructFType):
    body: FinchTensorFType

    @property
    def element_type(self):
        return self.body.element_type

    @property
    def fill_value(self):
        return self.body.fill_value

    @property
    def shape_type(self):
        return self.body.shape_type

    def __call__(self, *args, **kwargs):
        return LoTriMask(self.body(*args, **kwargs))

    def lower_freeze(self, ctx, tns, op):
        return self.body.lower_freeze(self, ctx, tns, op)

    def lower_thaw(self, ctx, tns, op):
        return self.body.lower_thaw(self, ctx, tns, op)

    def lower_unwrap(self, ctx, obj):
        return self.body.lower_unwrap(self, ctx, obj)

    def lower_increment(self, ctx, obj, val):
        return self.body.lower_increment(self, ctx, obj, val)

    def lower_declare(self, ctx, tns, init, op, shape):
        return self.body.lower_declare(self, ctx, tns, init, op, shape)

    def unfurl(self, ctx, tns, ext, mode, proto):
        return self.body.unfurl(ctx, tns, ext, mode, proto)

    def add_levels(self, idxs):
        return self.body.add_levels(idxs)

    def remove_levels(self, idxs):
        return self.body.remove_levels(idxs)

    def asm_unpack(self, ctx, var_n, val):
        val_body = asm.GetAttr(val, asm.Literal("body"))
        return self.body.asm_unpack(ctx, var_n, val_body)

    def asm_repack(self, ctx, lhs, obj):
        return self.body.asm_repack(ctx, lhs, obj)

    @property
    def struct_name(self):
        return "LoTriMaskFtype"

    @property
    def struct_fields(self):
        return [
            ("body", self.body),
            ("shape", TupleFType.from_tuple(self.shape_type)),
        ]


@dataclass(unsafe_hash=True)
class LoTriMaskColumnFType(LevelFType, asm.AssemblyStructFType):
    body: LevelFType
    column: ntn.Variable | None = None

    @property
    def element_type(self):
        return self.body.element_type

    @property
    def buffer_type(self):
        return self.body.buffer_type

    @property
    def position_type(self):
        return self.body.position_type

    @property
    def fill_value(self):
        return self.body.fill_value

    @property
    def shape_type(self):
        return self.body.shape_type

    @property
    def buffer_factory(self):
        return self.body.buffer_factory

    @property
    def ndim(self):
        return self.body.ndim

    @property
    def lvl_t(self):
        return self.body.lvl_t

    def lower_freeze(self, ctx, tns, op):
        return self.body.lower_freeze(self, ctx, tns, op)

    def lower_thaw(self, ctx, tns, op):
        return self.body.lower_thaw(self, ctx, tns, op)

    def lower_unwrap(self, ctx, obj):
        return self.body.lower_unwrap(self, ctx, obj)

    def lower_increment(self, ctx, obj, val):
        return self.body.lower_increment(self, ctx, obj, val)

    def lower_declare(self, ctx, tns, init, op, shape):
        return self.body.lower_declare(self, ctx, tns, init, op, shape)

    def unfurl(self, ctx, tns, ext, mode, proto):
        def child_accessor(ctx, idx):
            return self.body.unfurl(ctx, tns, ext, mode, proto)

        return lplt.Sequence(
            head=lambda ctx, idx: child_accessor(ctx, idx),
            split=lambda ctx, idx: self.column,
            tail=lambda ctx, idx: lplt.Run(
                lambda ctx: lplt.Leaf(
                    lambda ctx: ntn.Stack(asm.Literal(np.intp(0)), np.intp)
                )
            ),
        )

    def from_kwargs(self, **kwargs):
        return self.body.from_kwargs(**kwargs)

    def to_kwargs(self):
        return self.body.to_kwargs()

    def asm_unpack(self, ctx, var_n, val):
        val_body = asm.GetAttr(val, asm.Literal("body"))
        return self.body.asm_unpack(ctx, var_n, val_body)

    def asm_repack(self, ctx, lhs, obj):
        return self.body.asm_repack(ctx, lhs, obj)

    def get_fields_class(self, tns, buf_s, nind, pos, op):
        return self.body.get_fields_class(tns, buf_s, nind, pos, op)

    def __call__(self, *args, **kwargs):
        return LoTriMaskColumn(self, self.body(*args, **kwargs))

    @property
    def struct_name(self):
        return "LoTriMaskColumnFType"

    @property
    def struct_fields(self):
        return [("body", self.body)] + self.body.struct_fields


@dataclass
class LoTriMaskColumn(Level):
    """
    Lower triangular mask column level for storing column idx.
    """

    _format: LoTriMaskColumnFType = field(repr=False)
    lvl: Level

    @property
    def ftype(self) -> LoTriMaskColumnFType:
        return self._format

    @property
    def val(self) -> Any:
        return self.lvl.val

    @property
    def shape(self):
        return self.lvl.shape

    @property
    def stride(self):
        return self.lvl.stride

    @property
    def dimension(self):
        return self.lvl.dimension

    @property
    def body(self):
        return self.lvl


def tril(x: Tensor, /, *, k: int = 0) -> Tensor:
    if k != 0:
        raise Exception(f"Only k=0 is supported, but got: {k}")
    return LoTriMask(x)


register_property(LoTriMask, "asarray", "__attr__", lambda x: x)

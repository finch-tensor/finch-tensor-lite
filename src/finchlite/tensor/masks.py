from dataclasses import dataclass
from typing import Any

from finchlite.tensor.fiber_tensor import FiberTensor

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..compile import looplets as lplt
from ..interface import Scalar
from ..tensor import Level, LevelFType


@dataclass(unsafe_hash=True)
class LoTriMaskFType(LevelFType, asm.AssemblyStructFType):
    body: LevelFType

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

    def level_lower_freeze(self, ctx, tns, op, pos):
        return self.body.level_lower_freeze(self, ctx, tns, op, pos)

    def level_lower_thaw(self, ctx, tns, op, pos):
        return self.body.level_lower_thaw(self, ctx, tns, op, pos)

    def level_lower_unwrap(self, ctx, obj, pos):
        return self.body.level_lower_unwrap(self, ctx, obj, pos)

    def level_lower_increment(self, ctx, obj, val, pos):
        return self.body.level_lower_increment(self, ctx, obj, val, pos)

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        return self.body.level_lower_declare(self, ctx, tns, init, op, shape, pos)

    def level_unfurl(self, ctx, tns, ext, mode, proto, pos):
        def child_accessor(ctx, idx):
            return self.body.level_unfurl(ctx, tns, ext, mode, proto, pos)

        scalar = Scalar(self.fill_value, self.fill_value)
        return lplt.Sequence(
            head=lambda ctx, idx: child_accessor(ctx, idx),
            split=lambda ctx, idx, visited_idx: ctx.ctx(visited_idx[-1]),
            tail=lambda ctx, idx: lplt.Run(
                lambda ctx: lplt.Leaf(
                    lambda ctx: ntn.Stack(asm.Literal(scalar), scalar.ftype)
                )
            ),
        )

    def level_asm_unpack(self, ctx, var_n, val):
        val_body = asm.GetAttr(val, asm.Literal("body"))
        return self.body.level_asm_unpack(ctx, var_n, val_body)

    def level_lower_dim(self, ctx, obj, r):
        return self.body.level_lower_dim(ctx, obj, r)

    def next_level(self):
        return self.lvl_t

    def asm_repack(self, ctx, lhs, obj):
        return self.body.asm_repack(ctx, lhs, obj)

    def get_fields_class(self, tns, buf_s, pos, op, dirty_bit):
        return self.body.get_fields_class(tns, buf_s, pos, op, dirty_bit)

    def from_fields(self, lvl) -> "LoTriMask":
        return LoTriMask(lvl)

    def __str__(self):
        return f"LoTriMaskFType({self.body})"

    @property
    def struct_name(self):
        return "LoTriMaskFType"

    @property
    def struct_fields(self):
        return [("body", self.body)] + self.body.struct_fields


@dataclass
class LoTriMask(Level):
    """
    Lower triangular mask level.
    """

    lvl: Level

    @property
    def ftype(self) -> LoTriMaskFType:
        return LoTriMaskFType(self.lvl.ftype)  # type: ignore[abstract]

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


def tril(x, /, *, k: int = 0):
    # TODO: make a copy of x
    if k != 0:
        raise Exception(f"Only k=0 is supported, but got: {k}")
    lvl = x.lvl
    root_lvl = lvl
    for _ in range(x.ndim - 2):
        lvl = lvl.lvl
    low_tri_mask = LoTriMask(lvl.lvl)
    lvl.lvl = low_tri_mask
    return FiberTensor(root_lvl)

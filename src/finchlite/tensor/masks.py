from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from finchlite import finch_assembly as asm
from finchlite import finch_notation as ntn
from finchlite.algebra import ImmutableStructFType, ffuncs
from finchlite.compile import looplets as lplt

from .fiber_tensor import FiberTensor, FiberTensorFType, Level, LevelFType
from .scalar import Scalar


@dataclass(unsafe_hash=True)
class LoTriMaskFType(LevelFType, ImmutableStructFType):
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

    def construct(self, shape):
        return LoTriMask(self.lvl_t.construct(shape=shape))

    def __call__(self, val: Any) -> "LoTriMask":
        """
        Convert a level to this lower triangular mask level type.

        Args:
            val: A value to convert to this type.
        Returns:
            A LoTriMask instance of this type.
        """
        raise NotImplementedError(
            f"Level conversion not yet implemented for {type(self).__name__}"
        )

    def from_numpy(self, shape, arr):
        return LoTriMask(self.lvl_t.from_numpy(shape, arr))

    def level_lower_freeze(self, ctx, tns, op, pos):
        return self.body.level_lower_freeze(
            ctx, asm.GetAttr(tns, asm.Literal("body")), op, pos
        )

    def level_lower_thaw(self, ctx, tns, op, pos):
        return self.body.level_lower_thaw(
            ctx, asm.GetAttr(tns, asm.Literal("body")), op, pos
        )

    def level_lower_unwrap(self, ctx, obj, pos):
        body = ntn.Fiber(
            obj.root,
            ntn.Child(obj.lvl, "body"),
            obj.pos,
            FiberTensorFType(self.body),
            obj.idxs,
        )
        return self.body.level_lower_unwrap(
            ctx,
            body,
            pos,
        )

    def level_lower_increment(self, ctx, obj, op, val, pos):
        body = ntn.Fiber(
            obj.root,
            ntn.Child(obj.lvl, "body"),
            obj.pos,
            FiberTensorFType(self.body),
            obj.idxs,
        )
        return self.body.level_lower_increment(
            ctx,
            body,
            op,
            val,
            pos,
        )

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        return self.body.level_lower_declare(
            ctx, asm.GetAttr(tns, asm.Literal("body")), init, op, shape, pos
        )

    def level_unfurl(self, ctx, fiber: ntn.Fiber, ext, mode, proto, pos):
        tns = fiber

        def child_accessor(ctx, idx):
            body_view = ntn.Fiber(
                tns.root,
                ntn.Child(tns.lvl, "body"),
                tns.pos,
                FiberTensorFType(self.body),
                tns.idxs,
            )
            return self.body.level_unfurl(
                ctx, body_view, ext, mode, proto, pos
            )

        scalar = Scalar(self.fill_value, self.fill_value)
        visited_idxs = tns.idxs
        return lplt.Sequence(
            head=lambda ctx, idx: child_accessor(ctx, idx),
            split=lambda ctx, ext: ntn.Call(
                ntn.L(ffuncs.add), (visited_idxs[-1], ext.get_unit())
            ),
            tail=lambda ctx, idx: lplt.Run(scalar),
        )

    def level_lower_dim(self, ctx, obj, r):
        return self.body.level_lower_dim(ctx, asm.GetAttr(obj, asm.Literal("body")), r)

    def from_fields(self, lvl) -> "LoTriMask":
        return LoTriMask(lvl)

    def __str__(self):
        return f"LoTriMaskFType({self.body})"

    @property
    def struct_name(self):
        return "LoTriMaskFType"

    @property
    def struct_fields(self):
        return [("body", self.body), ("stride", self.body.dimension_type)]


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
    x = deepcopy(x)
    if k != 0:
        raise Exception(f"Only k=0 is supported, but got: {k}")
    lvl = x.lvl
    root_lvl = lvl
    for _ in range(x.ndim - 2):
        lvl = lvl.lvl
    low_tri_mask = LoTriMask(lvl.lvl)
    lvl.lvl = low_tri_mask
    return FiberTensor(root_lvl)

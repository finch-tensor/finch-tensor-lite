from dataclasses import dataclass
from typing import Any

import numpy as np

from finchlite.algebra import FType, ImmutableStructFType, ftype, ftypes
from finchlite.tensor.fiber_tensor import Level, LevelFType

_LOWERING_ERROR = "SparseByteMapLevelFType lowering is not implemented."


@dataclass(unsafe_hash=True)
class SparseByteMapLevelFType(LevelFType, ImmutableStructFType):
    _lvl_t: LevelFType
    dimension_type: FType = ftypes.intp

    def __post_init__(self) -> None:
        if not isinstance(self._lvl_t, LevelFType):
            raise TypeError("SparseByteMapLevelFType lvl_t must be a level type")
        self.dimension_type = ftype(self.dimension_type)

    @property
    def struct_name(self) -> str:
        return "SparseByteMapLevelFType"

    @property
    def struct_fields(self) -> list[tuple[str, FType]]:
        return [
            ("lvl", self.lvl_t),
            ("dimension", self.dimension_type),
            ("ptr", self.ptr_type),
            ("tbl", self.tbl_type),
            ("srt", self.srt_type),
        ]

    def __str__(self) -> str:
        return f"SparseByteMapLevelFType({self.lvl_t})"

    @property
    def ndim(self) -> int:
        return 1 + self.lvl_t.ndim

    @property
    def fill_value(self) -> Any:
        return self.lvl_t.fill_value

    @property
    def element_type(self) -> FType:
        return self.lvl_t.element_type

    @property
    def shape_type(self) -> tuple[FType, ...]:
        return (self.dimension_type, *self.lvl_t.shape_type)

    @property
    def position_type(self) -> FType:
        return self.lvl_t.position_type

    @property
    def buffer_type(self) -> FType:
        return self.lvl_t.buffer_type

    @property
    def buffer_factory(self) -> Any:
        return self.lvl_t.buffer_factory

    @property
    def ptr_type(self) -> FType:
        return self.buffer_factory(self.position_type)

    @property
    def tbl_type(self) -> FType:
        return self.buffer_factory(ftypes.bool_)

    @property
    def srt_type(self) -> FType:
        return self.buffer_factory(self.position_type)

    @property
    def lvl_t(self) -> LevelFType:
        return self._lvl_t

    def construct(self, *, shape) -> "SparseByteMapLevel":
        lvl = self.lvl_t.construct(shape=shape[1:])
        return SparseByteMapLevel(lvl, self.dimension_type(shape[0]))

    def __call__(self, val: Any) -> "SparseByteMapLevel":
        raise NotImplementedError(
            f"Level conversion not yet implemented for {type(self).__name__}"
        )

    def from_numpy(self, shape, val):
        raise NotImplementedError("sparse bytemap level doesn't support from_numpy")

    def level_asm_unpack(self, ctx, var_n, val):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_asm_repack(self, ctx, lvl_fields):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_dim(self, ctx, obj, r):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_freeze(self, ctx, tns, op, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_thaw(self, ctx, tns, op, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_increment(self, ctx, obj, op, val, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def level_unfurl(self, ctx, tns, ext, mode, proto, pos):
        raise NotImplementedError(_LOWERING_ERROR)

    def from_fields(self, lvl, dimension, ptr, tbl, srt) -> "SparseByteMapLevel":
        return SparseByteMapLevel(lvl, dimension, ptr, tbl, srt)


def sparse_bytemap(lvl_t, dimension_type=None):
    return SparseByteMapLevelFType(lvl_t, dimension_type or ftypes.intp)


@dataclass
class SparseByteMapLevel(Level):
    lvl: Level
    dimension: np.integer
    ptr: Any | None = None
    tbl: Any | None = None
    srt: Any | None = None

    def __post_init__(self) -> None:
        if self.ptr is None:
            self.ptr = self.lvl.buffer_factory(self.lvl.position_type)(len=0)
        if self.tbl is None:
            self.tbl = self.lvl.buffer_factory(ftypes.bool_)(len=0)
        if self.srt is None:
            self.srt = self.lvl.buffer_factory(self.lvl.position_type)(len=0)

    @property
    def shape(self) -> tuple:
        return (self.dimension, *self.lvl.shape)

    @property
    def stride(self) -> np.integer:
        return np.intp(0)

    @property
    def ftype(self) -> SparseByteMapLevelFType:
        return SparseByteMapLevelFType(self.lvl.ftype, ftype(self.dimension))  # type: ignore[abstract]

    @property
    def val(self) -> Any:
        return self.lvl.val

    def __str__(self) -> str:
        return f"SparseByteMapLevel(lvl={self.lvl}, dim={self.dimension})"

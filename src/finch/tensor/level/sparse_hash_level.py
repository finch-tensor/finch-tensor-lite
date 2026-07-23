from dataclasses import dataclass
from typing import Any

import numpy as np

from finch.algebra import FType, ImmutableStructFType, TupleFType, ftype, ftypes
from finch.tensor.fiber_tensor import Level, LevelFType

_LOWERING_ERROR = "SparseHashLevelFType lowering is not implemented."


@dataclass(unsafe_hash=True)
class SparseHashLevelFType(LevelFType, ImmutableStructFType):
    _lvl_t: LevelFType
    dimension_type: FType = ftypes.intp
    single_writer: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self._lvl_t, LevelFType):
            raise TypeError("SparseHashLevelFType lvl_t must be a level type")
        self.dimension_type = ftype(self.dimension_type)

    @property
    def struct_name(self) -> str:
        return "SparseHashLevelFType"

    @property
    def struct_fields(self) -> list[tuple[str, FType]]:
        return [
            ("lvl", self.lvl_t),
            ("dimension", self.dimension_type),
            ("subtables", ftypes.intp),
            ("ptr", self.ptr_type),
            ("tbl_ctrl", self.tbl_ctrl_type),
            ("tbl", self.tbl_type),
            ("pool", self.pool_type),
            ("perm", self.perm_type),
        ]

    def __str__(self) -> str:
        return f"SparseHashLevelFType({self.lvl_t})"

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
    def tbl_ctrl_type(self) -> FType:
        return self.buffer_factory(ftypes.uint8)

    @property
    def tbl_entry_type(self) -> TupleFType:
        return TupleFType.from_tuple(
            (self.position_type, self.dimension_type, self.position_type)
        )

    @property
    def tbl_type(self) -> FType:
        return self.buffer_factory(self.tbl_entry_type)

    @property
    def pool_type(self) -> FType:
        return self.buffer_factory(self.position_type)

    @property
    def perm_type(self) -> FType:
        return self.buffer_factory(self.position_type)

    @property
    def lvl_t(self) -> LevelFType:
        return self._lvl_t

    def level_format_properties(self, n):
        return self.lvl_t.level_format_properties(n + 1)

    def construct(self, shape: tuple[Any, ...], *, pos: int) -> "SparseHashLevel":
        lvl = self.lvl_t.construct(shape=shape[1:], pos=0)
        return SparseHashLevel(
            lvl,
            self.dimension_type(shape[0]),
            self.ptr_type(int(pos) + 1),
            self.tbl_ctrl_type(0),
            self.tbl_type(0),
            self.pool_type(0),
            self.perm_type(0),
            single_writer=self.single_writer,
        )

    def __call__(self, val: Any) -> "SparseHashLevel":
        raise NotImplementedError(
            f"Level conversion not yet implemented for {type(self).__name__}"
        )

    def from_numpy(self, shape, val):
        raise NotImplementedError("sparse hash level doesn't support from_numpy")

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

    def from_fields(
        self, lvl, dimension, subtables, ptr, tbl_ctrl, tbl, pool, perm
    ) -> "SparseHashLevel":
        return SparseHashLevel(
            lvl,
            dimension,
            ptr,
            tbl_ctrl,
            tbl,
            pool,
            perm,
            subtables=subtables,
            single_writer=self.single_writer,
        )


def sparse_hash(lvl_t, dimension_type=None, *, single_writer: bool = True):
    return SparseHashLevelFType(
        lvl_t, dimension_type or ftypes.intp, single_writer=single_writer
    )


@dataclass(init=False)
class SparseHashLevel(Level):
    lvl: Level
    dimension: np.integer
    ptr: Any | None
    tbl_ctrl: Any | None
    tbl: Any | None
    pool: Any | None
    perm: Any | None
    subtables: int
    single_writer: bool

    def __init__(
        self,
        lvl: Level,
        dimension: np.integer,
        ptr: Any | None = None,
        tbl_ctrl: Any | None = None,
        tbl: Any | None = None,
        pool: Any | None = None,
        perm: Any | None = None,
        *,
        subtables: int = 1,
        single_writer: bool = True,
    ) -> None:
        self.lvl = lvl
        self.dimension = dimension
        self.ptr = ptr
        self.tbl_ctrl = tbl_ctrl
        self.tbl = tbl
        self.pool = pool
        self.perm = perm
        self.subtables = int(subtables)
        self.single_writer = bool(single_writer)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.subtables < 1 or self.subtables & (self.subtables - 1) != 0:
            raise ValueError(
                "SparseHashLevel subtables must be a positive power of two"
            )
        if self.ptr is None:
            self.ptr = self.lvl.buffer_factory(self.lvl.position_type)(1)
        if self.tbl_ctrl is None:
            self.tbl_ctrl = self.lvl.buffer_factory(ftypes.uint8)(0)
        if self.tbl is None:
            tbl_entry_type = TupleFType.from_tuple(
                (self.lvl.position_type, ftype(self.dimension), self.lvl.position_type)
            )
            self.tbl = self.lvl.buffer_factory(tbl_entry_type)(0)
        if self.pool is None:
            self.pool = self.lvl.buffer_factory(self.lvl.position_type)(0)
        if self.perm is None:
            self.perm = self.lvl.buffer_factory(self.lvl.position_type)(0)

    @property
    def shape(self) -> tuple:
        return (self.dimension, *self.lvl.shape)

    @property
    def stride(self) -> np.integer:
        return np.intp(0)

    @property
    def ftype(self) -> SparseHashLevelFType:
        return SparseHashLevelFType(
            self.lvl.ftype,
            ftype(self.dimension),
            single_writer=self.single_writer,
        )  # type: ignore[abstract]

    @property
    def val(self) -> Any:
        return self.lvl.val

    def __str__(self) -> str:
        return f"SparseHashLevel(lvl={self.lvl}, dim={self.dimension})"

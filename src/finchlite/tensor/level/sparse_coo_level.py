from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from finchlite.algebra import FType, ImmutableStructFType, TupleFType, ftype, ftypes
from finchlite.tensor.fiber_tensor import Level, LevelFType


class SparseCOOLevelFType(ImmutableStructFType, LevelFType):
    def __init__(
        self,
        lvl_type: LevelFType,
        coo_shape_type: FType,
        ptr_type: FType,
        idx_type: FType,
        tbl_type: FType,
    ) -> None:
        if not isinstance(lvl_type, LevelFType):
            raise TypeError("SparseCOOLevelFType lvl_type must be a level type")
        if not isinstance(coo_shape_type, TupleFType):
            raise TypeError("SparseCOOLevelFType coo_shape_type must be a tuple type")
        if not isinstance(ptr_type, FType):
            raise TypeError("SparseCOOLevelFType ptr_type must be an ftype")
        if not isinstance(idx_type, FType):
            raise TypeError("SparseCOOLevelFType idx_type must be an ftype")
        if not isinstance(tbl_type, TupleFType):
            raise TypeError("SparseCOOLevelFType tbl_type must be a tuple type")
        self.lvl_type = lvl_type
        self.coo_shape_type = coo_shape_type
        self.ptr_type = ptr_type
        self.idx_type = idx_type
        self.tbl_type = tbl_type
        if self.coo_ndim < 1:
            raise ValueError("SparseCOOLevelFType requires at least one COO dimension")
        if self.coo_ndim != len(self.tbl_type.struct_fieldtypes):
            raise ValueError("SparseCOOLevelFType tbl arity must match COO dimensions")

    @property
    def struct_name(self) -> str:
        return "SparseCOOLevelFType"

    @property
    def struct_fields(self) -> list[tuple[str, FType]]:
        # `idx` is a lowerable alias for the coordinate table.
        return [
            ("lvl", self.lvl_t),
            ("coo_shape", self.coo_shape_type),
            ("ptr", self.ptr_type),
            ("idx", self.idx_type),
            ("tbl", self.tbl_type),
        ]

    def __str__(self) -> str:
        return f"SparseCOOLevelFType({self.coo_ndim}, {self.lvl_t})"

    @property
    def ndim(self) -> int:
        return self.coo_ndim + self.lvl_t.ndim

    @property
    def coo_ndim(self) -> int:
        return len(self.coo_shape_tuple_type.struct_fieldtypes)

    @property
    def fill_value(self) -> Any:
        return self.lvl_t.fill_value

    @property
    def element_type(self) -> FType:
        return self.lvl_t.element_type

    @property
    def position_type(self) -> FType:
        return self.lvl_t.position_type

    @property
    def shape_type(self) -> tuple[FType, ...]:
        return (
            *self.lvl_t.shape_type,
            *self.coo_shape_tuple_type.struct_fieldtypes,
        )

    @property
    def idx_buffer_type(self) -> FType:
        return self.tbl_tuple_type.struct_fieldtypes[0]

    @property
    def coo_shape_tuple_type(self) -> TupleFType:
        return cast(TupleFType, self.coo_shape_type)

    @property
    def tbl_tuple_type(self) -> TupleFType:
        return cast(TupleFType, self.tbl_type)

    @property
    def buffer_type(self) -> FType:
        return self.lvl_t.buffer_type

    @property
    def buffer_factory(self) -> Any:
        return self.lvl_t.buffer_factory

    @property
    def lvl_t(self) -> LevelFType:
        return self.lvl_type

    def level_format_properties(self, n):
        return []

    def construct(self, shape: tuple[Any, ...], **kwargs) -> "SparseCOOLevel":
        if kwargs:
            raise TypeError("SparseCOOLevelFType.construct does not accept kwargs")
        if len(shape) < self.coo_ndim:
            raise ValueError(
                "SparseCOOLevelFType shape must include all COO dimensions"
            )
        lvl_shape = shape[: -self.coo_ndim]
        coo_shape = shape[-self.coo_ndim :]
        lvl = self.lvl_t.construct(shape=lvl_shape)
        return SparseCOOLevel(lvl, self.coo_shape_type(coo_shape))

    def from_numpy(self, shape, val):
        raise NotImplementedError("SparseCOOLevelFType has no from_numpy yet.")

    def __call__(
        self, lvl, shape=None, ptr=None, idx=None, tbl=None
    ) -> "SparseCOOLevel":
        if shape is None:
            raise ValueError("SparseCOOLevel requires a COO shape")
        if tbl is None:
            tbl = idx
        return SparseCOOLevel(lvl, shape, ptr, tbl)

    def from_fields(self, lvl, coo_shape, ptr, idx, tbl=None) -> "SparseCOOLevel":
        if tbl is None:
            tbl = idx
        return SparseCOOLevel(lvl, coo_shape, ptr, tbl)

    def level_asm_unpack(self, ctx, var_n, val):
        raise NotImplementedError("SparseCOOLevelFType lowering is not implemented.")

    def level_asm_repack(self, ctx, lvl_fields):
        raise NotImplementedError("SparseCOOLevelFType lowering is not implemented.")

    def level_lower_dim(self, ctx, obj, r):
        raise NotImplementedError("SparseCOOLevelFType lowering is not implemented.")

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        raise NotImplementedError("SparseCOOLevelFType lowering is not implemented.")

    def level_lower_freeze(self, ctx, tns, op, pos):
        raise NotImplementedError("SparseCOOLevelFType lowering is not implemented.")

    def level_lower_thaw(self, ctx, tns, op, pos):
        raise NotImplementedError("SparseCOOLevelFType lowering is not implemented.")

    def level_lower_increment(self, ctx, obj, op, val, pos):
        raise NotImplementedError("SparseCOOLevelFType lowering is not implemented.")

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError("SparseCOOLevelFType lowering is not implemented.")

    def level_unfurl(self, ctx, tns, ext, mode, proto, pos):
        raise NotImplementedError("SparseCOOLevelFType lowering is not implemented.")


def sparse_coo(lvl_t, coo_ndim: int, dimension_type=None):
    if coo_ndim < 1:
        raise ValueError("SparseCOOLevelFType requires at least one COO dimension")
    dimension_type = ftype(dimension_type or ftypes.intp)
    coo_shape_type = TupleFType.from_tuple(
        tuple(dimension_type for _ in range(coo_ndim))
    )
    ptr_type = lvl_t.buffer_factory(lvl_t.position_type)
    idx_buffer_type = lvl_t.buffer_factory(dimension_type)
    tbl_type = TupleFType.from_tuple(tuple(idx_buffer_type for _ in range(coo_ndim)))
    return SparseCOOLevelFType(lvl_t, coo_shape_type, ptr_type, tbl_type, tbl_type)


@dataclass(init=False)
class SparseCOOLevel(Level):
    lvl: Level
    coo_shape: tuple[Any, ...]
    ptr: Any
    tbl: tuple[Any, ...]

    def __init__(self, lvl, shape, ptr=None, tbl=None):
        self.lvl = lvl
        self.coo_shape = tuple(shape)
        self.ptr = ptr
        self.tbl = tbl
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.coo_shape:
            raise ValueError("SparseCOOLevel requires at least one COO dimension")
        dimension_type = ftype(self.coo_shape[0])
        if self.ptr is None:
            self.ptr = self.lvl.buffer_factory(self.lvl.position_type)(len=0)
        if self.tbl is None:
            self.tbl = tuple(
                self.lvl.buffer_factory(dimension_type)(len=0) for _ in self.coo_shape
            )
        elif not isinstance(self.tbl, tuple):
            raise TypeError("SparseCOOLevel tbl must be a tuple")
        if len(self.tbl) != len(self.coo_shape):
            raise ValueError("SparseCOOLevel tbl length must match COO dimensions")

    @property
    def idx(self) -> tuple[Any, ...]:
        return self.tbl

    @idx.setter
    def idx(self, value) -> None:
        if not isinstance(value, tuple):
            raise TypeError("SparseCOOLevel idx must be a tuple")
        if len(value) != len(self.coo_shape):
            raise ValueError("SparseCOOLevel idx length must match COO dimensions")
        self.tbl = value

    @property
    def shape(self) -> tuple[Any, ...]:
        return (*self.lvl.shape, *self.coo_shape)

    @property
    def stride(self) -> np.intp:
        return np.intp(0)

    @property
    def ftype(self) -> SparseCOOLevelFType:
        return SparseCOOLevelFType(
            self.lvl.ftype,
            ftype(self.coo_shape),
            ftype(self.ptr),
            ftype(self.idx),
            ftype(self.tbl),
        )

    @property
    def val(self):
        return self.lvl.val

    def __str__(self) -> str:
        return f"SparseCOOLevel(lvl={self.lvl}, coo_shape={self.coo_shape})"

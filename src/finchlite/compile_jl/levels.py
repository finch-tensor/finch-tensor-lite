from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import dtypes
from .julia import jl
from .typing import JuliaObj, number


# Abstract Formats
class LevelFormat:
    @property
    @abstractmethod
    def shape_type(self) -> tuple: ...


class NestedLevelFormat(LevelFormat):
    @property
    def ndim(self) -> np.intp:
        return self.lvl.ndim + np.intp(1)

    @property
    def fill_value(self) -> Any:
        return self.element_type(self.lvl.fill_value)

    @property
    def element_type(self) -> Any:
        return self.lvl.element_type

    def __eq__(self, other):
        return type(other) is type(self) and self.lvl == other.lvl

    def __hash__(self):
        return hash((self.__class__.__name__, self.lvl.__hash__))

    @abstractmethod
    def create_jl_obj(self) -> JuliaObj: ...


class ElementFormat(LevelFormat):
    """Element level storage format for scalar tensor leaves.

    A subfiber of an element level is a scalar, initialized to a fill value.
    The element level is a leaf level used at the end of the tensor tree structure.
    """

    def __init__(self, fill_value: number, element_type: Any | None = None):
        self._fill_value = fill_value
        self._element_type = dtypes.to_fl_dtype(
            type(fill_value) if element_type is None else element_type
        )

    @property
    def ndim(self) -> np.intp:
        return np.intp(0)

    @property
    def fill_value(self) -> Any:
        return self.element_type(self._fill_value)

    @property
    def element_type(self) -> Any:
        return self._element_type

    def __eq__(self, other):
        return (
            isinstance(other, ElementFormat)
            and self._fill_value == other.fill_value
            and self._element_type == other.element_type
        )

    def __hash__(self):
        return hash((self.__class__.__name__, self._fill_value, self._element_type))

    def create_jl_obj(self) -> JuliaObj:
        # Cast through element_type so the real Julia element type matches
        # the requested dtype, rather than whatever Julia infers from
        # self._fill_value's own Python/numpy type.
        val = self.fill_value
        # PythonCall wraps numpy.bool_ as a 0-d PyArray (non-isbits), which
        # Finch's ElementLevel rejects -- unwrap to a native Python bool.
        # For all other numpy scalar types, pass them through directly so
        # Julia preserves the right type (e.g. np.uint8(0) → UInt8, not Int64).
        if isinstance(val, np.bool_):
            val = val.item()
        return jl.Element(val)

    @property
    def shape_type(self) -> tuple:
        return ()


@dataclass(frozen=True)
class DenseFormat(NestedLevelFormat):
    """Dense format wrapper type for Finch tensors.

    A dense format stores every slice of a tensor. A subfiber of a dense level
    is an array which stores every slice A[:, ..., :, i] as a distinct subfiber.
    Dense levels support both row-major and column-major access.
    """

    lvl: NestedLevelFormat
    dim_type: Any = dtypes.int_

    def create_jl_obj(self) -> JuliaObj:
        return jl.Dense(self.lvl.create_jl_obj())

    @property
    def shape_type(self) -> tuple:
        return self.lvl.shape_type + (self.dim_type,)


@dataclass(frozen=True)
class SparseListFormat(NestedLevelFormat):
    """Sparse list format wrapper type for Finch tensors.

    A sparse list format stores only potentially non-fill slices using a sorted list.
    Slices that are entirely fill_value are omitted. This format is efficient for
    tensors with sparse patterns and supports column-major reads and bulk updates.
    """

    lvl: NestedLevelFormat
    dim_type: Any = dtypes.int_

    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseList(self.lvl.create_jl_obj())

    @property
    def shape_type(self) -> tuple:
        return self.lvl.shape_type + (self.dim_type,)


@dataclass(frozen=True)
class SparseCOOFormat(NestedLevelFormat):
    """Coordinate (COO) format wrapper type for Finch tensors.

    A coordinate format stores sparse tensors as lists of coordinates.
    It uses N separate arrays to record which coordinates are stored,
    with coordinates sorted in column-major order. This is a legacy format
    maintained for backward compatibility.
    """

    lvl: NestedLevelFormat
    N: int = 2
    dim_type: tuple | None = dtypes.int_

    def create_jl_obj(self) -> JuliaObj:
        coo_type = jl.seval(f"Finch.SparseCOO{{{self.N}}}")
        return coo_type(self.lvl.create_jl_obj())

    @property
    def ndim(self) -> np.intp:
        return self.lvl.ndim + np.intp(self.N)  # FIXME: not sure about this fix.

    @property
    def shape_type(self) -> tuple:
        if self.dim_type is None:
            return self.lvl.shape_type + (dtypes.int_,) * self.N
        return self.lvl.shape_type + self.dim_type


@dataclass(frozen=True)
class SparseByteMapFormat(NestedLevelFormat):
    """Sparse byte map format wrapper type for Finch tensors.

    A sparse byte map format uses a dense bitmap to encode which slices
    are stored, similar to SparseList but supporting random access.
    Only potentially non-fill slices are stored as subfibers.
    """

    lvl: NestedLevelFormat
    dim_type: Any = dtypes.int_

    def create_jl_obj(self) -> JuliaObj:
        return jl.SparseByteMap(self.lvl.create_jl_obj())

    @property
    def shape_type(self) -> tuple:
        return self.lvl.shape_type + (self.dim_type,)


# Helper Methods
def jlobj_to_format(obj: JuliaObj) -> LevelFormat:
    """Construct a level hierarchy from a Julia Finch tensor object.

    Recursively constructs a Python representation of the tensor's level structure
    by inspecting the Julia object's levels.

    Parameters
    ----------
    obj : JuliaObj
        A Julia Finch tensor object whose levels will be inspected.
    fill_value : float or int
        The fill value used for the tensor's sparse representation.

    Returns
    -------
    LevelFormat
        A Python representation of the level hierarchy.

    Raises
    ------
    Exception
        If an unsupported level type is encountered.
    """
    if jl.isa(obj, jl.Finch.Element):
        obj_type = jl.typeof(obj)
        return ElementFormat(
            jl.Finch.level_fill_value(obj_type),
            dtypes.to_fl_dtype(jl.Finch.level_eltype(obj_type)),
        )
    if jl.isa(obj, jl.Finch.Dense):
        return DenseFormat(
            jlobj_to_format(obj.lvl), dtypes.to_fl_dtype(type(obj.shape))
        )
    if jl.isa(obj, jl.Finch.SparseList):
        return SparseListFormat(
            jlobj_to_format(obj.lvl), dtypes.to_fl_dtype(type(obj.shape))
        )
    if jl.isa(obj, jl.Finch.SparseCOO):
        N = jl.seval("Finch.level_ndims")(jl.typeof(obj))
        return SparseCOOFormat(jlobj_to_format(obj.lvl), N, None)
    if jl.isa(obj, jl.Finch.SparseByteMap):
        return SparseByteMapFormat(
            jlobj_to_format(obj.lvl), dtypes.to_fl_dtype(type(obj.shape))
        )
    raise Exception("Unhandled exception!")

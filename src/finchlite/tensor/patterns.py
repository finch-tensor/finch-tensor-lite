from __future__ import annotations

import builtins
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from finchlite.algebra import FType, TensorFType, ffuncs, ftype

from .override_tensor import OverrideTensor
from .scalar import Scalar


def _shape_size(shape: tuple) -> int:
    return int(np.prod(shape, dtype=np.intp)) if shape else 1


@dataclass(frozen=True, eq=False)
class IndexTensorFType(TensorFType):
    _element_type: FType
    _shape_type: tuple[FType, ...]

    def __init__(
        self,
        _element_type: FType | type = np.intp,
        _shape_type: tuple[FType | type, ...] = (),
    ):
        object.__setattr__(self, "_element_type", ftype(_element_type))
        object.__setattr__(
            self, "_shape_type", tuple(ftype(dim_t) for dim_t in _shape_type)
        )

    @property
    def fill_value(self):
        return self._element_type(0)

    @property
    def element_type(self) -> FType:
        return self._element_type

    @property
    def shape_type(self):
        return self._shape_type

    def from_numpy(self, arr):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, IndexTensorFType):
            return False
        return (
            builtins.bool(np.all(ffuncs.same(self.fill_value, other.fill_value)))
            and self._element_type == other._element_type
            and self._shape_type == other._shape_type
        )

    def __hash__(self):
        return hash(
            (ffuncs.samehash(self.fill_value), self._element_type, self._shape_type)
        )

    def construct(self, shape: tuple) -> IndexTensor:
        return IndexTensor(shape, self.element_type)

    def __call__(self, val: Any) -> IndexTensor:
        """
        Convert a tensor to this index tensor type.

        Args:
            val: A tensor to convert to this type.
        Returns:
            An IndexTensor instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )


@dataclass(frozen=True, eq=False)
class FillTensorFType(TensorFType):
    _fill_value: Any
    _element_type: FType
    _shape_type: tuple

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def element_type(self) -> FType:
        return self._element_type

    @property
    def shape_type(self):
        return self._shape_type

    def from_numpy(self, arr):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, FillTensorFType):
            return False
        return (
            builtins.bool(np.all(ffuncs.same(self._fill_value, other._fill_value)))
            and self._element_type == other._element_type
            and self._shape_type == other._shape_type
        )

    def __hash__(self):
        return hash(
            (ffuncs.samehash(self._fill_value), self._element_type, self._shape_type)
        )

    def construct(self, shape: tuple) -> FillTensor:
        return FillTensor(shape, self.fill_value)

    def __call__(self, val: Any) -> FillTensor:
        """
        Convert a tensor to this fill tensor type.

        Args:
            val: A tensor to convert to this type.
        Returns:
            A FillTensor instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )


class FillTensor(OverrideTensor):
    """
    A tensor that has a specific shape but contains no actual data.
    Used primarily for broadcasting operations where a tensor of a specific
    shape is needed but the values are irrelevant.
    """

    def __init__(self, shape, fill_value):
        self._shape = shape
        self._fill_value = fill_value

    def __getitem__(self, idxs):
        return Scalar(self._fill_value, fill_value=self._fill_value)

    def item(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to Python scalar.")
        return self._fill_value

    def to_numpy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_numpy.")

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_scipy.")

    @property
    def shape(self):
        return self._shape

    @property
    def fill_value(self) -> Any:
        """Default fill value."""
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        """Data type of the tensor's elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple[FType, ...]:
        """Shape type of the tensor."""
        return self.ftype.shape_type

    @property
    def ftype(self):
        return FillTensorFType(
            self._fill_value,
            ftype(self._fill_value),
            tuple(ftype(dim) for dim in self.shape),
        )


class IndexTensor(OverrideTensor):
    """
    A tensor that has a specific shape and returns the linear (flattened) index used
    to access it.
    """

    def __init__(self, shape, element_type: FType | type = np.intp):
        self._shape = tuple(shape)
        self._element_type = ftype(element_type)

    def __getitem__(self, idxs):
        if self.ndim == 0 and idxs in ((), Ellipsis, (...,)):
            return Scalar(self.fill_value, fill_value=self.fill_value)
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(idx for idx in idxs if idx is not Ellipsis)
        if len(idxs) != self.ndim:
            raise IndexError("Incorrect number of indices for IndexTensor.")
        flat_index = 0
        for idx, dim in zip(idxs, self.shape, strict=True):
            flat_index = flat_index * dim + idx
        flat_index = self._element_type(flat_index)
        return Scalar(flat_index, fill_value=self.fill_value)

    def item(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to Python scalar.")
        return self.fill_value

    def to_numpy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_numpy.")

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_scipy.")

    @property
    def shape(self):
        return self._shape

    @property
    def fill_value(self) -> Any:
        """Default fill value."""
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        """Data type of the tensor's elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple[FType, ...]:
        """Shape type of the tensor."""
        return self.ftype.shape_type

    @property
    def ftype(self):
        return IndexTensorFType(
            self._element_type,
            tuple(ftype(dim) for dim in self.shape),
        )


@dataclass(frozen=True, eq=False)
class PatternTensorFType(TensorFType):
    _fill_value: Any
    _pattern_value: Any
    _element_type: FType
    _shape_type: tuple
    _tensor_type: type[PatternTensor]
    _constructor_kwargs: tuple[tuple[str, Any], ...]

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def element_type(self) -> FType:
        return self._element_type

    @property
    def shape_type(self):
        return self._shape_type

    def from_numpy(self, arr):
        raise NotImplementedError

    def __eq__(self, other):
        match other:
            case PatternTensorFType(
                _fill_value=fill_value,
                _pattern_value=pattern_value,
                _element_type=element_type,
                _shape_type=shape_type,
                _tensor_type=tensor_type,
                _constructor_kwargs=constructor_kwargs,
            ):
                return (
                    builtins.bool(np.all(ffuncs.same(self._fill_value, fill_value)))
                    and builtins.bool(
                        np.all(ffuncs.same(self._pattern_value, pattern_value))
                    )
                    and self._element_type == element_type
                    and self._shape_type == shape_type
                    and self._tensor_type == tensor_type
                    and self._constructor_kwargs == constructor_kwargs
                )
            case _:
                return False

    def __hash__(self):
        return hash(
            (
                ffuncs.samehash(self._fill_value),
                ffuncs.samehash(self._pattern_value),
                self._element_type,
                self._shape_type,
                self._tensor_type,
                self._constructor_kwargs,
            )
        )

    def construct(self, shape: tuple) -> PatternTensor:
        return self._tensor_type(
            shape,
            dtype=self.element_type,
            **dict(self._constructor_kwargs),
        )

    def __call__(self, val: Any) -> PatternTensor:
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )


class PatternTensor(OverrideTensor):
    def __init__(
        self,
        shape,
        *,
        ndim: int = 2,
        dtype=None,
        default_dtype=np.float64,
        fill_value=0,
        pattern_value=1,
        **constructor_kwargs,
    ):
        if isinstance(shape, int):
            shape = (shape,) if ndim == 1 else (shape, shape)
        self._shape = tuple(shape)
        if len(self._shape) != ndim:
            raise ValueError(f"Expected a {ndim}D shape, got {self._shape}")
        self._element_type = ftype(dtype if dtype is not None else default_dtype)
        self._fill_value = self._element_type(fill_value)
        self._pattern_value = self._element_type(pattern_value)
        self._constructor_kwargs = tuple(constructor_kwargs.items())

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        if len(idxs) != self.ndim:
            raise ValueError(f"{type(self).__name__} requires one index per dimension.")
        val = self._pattern_value if self.contains(*idxs) else self._fill_value
        return Scalar(val, fill_value=self._fill_value)

    def contains(self, *idxs) -> bool:
        raise NotImplementedError

    def item(self):
        raise ValueError("Cannot convert non-scalar tensor to Python scalar.")

    def to_numpy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_numpy.")

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_scipy.")

    @property
    def shape(self):
        return self._shape

    @property
    def fill_value(self) -> Any:
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple[FType, ...]:
        return self.ftype.shape_type

    @property
    def ftype(self):
        return PatternTensorFType(
            self._fill_value,
            self._pattern_value,
            self._element_type,
            tuple(ftype(dim) for dim in self.shape),
            type(self),
            self._constructor_kwargs,
        )


class ReshapeMaskTensor(PatternTensor):
    def __init__(self, shape, new_shape=None, *, old_shape=None, dtype=None):
        if old_shape is None:
            if new_shape is None:
                raise TypeError("new_shape is required")
            old_shape = shape
            shape = (*tuple(old_shape), *tuple(new_shape))
        elif new_shape is None:
            raise TypeError("new_shape is required")

        self._old_shape = tuple(old_shape)
        self._new_shape = tuple(new_shape)
        if _shape_size(self._old_shape) != _shape_size(self._new_shape):
            raise ValueError(
                f"Cannot reshape array of size {_shape_size(self._old_shape)} "
                f"into shape {self._new_shape}"
            )
        self._old_ndim = len(self._old_shape)
        super().__init__(
            shape,
            ndim=self._old_ndim + len(self._new_shape),
            dtype=dtype,
            default_dtype=np.bool_,
            fill_value=False,
            pattern_value=True,
            old_shape=self._old_shape,
            new_shape=self._new_shape,
        )

    @staticmethod
    def _flat_index(idxs, shape) -> int:
        flat_index = 0
        for idx, dim in zip(idxs, shape, strict=True):
            flat_index = flat_index * dim + idx
        return flat_index

    def contains(self, *idxs) -> bool:
        old_idxs = idxs[: self._old_ndim]
        new_idxs = idxs[self._old_ndim :]
        return self._flat_index(old_idxs, self._old_shape) == self._flat_index(
            new_idxs,
            self._new_shape,
        )


class EyeTensor(PatternTensor):
    def __init__(self, shape, *, k: int = 0, dtype=None):
        self._k = k
        super().__init__(shape, dtype=dtype, k=self._k)

    def contains(self, i, j) -> bool:
        return j - i == self._k


class UpperTriangleTensor(PatternTensor):
    def __init__(self, shape, *, k: int = 0, dtype=None):
        self._k = k
        super().__init__(shape, dtype=dtype, k=self._k)

    def contains(self, i, j) -> bool:
        return j - i >= self._k


class LowerTriangleTensor(PatternTensor):
    def __init__(self, shape, *, k: int = 0, dtype=None):
        self._k = k
        super().__init__(shape, dtype=dtype, k=self._k)

    def contains(self, i, j) -> bool:
        return j - i <= self._k


class PairSumTensor(PatternTensor):
    def __init__(self, shape, *, dtype=None):
        super().__init__(shape, dtype=dtype)

    def contains(self, i, j) -> bool:
        return j == i * 2 or j == i * 2 + 1


class PairCarryTensor(PatternTensor):
    def __init__(self, shape, *, dtype=None):
        super().__init__(shape, dtype=dtype)

    def contains(self, i, j) -> bool:
        return i > 0 and j == (i - 1) // 2


class ReverseTensor(PatternTensor):
    def __init__(self, shape, *, dtype=None):
        super().__init__(shape, dtype=dtype)

    def contains(self, i, j) -> bool:
        return j == self.shape[1] - i - 1


class RollTensor(PatternTensor):
    def __init__(self, shape, *, k: int = 0, dtype=None):
        self._k = k
        super().__init__(shape, dtype=dtype, k=self._k)

    def contains(self, i, j) -> bool:
        axis_size = self.shape[1]
        return axis_size > 0 and j == (i - self._k) % axis_size


class RepeatTensor(PatternTensor):
    def __init__(self, shape, *, k: int = 0, dtype=None):
        self._k = k
        super().__init__(shape, dtype=dtype, k=self._k)

    def contains(self, i, j) -> bool:
        return self._k > 0 and j == i // self._k


class OddEvenMergeSortPartnerMaskTensor(PatternTensor):
    def __init__(self, shape, *, p: int, k: int, dtype=None):
        self._p = p
        self._k = k
        super().__init__(
            shape,
            ndim=2,
            dtype=dtype,
            default_dtype=np.bool_,
            fill_value=False,
            pattern_value=True,
            p=self._p,
            k=self._k,
        )

    def _is_left(self, i) -> bool:
        axis_size = self.shape[1]
        offset = self._k % self._p
        return (
            i + self._k < axis_size
            and i >= offset
            and (i - offset) % (2 * self._k) < self._k
            and i // (2 * self._p) == (i + self._k) // (2 * self._p)
        )

    def _partner_index(self, i):
        if self._is_left(i):
            return i + self._k
        if i >= self._k and self._is_left(i - self._k):
            return i - self._k
        return i

    def contains(self, i, j) -> bool:
        return j == self._partner_index(i)


class OddEvenMergeSortLowerMaskTensor(PatternTensor):
    def __init__(self, shape, *, p: int, k: int, dtype=None):
        self._p = p
        self._k = k
        super().__init__(
            shape,
            ndim=1,
            dtype=dtype,
            default_dtype=np.bool_,
            fill_value=False,
            pattern_value=True,
            p=self._p,
            k=self._k,
        )

    def _is_left(self, i) -> bool:
        axis_size = self.shape[0]
        offset = self._k % self._p
        return (
            i + self._k < axis_size
            and i >= offset
            and (i - offset) % (2 * self._k) < self._k
            and i // (2 * self._p) == (i + self._k) // (2 * self._p)
        )

    def contains(self, i) -> bool:
        return self._is_left(i)


class OneHotMaskTensor(PatternTensor):
    def __init__(self, shape, *, index: int, dtype=None):
        self._index = operator.index(index)
        super().__init__(
            shape,
            ndim=1,
            dtype=dtype,
            default_dtype=np.bool_,
            fill_value=False,
            pattern_value=True,
            index=self._index,
        )

    def contains(self, i) -> bool:
        return i == self._index


class ParityMaskTensor(PatternTensor):
    def __init__(self, shape, *, parity: int = 0, dtype=None):
        self._parity = parity
        super().__init__(
            shape,
            ndim=1,
            dtype=dtype,
            default_dtype=np.bool_,
            fill_value=False,
            pattern_value=True,
            parity=self._parity,
        )

    def contains(self, i) -> bool:
        return i % 2 == self._parity

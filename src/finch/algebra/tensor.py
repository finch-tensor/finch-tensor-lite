from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..algebra import register_property
from ..symbolic import FType, FTyped, ftype


class TensorFType(FType, ABC):
    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        return len(self.shape_type)

    @property
    @abstractmethod
    def fill_value(self) -> Any:
        """Default value to fill the tensor."""
        ...

    @property
    @abstractmethod
    def element_type(self):
        """Data type of the tensor elements."""
        ...

    @property
    @abstractmethod
    def shape_type(self) -> tuple:
        """Shape type of the tensor. The shape type is a tuple of the index
        types in the tensor. It's the type of each element in tns.shape. It
        should be an actual tuple, rather than a tuple type, so that it can hold
        e.g. dtypes, formats, or types, and so that we can easily index it."""
        ...


class Tensor(FTyped, ABC):
    """
    Abstract base class for tensor-like data structures. Tensors are
    multi-dimensional arrays that can be used to represent data in various
    formats. They support operations such as indexing, slicing, and reshaping,
    and can be used in mathematical computations. This class provides the basic
    interface for tensors to be used with lazy ops in Finch, though more
    advanced interfaces may be required for different backends.
    """

    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        return self.ftype.ndim

    @property
    @abstractmethod
    def shape(self):
        """Shape of the tensor as a tuple."""
        ...

    @property
    @abstractmethod
    def ftype(self) -> TensorFType:
        """FType of the tensor, which may include metadata about the tensor."""
        ...

    @property
    def fill_value(self) -> Any:
        """Default value to fill the tensor."""
        return self.ftype.fill_value

    @property
    def element_type(self):
        """Data type of the tensor elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple:
        """Shape type of the tensor. The shape type is a tuple of the index
        types in the tensor. It's the type of each element in tns.shape. It
        should be an actual tuple, rather than a tuple type, so that it can hold
        e.g. dtypes, formats, or types, and so that we can easily index it."""
        return self.ftype.shape_type


def fill_value(arg: Any) -> Any:
    """The fill value for the given argument.  The fill value is the
    default value for a tensor when it is created with a given shape and dtype,
    as well as the background value for sparse tensors.

    Args:
        arg: The argument to determine the fill value for.

    Returns:
        The fill value for the given argument.

    Raises:
        AttributeError: If the fill value is not implemented for the given type.
    """
    if hasattr(arg, "fill_value"):
        return arg.fill_value
    return ftype(arg).fill_value


def element_type(arg: Any):
    """The element type of the given argument.  The element type is the scalar type of
    the elements in a tensor, which may be different from the data type of the
    tensor.

    Args:
        arg: The tensor to determine the element type for.

    Returns:
        The element type of the given tensor.

    Raises:
        AttributeError: If the element type is not implemented for the given type.
    """
    if hasattr(arg, "element_type"):
        return arg.element_type
    return ftype(arg).element_type


def shape_type(arg: Any) -> tuple:
    """The shape type of the given argument. The shape type is a tuple holding
    the type of each value returned by arg.shape.

    Args:
        arg: The object to determine the shape type for.

    Returns:
        The shape type of the given object.

    Raises:
        AttributeError: If the shape type is not implemented for the given type.
    """
    if hasattr(arg, "shape_type"):
        return arg.shape_type
    return ftype(arg).shape_type


class NDArrayFType(TensorFType):
    """
    A ftype for NumPy arrays that provides metadata about the array.
    This includes the fill value, element type, and shape type.
    """

    def __init__(self, dtype: np.dtype, ndim: int):
        self._dtype = dtype
        self._ndim = ndim

    def __eq__(self, other):
        if not isinstance(other, NDArrayFType):
            return False
        return self._dtype == other._dtype and (
            # TODO: Remove `0` once logic compiler supports ndim inference
            self._ndim == other._ndim or self._ndim == 0 or other._ndim == 0
        )

    def __hash__(self):
        return hash((self._dtype, self._ndim))

    def __repr__(self) -> str:
        return f"NDArrayFType(dtype={repr(self._dtype)}, ndim={self._ndim})"

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def fill_value(self) -> Any:
        return np.zeros((), dtype=self._dtype)[()]

    @property
    def element_type(self):
        return self._dtype.type

    @property
    def shape_type(self) -> tuple:
        return tuple(np.int_ for _ in range(self._ndim))


register_property(
    np.ndarray, "ftype", "__attr__", lambda x: NDArrayFType(x.dtype, x.ndim)
)

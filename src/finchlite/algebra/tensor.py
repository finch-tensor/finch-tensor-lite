from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..symbolic import FType, FTyped


class TensorFType(FType, ABC):
    @property
    def ndim(self) -> np.intp:
        """Number of dimensions of the tensor."""
        return np.intp(len(self.shape_type))

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
    @abstractmethod
    def __call__(self, shape: tuple) -> Tensor:
        """
        Create a tensor instance with the given shape.

        Args:
            shape: The shape of the tensor to create.
        Returns:
            A tensor instance with the specified shape.
        """
        ...

    @abstractmethod
    def from_numpy(self, arr: np.ndarray) -> Tensor: ...


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
    def ndim(self) -> np.intp:
        """Number of dimensions of the tensor."""
        return self.ftype.ndim

    @property
    @abstractmethod
    def ftype(self) -> TensorFType:
        """FType of the tensor, which may include metadata about the tensor."""
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Shape of the tensor."""
        ...


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
    if isinstance(arg, TensorFType):
        return arg.fill_value
    if isinstance(arg, Tensor):
        return arg.ftype.fill_value
    raise AttributeError(f"Expected Tensor or TensorFType, instead got {type(arg)}")


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
    if isinstance(arg, TensorFType):
        return arg.element_type
    if isinstance(arg, Tensor):
        return arg.ftype.element_type
    raise AttributeError(f"Expected Tensor or TensorFType, instead got {type(arg)}")


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
    if isinstance(arg, TensorFType):
        return arg.shape_type
    if isinstance(arg, Tensor):
        return arg.ftype.shape_type
    raise AttributeError(f"Expected Tensor or TensorFType, instead got {type(arg)}")

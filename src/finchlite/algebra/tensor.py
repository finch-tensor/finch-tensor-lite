from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..symbolic import FType, FTyped, ftype


class TensorFType(FType, ABC):
    @property
    def ndim(self) -> np.intp:
        """Number of dimensions of the tensor."""
        return np.intp(len(self.shape_type))

    @property
    @abstractmethod
    def fill_value(self) -> Any:
        """Default value to fill the tensor."""
        ...

    @property
    @abstractmethod
    def element_type(self) -> Any:
        """Data type of the tensor elements."""
        ...

    @property
    @abstractmethod
    def shape_type(self) -> tuple[type, ...]:
        """Shape type of the tensor. The shape type is a tuple of the index
        types in the tensor. It's the type of each element in tns.shape. It
        should be an actual tuple, rather than a tuple type, so that it can hold
        e.g. dtypes, formats, or types, and so that we can easily index it."""
        ...

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

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Shape of the tensor."""
        ...
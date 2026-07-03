from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .devices import normalize_device, serial
from .ftypes import FType, FTyped


class TensorFType(FType, ABC):
    @property
    def ndim(self) -> np.intp:
        """Number of dimensions of the tensor."""
        return np.intp(len(self.shape_type))

    @property
    def size(self):
        size = 1
        for dim in self.shape:
            size *= int(dim)
        return size

    @property
    @abstractmethod
    def fill_value(self) -> Any:
        """Default value to fill the tensor."""
        ...

    @property
    @abstractmethod
    def element_type(self) -> FType:
        """Data type of the tensor elements."""
        ...

    @property
    def dtype(self):
        return self.element_type

    @property
    def device(self):
        return serial()

    @property
    def device_type(self) -> FType:
        return self.device.ftype

    @property
    @abstractmethod
    def shape_type(self) -> tuple[FType, ...]:
        """Shape type of the tensor. The shape type is a tuple of the index
        types in the tensor. It's the type of each element in tns.shape. It
        should be an actual tuple, rather than a tuple type, so that it can hold
        e.g. dtypes, formats, or types, and so that we can easily index it."""
        ...

    @abstractmethod
    def construct(self, shape: tuple) -> Tensor:
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
    def size(self):
        size = 1
        for dim in self.shape:
            size *= int(dim)
        return size

    @property
    @abstractmethod
    def ftype(self) -> TensorFType:
        """FType of the tensor, which may include metadata about the tensor."""
        ...

    @property
    def fill_value(self) -> Any:
        """The fill value for the tensor.  The fill value is the
        default value for a tensor when it is created with a given shape and dtype,
        as well as the background value for sparse tensors.
        """
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        """The element type of the tensor.  The element type is the scalar type of
        the elements in a tensor, which may be different from the data type of the
        tensor.
        """
        return self.ftype.element_type

    @property
    def dtype(self):
        return self.element_type

    @property
    def device(self):
        return getattr(self, "_device", self.ftype.device)

    @property
    def device_type(self) -> FType:
        return self.ftype.device_type

    def to_device(self, device, /, *, stream=None):
        if stream is not None:
            raise ValueError(f"stream argument is not supported; got {stream!r}")
        device = normalize_device(device)
        if device == self.device:
            return self
        raise ValueError(f"device argument is not supported; got {device!r}")

    @property
    def shape_type(self) -> tuple[FType, ...]:
        """Shape type of the tensor. The shape type is a tuple of the index
        types in the tensor. It's the type of each element in tns.shape. It
        should be an actual tuple, rather than a tupleftype, so that it can hold
        e.g. dtypes, formats, or types, and so that we can easily index it."""
        return self.ftype.shape_type

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Shape of the tensor."""
        ...

    @abstractmethod
    def item(self):
        """
        Return the Python scalar stored in a zero-dimensional tensor.

        Indexing tensor objects must keep returning tensor-like values, including for
        zero-dimensional results. `item()` is the explicit escape hatch for code that
        needs the underlying Python scalar value. Implementations should raise
        `ValueError` when called on a non-scalar tensor.
        """
        ...

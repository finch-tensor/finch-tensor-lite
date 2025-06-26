from abc import ABC, abstractmethod
from typing import Any
from ..algebra import query_property, register_property
from ..symbolic import Formattable, Format, format
from dataclasses import dataclass
import numpy as np


class TensorFormat(Format, ABC):
    @property
    @abstractmethod
    def fill_value(self):
        """Default value to fill the tensor."""
        ...
    
    @property
    @abstractmethod
    def element_type(self):
        """Data type of the tensor elements."""
        ...
    
    @property
    @abstractmethod
    def shape_type(self):
        """Shape type of the tensor."""
        ...

class Tensor(Formattable, ABC):
    """
    Abstract base class for tensor-like data structures. Tensors are multi-dimensional
    arrays that can be used to represent data in various formats. They support operations
    such as indexing, slicing, and reshaping, and can be used in mathematical computations.
    """

    @property
    @abstractmethod
    def ndim(self):
        """Number of dimensions of the tensor."""
        ...
    
    @property
    @abstractmethod
    def shape(self):
        """Shape of the tensor as a tuple."""
        ...

    @property
    @abstractmethod
    def format(self) -> TensorFormat:
        """Format of the tensor, which may include metadata about the tensor."""
        ...
    
    @property
    @abstractmethod
    def fill_value(self):
        """Default value to fill the tensor."""
        ...
    
    @property
    @abstractmethod
    def element_type(self):
        """Data type of the tensor elements."""
        ...
    
    @property
    @abstractmethod
    def shape_type(self):
        """Shape type of the tensor."""
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
    if hasattr(arg, "fill_value"):
        return arg.fill_value
    return query_property(arg, "fill_value", "__attr__")


def element_type(arg: Any) -> type:
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
    return query_property(arg, "element_type", "__attr__")

def shape_type(arg: Any) -> type:
    """The shape type of the given argument. The shape type is the type of
    the value returned by arg.shape.

    Args:
        arg: The object to determine the shape type for.

    Returns:
        The shape type of the given object.

    Raises:
        AttributeError: If the shape type is not implemented for the given type.
    """
    if hasattr(arg, "shape_type"):
        return arg.shape_type
    return query_property(arg, "shape_type", "__attr__")



@dataclass
class NDArrayFormat(TensorFormat):
    """
    A format for NumPy arrays that provides metadata about the array.
    This includes the fill value, element type, and shape type.
    """

    _dtype: np.dtype

    @property
    def fill_value(self):
        return np.zeros((), dtype=self._dtype)[()]

    @property
    def element_type(self):
        return self._dtype.type

    @property
    def shape_type(self):
        return tuple(type(dim) for dim in self._dtype.shape)


register_property(
    np.ndarray, "format", "__attr__", lambda x: NDArrayFormat(x.dtype)
)

register_property(
    np.ndarray, "fill_value", "__attr__", lambda x: format(x).fill_value
)

register_property(
    np.ndarray,
    "element_type",
    "__attr__",
    lambda x: format(x).element_type
)

register_property(
    np.ndarray,
    "shape_type",
    "__attr__",
    lambda x: format(x).shape_type
)
from abc import ABC, abstractmethod
from typing import Any

from .. import algebra
from ..algebra import query_property
from ..symbolic import Format, Formattable


class Buffer(Formattable, ABC):
    """
    Abstract base class for buffer-like data structures. Buffers support random access,
    and can be resized. They are used to store data in a way that allows for efficient
    reading and writing of elements.
    """

    @abstractmethod
    def __init__(self, length: int, dtype: type): ...

    @abstractmethod
    def length(self):
        """
        Return the length of the buffer.
        """
        ...

    @property
    def element_type(self):
        """
        Return the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self.format.element_type()

    @property
    def length_type(self):
        """
        Return the type of indices used to access elements in the buffer.
        This is typically an integer type.
        """
        return self.format.length_type()

    @abstractmethod
    def load(self, idx: int): ...

    @abstractmethod
    def store(self, idx: int, val): ...

    @abstractmethod
    def resize(self, len: int):
        """
        Resize the buffer to the new length.
        """
        ...


def length_type(arg: Any):
    """The length type of the given argument. The length type is the type of
    the value returned by len(arg).

    Args:
        arg: The object to determine the length type for.

    Returns:
        The length type of the given object.

    Raises:
        AttributeError: If the length type is not implemented for the given type.
    """
    if hasattr(arg, "length_type"):
        return arg.length_type
    return query_property(arg, "length_type", "__attr__")


def element_type(arg: Any):
    return algebra.element_type(arg)


class BufferFormat(Format):
    """
    Abstract base class for the format of arguments. The format defines how the
    data structures store data, and can construct a data structure with the call method.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Create an instance of an object in this format with the given arguments.
        """
        ...

    @property
    @abstractmethod
    def element_type(self):
        """
        Return the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        ...

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return int

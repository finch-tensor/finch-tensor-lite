from abc import ABC, abstractmethod


class AbstractBuffer(ABC):
    """
    Abstract base class for buffer-like data structures. Buffers support random access,
    and can be resized. They are used to store data in a way that allows for efficient
    reading and writing of elements.
    """

    @abstractmethod
    def __init__(self, length: int, dtype: type):
        pass

    @abstractmethod
    def get_format(self):
        """
        Return the format of the buffer. The format defines how the data is
        organized and accessed.
        """

    @abstractmethod
    def length(self):
        """
        Return the length of the buffer.
        """

    @abstractmethod
    def load(self, idx: int):
        pass

    @abstractmethod
    def store(self, idx: int, val):
        pass

    @abstractmethod
    def resize(self, len: int):
        """
        Resize the buffer to the new length.
        """


class AbstractFormat(ABC):
    """
    Abstract base class for the format of arguments. The format defines how the
    data structures store data, and can construct a data structure with the call method.
    """

    @abstractmethod
    def __call__(self, *args):
        """
        Create an instance of an object in this format with the given arguments.
        """


def isinstanceorformat(x, format_or_type):
    """
    Check if x is an instance of the given format or type.
    """
    if isinstance(format_or_type, AbstractFormat):
        return hasattr(x, "get_format") and x.get_format() == format_or_type
    return isinstance(x, format_or_type)

import ctypes
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

    #    @abstractmethod
    #    def get_format(self):
    #        """
    #        Return the format of the buffer. The format defines how the data is
    #        organized
    #        and accessed.
    #        """
    #        pass

    @abstractmethod
    def load(self, index: int):
        pass

    @abstractmethod
    def store(self, index: int, value):
        pass

    @abstractmethod
    def resize(self, new_length: int):
        """
        Resize the buffer to the new length.
        """


# class AbstractBufferFormat(ABC):
#    """
#    Abstract base class for the format of buffers. The format defines how the data
#    in an AbstractBuffer is organized and accessed.
#    """
#
#    @abstractmethod
#    def make_buffer(self, length):
#        """
#        Create a new buffer of the given length
#        """
#        pass


import numpy as np

from ..finch_assembly.abstract_buffer import Buffer, BufferFormat


class AbstractNumpyBuffer(Buffer):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the Buffer class.
    """

    def __init__(self, arr: np.ndarray):
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("NumPy array must be C-contiguous")
        self.arr = arr

    def length(self):
        return self.arr.size

    def load(self, index: int):
        return self.arr[index]

    def store(self, index: int, value):
        self.arr[index] = value

    def resize(self, new_length: int):
        self.arr = np.resize(self.arr, new_length)


class AbstractNumpyBufferFormat(BufferFormat):
    """
    A format for buffers that uses NumPy arrays. This is an implementation
    of the AbstractFormat class.
    """

    def __init__(self, dtype: type):
        self._dtype = dtype

    def __eq__(self, other):
        if not isinstance(other, BufferFormat):
            return False
        return self._dtype == other._dtype

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return int

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self._dtype

    def __hash__(self):
        return hash(self._dtype)

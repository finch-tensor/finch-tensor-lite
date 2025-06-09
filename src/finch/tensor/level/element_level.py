from ..tensor import Level, LevelFormat
from ...codegen import NumpyBufferFormat
import numpy as np
from abc import ABC, abstractmethod


class ElementLevelFormat(LevelFormat, ABC):

    def __init__(self, fill_value, element_type=None, position_type=None, buffer_format=None):
        """
        Initializes the ElementLevelFormat with an optional fill value.
        Args:
            fill_value: The value used to fill the fibers, or `None` if dynamic.
            position_type: The type of positions within the fibers.
        """
        self.fill_value = fill_value
        if element_type is None:
            element_type = type(fill_value)
        if position_type is None:
            position_type = np.intp
        if buffer_type is None:
            buffer_type = NumpyBufferFormat(element_type)
        self.position_type = position_type
    
    def __call__(self, fmt):
        """
        Creates an instance of ElementLevel with the given format.
        Args:
            fmt: The format to be used for the level.
        Returns:
            An instance of ElementLevel.
        """
        return ElementLevel(fmt)

    def ndims(self):
        return 0

    def fill_value(self):
        return self.fill_value

    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        return self.element_type

    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        return ()

    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        return self.position_type

    def buffer_format(self):
        """
        Returns the format of the buffer used for the fibers.
        """
        return self.buffer_format


class ElementLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    def __init__(self, fmt, val=None):
        """
        Initializes the ElementLevel with a format.
        Args:
            fmt: The format to be used for the level.
        """
        self.fmt = fmt
        if val is None:
            val = fmt.buffer_format()(len=0, dtype=fmt.element_type())
        self.val = val

    def get_format(self):
        return self.fmt
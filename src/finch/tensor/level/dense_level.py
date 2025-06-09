from ..tensor import Level, LevelFormat
from ...codegen import NumpyBufferFormat
import numpy as np
from abc import ABC, abstractmethod


class DenseLevelFormat(LevelFormat, ABC):

    def __init__(self, lvl, dimension_type=None):
        """
        Initializes the DenseLevelFormat with an optional fill value.
        Args:
            fill_value: The value used to fill the fibers, or `None` if dynamic.
            position_type: The type of positions within the fibers.
        """
        self.lvl = lvl
        if dimension_type is None:
            dimension_type = np.intp
        self.dimension_type = dimension_type
    
    def __call__(self, shape):
        """
        Creates an instance of DenseLevel with the given format.
        Args:
            fmt: The format to be used for the level.
        Returns:
            An instance of DenseLevel.
        """
        if not isinstance(shape[0], self.dimension_type):
            raise TypeError(f"Dimension must be of type {self.dimension_type}, got {type(dimension)}")
        lvl = self.lvl(shape[1:])
        return DenseLevel(lvl, shape[0], fmt=self)

    def ndims(self):
        return 1 + self.lvl.ndims()

    def fill_value(self):
        return self.lvl.fill_value()

    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        return self.lvl.element_type()

    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        return (self.dimension_type, *self.lvl.shape_type())

    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        return self.lvl.position_type()

    def buffer_format(self):
        """
        Returns the format of the buffer used for the fibers.
        """
        return self.lvl.buffer_format()


class DenseLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    def __init__(self, fmt, val=None):
        """
        Initializes the DenseLevel with a format.
        Args:
            fmt: The format to be used for the level.
        """
        self.fmt = fmt
        if val is None:
            val = fmt.buffer_format()(len=0, dtype=fmt.element_type())
        self.val = val

    def get_format(self):
        return self.fmt
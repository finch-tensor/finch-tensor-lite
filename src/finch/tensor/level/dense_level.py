from abc import ABC
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tensor import Level, LevelFormat


@dataclass
class DenseLevelFormat(LevelFormat, ABC):
    lvl: Any
    dimension_type: Any = None

    def __post_init__(self):
        if self.dimension_type is None:
            self.dimension_type = np.intp

    def __call__(self, shape):
        """
        Creates an instance of DenseLevel with the given format.
        Args:
            fmt: The format to be used for the level.
        Returns:
            An instance of DenseLevel.
        """
        lvl = self.lvl(shape[1:])
        return DenseLevel(self, lvl, self.dimension_type(shape[0]))

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

    def __init__(self, fmt, lvl, dimension):
        """
        Initializes the DenseLevel with a format.
        Args:
            fmt: The format to be used for the level.
            lvl: The underlying level that this dense level wraps.
        """
        self.fmt = fmt
        self.lvl = lvl
        self.dimension = dimension

    def shape(self):
        return (self.dimension, *self.lvl.shape())

    def get_format(self):
        return self.fmt

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
            shape: The shape to be used for the level. (mandatory)
        Returns:
            An instance of DenseLevel.
        """
        lvl = self.lvl(shape=shape[1:])
        return DenseLevel(self, lvl, self.dimension_type(shape[0]))

    @property
    def ndims(self):
        return 1 + self.lvl.ndims

    @property
    def fill_value(self):
        return self.lvl.fill_value

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        return self.lvl.element_type

    @property
    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        return (self.dimension_type, *self.lvl.shape_type)

    @property
    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        return self.lvl.position_type

    @property
    def buffer_factory(self):
        """
        Returns the format of the buffer used for the fibers.
        """
        return self.lvl.buffer_factory


@dataclass
class DenseLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    format: DenseLevelFormat
    lvl: Any
    dimension: Any

    @property
    def shape(self):
        return (self.dimension, *self.lvl.shape)

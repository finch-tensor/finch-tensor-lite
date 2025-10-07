from abc import ABC
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..fiber_tensor import Level, LevelFType


@dataclass(unsafe_hash=True)
class DenseLevelFType(LevelFType, ABC):
    lvl: Any
    dimension_type: Any = None

    def __post_init__(self):
        if self.dimension_type is None:
            self.dimension_type = np.intp

    def __call__(self, shape):
        """
        Creates an instance of DenseLevel with the given ftype.
        Args:
            shape: The shape to be used for the level. (mandatory)
        Returns:
            An instance of DenseLevel.
        """
        lvl = self.lvl(shape=shape[1:])
        return DenseLevel(self, lvl, self.dimension_type(shape[0]))

    @property
    def ndim(self):
        return 1 + self.lvl.ndim

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
        Returns the ftype of the buffer used for the fibers.
        """
        return self.lvl.buffer_factory


def dense(lvl, dimension_type=None):
    return DenseLevelFType(lvl, dimension_type=dimension_type)


@dataclass
class DenseLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    _format: DenseLevelFType
    lvl: Any
    dimension: Any

    @property
    def shape(self):
        return (self.dimension, *self.lvl.shape)

    @property
    def ftype(self):
        return self._format

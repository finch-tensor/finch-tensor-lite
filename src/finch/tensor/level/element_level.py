from dataclasses import dataclass
from typing import Any

import numpy as np

from ...codegen import NumpyBufferFormat
from ..tensor import Level, LevelFormat


@dataclass
class ElementLevelFormat(LevelFormat):
    fill_value_: Any
    element_type_: type | None = None
    position_type_: type | None = None
    val_format: Any = None

    def __post_init__(self):
        if self.element_type_ is None:
            self.element_type_ = type(self.fill_value_)
        if self.val_format is None:
            self.val_format = NumpyBufferFormat(self.element_type_)
        if self.position_type_ is None:
            self.position_type_ = np.intp
        self.element_type_ = self.val_format.element_type
        self.fill_value_ = self.element_type_(self.fill_value_)

    def __call__(self, shape):
        """
        Creates an instance of ElementLevel with the given format.
        Args:
            fmt: The format to be used for the level.
        Returns:
            An instance of ElementLevel.
        """
        return ElementLevel(self)

    def ndims(self):
        return 0

    def fill_value(self):
        return self.fill_value_

    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        return self.element_type_

    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        return ()

    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        return self.position_type_

    def buffer_format(self):
        """
        Returns the format of the buffer used for the fibers.
        """
        return self.val_format


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

    def shape(self):
        return ()

    def get_format(self):
        return self.fmt

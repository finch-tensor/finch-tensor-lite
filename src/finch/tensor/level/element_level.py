from typing import Any

import numpy as np

from ...codegen import NumpyBufferFormat
from ..tensor import Level, LevelFormat


class ElementLevelFormat(LevelFormat):
    fill_value_: Any
    element_type_: type | None = None
    position_type_: type | None = None
    val_format: Any = None

    def __init__(
        self, fill_value, element_type=None, position_type=None, val_format=None
    ):
        self.fill_value_ = fill_value
        self.element_type_ = element_type or type(fill_value)
        self.position_type_ = position_type or np.intp
        self.val_format = val_format or NumpyBufferFormat(self.element_type_)
        self.element_type_ = self.val_format.element_type
        self.fill_value_ = self.element_type_(self.fill_value_)

    def __eq__(self, other):
        if not isinstance(other, ElementLevelFormat):
            return False
        return (
            self.fill_value_ == other.fill_value_
            and self.element_type_ == other.element_type_
            and self.position_type_ == other.position_type_
            and self.val_format == other.val_format
        )

    def __hash__(self):
        return hash(
            (self.fill_value_, self.element_type_, self.position_type_, self.val_format)
        )

    def __call__(self, shape):
        """
        Creates an instance of ElementLevel with the given format.
        Args:
            fmt: The format to be used for the level.
        Returns:
            An instance of ElementLevel.
        """
        return ElementLevel(self)

    @property
    def ndims(self):
        return 0

    @property
    def fill_value(self):
        return self.fill_value_

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        return self.element_type_

    @property
    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        return ()

    @property
    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        return self.position_type_

    @property
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
            val = fmt.buffer_format(len=0, dtype=fmt.element_type())
        self.val = val

    @property
    def shape(self):
        return ()

    def get_format(self):
        return self.fmt
